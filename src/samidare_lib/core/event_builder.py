from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window

import click
import sys
import pathlib
from samidare_lib.core.prm_loader import check_input_file, get_fileinfo

_this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(_this_file_path.parent.parent.parent / "src"))

def get_spark_session(app_name="samidare-event-builder"):
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .getOrCreate()
    )

    return spark

def add_event_id_gap(df, *, time_col: str = "timestamp",
                     threshold: float = 0.001,
                     id_col: str = "event_id"):
    """
    直前レコードとの時間差が threshold を超えたらイベント番号を+1。
    null の時間は event_id を付与しない（nullのまま）。

    df: 任意の DataFrame（time_col を含む）
    time_col: ミリ秒などの昇順で並べる時刻列（数値/小数にキャスト可能であること）
    threshold: 新イベント判定のしきい値（time_col と同じ単位）
    id_col: 付与するイベントID列名
    """
    # タイブレーク用の安定IDを付与
    df1 = df.withColumn("__rid", F.monotonically_increasing_id())

    # event_id は time_col が null でない行だけで計算し、あとで join
    base = (
        df1
        .where(F.col(time_col).isNotNull())
        .select("__rid", F.col(time_col).cast("double").alias(time_col))
    )

    w = Window.orderBy(F.col(time_col).asc(), F.col("__rid").asc())
    dt = F.col(time_col) - F.lag(time_col).over(w)
    new_grp = F.when(dt.isNull() | (dt > F.lit(threshold)), 1).otherwise(0)
    event_id = F.sum(new_grp).over(w).cast("long")

    ids = base.select(F.col("__rid"), event_id.alias(id_col))
    out = df1.join(ids, on="__rid", how="left").drop("__rid")
    return out


def build_events(
    df,
    time_col="timestamp",
    threshold=0.001,
    id_col="event_id",
    device_name="tpc",
    additional_col=None
):
    
    df = add_event_id_gap(df, time_col=time_col, id_col=id_col, threshold=threshold)

    if additional_col is None:
        
        df_grouped = (df
            .groupBy(id_col)
            .agg(
                F.first(time_col).alias(time_col),
                F.collect_list("chip").alias(f"chip_{device_name}"),
                F.collect_list("channel").alias(f"channel_{device_name}"),
                F.collect_list("charge").alias(f"charge_{device_name}"),
                F.collect_list("peak").alias(f"peak_{device_name}"),
                F.collect_list("baseline").alias(f"baseline_{device_name}"),
                F.collect_list("pulse_segment").alias(f"pulse_{device_name}"),
                F.collect_list("time_segment").alias(f"time_{device_name}"),
            )
            .withColumn(
                f"qtot_{device_name}",
                F.round(
                    F.aggregate(
                        F.col(f"charge_{device_name}"),
                        F.lit(0.0),
                        lambda acc, x: acc + x
                    ),
                    3
                )
            )
        )

    return df_grouped


def common_options(func):
    @click.option('--file' , '-f', type=str, default=None, help='file name without .bin')
    @click.option('--time-col', type=str, default="timestamp_ns", help='time column name for event building (default: timestamp_ns)')
    @click.option('--id-col', type=str, default="event_id", help='event ID column name for event building (default: event_id)')
    @click.option('--threshold', type=float, default=0.001, help='time threshold for event building (default: 0.001)')
    @click.option('--debug', is_flag=True, help='enable debug mode (show intermediate DataFrames) (default: False)')
    @click.option('--save', is_flag=True, help='enable saving output to parquet (default: False)')
    
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@common_options
def main(file, time_col, id_col, threshold, debug, save):

    fileinfo = get_fileinfo()
    input_list = check_input_file(fileinfo, file)

    input_path = fileinfo["base_output_path"]  + "/" + pathlib.Path(input_list['found']).stem + "_pulse.parquet"
    output_path = fileinfo["base_output_path"]  + "/" + pathlib.Path(input_list['found']).stem + "_event.parquet"
    
    if not pathlib.Path(input_path).exists():
        print(f"Input file does not exist: {input_path}")
        return
    
    else:
        print(f"input_path: {input_path}")

    spark = get_spark_session()
    df = spark.read.parquet(input_path)

    df = build_events(df, time_col=time_col, id_col=id_col, threshold=threshold)


    if debug:
        df.show(10, truncate=True)
        # df.printSchema()

    if save:
        out_dir = pathlib.Path(output_path).expanduser().resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        
        df.write.mode("overwrite").parquet(output_path)
        print(f"Output saved to: {output_path}")
    
    spark.stop()

if __name__ == '__main__':
    main()