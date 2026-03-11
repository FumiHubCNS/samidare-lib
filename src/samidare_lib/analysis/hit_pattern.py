from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F, types as T

import pandas as pd
import sys
import pathlib

_this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(_this_file_path.parent.parent.parent / "src"))

schema_map = (
    T.StructType()
    .add("sampaNo",   T.IntegerType())
    .add("sampaID",   T.IntegerType())
    .add("samidareID",T.IntegerType())
    .add("tpcID",     T.IntegerType())
    .add("padID",     T.IntegerType())
    .add("gid",       T.StringType())  
)

def get_spark_session(app_name="samidare-hit-pattern"):
    spark = (
        SparkSession.builder
        .config("spark.driver.memory", "8g")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "128") 
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.files.maxPartitionBytes", 32 * 1024 * 1024)
        .appName(app_name)
        .getOrCreate()
        )

    spark.sparkContext.setLogLevel("ERROR")

    return spark


def load_map(
    map_path='prm/cat/minitpc.map',
    file_type='spark',
    spark=None, 
    debug=False    
):
    if file_type == 'spark':
        if spark is None:
            spark = get_spark_session()

        df_map_raw = (spark.read
            .option("header", True)
            .schema(schema_map)
            .csv(map_path)
        )

        df_map = (df_map_raw
            .withColumn(
                "gid_mapped",
                F.when(F.col("gid") == F.lit("G"), F.lit(-1)).otherwise(F.col("gid").cast("int"))
            )
            .select(
                F.col("samidareID").cast("long").alias("map_samidare_id"),
                F.col("gid_mapped"),
                F.col("tpcID").cast("int").alias("dev_id"),
                F.col("padID").cast("int").alias("pad_id")
            )

        )

        if debug:
            print(f"path: {map_path}")
            df_map.show(10)
        
        return df_map
    
    elif file_type == 'pandas':
        mapdf = pd.read_csv(map_path)
        mapdf['padID'] = pd.to_numeric(mapdf['padID'], errors='coerce')
        mapdf['gid'] = pd.to_numeric(mapdf['gid'], errors='coerce')

        mapdf['tpcID'] = mapdf['tpcID'].astype(int)
        mapdf['padID'] = mapdf['padID'].fillna(-1).astype(int)
        mapdf['gid'] = mapdf['gid'].fillna(-1).astype(int)
        mapdf['sampaNo'] = mapdf['sampaNo'].astype(int)
        mapdf['sampaID'] = mapdf['sampaID'].astype(int)
        mapdf['samidareID'] = mapdf['samidareID'].astype(int)

        mapdf = mapdf.reset_index(drop=True)

        return mapdf


def asign_map(
    data_path='output/minitpc_demo_pulse.parquet',
    map_path='prm/cat/minitpc.map',
    spark=None,
    debug=False
):
    if spark is None:
        spark = get_spark_session()


    df_map = load_map(map_path, spark=spark, debug=debug)
    
    df = spark.read.parquet(data_path)  
    df = df.withColumn("samidare_id", F.col("chip") * F.lit(32) + F.col("channel") )
    # df.show(10)


    ###
    df_gid = ( df
        .join(
            F.broadcast(df_map),
            df["samidare_id"] == df_map["map_samidare_id"], 
            "left"
        )
        .drop("map_samidare_id")
        .withColumnRenamed("gid_mapped", "tpc_id")
    )

    if debug:
        print(f"parquet path: {data_path}")
        df_gid.show(10)
    
    return df_gid


def calculate_energy_depoist(
    df=None,
    device_name="mini1",
    pol1=0.00759,
    pol0=0.29076,
    debug=False,

):
    if df is None:
        raise ValueError("DataFrame is required for energy deposition calculation.")

    df = df.withColumn(
        f"de_{device_name}",
        F.col(f"qtot_{device_name}") * F.lit(pol1) + F.lit(pol0)
    )

    if debug:
        df.show(10)

    return df


def add_merged_events_by_time_window(
    ref_parquet,
    add_parquet,
    ref_dev_name="mini1",
    add_dev_name="mini2",
    debug=False,
    *,
    window_ns,
    ts_col: str = "timestamp_ns",
    out_event_col: str = "merged_event_id",
):

    df0 = ref_parquet.withColumn("dev", F.lit(ref_dev_name))
    df1 = add_parquet.withColumn("dev", F.lit(add_dev_name))

    df = df0.unionByName(df1, allowMissingColumns=True)
    df = df.withColumn("merged_ts", F.col(ts_col).cast("long"))
    df = df.withColumn("__rid", F.monotonically_increasing_id())

    w = Window.orderBy(F.col("merged_ts").asc(), F.col("__rid").asc())
    prev_ts = F.lag("merged_ts").over(w)
    dt = F.col("merged_ts") - prev_ts

    is_new = F.when(prev_ts.isNull() | (dt > F.lit(window_ns)), 1).otherwise(0)

    df2 = (
        df
        .withColumn(out_event_col, (F.sum(is_new).over(w) - 1).cast("long"))
        .drop("__rid")
    )

    if debug:
        df2.show(10)

    return df2


def build_events(
    df,
    dev="mini1",
    out_event_col: str = "merged_event_id",
    debug=False
):
    
    cols_to_lift = [
        "event_id",
        "timestamp_ns",
        f"chip_{dev}",
        f"channel_{dev}",
        f"charge_{dev}",
        f"peak_{dev}",
        f"baseline_{dev}",
        f"pulse_{dev}",
        f"time_{dev}",
        f"qtot_{dev}",
        f"de_{dev}",
        "merged_ts",
        "merged_event_id",
    ]

    df_grouped = (
        df
        .orderBy("merged_ts")
        .filter(F.col("dev") == dev)
        .select(*cols_to_lift)
        .groupBy(out_event_col)
        .agg(
            F.sort_array(
                F.collect_list(F.struct(*cols_to_lift))
            ).alias("rows")
        )
        .select(
            out_event_col,
            *[F.expr(f"transform(rows, x -> x.{c})").alias(c) for c in cols_to_lift[:-1]],
        )
        .withColumnRenamed("event_id", f"event_id_{dev}")
        .withColumnRenamed("timestamp_ns", f"timestamp_ns_{dev}")
        .withColumnRenamed("merged_ts", f"merged_ts_{dev}")
    )

    if debug:
        df_grouped.show(10)
        df_grouped.printSchema()

    return df_grouped