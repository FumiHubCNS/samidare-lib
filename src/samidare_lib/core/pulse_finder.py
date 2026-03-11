from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    ArrayType, StructType, StructField,
    IntegerType, DoubleType
)

import click
import sys
import pathlib
from samidare_lib.core.prm_loader import check_input_file, get_fileinfo

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

def get_spark_session(app_name="samidare-pulse-finder"):
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .getOrCreate()
    )

    return spark

def load_parquet(spark, file_path, start=0, data_size=60):
    df = spark.read.parquet(file_path)

    df = ( df
        .filter(F.col("start") >= start)
        .filter(F.col("data_size") == data_size) 
    )

    return df


def get_raw_pulses(df):
    # 期待する sample 最大値（たとえば 63）
    expected_max = df.agg(F.max("sample").alias("mx")).first()["mx"]

    # 2) start 順に見て sample が巻き戻ったら新イベント
    w = Window.orderBy("start")

    df1 = (
        df
        .withColumn("prev_sample", F.lag("sample").over(w))
        .withColumn(
            "is_new_event",
            F.when(F.col("prev_sample").isNull(), 1)
            .when(F.col("sample") <= F.col("prev_sample"), 1)
            .otherwise(0)
        )
        .withColumn(
            "event_id",
            F.sum("is_new_event").over(
                w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
            ) - 1
        )
        .drop("prev_sample", "is_new_event")
    )

    # +---------+-----+-----------+----+------+--------------------+--------+
    # |data_size|start|  timestamp|chip|sample|              values|event_id|
    # +---------+-----+-----------+----+------+--------------------+--------+
    # |       60|   36|19071822856|   3|    62|[61, 74, 78, 90, ...|       0|
    # |       60|   96|19071822888|   3|    63|[62, 74, 77, 90, ...|       0|
    # |       60|  156|19075020840|   3|     0|[64, 74, 76, 90, ...|       1|
    # |       60|  312|        618|   0|     0|[83, 67, 75, 73, ...|       2|
    # |       60|  372|        638|   2|     0|[72, 72, 70, 77, ...|       3|
    # |       60|  432|        670|   2|     1|[72, 74, 71, 79, ...|       3|
    # |       60|  492|        702|   2|     2|[71, 74, 71, 77, ...|       3|
    # |       60|  552|        734|   2|     3|[71, 73, 70, 77, ...|       3|
    # |       60|  612|        766|   2|     4|[72, 72, 71, 77, ...|       3|
    # |       60|  712|        798|   2|     5|[73, 72, 70, 78, ...|       3|
    # +---------+-----+-----------+----+------+--------------------+--------+

    # 3) イベントごとの正常性チェック
    event_stats = (
        df1.groupBy("event_id")
        .agg(
            F.count("*").alias("n_rows"),
            F.min("sample").alias("min_sample"),
            F.max("sample").alias("max_sample"),
            F.countDistinct("sample").alias("n_distinct_sample"),
            F.first("chip").alias("chip")
        )
        .withColumn(
            "is_valid",
            (F.col("n_rows") == F.lit(expected_max + 1)) &
            (F.col("min_sample") == F.lit(0)) &
            (F.col("max_sample") == F.lit(expected_max)) &
            (F.col("n_distinct_sample") == F.lit(expected_max + 1))
        )
    )

    # +--------+------+----------+----------+-----------------+----+--------+
    # |event_id|n_rows|min_sample|max_sample|n_distinct_sample|chip|is_valid|
    # +--------+------+----------+----------+-----------------+----+--------+
    # |       0|     2|        62|        63|                2|   3|   false|
    # |       1|     1|         0|         0|                1|   3|   false|
    # |       2|     1|         0|         0|                1|   0|   false|
    # |       3|    64|         0|        63|               64|   2|    true|
    # |       4|    64|         0|        63|               64|   3|    true|
    # |       5|    63|         1|        63|               63|   0|   false|
    # |       6|    64|         0|        63|               64|   1|    true|
    # |       7|     1|         0|         0|                1|   3|   false|
    # |       8|    64|         0|        63|               64|   0|    true|
    # |       9|    64|         0|        63|               64|   1|    true|
    # +--------+------+----------+----------+-----------------+----+--------+

    # 4) 正常イベントだけ残す
    df_valid = (
        df1.join(
            event_stats.filter(F.col("is_valid")).select("event_id"),
            on="event_id",
            how="inner",
        )
    )

    # +--------+---------+-----+---------+----+------+--------------------+
    # |event_id|data_size|start|timestamp|chip|sample|              values|
    # +--------+---------+-----+---------+----+------+--------------------+
    # |       3|       60|  372|      638|   2|     0|[72, 72, 70, 77, ...|
    # |       3|       60|  432|      670|   2|     1|[72, 74, 71, 79, ...|
    # |       3|       60|  492|      702|   2|     2|[71, 74, 71, 77, ...|
    # |       3|       60|  552|      734|   2|     3|[71, 73, 70, 77, ...|
    # |       3|       60|  612|      766|   2|     4|[72, 72, 71, 77, ...|
    # |       3|       60|  712|      798|   2|     5|[73, 72, 70, 78, ...|
    # |       3|       60|  772|      830|   2|     6|[71, 74, 71, 78, ...|
    # |       3|       60|  832|      862|   2|     7|[72, 74, 73, 77, ...|
    # |       3|       60|  892|      894|   2|     8|[72, 73, 72, 78, ...|
    # |       3|       60|  952|      926|   2|     9|[71, 75, 70, 79, ...|
    # +--------+---------+-----+---------+----+------+--------------------+

    # 5) sample/timestamp/values をイベント単位で集めて sample 順に整列
    grouped = (
        df_valid.groupBy("event_id")
        .agg(
            F.first("chip").alias("chip"),
            F.collect_list(
                F.struct("sample", "timestamp", "values")
            ).alias("sample_rows")
        )
        .withColumn("sample_rows", F.sort_array("sample_rows"))
    )

    # +--------+----+--------------------+
    # |event_id|chip|         sample_rows|
    # +--------+----+--------------------+
    # |       3|   2|[{0, 638, [72, 72...|
    # |       4|   3|[{0, 624, [62, 74...|
    # |       6|   1|[{0, 644, [65, 89...|
    # |       8|   0|[{0, 24308458, [8...|
    # |       9|   1|[{0, 24308452, [6...|
    # |      10|   2|[{0, 24308446, [7...|
    # |      13|   0|[{0, 56308106, [8...|
    # |      14|   1|[{0, 56308100, [6...|
    # |      15|   2|[{0, 56308094, [7...|
    # |      18|   0|[{0, 88307754, [8...|
    # +--------+----+--------------------+

    # 6) timepoints と rows_of_values を作る
    grouped2 = (
        grouped
        .withColumn(
            "timepoints",
            F.expr("transform(sample_rows, x -> x.timestamp)")
        )
        .withColumn(
            "rows_of_values",
            F.expr("transform(sample_rows, x -> x.values)")
        )
    )

    # +--------+----+--------------------+--------------------+--------------------+
    # |event_id|chip|         sample_rows|          timepoints|      rows_of_values|
    # +--------+----+--------------------+--------------------+--------------------+
    # |       3|   2|[{0, 638, [72, 72...|[638, 670, 702, 7...|[[72, 72, 70, 77,...|
    # |       4|   3|[{0, 624, [62, 74...|[624, 656, 688, 7...|[[62, 74, 76, 89,...|
    # |       6|   1|[{0, 644, [65, 89...|[644, 676, 708, 7...|[[65, 89, 75, 85,...|
    # |       8|   0|[{0, 24308458, [8...|[24308458, 243084...|[[82, 68, 77, 78,...|
    # |       9|   1|[{0, 24308452, [6...|[24308452, 243084...|[[61, 89, 76, 86,...|
    # |      10|   2|[{0, 24308446, [7...|[24308446, 243084...|[[71, 72, 70, 77,...|
    # |      13|   0|[{0, 56308106, [8...|[56308106, 563081...|[[86, 73, 74, 79,...|
    # |      14|   1|[{0, 56308100, [6...|[56308100, 563081...|[[64, 89, 77, 87,...|
    # |      15|   2|[{0, 56308094, [7...|[56308094, 563081...|[[72, 73, 74, 78,...|
    # |      18|   0|[{0, 88307754, [8...|[88307754, 883077...|[[86, 68, 76, 76,...|
    # +--------+----+--------------------+--------------------+--------------------+

    # values 配列長をデータから取得（通常は 32 のはず）
    n_channels = df.select(F.size("values").alias("n")).first()["n"]

    # 7) 転置して 1 channel = 1 row にする
    transposed = (
        grouped2
        .withColumn("channel_idx", F.sequence(F.lit(0), F.lit(n_channels - 1)))
        .withColumn(
            "channel_waveforms",
            F.expr(f"""
                transform(
                    channel_idx,
                    ch -> named_struct(
                        'channel', ch,
                        'waveform',
                        transform(rows_of_values, row -> element_at(row, ch + 1)),
                        'timepoints',
                        timepoints
                    )
                )
            """)
        )
        .select(
            "event_id",
            "chip",
            F.explode("channel_waveforms").alias("cw")
        )
        .select(
            "chip",
            F.col("cw.channel").alias("channel"),
            F.col("cw.timepoints").getItem(0).alias("timestamp"),
            F.col("cw.waveform").alias("pulse_raw"),
            F.col("cw.timepoints").alias("time_raw")
        )
    )

    # +----+-------+---------+--------------------+--------------------+
    # |chip|channel|timestamp|           pulse_raw|            time_raw|
    # +----+-------+---------+--------------------+--------------------+
    # |   2|      0|      638|[72, 72, 71, 71, ...|[638, 670, 702, 7...|
    # |   2|      1|      638|[72, 74, 74, 73, ...|[638, 670, 702, 7...|
    # |   2|      2|      638|[70, 71, 71, 70, ...|[638, 670, 702, 7...|
    # |   2|      3|      638|[77, 79, 77, 77, ...|[638, 670, 702, 7...|
    # |   2|      4|      638|[72, 72, 72, 72, ...|[638, 670, 702, 7...|
    # |   2|      5|      638|[90, 90, 89, 89, ...|[638, 670, 702, 7...|
    # |   2|      6|      638|[68, 70, 70, 69, ...|[638, 670, 702, 7...|
    # |   2|      7|      638|[78, 79, 79, 79, ...|[638, 670, 702, 7...|
    # |   2|      8|      638|[63, 63, 60, 60, ...|[638, 670, 702, 7...|
    # |   2|      9|      638|[82, 84, 84, 83, ...|[638, 670, 702, 7...|
    # +----+-------+---------+--------------------+--------------------+

    return transposed


def subtracte_pulses(
    df,
    head_start_idx = 0,
    head_end_idx = 10,
    tail_start_idx = 53,
    tail_end_idx = 63

    ):

    head_length = head_end_idx - head_start_idx
    tail_length = tail_end_idx - tail_start_idx

    baseline = ( df
        .withColumn(
            "baseline_head",
            F.expr(
                f"""
                aggregate(
                    slice(pulse_raw, {head_start_idx + 1}, {head_length}),
                    cast(0.0 as double),
                    (acc, x) -> acc + x,
                    acc -> acc / {head_length}
                )
                """
            )
        )
        .withColumn(
            "baseline_tail",
            F.expr(
                f"""
                aggregate(
                    slice(pulse_raw, {tail_start_idx + 1}, {tail_length}),
                    cast(0.0 as double),
                    (acc, x) -> acc + x,
                    acc -> acc / {tail_length}
                )
                """
            )
        )
    )

    # +----+-------+---------+--------------------+--------------------+-------------+-------------+
    # |chip|channel|timestamp|           pulse_raw|            time_raw|baseline_head|baseline_tail|
    # +----+-------+---------+--------------------+--------------------+-------------+-------------+
    # |   2|      0|      638|[72, 72, 71, 71, ...|[638, 670, 702, 7...|         71.7|         72.2|
    # |   2|      1|      638|[72, 74, 74, 73, ...|[638, 670, 702, 7...|         73.3|         73.0|
    # |   2|      2|      638|[70, 71, 71, 70, ...|[638, 670, 702, 7...|         70.9|         71.3|
    # |   2|      3|      638|[77, 79, 77, 77, ...|[638, 670, 702, 7...|         77.7|         77.5|
    # |   2|      4|      638|[72, 72, 72, 72, ...|[638, 670, 702, 7...|         72.2|         72.8|
    # |   2|      5|      638|[90, 90, 89, 89, ...|[638, 670, 702, 7...|         89.4|         89.3|
    # |   2|      6|      638|[68, 70, 70, 69, ...|[638, 670, 702, 7...|         69.9|         69.6|
    # |   2|      7|      638|[78, 79, 79, 79, ...|[638, 670, 702, 7...|         78.3|         78.9|
    # |   2|      8|      638|[63, 63, 60, 60, ...|[638, 670, 702, 7...|         61.8|         61.9|
    # |   2|      9|      638|[82, 84, 84, 83, ...|[638, 670, 702, 7...|         81.8|         81.6|
    # +----+-------+---------+--------------------+--------------------+-------------+-------------+

    subtraction = (
        baseline
        .withColumn("baseline", F.round((F.col("baseline_head") + F.col("baseline_tail")) / 2.0, 3))
        .withColumn(
            "pulse_sub",
            F.transform("pulse_raw", lambda x: F.round(x - F.col("baseline"), 3))
        )
    )

    # +----+-------+---------+--------------------+--------------------+-------------+-------------+--------+--------------------+
    # |chip|channel|timestamp|           pulse_raw|            time_raw|baseline_head|baseline_tail|baseline|           pulse_sub|
    # +----+-------+---------+--------------------+--------------------+-------------+-------------+--------+--------------------+
    # |   2|      0|      638|[72, 72, 71, 71, ...|[638, 670, 702, 7...|         71.7|         72.2|   71.95|[0.05, 0.05, -0.9...|
    # |   2|      1|      638|[72, 74, 74, 73, ...|[638, 670, 702, 7...|         73.3|         73.0|   73.15|[-1.15, 0.85, 0.8...|
    # |   2|      2|      638|[70, 71, 71, 70, ...|[638, 670, 702, 7...|         70.9|         71.3|    71.1|[-1.1, -0.1, -0.1...|
    # |   2|      3|      638|[77, 79, 77, 77, ...|[638, 670, 702, 7...|         77.7|         77.5|    77.6|[-0.6, 1.4, -0.6,...|
    # |   2|      4|      638|[72, 72, 72, 72, ...|[638, 670, 702, 7...|         72.2|         72.8|    72.5|[-0.5, -0.5, -0.5...|
    # |   2|      5|      638|[90, 90, 89, 89, ...|[638, 670, 702, 7...|         89.4|         89.3|   89.35|[0.65, 0.65, -0.3...|
    # |   2|      6|      638|[68, 70, 70, 69, ...|[638, 670, 702, 7...|         69.9|         69.6|   69.75|[-1.75, 0.25, 0.2...|
    # |   2|      7|      638|[78, 79, 79, 79, ...|[638, 670, 702, 7...|         78.3|         78.9|    78.6|[-0.6, 0.4, 0.4, ...|
    # |   2|      8|      638|[63, 63, 60, 60, ...|[638, 670, 702, 7...|         61.8|         61.9|   61.85|[1.15, 1.15, -1.8...|
    # |   2|      9|      638|[82, 84, 84, 83, ...|[638, 670, 702, 7...|         81.8|         81.6|    81.7|[0.3, 2.3, 2.3, 1...|
    # +----+-------+---------+--------------------+--------------------+-------------+-------------+--------+--------------------+

    return subtraction

def convert_timestamps(df, ref_label="time_raw", clock_ref = 320e6):
    realtime = (
        df
        .withColumn(
            "time_ns",
            F.transform(ref_label, lambda x: F.round( x / F.lit(clock_ref) * F.lit(1e9), 5))
        )
        .withColumn("timestamp_ns", (F.col("timestamp") / F.lit(clock_ref) * F.lit(1e9)).cast(DoubleType()))
    )

    # +----+-------+---------+--------------------+--------------------+-------------+-------------+--------+--------------------+--------------------+
    # |chip|channel|timestamp|           pulse_raw|            time_raw|baseline_head|baseline_tail|baseline|           pulse_sub|             time_ns|
    # +----+-------+---------+--------------------+--------------------+-------------+-------------+--------+--------------------+--------------------+
    # |   2|      0|      638|[72, 72, 71, 71, ...|[638, 670, 702, 7...|         71.7|         72.2|   71.95|[0.05, 0.05, -0.9...|[1993.75, 2093.75...|
    # |   2|      1|      638|[72, 74, 74, 73, ...|[638, 670, 702, 7...|         73.3|         73.0|   73.15|[-1.15, 0.85, 0.8...|[1993.75, 2093.75...|
    # |   2|      2|      638|[70, 71, 71, 70, ...|[638, 670, 702, 7...|         70.9|         71.3|    71.1|[-1.1, -0.1, -0.1...|[1993.75, 2093.75...|
    # |   2|      3|      638|[77, 79, 77, 77, ...|[638, 670, 702, 7...|         77.7|         77.5|    77.6|[-0.6, 1.4, -0.6,...|[1993.75, 2093.75...|
    # |   2|      4|      638|[72, 72, 72, 72, ...|[638, 670, 702, 7...|         72.2|         72.8|    72.5|[-0.5, -0.5, -0.5...|[1993.75, 2093.75...|
    # |   2|      5|      638|[90, 90, 89, 89, ...|[638, 670, 702, 7...|         89.4|         89.3|   89.35|[0.65, 0.65, -0.3...|[1993.75, 2093.75...|
    # |   2|      6|      638|[68, 70, 70, 69, ...|[638, 670, 702, 7...|         69.9|         69.6|   69.75|[-1.75, 0.25, 0.2...|[1993.75, 2093.75...|
    # |   2|      7|      638|[78, 79, 79, 79, ...|[638, 670, 702, 7...|         78.3|         78.9|    78.6|[-0.6, 0.4, 0.4, ...|[1993.75, 2093.75...|
    # |   2|      8|      638|[63, 63, 60, 60, ...|[638, 670, 702, 7...|         61.8|         61.9|   61.85|[1.15, 1.15, -1.8...|[1993.75, 2093.75...|
    # |   2|      9|      638|[82, 84, 84, 83, ...|[638, 670, 702, 7...|         81.8|         81.6|    81.7|[0.3, 2.3, 2.3, 1...|[1993.75, 2093.75...|
    # +----+-------+---------+--------------------+--------------------+-------------+-------------+--------+--------------------+--------------------+

    return realtime


def find_pulses(
    df,
    rise_threshold=50.0,
    fall_threshold=50.0,
    min_length=3,
    pre_sample=3,
    pos_sample=3
):


    def detect_pulses(
        pulse_sub: list[float],
        time_raw: list[any],
        rise_threshold: float,
        fall_threshold: float,
        min_length: int,
        pre_sample=pre_sample,
        pos_sample=pos_sample
    ) -> list[dict]:
        """
        1本の波形から複数パルスを検出する。
        fall 判定は rise 位置から min_length 後ろから開始する。
        """
        if pulse_sub is None or time_raw is None:
            return []

        n = min(len(pulse_sub), len(time_raw))
        pulses = []
        i = 0

        while i < n:
            # 1) 立ち上がり探索
            while i < n and pulse_sub[i] < rise_threshold:
                i += 1

            if i >= n:
                break

            rise_idx = i

            # 2) 立ち下がり探索開始位置
            search_start = rise_idx + min_length
            if search_start >= n:
                # 長さ不足ならここでは採用しない
                break

            j = search_start

            # 3) 立ち下がり探索
            while j < n and pulse_sub[j] > fall_threshold:
                j += 1

            if j >= n:
                # 最後まで fall しなければ未完了パルスとして無視
                break

            fall_idx = j

            pulses.append({
                "rise_idx": rise_idx,
                "fall_idx": fall_idx,
                "length": fall_idx - rise_idx + 1 + pre_sample + pos_sample,
                "rise_time": time_raw[rise_idx],
                "fall_time": time_raw[fall_idx],
                "pulse_segment": pulse_sub[rise_idx - pre_sample:fall_idx + 1 + pos_sample],
                "time_segment": time_raw[rise_idx - pre_sample:fall_idx + 1 + pos_sample],
                "peak": max(pulse_sub[rise_idx:fall_idx + 1]),
                "charge": sum(pulse_sub[rise_idx - pre_sample:fall_idx + 1 + pos_sample]),
            })

            # 4) 次の探索へ
            i = fall_idx + 1

        return pulses

    pulse_schema = ArrayType(
        StructType([
            StructField("rise_idx", IntegerType(), True),
            StructField("fall_idx", IntegerType(), True),
            StructField("length", IntegerType(), True),
            StructField("rise_time", DoubleType(), True),
            StructField("fall_time", DoubleType(), True),
            StructField("pulse_segment", ArrayType(DoubleType(), True), True),
            StructField("time_segment", ArrayType(DoubleType(), True), True),
            StructField("peak", DoubleType(), True),
            StructField("charge", DoubleType(), True),
        ])
    )

    detect_pulses_udf = F.udf(
        lambda pulse_sub, time_raw: detect_pulses(
            pulse_sub=pulse_sub,
            time_raw=time_raw,
            rise_threshold=rise_threshold,
            fall_threshold=fall_threshold,
            min_length=min_length,
        ),
        pulse_schema,
    )

    df_detected = df.withColumn(
        "detected_pulses",
        detect_pulses_udf("pulse_sub", "time_ns")
    )

    # +----+-------+---------+--------------------+--------------------+-------------+-------------+--------+--------------------+--------------------+---------------+
    # |chip|channel|timestamp|           pulse_raw|            time_raw|baseline_head|baseline_tail|baseline|           pulse_sub|             time_ns|detected_pulses|
    # +----+-------+---------+--------------------+--------------------+-------------+-------------+--------+--------------------+--------------------+---------------+
    # |   2|      0|      638|[72, 72, 71, 71, ...|[638, 670, 702, 7...|         71.7|         72.2|   71.95|[0.05, 0.05, -0.9...|[1993.75, 2093.75...|             []|
    # |   2|      1|      638|[72, 74, 74, 73, ...|[638, 670, 702, 7...|         73.3|         73.0|   73.15|[-1.15, 0.85, 0.8...|[1993.75, 2093.75...|             []|
    # |   2|      2|      638|[70, 71, 71, 70, ...|[638, 670, 702, 7...|         70.9|         71.3|    71.1|[-1.1, -0.1, -0.1...|[1993.75, 2093.75...|             []|
    # |   2|      3|      638|[77, 79, 77, 77, ...|[638, 670, 702, 7...|         77.7|         77.5|    77.6|[-0.6, 1.4, -0.6,...|[1993.75, 2093.75...|             []|
    # |   2|      4|      638|[72, 72, 72, 72, ...|[638, 670, 702, 7...|         72.2|         72.8|    72.5|[-0.5, -0.5, -0.5...|[1993.75, 2093.75...|             []|
    # |   2|      5|      638|[90, 90, 89, 89, ...|[638, 670, 702, 7...|         89.4|         89.3|   89.35|[0.65, 0.65, -0.3...|[1993.75, 2093.75...|             []|
    # |   2|      6|      638|[68, 70, 70, 69, ...|[638, 670, 702, 7...|         69.9|         69.6|   69.75|[-1.75, 0.25, 0.2...|[1993.75, 2093.75...|             []|
    # |   2|      7|      638|[78, 79, 79, 79, ...|[638, 670, 702, 7...|         78.3|         78.9|    78.6|[-0.6, 0.4, 0.4, ...|[1993.75, 2093.75...|             []|
    # |   2|      8|      638|[63, 63, 60, 60, ...|[638, 670, 702, 7...|         61.8|         61.9|   61.85|[1.15, 1.15, -1.8...|[1993.75, 2093.75...|             []|
    # |   2|      9|      638|[82, 84, 84, 83, ...|[638, 670, 702, 7...|         81.8|         81.6|    81.7|[0.3, 2.3, 2.3, 1...|[1993.75, 2093.75...|             []|
    # +----+-------+---------+--------------------+--------------------+-------------+-------------+--------+--------------------+--------------------+---------------+

    df_pulses = (
        df_detected
        .withColumn("pulse_obj", F.explode("detected_pulses"))
        .select(
            "chip",
            "channel",
            "timestamp_ns",
            "baseline",
            F.col("pulse_obj.rise_idx").alias("rise_idx"),
            F.col("pulse_obj.fall_idx").alias("fall_idx"),
            F.col("pulse_obj.length").alias("pulse_length"),
            F.col("pulse_obj.rise_time").alias("rise_time"),
            F.col("pulse_obj.fall_time").alias("fall_time"),
            F.col("pulse_obj.peak").alias("peak"),
            F.col("pulse_obj.charge").alias("charge"),
            F.col("pulse_obj.pulse_segment").alias("pulse_segment"),
            F.col("pulse_obj.time_segment").alias("time_segment"),
        )
    )

    # +----+-------+---------+--------+--------+------------+--------------+--------------+----------+--------------------+--------------------+
    # |chip|channel|timestamp|rise_idx|fall_idx|pulse_length|     rise_time|     fall_time|pulse_peak|       pulse_segment|        time_segment|
    # +----+-------+---------+--------+--------+------------+--------------+--------------+----------+--------------------+--------------------+
    # |   0|      0| 24308458|      16|      20|           5| 7.596553125E7| 7.596593125E7|    799.45|[722.45, 799.45, ...|[7.596553125E7, 7...|
    # |   0|      0| 56308106|      15|      20|           6|1.7596433125E8|1.7596483125E8|    863.35|[64.35, 863.35, 6...|[1.7596433125E8, ...|
    # |   0|      0| 88307754|      15|      19|           5|2.7596323125E8|2.7596363125E8|     896.0|[309.0, 896.0, 54...|[2.7596323125E8, ...|
    # |   0|      0|120307370|      16|      20|           5|3.7596213125E8|3.7596253125E8|    855.75|[580.75, 855.75, ...|[3.7596213125E8, ...|
    # |   0|      0|152307018|      16|      20|           5|4.7596103125E8|4.7596143125E8|    789.85|[789.85, 759.85, ...|[4.7596103125E8, ...|
    # |   0|      0|184306666|      15|      19|           5|5.7595983125E8|5.7596023125E8|     883.2|[135.2, 883.2, 63...|[5.7595983125E8, ...|
    # |   0|      0|216306314|      15|      19|           5|6.7595873125E8|6.7595913125E8|     895.1|[405.1, 895.1, 50...|[6.7595873125E8, ...|
    # |   0|      0|248305930|      16|      20|           5|7.7595763125E8|7.7595803125E8|     828.8|[663.8, 828.8, 38...|[7.7595763125E8, ...|
    # |   0|      0|280305578|      16|      20|           5|8.7595653125E8|8.7595693125E8|     835.0|[835.0, 718.0, 27...|[8.7595653125E8, ...|
    # |   0|      0|312305226|      15|      19|           5|9.7595533125E8|9.7595573125E8|     897.8|[216.8, 897.8, 59...|[9.7595533125E8, ...|
    # +----+-------+---------+--------+--------+------------+--------------+--------------+----------+--------------------+--------------------+

    return df_pulses

def common_options(func):
    @click.option('--file' , '-f', type=str, default=None, help='file name without .bin')
    @click.option('--start-pos', type=int, default=0, help='start position binary file in bytes (default: 0)')
    @click.option('--block-size', type=int, default=60, help='data block size in bytes (default: 60, which corresponds to 1 event of 20 channels x 3 samples)')
    @click.option('--baseline-head-start', type=int, default=0, help='start index of baseline head in samples (default: 0)')
    @click.option('--baseline-head-end', type=int, default=10, help='end index of baseline head in samples (default: 10)')
    @click.option('--baseline-tail-start', type=int, default=53, help='start index of baseline tail in samples (default: 53)')
    @click.option('--baseline-tail-end', type=int, default=63, help='end index of baseline tail in samples (default: 63)')
    @click.option('--timsstamp-label', type=str, default="time_raw", help='reference label for timestamp conversion (default: time_raw)')
    @click.option('--clock-ref', type=float, default=320e6, help='reference clock frequency for timestamp conversion in Hz (default: 320e6)')
    @click.option('--rising-threshold', type=int, default=50, help='threshold for rising edge detection in ADC counts (default: 50)')
    @click.option('--falling-threshold', type=int, default=50, help='threshold for falling edge detection in ADC counts (default: 50)')
    @click.option('--min-length', type=int, default=3, help='minimum pulse length in sample (default: 3)')
    @click.option('--pre-sample', type=int, default=5, help='pre sample (default: 5)')
    @click.option('--post-sample', type=int, default=5, help='post sample (default: 5)')
    @click.option('--debug', is_flag=True, help='enable debug mode (show intermediate DataFrames) (default: False)')
    @click.option('--save', is_flag=True, help='enable saving output to parquet (default: False)')
    

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@common_options
def main(
    file, start_pos, block_size, 
    baseline_head_start, baseline_head_end, baseline_tail_start, baseline_tail_end,
    timsstamp_label, clock_ref,
    rising_threshold, falling_threshold, min_length,
    pre_sample,post_sample,
    debug, save
):

    fileinfo = get_fileinfo()
    input_list = check_input_file(fileinfo, file)

    input_path = fileinfo["base_output_path"]  + "/" + pathlib.Path(input_list['found']).stem + "_raw.parquet"
    output_path = fileinfo["base_output_path"]  + "/" + pathlib.Path(input_list['found']).stem + "_pulse.parquet"
    
    if not pathlib.Path(input_path).exists():
        print(f"Input file does not exist: {input_path}")
        return
    
    else:
        print(f"input_path: {input_path}")

    spark = get_spark_session()
    df = load_parquet(spark, input_path, start_pos, block_size)
    df = get_raw_pulses(df)
    df = subtracte_pulses(df, baseline_head_start, baseline_head_end, baseline_tail_start, baseline_tail_end)
    df = convert_timestamps(df, timsstamp_label, clock_ref)
    df = find_pulses(df, rising_threshold, falling_threshold, min_length, pre_sample, post_sample)

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