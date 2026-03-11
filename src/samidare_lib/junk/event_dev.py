import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")

with app.setup:
    import os
    import marimo_lib.util as molib
    from plotly.subplots import make_subplots
    import pandas as pd
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from pyspark.sql.types import (
        ArrayType, StructType, StructField,
        IntegerType, LongType, DoubleType
    )


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("parquet-read-sample")
        .getOrCreate()
    )
    return (spark,)


@app.cell
def _(mo):
    mo.md(r"""
    # Pulse Reconstruction and Pulse Finder Development
    """)
    return


@app.cell
def _():
    path = "/Users/fendo/Work/Data/root/sampa-minitpc-test/output/10hz_raw.parquet"
    return (path,)


@app.cell
def _(mo):
    mo.md(r"""
    まず読み込んでダンプ
    """)
    return


@app.cell
def _():
    truncate_flag = True
    return (truncate_flag,)


@app.cell
def _(path, spark, truncate_flag):
    df = spark.read.parquet(path)

    df = ( df
        .filter(F.col("start") >= 0)
        .filter(F.col("data_size") == 60)
        # .filter(F.col("chip")==0)
    )

    df.printSchema()
    df.show(10, truncate=truncate_flag)
    return (df,)


@app.cell
def _(df, truncate_flag):
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

    df1.show(10, truncate=truncate_flag)
    return df1, expected_max


@app.cell
def _(df1, expected_max, truncate_flag):
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

    event_stats.show(10, truncate=truncate_flag)
    return (event_stats,)


@app.cell
def _(df1, event_stats, truncate_flag):
    # 4) 正常イベントだけ残す
    df_valid = (
        df1.join(
            event_stats.filter(F.col("is_valid")).select("event_id"),
            on="event_id",
            how="inner",
        )
    )

    df_valid.show(10, truncate=truncate_flag)
    return (df_valid,)


@app.cell
def _(df_valid, truncate_flag):
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

    grouped.show(10, truncate=truncate_flag)
    return (grouped,)


@app.cell
def _(grouped, truncate_flag):
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

    grouped2.show(10, truncate=truncate_flag)
    return (grouped2,)


@app.cell
def _(df, grouped2, truncate_flag):
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

    transposed.show(10, truncate=truncate_flag)

    transposed.printSchema()
    return (transposed,)


@app.cell
def _(transposed, truncate_flag):
    head_start_idx = 0
    head_end_idx = 10
    head_length = head_end_idx - head_start_idx

    tail_start_idx = 53
    tail_end_idx = 63
    tail_length = tail_end_idx - tail_start_idx

    baseline = ( transposed
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

    baseline.show(10, truncate=truncate_flag)
    return (baseline,)


@app.cell
def _(baseline, truncate_flag):
    subtraction = (
        baseline
        .withColumn("baseline", F.round((F.col("baseline_head") + F.col("baseline_tail")) / 2.0, 3))
        .withColumn(
            "pulse_sub",
            F.transform("pulse_raw", lambda x: F.round(x - F.col("baseline"), 3))
        )
    )

    subtraction.show(10, truncate=truncate_flag)
    return (subtraction,)


@app.cell
def _(subtraction, truncate_flag):
    clock_ref = 320e6
    realtime = (
        subtraction
        .withColumn(
            "time_ns",
            F.transform("time_raw", lambda x: F.round( x / F.lit(clock_ref) * F.lit(1e9), 5))
        )
    )

    realtime.show(10, truncate=truncate_flag)
    return (realtime,)


@app.function
def detect_pulses(
    pulse_sub: list[float],
    time_raw: list[any],
    rise_threshold: float,
    fall_threshold: float,
    min_length: int,
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
            "length": fall_idx - rise_idx + 1,
            "rise_time": time_raw[rise_idx],
            "fall_time": time_raw[fall_idx],
            "pulse_segment": pulse_sub[rise_idx:fall_idx + 1],
            "time_segment": time_raw[rise_idx:fall_idx + 1],
            "peak": max(pulse_sub[rise_idx:fall_idx + 1]),
        })

        # 4) 次の探索へ
        i = fall_idx + 1

    return pulses


@app.cell
def _(realtime, truncate_flag):
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
        ])
    )

    detect_pulses_udf = F.udf(
        lambda pulse_sub, time_raw: detect_pulses(
            pulse_sub=pulse_sub,
            time_raw=time_raw,
            rise_threshold=50.0,
            fall_threshold=50.0,
            min_length=3,
        ),
        pulse_schema,
    )

    df_detected = realtime.withColumn(
        "detected_pulses",
        detect_pulses_udf("pulse_sub", "time_ns")
    )


    df_detected.show(10, truncate=truncate_flag)
    return (df_detected,)


@app.cell
def _(df_detected, truncate_flag):
    df_pulses = (
        df_detected
        .withColumn("pulse_obj", F.explode("detected_pulses"))
        .select(
            "chip",
            "channel",
            "timestamp",
            "baseline",
            F.col("pulse_obj.rise_idx").alias("rise_idx"),
            F.col("pulse_obj.fall_idx").alias("fall_idx"),
            F.col("pulse_obj.length").alias("pulse_length"),
            F.col("pulse_obj.rise_time").alias("rise_time"),
            F.col("pulse_obj.fall_time").alias("fall_time"),
            F.col("pulse_obj.peak").alias("pulse_peak"),
            F.col("pulse_obj.pulse_segment").alias("pulse_segment"),
            F.col("pulse_obj.time_segment").alias("time_segment"),
        )
    )

    df_pulses.filter(F.col("channel")==0).show(10, truncate=truncate_flag)
    return


@app.cell
def _(path, truncate_flag):
    import samidare_lib.core.pulse_finder as dev

    dev_spark = dev.get_spark_session()
    dev_df = dev.load_parquet(dev_spark, path)
    dev_df = dev.get_raw_pulses(dev_df)
    dev_df = dev.subtracte_pulses(dev_df)
    dev_df = dev.convert_timestamps(dev_df)
    dev_df = dev.find_pulses(dev_df)

    dev_df.filter(F.col("channel")==0).show(10, truncate=truncate_flag)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
