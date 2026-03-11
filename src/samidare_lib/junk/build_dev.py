import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")

with app.setup:
    import os
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
    import marimo_lib.util as molib
    return mo, molib


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
    # event build
    """)
    return


@app.cell
def _():
    path = "/Users/fendo/Work/Data/root/sampa-minitpc-test/output/10hz_pulse.parquet"
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
    df.printSchema()
    df.show(10, truncate=truncate_flag)
    return (df,)


@app.cell
def _(df, molib):
    _data = df.select("peak", "charge").toPandas()

    _x = _data["charge"].to_numpy()
    _y = _data["peak"].to_numpy()


    _fig = make_subplots(2,2,subplot_titles=(["a","b","c"]))

    molib.plot.add_sub_plot(_fig,1,1,[_x],func=molib.plot.go_Histogram)
    molib.plot.add_sub_plot(_fig,1,2,[_y],func=molib.plot.go_Histogram)
    molib.plot.add_sub_plot(_fig,2,1,[_x,_y],func=molib.plot.go_Heatmap,logz_option=True)
    molib.plot.align_colorbar(_fig, 20)
    _fig.update_layout(height=800, width=1000, showlegend=False, title_text="")
    return


@app.cell
def _(df, molib):
    _data = ( df
        .filter(F.col("chip")==0)
        # .filter(F.col("channel")==1)
        .orderBy("timestamp_ns")
        .withColumn("timestamp_s", F.round( F.col("timestamp_ns") / 1e9 , 3))
        .limit(200)
        .select("timestamp_s","peak")
        .toPandas()
    )

    _x = _data["timestamp_s"].to_numpy() 
    _y = _data["peak"].to_numpy()


    _fig = make_subplots(1,1,subplot_titles=(["a"]))
    molib.plot.add_sub_plot(_fig,1,1,[_x,_y],func=molib.plot.go_Scatter)
    _fig.update_layout(height=400, width=1000, showlegend=False, title_text="")
    return


@app.cell
def _():
    import samidare_lib.core.event_builder as dev
    return (dev,)


@app.cell
def _(dev, df):
    _df = df.withColumn("timestamp_s", F.round( F.col("timestamp_ns") / 1e9 , 3))
    _df = dev.add_event_id_gap(_df, time_col="timestamp_s",id_col="event_id", threshold=0.01 )
    _df.show(10, truncate=False)

    df_grouped = (_df
        .groupBy("event_id")
          .agg(
              F.first("timestamp_ns").alias("timestamp_ns"),
              F.collect_list("chip").alias("chip"),
              F.collect_list("channel").alias("channel"),
              F.collect_list("charge").alias("charge"),
              F.collect_list("peak").alias("peak"),
              F.collect_list("baseline").alias("baseline"),
              F.collect_list("pulse_segment").alias("pulse"),
              F.collect_list("time_segment").alias("time"),
          )
        .withColumn(
            "qtot",
            F.round(
                F.aggregate(
                    F.col("charge"),
                    F.lit(0.0),
                    lambda acc, x: acc + x
                ),
                3
            )
        )
    )

    df_grouped.show()
    return


@app.cell
def _(dev, df):
    _df = df.withColumn("timestamp_s", F.round( F.col("timestamp_ns") / 1e9 , 3))
    _df = dev.build_events(_df, time_col="timestamp_s",id_col="event_id", threshold=0.01 )
    _df.show(10, truncate=False)
    return


@app.cell
def _(spark):
    spark.stop()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
