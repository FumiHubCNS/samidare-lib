import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import samidare_lib as dev
    import sys
    import numpy as np
    import pathlib

    this_file_path = pathlib.Path(__file__).parent


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 解析方法のデモ
    """)
    return


@app.cell
def _():
    import marimo_lib.util as molib
    import subprocess, marimo as mo

    return mo, molib, subprocess


@app.function
def comandline_arg(text='test.py --test 1 --debug 2'):
    return text.split(' ')


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    このノートブックでは`samidare-lib`を使った解析について一例をします。

    デコードの際に使用するコードは`src/samidare_lib/core/decoder.py`です。

    このコードはコマンドライン引数で、実行するデコーダーを選択できます。

    デコーダーの種類については以下で確認できます。
    """)
    return


@app.cell
def _(mo, subprocess):
    _result = subprocess.run(
        comandline_arg("uv run src/samidare_lib/core/decoder.py --help"),
        cwd=str(this_file_path.parent),
        check=True,
        text=True,
        capture_output=True,
    )

    mo.md(f"```text\n{_result.stdout}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    上記のように`event`, `pulse`, `v0`, `v1`があり、対応は以下の通りです。

    | option  | description                                |
    | :----:  | :----------------------------------------: |
    | `event` | イベントビルド用コード(sparkが必要)　　　　　　  |
    | `pulse` | 波形抽出コード(sparkが必要)　　　　　　　 　　　 |
    | `v1`    | 生データデコーダー　　　　　　　　　　　　　　　　 |
    | `v0`    | プロトタイプ生データデコーダー（現在は使用しない） |

    これらのコマンドを指定した後に`--help`または`-h`をつけることでそれぞれのオプションにおける引数も確認できます。
    """)
    return


@app.cell
def _(mo, subprocess):
    _result = subprocess.run(
        comandline_arg("uv run src/samidare_lib/core/decoder.py v1 --help"),
        cwd=str(this_file_path.parent),
        check=True,
        text=True,
        capture_output=True,
    )

    mo.md(f"```text\n{_result.stdout}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 生データデコード
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    まず生データのデコードでについては`README.md`に書いた以下のコマンドを実行すれば良いです。

    ```zsh
    uv run python src/samidare_lib/core/decoder.py v1 -f [FILE_PATH]  --save
    ```

    このノートブックでは以下のセルにて同等の操作を行うことができます。（実際に行う際は`if 0`を`if 1`へ変更してください。）
    """)
    return


@app.cell
def _():
    binary_file_path = pathlib.Path('src/samidare_lib/example/minitpc_demo.bin')
    parquet_filename = 'output/' +  binary_file_path.stem 
    return binary_file_path, parquet_filename


@app.cell
def _(binary_file_path, subprocess):
    if 0:
        _result = subprocess.run(
            comandline_arg(f"uv run python src/samidare_lib/core/decoder.py v1 -f {binary_file_path} --save"),
            cwd=str(this_file_path.parent),
            check=True,
            text=True,
            capture_output=True,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    アウトプットファイルは`.parquet`になっています。

    ここでは`PySpark`を使って開きます。（もちろんなくても開くことは可能です。）

    まず下記のコマンドでセッションを立てます。
    """)
    return


@app.cell
def _():
    spark = dev.core.pulse_finder.get_spark_session()
    return (spark,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    そして以下のコマンドでデータを読み込んで中身の一部を確認できます。
    """)
    return


@app.cell
def _(parquet_filename, spark):
    _df = dev.core.pulse_finder.load_parquet(
        spark,
        parquet_filename + '_raw.parquet'
    )

    dev.util.parquetinfo.md_dump_parquet(_df, n=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    上記のように最初の数ブロックはもともとBufferに入っているデータが流れてくるので、timestampを見てリセットしているブロックまでスキップして読み出します。

    startがブロックの開始位置なので、以下のようにして読み飛ばします。
    """)
    return


@app.cell
def _(parquet_filename, spark):
    df_raw = dev.core.pulse_finder.load_parquet(
        spark,
        parquet_filename + '_raw.parquet',
        start=256
    )

    dev.util.parquetinfo.md_dump_parquet(df_raw, n=10)
    return (df_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    上記のデータを確認してみます。

    生データのデコードでは、1行1データブロックに対応しています。

    `values`の値は各チャンネルの1サンプルであり、サンプル番号はsampleに格納されています。

    そのため描画に際し、以下の処理を行い、図を確認します。
    """)
    return


@app.cell
def _(df_raw, molib):
    _nmax = 100000

    _df = df_raw.select("values","sample").limit(_nmax).toPandas()

    _arr_obj =_df["values"].to_numpy()
    _sam_obj =_df["sample"].to_numpy()

    _lengths = np.fromiter((len(_v) for _v in _arr_obj), dtype=np.int64, count=len(_arr_obj))
    _inds = np.repeat(_sam_obj, _lengths)
    _vals = np.stack(_arr_obj.tolist()).ravel() 


    ############## plot ###############
    _fig = molib.plot.get_subplots_object(1,1)

    molib.plot.add_sub_plot(
        _fig, 1, 1,
        data=[_inds, _vals],
        axes_title=['index', 'sample'],
        func=molib.plot.go_Heatmap,
        bins=[64,100],
        xrange=[0,64],
        yrange=[0,1025],
        logz_option=True
    )

    _fig.update_layout(height=500, width=1000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pulse切り出し解析

    次に波形に切り出し手法について確認していきます。
    """)
    return


@app.cell
def _(mo, subprocess):
    _result = subprocess.run(
        comandline_arg("uv run src/samidare_lib/core/decoder.py pulse --help"),
        cwd=str(this_file_path.parent),
        check=True,
        text=True,
        capture_output=True,
    )

    mo.md(f"```text\n{_result.stdout}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    こちらも同様にデコードできます。
    """)
    return


@app.cell
def _(binary_file_path, subprocess):
    if 0:
        _result = subprocess.run(
            comandline_arg(f"uv run python src/samidare_lib/core/decoder.py pulse -f {binary_file_path} --save --start-pos 256"),
            cwd=str(this_file_path.parent),
            check=True,
            text=True,
            capture_output=True,
        )
    return


@app.cell
def _(parquet_filename, spark):
    df_pulse =  spark.read.parquet(
        parquet_filename + '_pulse.parquet',
    )

    dev.util.parquetinfo.md_dump_parquet(df_pulse, n=10)
    return (df_pulse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    試しに切り出し結果について描画してみます。

    下記のコードでは読み込んだデータを`Pandas`の`DataFrame`に詰めていますが、関数のざっくりとした説明は以下のとおりです。


    - `.select("hoge", "huga")`: 抽出する列名を取得
    - `.limit(number)`: 取得する行数
    """)
    return


@app.cell
def _(df_pulse, molib):
    _nmax = 10000
    _df = df_pulse.select("pulse_segment", "peak", "charge").limit(_nmax).toPandas()

    _pulse = _df['pulse_segment'].to_numpy()
    _peak = _df['peak'].to_numpy()
    _charge = _df['charge'].to_numpy()

    _px = []
    _py = []

    for _i in range(_nmax):
        _py += _pulse[_i]
        _px += list(np.linspace(0,len(_pulse[_i])-1,len(_pulse[_i])))

    ############### plot ###############
    _fig = molib.plot.get_subplots_object(
        rows=1,
        cols=2,
        vertical_spacing=0.15,
        horizontal_spacing=0.175,
        subplot_titles=["Pulse", "peak vs charge"]
    )

    molib.plot.add_sub_plot(
        _fig, 1, 1,
        data=[_px, _py],
        axes_title=['index', 'sample'],
        func=molib.plot.go_Heatmap,
        bins=[20,100],
        xrange=[0,20],
        yrange=[-100,900]
    )

    molib.plot.add_sub_plot(
        _fig, 1, 2,
        data=[_charge, _peak],
        axes_title=['charge', 'max sample'],
        func=molib.plot.go_Heatmap
    )

    molib.plot.align_colorbar(_fig, 20)

    _fig.update_layout(height=500, width=1000, showlegend=True, title_text='Demo')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    取得波形の波高と時間を散布図と表示してみます。
    """)
    return


@app.cell
def _(df_pulse, molib):
    _nmax = 2000
    _df = df_pulse.select("timestamp_ns", "peak").orderBy("timestamp_ns").limit(_nmax).toPandas()

    _t = _df["timestamp_ns"].to_numpy()
    _p = _df["peak"].to_numpy()

    _dt = _t[1:] - _t[:-1]

    _fig = molib.plot.get_subplots_object(1,1)

    molib.plot.add_sub_plot(
        _fig,
        data=[_t, _p],
        axes_title=['timestamp [ns]', 'peak'],
        func=molib.plot.go_Scatter   
    )

    _fig.update_layout(height=300, width=1000, showlegend=True, title_text='Demo')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## イベントビルド（イベント番号付与）

    イベントビルドは`event`オプションで行います。

    このコードで行われるイベントビルドは厳密にはイベント番号を新たな列として追加することに対応しています。

    イベント番号の振り方は、時間でデータをソートしたのちに、直前データとの時間差を見ながら、一定の値を超えるとイベント番号を更新します。
    """)
    return


@app.cell
def _(mo, subprocess):
    _result = subprocess.run(
        comandline_arg("uv run src/samidare_lib/core/decoder.py event --help"),
        cwd=str(this_file_path.parent),
        check=True,
        text=True,
        capture_output=True,
    )

    mo.md(f"```text\n{_result.stdout}\n```")
    return


@app.cell
def _(binary_file_path, subprocess):
    if 0:
        _result = subprocess.run(
            comandline_arg(f"uv run python src/samidare_lib/core/decoder.py event -f {binary_file_path} --save --threshold 3e6"),
            cwd=str(this_file_path.parent),
            check=True,
            text=True,
            capture_output=True,
        )
    return


@app.cell
def _(parquet_filename, spark):
    df_event =  spark.read.parquet(
        parquet_filename + '_event.parquet',
    )

    dev.util.parquetinfo.md_dump_parquet(df_event, n=10)
    return (df_event,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    付与したイベント番号とタイムスタンプを相関を見てみましょう。
    """)
    return


@app.cell
def _(df_event, molib):
    _nmax = 200

    _df = df_event.select("timestamp_ns", "event_id").orderBy("timestamp_ns").limit(_nmax).toPandas()

    _x = _df["timestamp_ns"].to_numpy() / 1e9
    _y = _df["event_id"].to_numpy()

    _fig = molib.plot.get_subplots_object(1,1)

    molib.plot.add_sub_plot(
        _fig,
        data=[_x, _y],
        axes_title=['timestamp [s]', 'event_id'],
        func=molib.plot.go_Scatter   
    )

    _fig.update_layout(height=300, width=1000, title_text='Demo')
    return


if __name__ == "__main__":
    app.run()
