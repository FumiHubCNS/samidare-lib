import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    from pyspark.sql import functions as F
    import samidare_lib.analysis as analysis
    import catm_lib.util.catmviewer as cat
    import samidare_lib.core as core
    import samidare_lib.util as util
    import marimo_lib.util as molib


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ヒットパターン解析デモ
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from plotly.subplots import make_subplots

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## イベントビルド

    このノートではヒットパターンの描画方法を紹介します。

    このノートでは主に`src/samidare_lib.analysis/hit_pattern.py`を使って解析を行なっています。

    必要なファイルは波形抽出解析済みの`parquet`とマップファイル`csv`です。

    まずはそれらのパスを宣言します。

    ```python
    map_path = 'prm/cat/minitpc.map'
    input = 'output/minitpc_demo_pulse.parquet'
    ```
    """)
    return


@app.cell
def _():
    map_path = 'prm/cat/minitpc.map'
    input = 'output/minitpc_demo_pulse.parquet'
    return input, map_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    次に以下の関数でマップファイルの情報を波形解析済みのデータに埋め込みます。

    引数の`debug`を`True`にするとそれぞれのファイルのデータの一例を表示できます。

    この関数では元の`parquet`に対して`samidare_id`, `tpc_id`, `dev_id`, `pad_id`を追加します。

    これら列の意味については別のノートブック`mapfile_check.py`を参照ください。
    """)
    return


@app.cell
def _(input, map_path):
    df = analysis.hit_pattern.asign_map(input, map_path, debug=True)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    読み込んだらTPCのデバイス番号ごとにデータを抽出し、それぞれでイベント番号を付与します。

    この回生コードでは直前データとの時間差を見ながら、`threshold`にて指定した値を超えるまでのデータを1イベントと定義しています。

    直下のセルが`dev_id`が1のTPCのデータでのイベント付与結果、その下が2のTPCのデータでのイベント付与結果です。
    """)
    return


@app.cell
def _(df):
    _pol1 = 0.0075948252197827
    _pol0 = 0.2907597292526938

    df_tpc1 = df.filter(F.col("dev_id")==1)

    df_evt1 = core.event_builder.build_events(
        df_tpc1,
        time_col='timestamp_ns',
        id_col='event_id',
        threshold=1.5e6,
        device_name='mini1'
    )

    df_evt1 = analysis.hit_pattern.calculate_energy_depoist(
        df_evt1,
        device_name='mini1',
        pol0=_pol0,
        pol1=_pol1
    )
    return (df_evt1,)


@app.cell
def _(df):
    _pol1 = 0.00651260438912663
    _pol0 = 0.24932806669203858

    df_tpc2 = df.filter(F.col("dev_id")==2)

    df_evt2 = core.event_builder.build_events(
        df_tpc2,
        time_col='timestamp_ns',
        id_col='event_id',
        threshold=1.5e6,
        device_name='mini2'
    )

    df_evt2 = analysis.hit_pattern.calculate_energy_depoist(
        df_evt2,
        device_name='mini2',
        pol0=_pol0,
        pol1=_pol1
    )
    return (df_evt2,)


@app.cell
def _(df_evt1):
    util.parquetinfo.md_dump_parquet(df_evt1, n=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    それぞれに別々にイベント番号を付与した後に、それぞれのデータをまとめて再度イベント付与を行います。

    新しいイベント番号は`merged_event_id`に格納されます。
    """)
    return


@app.cell
def _(df_evt1, df_evt2):
    df_re = analysis.hit_pattern.add_merged_events_by_time_window(
        ref_parquet=df_evt1,
        add_parquet=df_evt2,
        ref_dev_name='mini1',
        add_dev_name='mini2',
        window_ns=3e6,
    )


    util.parquetinfo.md_dump_parquet(df_re.orderBy("merged_ts"), n=50)
    return (df_re,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    理想的には`merged_event_id`はMini1, Mini2それぞれで重複があって欲しくないですが、アクシデンタルコインシデンスなどで、時間的に近い領域に複数のイベントが入り込む可能性があります。

    そのため、再度以下で、同じイベント番号同士のデータをグルーピングして、データを配列に変換します
    """)
    return


@app.cell
def _(df_re):
    df_evt1_m = analysis.hit_pattern.build_events(df_re,'mini1')
    df_evt2_m = analysis.hit_pattern.build_events(df_re,'mini2')


    util.parquetinfo.md_dump_parquet(df_evt1_m, n=10)
    return df_evt1_m, df_evt2_m


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    その後、`merged_event_id`をリファレンスとしてデータの結合を行います。
    """)
    return


@app.cell
def _(df_evt1_m, df_evt2_m):
    df_joined_evt = df_evt1_m.join(df_evt2_m, on="merged_event_id", how="left")

    util.parquetinfo.md_dump_parquet(df_joined_evt, n=10)
    return (df_joined_evt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## データ確認


    以下ではアルファ線が両方のTPCに到達したイベントを選択して、データの図示を行います。

    まずは、それぞれのTPCにて落としてエネルギー損失を図示してみます。

    `spark`を使った解析で使える関数などは省略しますが以下のコードでは、これまでのコードで解析したデータから、チップ番号、チャンネル番号、各TPCでの多重度、各々のパッドでのエネルギーの和などを抽出しています。

    また抽出の際は下流側のTPCの多重度が8以上であるイベントにデータをフィルターしています。
    """)
    return


@app.cell
def _(df_joined_evt):
    _df = ( df_joined_evt
        .filter(F.col("event_id_mini2").isNotNull())
        .filter(F.size("de_mini1") == 1)
        .filter(F.size("de_mini2") == 1)
        .select(
            "de_mini1",
            "de_mini2", 
            "charge_mini1", 
            "charge_mini2", 
            "chip_mini1", 
            "chip_mini2", 
            "channel_mini1", 
            "channel_mini2",
            "charge_mini1",
            "charge_mini2"
        )
        .withColumn("de1", F.explode("de_mini1"))
        .withColumn("de2", F.explode("de_mini2"))
        .withColumn("_1", F.explode("charge_mini1"))
        .withColumn("_2", F.explode("charge_mini2"))
        .withColumn("n1", F.size("_1"))
        .withColumn("n2", F.size("_2"))   
        .withColumn("chip1", F.explode("chip_mini1"))
        .withColumn("chip2", F.explode("chip_mini2"))
        .withColumn("channel1", F.explode("channel_mini1"))
        .withColumn("channel2", F.explode("channel_mini2"))
        .withColumn("q1", F.explode("charge_mini1"))
        .withColumn("q2", F.explode("charge_mini2")) 
    )

    df_hit = ( _df
        .select("de1","de2","n1", "n2", "chip1", "chip2", "channel1", "channel2", "q1", "q2")
        .filter(F.col("n2") > 8)
    )
    return (df_hit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    そして以下のコードで描画を行います。

    描画には`Plotly`を使っています。

    ただし、`marimo`のノートブックで使いやすいように加工したライブラリ ([matimo-lib](https://github.com/FumiHubCNS/marimo-lib)) を使用しています。
    """)
    return


@app.cell
def _(df_hit):
    ### ① -> Pandas -> numpy array
    _df =  df_hit.toPandas()

    _c1 = _df ["de1"].to_numpy()
    _c2 = _df ["de2"].to_numpy()


    ### ② Plot data with Plotly
    _fig = molib.plot.get_subplots_object(
        rows = 1,
        cols = 2,
        subplot_titles =[
            "Total dE in Mini TPC 1",
            "Total dE inMini TPC 2"
        ]
    )

    molib.plot.add_sub_plot(
        _fig,1,1,
        data=[_c1 ],
        axes_title=['Energy Deposit [keV]','Counts'],
        func=molib.plot.go_Histogram,
        xrange=[0,300, 3]

    )

    molib.plot.add_sub_plot(
        _fig,1,2,
        data=[_c2],
        axes_title=['Energy Deposit [keV]','Counts'],
        func=molib.plot.go_Histogram,
        xrange=[0,300, 3]
    )

    _fig.update_layout(height=400, width=1000, showlegend=True, title_text="Alpha Particle Data")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    最後にイベントディスプレイを表示してみます。

    以下のコードでヒットパータンの解析を行います。

    手順は以下の通り

    1. マップファイルを読み込みます。
    2. TPCの構造や配置を設定します。
    3. `chip`, `channel`からSAMIDARE内でユニークなIDを付与します。
    4. ヒットパターン描画用のリストをイベント毎に計算し、配列に格納します。
    """)
    return


@app.cell
def _(df_hit, map_path):
    ### ①  ファイル読み込み（あらたな列の付与おも行なっている）
    mapdf = analysis.hit_pattern.load_map(map_path,file_type='pandas')

    ### ② 読み出し電極のパターンとジオメトリを設定
    offset = -3.031088913245535
    pad1 = util.padinfo.get_tpc_info(offset+45)
    pad2 = util.padinfo.get_tpc_info(offset+136.5,False)
    tpcs = util.padinfo.marge_padinfos(pad1,pad2)

    ### ③ samidare_id = 32 * chip + channelという値を付与
    _df = ( df_hit 
        .withColumn(
            "samidare_id1",
            F.expr("transform(arrays_zip(chip1, channel1), x -> x.chip1 * 32 + x.channel1)")
        )
        .withColumn(
            "samidare_id2",
            F.expr("transform(arrays_zip(chip2, channel2), x -> x.chip2 * 32 + x.channel2)")
        )
    )

    ### ④ イベントbyイベントにヒットパターンを計算
    hits_data = []

    for _ev in _df.toLocalIterator():
        _sid = _ev['samidare_id1'] + _ev['samidare_id2']
        _q = _ev['q1'] + _ev['q2']

        _gid = []

        for _id in _sid:
            _gid.append(util.mapfile.get_any_from_mapdf_using_ref(mapdf, refLabel='samidareID', refID=_id, label='gid'))

        # tpcid_arr = _gid
        # de_arr    = _q 

        # reflist = de_arr
        _q_lst = [0] * len(tpcs.ids)

        for _i in range(len(_gid)):
            _q_lst[_gid[_i]] = _q[_i]


        _edges, _colors = cat.get_color_bins(_q_lst, n_bins=20, cmap_name="ocean_r", fmt="hex")
        _color_array = cat.get_color_array(_q_lst, _edges, _colors)


        hits_data.append(_color_array)    
    return hits_data, tpcs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    以下の二つのセルでヒットパターンの描画を行います。

    直下のセルはインタラクティブなスライダーを作っています。

    その下のセルが実際に描画を行なっている関数で、スライダーの位置で描画するイベント番号をできます。
    """)
    return


@app.cell
def _(hits_data, mo):
    idx = mo.ui.slider(start=0, stop=len(hits_data)-1, step=1, value=0, label="event index")
    idx
    return (idx,)


@app.cell
def _(hits_data, idx, tpcs):
    tpcs.show_pads(
        plot_type='map',
        plane='zx',
        color_map=hits_data[idx.value], 
        check_id = True,
        check_size = 7,
        check_data = tpcs.ids,
        canvassize = [12,7],
        return_flag=True
    )
    return


if __name__ == "__main__":
    app.run()
