import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import pandas as pd
    import catm_lib.util.catmviewer as catview
    import samidare_lib.util as util


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Mapファイル&ジオメトリ確認用サンプルコード

    実機で取得したデータを解析する際に、イベントディスプレイを表示させるためのクラスの使い方を書いておきます。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 読み出し電極の描画用のジオメトリを作成する。

    `samidare-lib`では[catm-lib](https://github.com/FumiHubCNS/catm-lib)のコード群が使えるようになっています。

    `src/util/padinfo.py`に、`catm-lib`を使って読み出し電極の位置情報を格納しているクラスを生成できる関数が用意されています。

    例えば以下のようにして配置を生成、表示できます。
    """)
    return


@app.cell
def _():
    offset = -3.031088913245535
    pad1 = util.padinfo.get_tpc_info(offset-5)
    pad2 = util.padinfo.get_tpc_info(offset+5,False)
    tpcs = util.padinfo.marge_padinfos(pad1,pad2)

    tpcs.show_pads(return_flag=True)
    return (tpcs,)


@app.cell
def _(mo):
    mo.md(r"""
    ## マップファイルを読み込み、

    以下のコードでファイルを読み込み、データの前処理を行います。

    書く列名の意味については以下のとおりです。

    | 列名        | 説明 |
    | :--------: | :---: |
    | sampaNo    | SAMIDARE内のSAMPAチップの番号　|
    | sampaID    | 各SAMPAチップのchannel番号 |
    | samidareID | SAMIDAREのチャンネル番号 (sampaNo * 32 + sampaID )|
    | tpcID      | ジオメトリクラスに追加した読み出しパット群(tpc)の識別番号 |
    | padID      | 各パッドのID |
    | gid        | ジオメトリクラスでのグローバルID ( (tpcID(i)-1) * max(padID(i)) + padID(i) |
    """)
    return


@app.cell
def _():
    tpc_map = 'prm/cat/minitpc.map'

    mapdf = pd.read_csv(tpc_map)
    mapdf['padID'] = pd.to_numeric(mapdf['padID'], errors='coerce')
    mapdf['gid'] = pd.to_numeric(mapdf['gid'], errors='coerce')

    mapdf['tpcID'] = mapdf['tpcID'].astype(int)
    mapdf['padID'] = mapdf['padID'].fillna(-1).astype(int)
    mapdf['gid'] = mapdf['gid'].fillna(-1).astype(int)
    mapdf['sampaNo'] = mapdf['sampaNo'].astype(int)
    mapdf['sampaID'] = mapdf['sampaID'].astype(int)
    mapdf['samidareID'] = mapdf['samidareID'].astype(int)

    mapdf = mapdf.reset_index(drop=True)

    mapdf.head(10)
    return (mapdf,)


@app.cell
def _(mo):
    mo.md(r"""
    `tpcs.show_pads()`では書くパッドに色をつけたり、数値を表示したりできます。

    ただし、その際には、gidの昇順に値を格納したリストを渡す必要があります。

    そこで以下のコードで対応リストを作成します。
    """)
    return


@app.cell
def _(mapdf, mo):
    tpc_chip = []
    tpc_channel = []
    tpc_dev =[]
    rows = []

    for _i in range(120):
        _chip = util.mapfile.get_any_from_mapdf_using_ref(mapdf, refLabel='gid', refID=_i, label='sampaNo')
        _channel = util.mapfile.get_any_from_mapdf_using_ref(mapdf, refLabel='gid', refID=_i, label='sampaID')
        _id = util.mapfile.get_any_from_mapdf_using_ref(mapdf, refLabel='gid', refID=_i, label='samidareID')
        _dev = util.mapfile.get_any_from_mapdf_using_ref(mapdf, refLabel='gid', refID=_i, label='tpcID')
        tpc_chip.append(_chip)
        tpc_channel.append(_channel)
        tpc_dev.append(_dev)

        rows.append({
            "tpc_id": _i,
            "samidare_id": _id,
            "sampa_chip": _chip,
            "sampa_channel": _channel,
            "dev_id": _dev
        })

    _df = pd.DataFrame(rows)
    mo.ui.table(_df)
    return tpc_channel, tpc_chip, tpc_dev


@app.cell
def _(tpc_channel, tpc_chip, tpc_dev, tpcs):
    textdict = { "tpcid" : tpcs.ids , "chipid" : tpc_chip , "sampach" : tpc_channel, "dev": tpc_dev}
    return (textdict,)


@app.cell
def _(mo):
    mo.md(r"""
    あとは作ったリストを使用し、`catview.get_color_array`, `get_color_list`を実行しのちに戻り値を渡せば描画できます。

    ここでは引数として以下の値を入れていますが、対応関係は下記の表の通り。


    | 引数名      | 値                | 説明              |
    | :--------: | :---------------: | :--------------: |
    | check_id   | True              | 数値の表示オプション |
    | check_size | 13                | 数値の文字サイズ    |
    | plot_type  | 'map'             | 描画のスタイル      |
    | color_map  | color_array       | 色分布の値リスト    |
    | check_data | textdict["tpcid"] | 数値リスト         |
    """)
    return


@app.cell
def _(textdict, tpcs):
    cehck_list = textdict["tpcid"]

    color_array = catview.get_color_array(
        cehck_list,
        *catview.get_color_list(
            cehck_list,
            cmap_name="rainbow",
            fmt="hex"
        )
    )

    tpcs.show_pads(
        check_id=True, 
        check_size=13, 
        canvassize=[12,4],
        plot_type='map',
        color_map=color_array, 
        check_data=textdict["tpcid"], 
        return_flag=True
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## その他

    読み出し電極の作り方の詳細は[hoge](demo)に任せますが、頑張ると以下のようなデモが作れるようになります。
    """)
    return


@app.cell
def _():
    import catm_lib as catlib

    catlib.readoutpad.catm.check_pad_view()
    return


if __name__ == "__main__":
    app.run()
