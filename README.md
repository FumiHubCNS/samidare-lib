# samidare-lib

このリポジトリは連続読み出し用波形ディジタイザSAMIDAREを使用した取得したデータの解析ツール群です。

## Requirements

- pixi
- (PySpark 用) Java 17 以上
  - 本リポジトリでは pixi により openjdk 17 を環境に含めます

## Install

### pixi のインストール

macOS / Linux:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### リポジトリの取得&インストール

```bash
git clone https://github.com/FumiHubCNS/samidare-lib
cd samidare-lib
pixi install
```

## コマンド実行方法

このライブラリでは`src/samidare_lib/core/decoder.py`を使ってデータのデコードを行います。

> 具体的な入出力ファイルやオプションは `--help` を参照してください。

```bash
# decode (version 1)
pixi run python src/samidare_lib/core/decoder.py v1 -f [filename] --save

# pulse 抽出
pixi run python src/samidare_lib/core/decoder.py pulse -f [filename] --save

# event 構築
pixi run python src/samidare_lib/core/decoder.py event -f [filename] --save
```

### Quick start

```bash
pixi run python src/samidare_lib/core/decoder.py v1 -d -p -s
```

---

## Notebooks (marimo)

`pixi run marimo edit`にてノートブックを開くことができます。

実行方法やその結果得られるデータの表示は`notebook/how_to_use_demo.py`で確認できます。

以下のコマンドでノートブックを開き、ひとまず実行してみましょう。

```zsh 
pixi run marimo edit notebook/how_to_use_demo.py
```


