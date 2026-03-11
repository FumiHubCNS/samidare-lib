# samidare-lib

このリポジトリは連続読み出し用波形ディジタイザSAMIDAREを使用した取得したデータの解析ツール群です。

## Requirements

* **Python >= 3.13**
* **uv**（依存解決・仮想環境管理）
* （PySpark を使う場合）**JDK（推奨: 17） + Apache Spark**

  * `pyspark>=4.0.0` 
  * Spark 4 系は Java 17が必要

## Installデモ

### uv のインストール

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```



### JDK + Spark　（PySparkを使う場合） ※Homebrew 例

**JDK 17:**

```bash
brew install openjdk@17
```

```bash
sudo ln -sfn "$(brew --prefix)/opt/openjdk@17/libexec/openjdk.jdk" \
  /Library/Java/JavaVirtualMachines/openjdk-17.jdk
```

**Spark:**

```bash
brew install apache-spark
```

動作確認:

```bash
java -version
spark-submit --version
```

### リポジトリの取得&インストール

```bash
git clone https://github.com/FumiHubCNS/samidare-lib
cd samidare-lib
uv sync --group dev
```

## コマンド実行方法

このライブラリでは`src/samidare_lib/core/decoder.py`を使ってデータのデコードを行います。

> 具体的な入出力ファイルやオプションは `--help` を参照してください。

```bash
# decode (version 1)
uv run python src/samidare_lib/core/decoder.py v1 -f [filename] --save

# pulse 抽出
uv run python src/samidare_lib/core/decoder.py pulse -f [filename] --save

# event 構築
uv run python src/samidare_lib/core/decoder.py event -f [filename] --save
```

### Quick start

```bash
uv run python src/samidare_lib/core/decoder.py v1 -d -p -s
```

---

## Notebooks (marimo)

`uv run marimo edit`にてノートブックを開くことができます。

実行方法やその結果得られるデータの表示は`notebook/how_to_use_demo.py`で確認できます。

以下のコマンドでノートブックを開き、ひとまず実行してみましょう。

```zsh 
uv run marimo edit notebook/how_to_use_demo.py
```


