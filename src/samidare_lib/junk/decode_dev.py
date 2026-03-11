import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path
    from typing import Optional, Dict, Iterator, Any, Optional
    import mmap
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    import pyarrow as pa
    import pyarrow.parquet as pq
    import samidare_lib.core.viewer as v

    # tmp
    import os
    import marimo_lib.util as molib
    from plotly.subplots import make_subplots
    import pandas as pd
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # デコーダー再開発
    """)
    return


@app.function
def dump_range(data: bytes, start: int, end: int, width: int = 16):
    """
    data[start:end] を 16進でダンプする。

    Args:
        data: 対象バイト列
        start: 開始オフセット
        end: 終了オフセット（この位置は含まない）
        width: 1行あたりの表示バイト数
    """
    if start < 0:
        start = 0
    if end > len(data):
        end = len(data)
    if start >= end:
        print(f"[empty] start={start}, end={end}")
        return

    chunk = data[start:end]

    print(f"dump range: 0x{start:08x} - 0x{end:08x} ({end - start} bytes)")
    for i in range(0, len(chunk), width):
        row = chunk[i:i+width]
        hex_part = " ".join(f"{b:02x}" for b in row)
        print(f"0x{start + i:08x}: {hex_part}")


@app.function
def shifted_view(data: bytes, bit_shift: int) -> bytes:
    """
    data を bit_shift (0..7) ビット右シフトした論理バイト列として返す。
    """
    if not (0 <= bit_shift <= 7):
        raise ValueError("bit_shift must be in range 0..7")

    if bit_shift == 0:
        return data

    if len(data) < 2:
        return b""

    out = bytearray()
    s = bit_shift
    mask = (1 << s) - 1

    for i in range(len(data) - 1):
        a = data[i]
        b = data[i + 1]
        shifted = ((a >> s) | ((b & mask) << (8 - s))) & 0xFF
        out.append(shifted)

    return bytes(out)


@app.function
def find_one_block_v0(
    data: bytes,
    offset: int = 0,
    bit_shift: int = 0,
    search_limit: Optional[int] = 256,
) -> Optional[Dict]:
    """
    offset 以降から最初の afaf xxxx ... fafa xxxx を1つ探す。
    fafa xxxx が search_limit 内で見つからなければ、
    search_limit を超えて次の afaf の位置を探し、そこを終端として返す。

    Returns:
        {
            "start": afaf の開始位置,
            "end": ブロック終端の次位置,
            "xxxx": afaf直後2バイト,
            "footer_start": fafa の開始位置。なければ None,
            "next_header_start": 次の afaf の開始位置。なければ None,
            "bit_shift": bit_shift,
            "reason": "footer" or "next_afaf" or "eof"
        }
    """
    afaf = b"\xaf\xaf"
    fafa = b"\xfa\xfa"

    pos = data.find(afaf, offset)
    if pos == -1:
        return None

    if pos + 4 > len(data):
        return None

    xxxx = data[pos + 2:pos + 4]

    # fafa xxxx は search_limit 内だけ見る
    tail_raw = data[pos + 2:]
    if search_limit is not None:
        tail_raw = tail_raw[:search_limit]

    tail_shifted = shifted_view(tail_raw, bit_shift)
    target = fafa + xxxx

    rel = tail_shifted.find(target)
    if rel != -1:
        footer_start = pos + 2 + rel
        end = footer_start + 4
        return {
            "start": pos,
            "end": end,
            "xxxx": xxxx,
            "footer_start": footer_start,
            "next_header_start": None,
            "bit_shift": bit_shift,
            "reason": "footer",
        }

    # fafa が見つからなければ、search_limit を超えてよいので次の afaf を探す
    next_pos = data.find(afaf, pos + 2)

    if next_pos != -1:
        return {
            "start": pos,
            "end": next_pos,
            "xxxx": xxxx,
            "footer_start": None,
            "next_header_start": next_pos,
            "bit_shift": bit_shift,
            "reason": "next_afaf",
        }

    # 次の afaf もなければ EOF まで
    return {
        "start": pos,
        "end": len(data),
        "xxxx": xxxx,
        "footer_start": None,
        "next_header_start": None,
        "bit_shift": bit_shift,
        "reason": "eof",
    }


@app.function
def find_one_block(
    data: bytes,
    offset: int = 0,
    bit_shift: int = 0,
    footer_offset_from_header: int = 60,
) -> Optional[Dict]:
    """
    offset 以降から最初の afaf を1つ探す。
    footer は探索せず、afaf 開始位置からちょうど footer_offset_from_header byte 先だけを直接比較する。

    判定:
      shifted_view(data[pos : pos + footer_offset_from_header + 4], bit_shift)
      のうち、[footer_offset_from_header : footer_offset_from_header + 4]
      が b'\\xfa\\xfa' + xxxx と一致すれば footer とみなす。

    Returns:
        {
            "start": afaf の開始位置,
            "end": ブロック終端の次位置,
            "xxxx": afaf直後2バイト,
            "footer_start": fafa の開始位置。なければ None,
            "next_header_start": 次の afaf の開始位置。なければ None,
            "bit_shift": bit_shift,
            "reason": "footer" or "next_afaf" or "eof"
        }
    """
    afaf = b"\xaf\xaf"
    fafa = b"\xfa\xfa"

    pos = data.find(afaf, offset)
    if pos == -1:
        return None

    # afaf xxxx を読むのに最低4byte必要
    if pos + 4 > len(data):
        return None

    xxxx = data[pos + 2:pos + 4]
    target = fafa + xxxx

    # 固定位置だけ比較するために必要な範囲だけ切り出す
    need_end = pos + footer_offset_from_header + 4
    raw = data[pos:need_end]

    # shifted_view をかけて、固定位置の4byteだけ比較
    shifted = shifted_view(raw, bit_shift)

    has_footer = False
    footer_start = None

    if len(shifted) >= footer_offset_from_header + 4:
        candidate = shifted[
            footer_offset_from_header: footer_offset_from_header + 4
        ]
        if candidate == target:
            has_footer = True
            footer_start = pos + footer_offset_from_header

    if has_footer:
        return {
            "start": pos,
            "end": footer_start + 4,
            "xxxx": xxxx,
            "footer_start": footer_start,
            "next_header_start": None,
            "bit_shift": bit_shift,
            "reason": "footer",
        }

    # footer が固定位置に無ければ、次の afaf を探してそこまでを1ブロックとみなす
    next_pos = data.find(afaf, pos + 18)
    if next_pos != -1:
        return {
            "start": pos,
            "end": next_pos,
            "xxxx": xxxx,
            "footer_start": None,
            "next_header_start": next_pos,
            "bit_shift": bit_shift,
            "reason": "next_afaf",
        }

    # 次の afaf もなければ EOF まで
    return {
        "start": pos,
        "end": len(data),
        "xxxx": xxxx,
        "footer_start": None,
        "next_header_start": None,
        "bit_shift": bit_shift,
        "reason": "eof",
    }


@app.cell
def _(mo):
    mo.md(r"""
    ## コード確認
    """)
    return


@app.cell
def _():
    path = "/Users/fendo/Work/Data/root/sampa-minitpc-test/data2025/catmini/10hz.bin"
    data = Path(path).read_bytes()
    return data, path


@app.cell
def _(mo):
    mo.md(r"""
    まずブロック発見関数の確認
    """)
    return


@app.cell
def _(data):
    _hit = find_one_block(data, offset=0, bit_shift=0, footer_offset_from_header=120)
    print(_hit)
    return


@app.cell
def _(data):
    dump_range(data, start=36, end=36+60)
    _hit = find_one_block(data, offset=0, bit_shift=0, footer_offset_from_header=120)
    print(_hit)
    return


@app.cell
def _():
    print(f"{96-36:08x}")
    return


@app.cell
def _(data):
    _offset = 0
    _max_block = 60
    for _i in range(10):
        _hit = find_one_block(data, offset=_offset, bit_shift=0, footer_offset_from_header=_max_block)
        _offset = _hit['end'] if _hit['end'] is not None else _hit['next_header_start'] 
        print(_i, _hit, (_hit['end']-_hit['start']))
    return


@app.cell
def _(data):
    dump_range(data, start=36, end=36+60)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## イテレーション用の関数など
    """)
    return


@app.function
def iter_blocks(
    data: bytes,
    offset: int = 0,
    bit_shift: int = 0,
    search_limit: Optional[int] = 60,
    max_blocks: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    data 内のブロックを順次返すジェネレータ。
    """
    count = 0
    current = offset

    while current < len(data):
        hit = find_one_block(
            data,
            offset=current,
            bit_shift=bit_shift,
            footer_offset_from_header=search_limit,
        )
        if hit is None:
            break

        yield hit

        next_offset = hit.get("end")
        if next_offset is None or next_offset <= current:
            break

        current = next_offset
        count += 1

        if max_blocks is not None and count >= max_blocks:
            break


@app.cell
def _(data):
    for _i, _hit in enumerate(iter_blocks(data, offset=0, bit_shift=0, search_limit=60, max_blocks=10)):
        print(_i, _hit, _hit["end"] - _hit["start"])
    return


@app.function
def decode_10bit_32ch_v0(bitstream: bytes, msb_first: bool = True) -> list[int]:
    """
    bitstream を 10bit 単位で切れるところまで展開する。
    本来は 40 bytes (= 320 bits = 10bit x 32ch) を想定するが、
    デバッグ用途では不足していても、読める分だけ返す。

    Returns:
        10bit 値のリスト
    """
    values = []

    total_bits = len(bitstream) * 8
    n_values = total_bits // 10  # 完全な10bit単位の個数
    if n_values == 0:
        return values

    if msb_first:
        bitbuf = int.from_bytes(bitstream, byteorder="big", signed=False)

        for i in range(n_values):
            shift = total_bits - (i + 1) * 10
            v = (bitbuf >> shift) & 0x3FF
            values.append(v)
    else:
        bitbuf = int.from_bytes(bitstream, byteorder="little", signed=False)

        for i in range(n_values):
            shift = i * 10
            v = (bitbuf >> shift) & 0x3FF
            values.append(v)

    return values


@app.function
def build_error_flag(
    ok_size: bool,
    ok_afaf: bool,
    ok_fafa: bool,
    ok_affa: bool,
    ok_faaf: bool,
    ok_fffa: bool,
) -> int:
    """
    True -> 0, False -> 1 として
    ⑤④③②①⑥ = fffa, faaf, affa, fafa, afaf, size
    の順に 6bit を並べて 10進数化する。
    """
    bits = [
        0 if ok_fffa else 1,  # ⑤
        0 if ok_faaf else 1,  # ④
        0 if ok_affa else 1,  # ③
        0 if ok_fafa else 1,  # ②
        0 if ok_afaf else 1,  # ①
        0 if ok_size else 1,  # ⑥
    ]
    bitstr = "".join(str(b) for b in bits)
    return int(bitstr, 2)


@app.function
def parse_block(data: bytes, start: int, end: int, msb_first: bool = True) -> Dict[str, Any]:
    """
    data[start:end] を1ブロックとして検証・パースする。

    Returns:
        {
            "start": 開始位置,
            "end": 終了位置,
            "size": ブロック長,
            "chip": chip番号,
            "sample": sample番号,
            "timestamp": timestamp値,
            "samples_32ch": 32ch分の10bit値リスト or None,
            "flag": フラグ(10進),
            "flag_bits": '⑤④③②①⑥' の6bit文字列,
            "checks": {...各構造チェック...},
        }
    """
    block = data[start:end]
    size = len(block)

    # サイズチェック
    ok_size = (size == 60)

    # デフォルト値
    chip: Optional[int] = None
    sample: Optional[int] = None
    timestamp: Optional[int] = None
    samples_32ch: Optional[list[int]] = None

    # 固定オフセット参照できる最低限の長さがない場合にも落ちないようにする
    def get_slice(a: int, b: int) -> bytes:
        if a >= len(block):
            return b""
        return block[a:min(b, len(block))]

    def get_byte(i: int) -> Optional[int]:
        return block[i] if i < len(block) else None

    # 構造チェック
    ok_afaf = (get_slice(0, 2) == b"\xaf\xaf")
    ok_affa = (get_slice(4, 6) == b"\xaf\xfa")
    ok_faaf = (get_slice(8, 10) == b"\xfa\xaf")
    ok_fffa = (get_slice(12, 14) == b"\xff\xfa")
    ok_fafa = (get_slice(56, 58) == b"\xfa\xfa")

    # chip, sample
    x1 = get_byte(2)
    y1 = get_byte(3)
    x2 = get_byte(58)
    y2 = get_byte(59)

    if x1 is not None:
        chip = int(x1)
    if y1 is not None:
        sample = int(y1)

    # timestamp
    tttt_b = get_slice(6, 8)
    uuuu_b = get_slice(10, 12)
    vvvv_b = get_slice(14, 16)

    if len(tttt_b) == 2 and len(uuuu_b) == 2 and len(vvvv_b) == 2:
        tttt = int.from_bytes(tttt_b, byteorder="big", signed=False)
        uuuu = int.from_bytes(uuuu_b, byteorder="big", signed=False)
        vvvv = int.from_bytes(vvvv_b, byteorder="big", signed=False)
        timestamp = (tttt << 32) | (uuuu << 16) | vvvv

    # フラグ
    flag = build_error_flag(
        ok_size=ok_size,
        ok_afaf=ok_afaf,
        ok_fafa=ok_fafa,
        ok_affa=ok_affa,
        ok_faaf=ok_faaf,
        ok_fffa=ok_fffa,
    )

    flag_bits = format(flag, "06b")

    # bitstream デコード
    # まず固定長構造が最低限成立しているときだけ行う
    bitstream = get_slice(16, 56)
    # if len(bitstream) == 40:
    try:
        samples_32ch = decode_10bit_32ch_v0(bitstream, msb_first=msb_first)
    except Exception:
        samples_32ch = None

    return {
        "start": start,
        "end": end,
        "size": size,
        "chip": chip,
        "sample": sample,
        "timestamp": timestamp,
        "samples_32ch": samples_32ch,
        "flag": flag,
        "flag_bits": flag_bits,
        "checks": {
            "afaf": ok_afaf,
            "fafa": ok_fafa,
            "affa": ok_affa,
            "faaf": ok_faaf,
            "fffa": ok_fffa,
            "size60": ok_size, 
            "chip_footer_match": (x1 is not None and x2 is not None and x1 == x2),
            "sample_footer_match": (y1 is not None and y2 is not None and y1 == y2),
        },
        "raw": {
            "header_xx": x1,
            "header_yy": y1,
            "footer_xx": x2,
            "footer_yy": y2,
            "tttt_hex": tttt_b.hex() if len(tttt_b) == 2 else None,
            "uuuu_hex": uuuu_b.hex() if len(uuuu_b) == 2 else None,
            "vvvv_hex": vvvv_b.hex() if len(vvvv_b) == 2 else None,
        },
    }


@app.function
def check_block(data, hit):
    result = parse_block(data, hit["start"], hit["end"])
    hex_list = [format(x, '02x') for x in result["samples_32ch"]]
    print('size', result["size"])
    print('chip',hex(result["chip"]))
    print('index',hex(result["sample"]))
    print('time', result["timestamp"])
    print('val',*hex_list)


@app.cell
def _(data):
    _hit = find_one_block(data, offset=0, bit_shift=0, footer_offset_from_header=60)

    if _hit is not None:
        _result = parse_block(data, _hit["start"], _hit["end"])
        _hex_list = [format(x, '02x') for x in _result["samples_32ch"]]
        print('size', _result["size"])
        print('chip',hex(_result["chip"]))
        print('index',hex(_result["sample"]))
        print('time', _result["timestamp"])
        print('val',*_hex_list)
    return


@app.cell
def _(data):
    for _i, _hit in enumerate(iter_blocks(data, offset=0, bit_shift=0, search_limit=60, max_blocks=10)):
        # print(_i, _hit, _hit["end"] - _hit["start"])
        _result = parse_block(data, _hit["start"], _hit["end"])
        _hex_list = [format(x, '02x') for x in _result["samples_32ch"]]
        _val = ' '.join(_hex_list)
        print(f"{_i}, ({hex(_result['chip']):<4},{hex(_result['sample']):<4}) val: {_val:<96}, {_result['size']:<4}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 並列化
    """)
    return


@app.function
def decode_10bit_32ch(bitstream: bytes, msb_first: bool = True) -> list[int]:
    values = []

    total_bits = len(bitstream) * 8
    n_values = total_bits // 10
    if n_values == 0:
        return values

    if msb_first:
        bitbuf = int.from_bytes(bitstream, byteorder="big", signed=False)
        for i in range(n_values):
            shift = total_bits - (i + 1) * 10
            v = (bitbuf >> shift) & 0x3FF
            values.append(v)
    else:
        bitbuf = int.from_bytes(bitstream, byteorder="little", signed=False)
        for i in range(n_values):
            shift = i * 10
            v = (bitbuf >> shift) & 0x3FF
            values.append(v)

    return values


@app.function
def parse_block_bytes(block: bytes, start: int, msb_first: bool = True) -> Dict[str, Any]:
    end = start + len(block)

    def get_slice(a: int, b: int) -> bytes:
        if a >= len(block):
            return b""
        return block[a:min(b, len(block))]

    def get_byte(i: int) -> Optional[int]:
        return block[i] if i < len(block) else None

    size = len(block)

    ok_size = (size == 60)
    ok_afaf = (get_slice(0, 2) == b"\xaf\xaf")
    ok_affa = (get_slice(4, 6) == b"\xaf\xfa")
    ok_faaf = (get_slice(8, 10) == b"\xfa\xaf")
    ok_fffa = (get_slice(12, 14) == b"\xff\xfa")
    ok_fafa = (get_slice(56, 58) == b"\xfa\xfa")

    x1 = get_byte(2)
    y1 = get_byte(3)
    x2 = get_byte(58)
    y2 = get_byte(59)

    chip = int(x1) if x1 is not None else None
    sample = int(y1) if y1 is not None else None

    tttt_b = get_slice(6, 8)
    uuuu_b = get_slice(10, 12)
    vvvv_b = get_slice(14, 16)

    timestamp = None
    if len(tttt_b) == 2 and len(uuuu_b) == 2 and len(vvvv_b) == 2:
        tttt = int.from_bytes(tttt_b, byteorder="big", signed=False)
        uuuu = int.from_bytes(uuuu_b, byteorder="big", signed=False)
        vvvv = int.from_bytes(vvvv_b, byteorder="big", signed=False)
        timestamp = (tttt << 32) | (uuuu << 16) | vvvv

    # ⑤④③②①⑥ = fffa, faaf, affa, fafa, afaf, size60
    bits = [
        0 if ok_fffa else 1,
        0 if ok_faaf else 1,
        0 if ok_affa else 1,
        0 if ok_fafa else 1,
        0 if ok_afaf else 1,
        0 if ok_size else 1,
    ]
    flag_bits = "".join(str(b) for b in bits)
    flag = int(flag_bits, 2)

    bitstream = get_slice(16, 56)
    samples_32ch = decode_10bit_32ch(bitstream, msb_first=msb_first)

    return {
        "start": start,
        "end": end,
        "size": size,
        "chip": chip,
        "sample": sample,
        "timestamp": timestamp,
        "samples_32ch": samples_32ch,
        "flag": flag,
        "flag_bits": flag_bits,
        "checks": {
            "afaf": ok_afaf,
            "fafa": ok_fafa,
            "affa": ok_affa,
            "faaf": ok_faaf,
            "fffa": ok_fffa,
            "size60": ok_size,
            "chip_footer_match": (x1 is not None and x2 is not None and x1 == x2),
            "sample_footer_match": (y1 is not None and y2 is not None and y1 == y2),
        },
    }


@app.function
def parse_block_task(task):
    seq, start, block = task
    result = parse_block_bytes(block, start=start, msb_first=True)
    result["_seq"] = seq
    return result


@app.function
def iter_parsed_blocks_parallel(
    path: str,
    bit_shift: int = 0,
    search_limit: int = 60,
    max_workers: int | None = None,
    max_inflight: int = 1000,
):
    """
    ファイルを逐次スキャンし、見つけたブロックを並列でパースして結果を順次返す。

    - ブロック探索ループ自体は逐次
    - パースだけ ProcessPool で並列
    - max_inflight で未完了ジョブ数を制限
    """
    with open(path, "rb") as f, \
         mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm, \
         ProcessPoolExecutor(max_workers=max_workers) as ex:

        pending = set()
        offset = 0
        seq = 0
        file_size = len(mm)

        while True:
            # queue に空きがあるだけ順次スキャンして投げる
            while len(pending) < max_inflight and offset < file_size:
                hit = find_one_block(
                    mm,
                    offset=offset,
                    bit_shift=bit_shift,
                    search_limit=search_limit,
                )
                if hit is None:
                    break

                start = hit["start"]
                end = hit["end"]

                # 念のため無限ループ防止
                if end is None or end <= start or end <= offset:
                    break

                # 巨大ファイル全体は渡さず、見つけたブロックだけコピーして送る
                block = bytes(mm[start:end])

                fut = ex.submit(parse_block_task, (seq, start, block))
                pending.add(fut)

                offset = end
                seq += 1

            if not pending:
                break

            done, pending = wait(pending, return_when=FIRST_COMPLETED)

            for fut in done:
                yield fut.result()


@app.cell
def _(mo):
    mo.md(r"""
    ## save
    """)
    return


@app.cell
def _():
    def make_parquet_schema():
        return pa.schema([
            pa.field("data_size", pa.int32()),
            pa.field("start", pa.int64()),
            pa.field("timestamp", pa.int64()),
            pa.field("chip", pa.int32()),
            pa.field("sample", pa.int32()),
            pa.field("values", pa.list_(pa.int32())),
        ])


    def results_to_table(results: list[dict]) -> pa.Table:
        rows = {
            "data_size": [],
            "start": [],
            "timestamp": [],
            "chip": [],
            "sample": [],
            "values": [],
        }

        for r in results:
            rows["data_size"].append(r.get("size"))
            rows["start"].append(r.get("start"))
            rows["timestamp"].append(r.get("timestamp"))
            rows["chip"].append(r.get("chip"))
            rows["sample"].append(r.get("sample"))
            rows["values"].append(r.get("samples_32ch") or [])

        return pa.table(rows, schema=make_parquet_schema())
    return make_parquet_schema, results_to_table


@app.cell
def _(make_parquet_schema, results_to_table):
    def iter_parsed_blocks_parallel_save(
        path: str,
        bit_shift: int = 0,
        search_limit: int = 60,
        max_workers: int | None = None,
        max_inflight: int = 1000,
        save_flag: bool = False,
        parquet_path: Optional[str] = None,
        save_batch_size: int = 1000,
    ) -> Iterator[dict]:
        """
        ファイルを逐次スキャンし、見つけたブロックを並列でパースして結果を順次返す。

        save_flag=True の場合、結果を parquet に追記保存する。
        保存列:
          - data_size
          - start
          - timestamp
          - chip
          - sample
          - values
        """
        if save_flag and not parquet_path:
            raise ValueError("save_flag=True のときは parquet_path を指定してください")

        writer = None
        save_buffer: list[dict] = []

        try:
            with open(path, "rb") as f, \
                 mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm, \
                 ProcessPoolExecutor(max_workers=max_workers) as ex:

                pending = set()
                offset = 0
                seq = 0
                file_size = len(mm)

                if save_flag:
                    writer = pq.ParquetWriter(parquet_path, make_parquet_schema())

                while True:
                    # 空きがあるだけ順次スキャンして投げる
                    while len(pending) < max_inflight and offset < file_size:
                        hit = find_one_block(
                            mm,
                            offset=offset,
                            bit_shift=bit_shift,
                            footer_offset_from_header=search_limit,
                        )
                        if hit is None:
                            break

                        start = hit["start"]
                        end = hit["end"]

                        # 無限ループ防止
                        if end is None or end <= start or end <= offset:
                            break

                        # 見つけたブロックだけ切り出して worker へ渡す
                        block = bytes(mm[start:end])

                        fut = ex.submit(parse_block_task, (seq, start, block))
                        pending.add(fut)

                        offset = end
                        seq += 1

                    if not pending:
                        break

                    done, pending = wait(pending, return_when=FIRST_COMPLETED)

                    for fut in done:
                        result = fut.result()

                        if save_flag:
                            save_buffer.append(result)
                            if len(save_buffer) >= save_batch_size:
                                table = results_to_table(save_buffer)
                                writer.write_table(table)
                                save_buffer.clear()

                        yield result

                # 余りを flush
                if save_flag and save_buffer:
                    table = results_to_table(save_buffer)
                    writer.write_table(table)
                    save_buffer.clear()

        finally:
            if writer is not None:
                writer.close()
    return (iter_parsed_blocks_parallel_save,)


@app.cell
def _(iter_parsed_blocks_parallel_save, path):
    if 0:
        for _result in iter_parsed_blocks_parallel_save(
            path,
            bit_shift=0,
            search_limit=60,
            max_workers=4,
            max_inflight=1000,
            save_flag=True,
            parquet_path="output/parsed_blocks.parquet",
            save_batch_size=2000,
        ):
            pass
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## データ確認
    """)
    return


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
def _(spark):
    df = spark.read.parquet("output/parsed_blocks.parquet")

    df.printSchema()
    df.show(10, truncate=False)
    return (df,)


@app.cell
def _(df):
    _df = ( df
        .filter(F.col("start") >= 312)
        .orderBy("start")
    )

    # _dum = _df.toPandas()
    # _s = _dum["start"].to_numpy()
    # _d = _dum["data_size"].to_numpy()

    # print(_s[-1],_d[-1])

    _df.show(100, truncate=False)
    return


@app.cell
def _(df, path):
    _df = df.toPandas()
    _pos = _df["start"].to_numpy()
    _val = _df["data_size"].to_numpy()


    size = os.path.getsize(path)
    print(f"original: {size} bytes")
    print(f"decoded : {_pos[0]+_val.sum()} bytes")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ダメなデータの情報を見てみる。
    """)
    return


@app.cell
def _(df):
    _df = ( df
        .filter(F.col("start") >= 312)
        .filter(F.col("data_size") != 60)
        .toPandas()
    )
    _df
    return


@app.cell
def _(mo):
    mo.md(r"""
    割合などを見ていく。
    """)
    return


@app.cell
def _(df):
    _df_no  = ( df.filter(F.col("start") >= 312).filter(F.col("data_size") != 60).toPandas())
    _df_yes = ( df.filter(F.col("start") >= 312).filter(F.col("data_size") == 60).toPandas())

    _no = len(_df_no)
    _yes = len(_df_yes)

    print( _no / (_yes + _no) * 100)
    return


@app.cell
def _(mo):
    mo.md(r"""
    クラウディオ曰く、失敗したら再度データの転送を試みるらしい。
    なので、次のブロックを見てみる。
    """)
    return


@app.cell
def _(data, df):
    _df = ( df
        .filter(F.col("start") >= 312)
        .filter(F.col("data_size") != 60)
        .toPandas()
    )


    for _i, _row in enumerate(_df.itertuples(index=False)):
        # print(_i, _row.start, _row.data_size)
        print(f"============================ {_i} ============================")
        dump_range(data, start=_row.start, end=_row.start+_row.data_size)
        print(f"---------- next  ----------")
        dump_range(data, start=_row.start+_row.data_size, end=_row.start+_row.data_size+60)

        if _i == 1:
            break
    return


@app.cell
def _(df):
    if 0:
        _df = (df.filter(F.col("start") >= 312).filter(F.col("data_size") != 60).toPandas())

        _chk = 0

        for _i, _row in enumerate(_df.itertuples(index=False)):
            _curr = df.filter(F.col("start") == _row.start).toPandas()
            _next = df.filter(F.col("start") == _row.start + _row.data_size).toPandas()

            _f1 = ((_curr["chip"][0] == _next["chip"][0] ) * (_curr["sample"][0] == _next["sample"][0]))
            _n = len(_curr["values"][0])
            _f2 = (_curr["values"][0][:_n] == _next["values"][0][:_n])

            _flag = _f1 *_f2
            _flag = 1 if _flag == True else 0
            _chk += 1 - _flag

        print(_chk)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## イベントディスプレイ用
    """)
    return


@app.cell
def _(iter_parsed_blocks_parallel_save):
    _path = "/Users/fendo/Work/Data/root/sampa-minitpc-test/data2025/0819/20250919_test_po_20mVfC_160ns_64sample_16presample_001.bin"
    _i = 0

    for _result in iter_parsed_blocks_parallel_save(
        _path,
        bit_shift=0,
        search_limit=60,
        max_workers=8,
        max_inflight=1000,
    ):

        print(_result)
        # _timestamp = _result["timestamp"]
        # _samples_32ch = _result["samples_32ch"]
        # _timestamp_32ch = [_timestamp] * len(_samples_32ch)


        # print("timestamp_23ch =", _timestamp_32ch)
        # print("samples_32ch =", _samples_32ch)

        _i += 1

        if _i == 1:
            break
    return


@app.cell
def _(iter_parsed_blocks_parallel_save):
    if 1:
        _path = "/Users/fendo/Work/Data/root/sampa-minitpc-test/data2025/catmini/10hz.bin"

        _proc, _data_queue = v.launch_pyqtgraph_stream_4x32_queue(
            max_points=64,
            interval_ms=300,
        )

        _i = 0

        for _result in iter_parsed_blocks_parallel_save(
            _path,
            bit_shift=0,
            search_limit=60,
            max_workers=8,
            max_inflight=1000,
        ):

            if _result["size"] == 60:
                _timestamp = _result["timestamp"]
                _samples_32ch = _result["samples_32ch"]
                _chip = _result["chip"]

                v.push_board32_point(
                    _data_queue,
                    x=_timestamp,
                    board=_chip,
                    values_32ch=_samples_32ch,
                )

                _i += 1

            if _i == 100000:
                break

        _data_queue.put(None)
    return


@app.cell
def _():


    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
