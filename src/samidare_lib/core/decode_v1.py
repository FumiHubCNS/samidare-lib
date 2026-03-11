import pathlib
from typing import Optional, Dict, Iterator, Any
import mmap
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import pyarrow as pa
import pyarrow.parquet as pq
import samidare_lib.core.viewer as v
import click
import sys

from samidare_lib.core.prm_loader import check_input_file, get_fileinfo

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

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


def dump_range(data: bytes, start: int, end: int, width: int = 16, summary: bool = True):
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

    if summary:
        print(f"dump range: 0x{start:08x} - 0x{end:08x} ({end - start} bytes)")
    
    for i in range(0, len(chunk), width):
        row = chunk[i:i+width]
        hex_part = " ".join(f"{b:02x}" for b in row)
        print(f"0x{start + i:08x}: {hex_part}")


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


def parse_block_task(task):
    seq, start, block = task
    result = parse_block_bytes(block, start=start, msb_first=True)
    result["_seq"] = seq
    return result


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

            if save_flag and save_buffer:
                table = results_to_table(save_buffer)
                writer.write_table(table)
                save_buffer.clear()

    finally:
        if writer is not None:
            writer.close()
    

def common_options(func):
    @click.option('--file' , '-f', type=str, default=None, help='file name without .bin')
    @click.option('--plot' , '-p', is_flag=True, help='plot flag (default: False)')
    @click.option('--dump' , '-d', is_flag=True, help='dump flag (default: False)')
    @click.option('--save' , '-s', is_flag=True, help='output file generation flag (default: False)')
    @click.option('--max-blocks', type=int, default=-1, help='maximum number of blocks to process for testing (default: -1 for no limit)')
    @click.option('--plot-size', type=int, default=64, help='number of points to show in the plot window (default: 64)')
    @click.option('--plot-interval', type=int, default=100, help='plot interval in milliseconds (default: 100)')

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@common_options
def main(file, plot, dump, save, max_blocks, plot_size, plot_interval):


    fileinfo = get_fileinfo()
    input_list = check_input_file(fileinfo, file)
    output_path = fileinfo["base_output_path"]  + "/" + pathlib.Path(input_list['found']).stem + "_raw.parquet"

    if plot:
        proc, data_queue = v.launch_pyqtgraph_stream_4x32_queue(
            max_points = plot_size,
            interval_ms = plot_interval,
        )

    print(f"found input file: {input_list['found']}")
    
    if save:
        out_dir = pathlib.Path(output_path).expanduser().resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"output parquet path: {output_path}")

    max_blocks

    _i = 0

    if dump:
        data = pathlib.Path(input_list["found"]).read_bytes()

    for _result in iter_parsed_blocks_parallel_save(
        input_list["found"],
        bit_shift=0,
        search_limit=60,
        max_workers=4,
        max_inflight=1000,
        save_flag=save,
        parquet_path=output_path,
        save_batch_size=2000,
    ):
        if plot:
            if _result["size"] == 60:
                _timestamp = _result["timestamp"]
                _samples_32ch = _result["samples_32ch"]
                _chip = _result["chip"]

                v.push_board32_point(
                    data_queue,
                    x=_timestamp,
                    board=_chip,
                    values_32ch=_samples_32ch,
                )

        if dump:
            dump_range(data, _result["start"], _result["end"], width=60, summary=False)
        
        if _i == max_blocks:
            print(f"reached max block limit: {max_blocks}, stopping.")
            break

        _i += 1

    if plot:
        data_queue.put(None)


if __name__ == '__main__':
    main()
