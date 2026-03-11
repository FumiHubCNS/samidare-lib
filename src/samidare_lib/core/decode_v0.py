import pathlib
import sys
import click
import toml

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from typing import Iterable
from typing import List, Optional
from typing import Dict, Tuple
from statistics import mode
from typing import Union
from collections import Counter

from samidare_lib.core.appender import PulseParquetAppender
from samidare_lib.core.appender import SAMPADataParquetAppender
from samidare_lib.core.prm_loader import check_input_file, get_fileinfo

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

Pair = Tuple[int, int]
Color = Tuple[float, float, float, float]

def value_counts(lst):
    """リスト内の各要素の出現回数をカウントし、ソート済みの辞書で返す。
    
    Args:
        lst: 要素をカウントするリスト
    
    Returns:
        ソート済みの辞書（キー: 値、値: 出現回数）
    """
    return dict(sorted(Counter(lst).items()))


def _colors32(cmap: Union[str, mcolors.Colormap]) -> np.ndarray:
    """カラーマップから32色のカラーテーブルを取得する（内部関数）。
    
    Args:
        cmap: カラーマップ名（文字列）またはColormapオブジェクト
    
    Returns:
        32色のRGBAテーブル（numpy配列）
    """
    if isinstance(cmap, str):

        cmap = cm.get_cmap(cmap, 32)

    table = cmap(np.linspace(0, 1, 32))
    return np.asarray(table)


def color32(value: int,
            cmap: Union[str, mcolors.Colormap] = "viridis",
            *,
            as_hex: bool = False,
            clip: bool = False) -> Union[Color, str]:
    """0～31の値に対応する色を取得する。
    
    Args:
        value: 0～31の整数値
        cmap: カラーマップ名またはColormapオブジェクト（デフォルト: "viridis"）
        as_hex: 16進数文字列として返すかどうか（デフォルト: False）
        clip: 値を[0,31]にクリップするかどうか（デフォルト: False）
    
    Returns:
        RGBA タプル或いは16進数カラーコード
    
    Raises:
        ValueError: clipがFalseで値が[0,31]範囲外の場合
    """
    v = int(value)
    if clip:
        v = max(0, min(31, v))
    elif not (0 <= v <= 31):
        raise ValueError(f"value must be in [0,31], got {value}")

    table = _colors32(cmap)
    rgba = tuple(map(float, table[v]))
    return mcolors.to_hex(rgba) if as_hex else rgba


def color32_many(values: Iterable[int],
                 cmap: Union[str, mcolors.Colormap] = "viridis",
                 *,
                 as_hex: bool = False,
                 clip: bool = False) -> List[Union[Color, str]]:
    """複数の値（0～31）に対応する色をリストで取得する。
    
    Args:
        values: 0～31の整数値のイテラブル
        cmap: カラーマップ名またはColormapオブジェクト（デフォルト: "viridis"）
        as_hex: 16進数文字列として返すかどうか（デフォルト: False）
        clip: 値を[0,31]にクリップするかどうか（デフォルト: False）
    
    Returns:
        色のリスト（RGBAタプル或いは16進数コード）
    
    Raises:
        ValueError: clipがFalseで値が[0,31]範囲外の場合
    """
    table = _colors32(cmap)
    out = []
    for value in values:
        v = int(value)
        if clip:
            v = max(0, min(31, v))
        elif not (0 <= v <= 31):
            raise ValueError(f"value must be in [0,31], got {value}")
        rgba = tuple(map(float, table[v]))
        out.append(mcolors.to_hex(rgba) if as_hex else rgba)
    return out


def pack_many_inverted(*flags: bool) -> tuple[int, int]:
    """複数のブール値をビットで逆転させてパックする。
    
    Args:
        *flags: パックするブール値（True→0, False→1）
    
    Returns:
        (パック済み整数, フラグ数) のタプル
    """
    n = 0
    for i, f in enumerate(flags):
        # True→0, False→1
        bit = 0 if f else 1   
        n |= bit << i
    return n, len(flags)


def unpack_inverted(n: int, width: int) -> list[bool]:
    """パック済み整数からブール値をアンパックする（逆転）。
    
    Args:
        n: アンパック対象の整数
        width: ビット幅
    
    Returns:
        アンパック済みのブール値リスト（0→True, 1→False）
    """
    return [((n >> i) & 1) == 0 for i in range(width)]


def build_posmap(pairs: List[Pair]) -> Dict[int, int]:
    """(位置, バイト)ペアから位置マップを構築する。
    
    Args:
        pairs: (位置, バイト)ペアのリスト
    
    Returns:
        {位置: バイト値} の辞書
    """
    return {pos: b & 0xFF for pos, b in pairs}


def read_bytes_after(posmap: Dict[int, int], marker_pos: int, n: int = 2) -> Optional[bytes]:
    """マーカー位置の後にあるバイトを読む。
    
    Args:
        posmap: 位置→バイト値 のマップ
        marker_pos: マーカーの位置
        n: 読み込むバイト数（デフォルト: 2）
    
    Returns:
        読み込んだバイト列、位置にデータがない場合はNone
    """
    start = marker_pos + 2
    out = []
    for i in range(n):
        p = start + i
        if p not in posmap:
            return None
        out.append(posmap[p])
    return bytes(out)


def extract_values_after_markers(
    pairs: List[Pair],
    marker_hits: Dict[str, List[int]],
    markers=("affa","faaf","fffa","fafa"),
    after_len: int = 2,
    hit_index: int = 0, 
) -> Dict[str, Optional[str]]:
    """マーカーの後にある値を16進数文字列で抽出する。
    
    Args:
        pairs: (位置, バイト)ペアのリスト
        marker_hits: {マーカー: [位置リスト]} のマップ
        markers: 検索対象のマーカー（デフォルト: ("affa","faaf","fffa","fafa")）
        after_len: マーカー後の読み込みバイト数（デフォルト: 2）
        hit_index: 複数ヒット時の取得インデックス（デフォルト: 0）
    
    Returns:
        {マーカー: 16進数文字列} の辞書、見つからない場合はNone
    """
    posmap = build_posmap(pairs)
    result: Dict[str, Optional[str]] = {}
    for key in markers:
        poss = marker_hits.get(key) or []
        if len(poss) <= hit_index:
            result[key] = None
            continue
        pos = poss[hit_index]
        bs = read_bytes_after(posmap, pos, n=after_len)
        result[key] = bs.hex() if bs is not None else None
    return result


def extract_timestamp_bytes(t1, t2, t3, debug):
    """3つのバイトからタイムスタンプを構築する。
    
    Args:
        t1: 下位バイト（0～15ビット）
        t2: 中位バイト（16～31ビット）
        t3: 上位バイト（32～47ビット）
        debug: デバッグ出力するかどうか
    
    Returns:
        構築されたタイムスタンプ値
    """
    timestamp = (t3 << 32) | (t2 << 16) | t1
    if debug:
        print(f"t3={hex(t3)}, t2={hex(t2)}, t1={hex(t1)} => timestamp={timestamp}, ({hex(timestamp)})")
    return timestamp


def format_2byte_groups_colorized_from_pairs(pairs, colors, reset="\x1b[0m", last_byte_color=None):
    """ペアからバイトを抽出し、2バイトグループで色付けしてフォーマットする。
    
    Args:
        pairs: (位置, バイト)ペアのリスト
        colors: {2バイト16進数: ANSIカラーコード} の辞書
        reset: リセット用ANSIコード（デフォルト: "\\x1b[0m"）
        last_byte_color: 奇数バイト末尾用のANSIコード
    
    Returns:
        色付きの16進数文字列（スペース区切り）
    """
    bs = bytes(b for _, b in pairs)
    out = []
    i = 0
    n = len(bs)
    while i + 1 < n:
        token = f"{bs[i]:02x}{bs[i+1]:02x}"
        c = colors.get(token.lower())
        out.append(f"{c}{token}{reset}" if c else token)
        i += 2
    if i < n:
        last = f"{bs[i]:02x}"
        out.append(f"{last_byte_color}{last}{reset}" if last_byte_color else last)
    return " ".join(out)


def gap_size_bytes_from_pairs(gap_pairs, start_pos=None, end_before=None):
    """指定範囲内のギャップペアのバイト数をカウントする。
    
    Args:
        gap_pairs: (位置, バイト)ペアのリスト
        start_pos: 開始位置（Noneの場合は制限なし）
        end_before: 終了位置（この値より前、Noneの場合は制限なし）
    
    Returns:
        範囲内のペア数（バイト数）
    """
    cnt = 0
    for pos, _b in gap_pairs:
        if start_pos is not None and pos < start_pos:
            continue
        if end_before is not None and pos >= end_before:
            continue
        cnt += 1
    return cnt


def gap_bytes_and_posmap(gap_pairs, start_pos=None, end_before=None):
    """指定範囲内のギャップペアからバイト列と位置リストを取得する。
    
    Args:
        gap_pairs: (位置, バイト)ペアのリスト
        start_pos: 開始位置（Noneの場合は制限なし）
        end_before: 終了位置（この値より前、Noneの場合は制限なし）
    
    Returns:
        (バイト列, 位置リスト) のタプル
    """
    bs_list = []
    posmap = []
    for pos, b in gap_pairs:
        if start_pos is not None and pos < start_pos:
            continue
        if end_before is not None and pos >= end_before:
            continue
        bs_list.append(b)
        posmap.append(pos)
    return bytes(bs_list), posmap


def find_markers_in_gap_pairs(gap_pairs, markers, start_pos=None, end_before=None):
    """ギャップペア内でマーカーを検索し、そのすべての位置を返す。
    
    Args:
        gap_pairs: (位置, バイト)ペアのリスト
        markers: 検索対象のマーカー（int/bytes/str のリスト、16進数など）
        start_pos: 検索開始位置（Noneの場合は制限なし）
        end_before: 検索終了位置（この値より前、Noneの場合は制限なし）
    
    Returns:
        {マーカー16進数: [位置リスト]} の辞書
    """
    bs, posmap = gap_bytes_and_posmap(gap_pairs, start_pos, end_before)

    def norm_marker(m):
        if isinstance(m, int):
            m &= 0xFFFF
            return m.to_bytes(2, 'big')
        if isinstance(m, bytes):
            if len(m) != 2:
                raise ValueError("marker bytes must be length 2")
            return m
        if isinstance(m, str):
            h = m.replace(" ", "").replace("_", "")
            if len(h) != 4:
                raise ValueError("marker hex must be 4 hex chars for 2-byte marker")
            return bytes.fromhex(h)
        raise TypeError("unsupported marker type")
    needles = [(norm_marker(m), m) for m in markers]

    hits = {}
    for needle_bytes, orig in needles:
        key = needle_bytes.hex()  
        hits[key] = []

        i = 0
        n = len(needle_bytes)
        while True:
            j = bs.find(needle_bytes, i)
            if j == -1:
                break

            hits[key].append(posmap[j])
            i = j + 1
    return hits


def _bytes_from_pairs_interval(
    pairs: List[Pair],
    start_pos: int,
    end_before: int,
    *,
    fill_missing: Optional[int] = None, 
) -> bytes:
    """指定区間内のバイトをペアから抽出する。
    
    Args:
        pairs: (位置, バイト)ペアのリスト
        start_pos: 開始位置（含む）
        end_before: 終了位置（含まない）
        fill_missing: 欠落位置に埋めるバイト値（Noneの場合はエラー）
    
    Returns:
        抽出されたバイト列
    
    Raises:
        KeyError: fill_missingがNoneで位置にデータがない場合
    """
    if start_pos < 0 or end_before < start_pos:
        return b""

    posmap: Dict[int, int] = {p: (b & 0xFF) for p, b in pairs}
    out = bytearray()
    for p in range(start_pos, end_before):
        if p in posmap:
            out.append(posmap[p])
        else:
            if fill_missing is None:
                raise KeyError(f"missing byte at pos={p}")
            out.append(fill_missing & 0xFF)

    return bytes(out)


def expand_10bit_units_from_pairs(
    pairs: List[Pair],
    start_pos: int,
    end_before: int,
    *,
    msb_first: bool = True,      
    fill_missing: Optional[int] = None,  
    pad_final_with_zeros: bool = False, 
) -> List[int]:
    """ペアからバイト列を抽出し、10ビット単位に展開する。
    
    Args:
        pairs: (位置, バイト)ペアのリスト
        start_pos: 開始位置
        end_before: 終了位置（含まない）
        msb_first: MSB優先（True）またはLSB優先（False）
        fill_missing: 欠落バイトに埋める値
        pad_final_with_zeros: 最終的な不完全な10ビット単位をパディングするか
    
    Returns:
        抽出された10ビット整数のリスト
    """
    data = _bytes_from_pairs_interval(pairs, start_pos, end_before, fill_missing=fill_missing)
    out: List[int] = []

    if msb_first:
        acc = 0      
        acc_bits = 0 
        for b in data:
            acc = ((acc << 8) | (b & 0xFF)) & ((1 << (acc_bits + 8)) - 1 if acc_bits + 8 <= 64 else (acc << 8) | (b & 0xFF))
            acc_bits += 8
    
            while acc_bits >= 10:
                val = (acc >> (acc_bits - 10)) & 0x3FF
                out.append(val)
                acc_bits -= 10

                acc &= (1 << acc_bits) - 1 if acc_bits > 0 else 0
        if acc_bits > 0 and pad_final_with_zeros:
          
            val = (acc << (10 - acc_bits)) & 0x3FF
            out.append(val)
    else:
        acc = 0
        acc_bits = 0
        for b in data:
            for i in range(8):  # 0..7
                bit = (b >> i) & 1
                acc |= (bit << acc_bits)
                acc_bits += 1
                if acc_bits >= 10:
                    val = acc & 0x3FF
                    out.append(val)
                    acc >>= 10
                    acc_bits -= 10
        if acc_bits > 0 and pad_final_with_zeros:
            val = acc & 0x3FF  
            out.append(val)

    return out


def byte_to_hex_and_int(val: int):
    """バイト値を16進数文字列と整数に変換する。
    
    Args:
        val: 0～255の整数値
    
    Returns:
        (16進数文字列, 整数) のタプル
    
    Raises:
        TypeError: valが整数でない場合
        ValueError: valが0～255の範囲外の場合
    """
    try:
        if not isinstance(val, int):
            raise TypeError(f"val must be int, got {type(val).__name__}")
        if not (0 <= val <= 0xFF):
            raise ValueError(f"val out of range: {val}")

        b = bytes([val])  
        hex_str = b.hex() 

        if not hex_str.isdigit():

            raise ValueError(f"hex_str contains a-f: {hex_str}")

        n = int(hex_str, 16) 
        return hex_str, n

    except Exception as e:
        raise


def scan_stream(path: str, header, footer, timestamp1, timestamp2, timestamp3, 
                chunk, limit, max_gap_bytes=None, footer_search_limit=58, 
                output1=None, output2=None, binary_checker_flag=False, event_check_flag=False):
    """バイナリストリームをスキャンしてヘッダー/フッターマーカーを検出し、
    イベントデータを解析・出力する。
    
    Args:
        path: スキャン対象のバイナリファイルパス
        header: ヘッダーマーカー値
        footer: フッターマーカー値
        timestamp1, timestamp2, timestamp3: タイムスタンプマーカー値
        chunk: 1回の読み込みサイズ（バイト）
        limit: 処理する最大バイト数（Noneで無制限）
        max_gap_bytes: 最大ギャップサイズ（現在未使用）
        footer_search_limit: フッター検索範囲の上限（バイト数）
        output1: SAMPAデータ出力先（Parquetファイル）
        output2: パルスデータ出力先（Parquetファイル）
        binary_checker_flag: デバッグ出力フラグ
        event_check_flag: イベント検査（プロット）フラグ
    
    Returns:
        なし
    """
    logger1 = None
    logger2 = None

    try:
        if output1 is not None:
            logger1 = SAMPADataParquetAppender(output1)
        if output2 is not None:
            logger2 = PulseParquetAppender(output2)

        if event_check_flag:
            fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
            ax = axes.ravel()    

            for i in range(4):
                ax[i].set_title(f"pulse @ chip{i}")

        if output1 is not None:
            logger1 = SAMPADataParquetAppender(output1)

        if output2 is not None:
            logger2 = PulseParquetAppender(output2)


        COLORS = {
            "afaf": "\x1b[31m",  
            "affa": "\x1b[32m",
            "fafa": "\x1b[34m",
            "fffa": "\x1b[35m",
            "faaf": "\x1b[33m",
        }

        RESET = "\x1b[0m"

        head0 = (header >> 8) & 0xFF
        head1 = header & 0xFF
        foot0 = (footer >> 8) & 0xFF
        foot1 = footer & 0xFF

        prev_shifted = [None] * 8
        have_prev_shifted = [False] * 8

        prev_raw = None
        prev_pos = -1
        total_bytes = 0

        active = None

        sample_block = []
        event_block = []

        def sample_builder(sample_block: list = [], row: Dict = {}, event_data_check_flag: bool = False):
            """サンプルブロックを組み立て、イベント単位でまとめる内部ヘルパー。

            この関数は連続するサンプル行を受け取り、チップごとのサンプル列を
            まとめて `sample_build_data` を作成する。条件を満たせば `output2` へ書き出し、
            イベントとして `event_block` に追加する。`event_data_check_flag` が True のとき
            は可視化（プロット）を行う。

            Args:
                sample_block: 連続サンプルを一時保持するリスト（参照で更新される）
                row: 現在のSAMPAサンプル行（辞書）
                event_data_check_flag: プロット検査を行うかどうかのフラグ
            """
            current_data = list(row.values())

            if len(sample_block) == 0:
                sample_block.append(current_data)
                
            else:
                previous_chip = sample_block[len(sample_block)-1][3]
                previous_sample_index = sample_block[len(sample_block)-1][4]

                current_chip = current_data[3]
                current_sample_index = current_data[4]

                if ( previous_chip == current_chip ) and ( previous_sample_index < current_sample_index ):
                    sample_block.append(current_data)

                else:

                    transposed_sample_block = [list(col) for col in zip(*sample_block)]
                    fadc_samples = [list(col) for col in zip(*transposed_sample_block[5])]
                    
                    sample_build_data = {
                        "chip" : transposed_sample_block[3][0],
                        "timestamp": transposed_sample_block[2],
                        "sample_index":  transposed_sample_block[4],
                        "samples_value" : fadc_samples
                    }

                    if output2 is not None:
                        logger2.append(sample_build_data)

                    if len(event_block) == 0:
                        event_block.append(sample_build_data)
                    
                    else:
                        previous_event_block = event_block[len(event_block)-1]
                        current_event_block  = sample_build_data
                        
                        if abs( current_event_block['timestamp'][0] - previous_event_block['timestamp'][0] ) < 50:
                            event_block.append(sample_build_data)
                        
                        else:

                            if event_data_check_flag:
                                for data in event_block:
                                    chip_number = data['chip']
                                    sample_indices = data['sample_index']

                                    ich = 0
                                    
                                    for sample_values in data['samples_value']:
                                        baseline = np.array(mode(sample_values))
                                        x = np.array(sample_indices)
                                        y = np.array(sample_values) - baseline
                                        ax[chip_number].plot(x, y, lw=1, alpha=0.5, marker="o", markersize=2, label=f"ch{ich}", color=color32(ich,"brg"))
                                        ich += 1
                                
                                for i in range(4):
                                    ax[i].legend(loc='upper right', ncol=3, fontsize=6)

                                plt.show(block=False)   
                                plt.pause(0.01)   

                                for i in range(4):
                                    ax[i].clear()
                                    ax[i].set(xlim=(0, 64), ylim=(-300, 725))
                                    ax[i].set_title(f"pulse @ chip{i}")     
                                    ax[i].set_xlabel("Sample index")
                                    ax[i].set_ylabel("Sample value - Base line(mode value)")     

                            event_block.clear()
                            event_block.append(sample_build_data)

                    sample_block.clear()
                    sample_block.append(current_data)
        
        def fmt_pairs_color(pairs):
            """(デバッグ用) ペア列を読み取り、2バイトごとに色付けして文字列化する。

            Args:
                pairs: (位置, バイト) ペアのリスト

            Returns:
                色付けされた16進表記の文字列（スペース区切り）
            """
            bs = bytes(b for _, b in pairs)
            out = []
            i = 0
            n = len(bs)
            while i + 1 < n:
                token = f"{bs[i]:02x}{bs[i+1]:02x}"
                c = COLORS.get(token.lower())
                out.append(f"{c}{token}{RESET}" if c else token)
                i += 2
            if i < n:
                out.append(f"{bs[i]:02x}")
            return " ".join(out)

        def emit_block(start_off, pairs, end_before_abs, tag_reason, dump_flag, event_data_check_flag):
            """検出したデータブロックを解析して出力・デバッグ表示を行う内部関数。

            主な処理:
            - ギャップ領域のトリムとマーカー検出
            - タイムスタンプやチップ番号・サンプルインデックスの抽出
            - 10ビット単位のサンプル展開
            - `sample_builder` の呼び出しおよび `logger1`/`logger2` への書き込み

            Args:
                start_off: ブロック開始オフセット（絶対位置）
                pairs: ブロックを構成する (位置, バイト) ペアのリスト
                end_before_abs: ブロック終了位置（絶対）
                tag_reason: ブロック検出理由のタグ文字列
                dump_flag: デバッグ出力を行うかどうか
                event_data_check_flag: イベント検査（プロット）フラグ
            """

            timestamp = None
            trimmed = [p for p in pairs if p[0] < end_before_abs]
            gap_str = fmt_pairs_color(trimmed)
            gap_size = len(trimmed)

            markers = [0xaffa, 0xfaaf, 0xfffa, 0xfafa]
            marker_hits = find_markers_in_gap_pairs(trimmed, markers)

            datasize_flag = ( gap_size == 58 )
            timestamp1_flag = ( len(marker_hits['affa'])>0 )
            timestamp2_flag = ( len(marker_hits['faaf'])>0 )
            timestamp3_flag = ( len(marker_hits['fffa'])>0 )

            vals_hex = extract_values_after_markers(pairs, marker_hits)
            vals_int = {k: (int(v,16) if v is not None else None) for k, v in vals_hex.items()}
        
            head_hex_col = f"{COLORS.get('afaf','')}{'afaf'}{RESET}" if COLORS.get('afaf') else "afaf"

            if 3 - timestamp1_flag - timestamp2_flag - timestamp3_flag == 0:
                timestamp = extract_timestamp_bytes(vals_int['fffa'], vals_int['faaf'], vals_int['affa'],False)

            chip_number_byte = read_bytes_after(build_posmap(pairs), start_off, n=1)
            sample_index_byte = read_bytes_after(build_posmap(pairs), start_off+1, n=1)

            vals= None
            
            if tag_reason == "ok:footer-in-60B":
                fafa_abs = end_before_abs - 4
                footer_flag = True
            else:
                footer_flag = False

            flag_binary, length = pack_many_inverted(datasize_flag, timestamp1_flag, timestamp2_flag, timestamp3_flag, footer_flag)  
            flag_debuger = flag_binary

            if 2 - footer_flag - timestamp3_flag == 0:
                start_pos = marker_hits['fffa'][0] + 4
                end_before = fafa_abs

                vals = expand_10bit_units_from_pairs(pairs, start_pos, end_before, msb_first=True, fill_missing=0, pad_final_with_zeros=False)

            chip_number = int(chip_number_byte.hex(),16)
            sample_index= int(sample_index_byte.hex(),16)

            row = {
                "data_block": gap_size+2,
                "error_level": flag_debuger,
                "timestamp": timestamp,
                "chip": chip_number,
                "sample_index": sample_index,
                "samples_value":vals
            }
            
            if gap_size+2 == 60 and flag_debuger == 0:
                sample_builder(sample_block, row, event_data_check_flag)

            if output1 is not None:
                logger1.append(row)

            if dump_flag:
                if flag_debuger >= 0:
                    print(
                        f"0x{start_off:08x}: {head_hex_col}"
                        + (f" {gap_str}" if gap_str else "")
                        + f"  ({gap_size+2}B)  {tag_reason}  error:{flag_debuger} timestamp:{timestamp} ({chip_number_byte.hex()}, {sample_index_byte.hex()})"
                    )

        def rebase_active_to(start_off_new):
            """現在の `active` 状態を新しい開始オフセットに合わせて再ベースする。

            このヘルパーは、既存の `active['pairs']` から新しい開始位置以降のペアを
            抜き出し、`active` をその範囲で再構成する。`first2` は新しいペア列から
            再計算される。

            Args:
                start_off_new: 新しい開始オフセット（絶対位置）
            """
            nonlocal active
            new_pairs = [p for p in active['pairs'] if p[0] >= start_off_new + 2]
            new_first2 = None
            if len(new_pairs) >= 2:
                new_first2 = bytes([new_pairs[0][1], new_pairs[1][1]])
            active = {
                'start': start_off_new,
                'align': active['align'],
                'pairs': new_pairs,
                'first2': new_first2,
                'footer_abs': None,
                'queued_head': None,
                'queued_used': True,
            }

        with open(path, "rb") as f:
            while True:
                buf = f.read(chunk)
                if not buf:
                    break

                for b in buf:
                    if limit is not None and total_bytes >= limit:
                        return

                    if prev_raw is not None and active is not None:
                        if prev_pos >= active['start'] + 2:
                            active['pairs'].append((prev_pos, prev_raw))

                            if active['first2'] is None and len(active['pairs']) >= 2:
                                active['first2'] = bytes([active['pairs'][0][1], active['pairs'][1][1]])

                            window_len = prev_pos - (active['start'] + 2) + 1
                            if active['first2'] is not None and window_len <= footer_search_limit:
                                bs = bytes(bb for _, bb in active['pairs'])
                                n = len(bs)
                                k = 0
                                while k + 3 < n:
                                    if bs[k] == 0xFA and bs[k+1] == 0xFA:
                                        after_footer2 = bs[k+2:k+4]
                                        if after_footer2 == active['first2']:
                                            active['footer_abs'] = active['start'] + 2 + k + 4
                                            emit_block(
                                                active['start'],
                                                active['pairs'],
                                                active['footer_abs'],
                                                "ok:footer-in-60B",
                                                dump_flag = binary_checker_flag,
                                                event_data_check_flag = event_check_flag
                                            )
                                            active = None
                                            break
                                    k += 2

                            if active is not None and active['footer_abs'] is None:
                                if window_len > footer_search_limit and active['queued_head'] is not None:
                                    emit_block(
                                        active['start'],
                                        active['pairs'],
                                        end_before_abs = active['queued_head'],
                                        tag_reason = "fallback:queued-head-on-expire",
                                        dump_flag = binary_checker_flag,
                                        event_data_check_flag = event_check_flag
                                    )
                                    rebase_active_to(active['queued_head'])

                    if prev_raw is None:
                        prev_raw = b
                        prev_pos = total_bytes
                        total_bytes += 1
                        continue

                    curr_pos = total_bytes

                    for s in range(8):
                        if s == 0:
                            shifted = prev_raw
                            out_pos = prev_pos
                        else:
                            mask = (1 << s) - 1
                            shifted = ((prev_raw >> s) | ((b & mask) << (8 - s))) & 0xFF
                            out_pos = prev_pos

                        if have_prev_shifted[s]:

                            if prev_shifted[s] == head0 and shifted == head1:
                                start_byte = out_pos - 1

                                if start_byte >= 0:
                                    if active is None:

                                        active = {
                                            'start': start_byte,
                                            'align': s,
                                            'pairs': [],
                                            'first2': None,
                                            'footer_abs': None,
                                            'queued_head': None,
                                            'queued_used': False,
                                        }

                                    else:
                                    
                                        if s != active['align']:
                                    
                                            pass
                                        else:
                                            window_len_now = prev_pos - (active['start'] + 2) + 1
                                            if active['footer_abs'] is None and window_len_now <= footer_search_limit:
                                            
                                                if active['queued_head'] is None:
                                                    active['queued_head'] = start_byte
                                            else:
                                                if active['queued_head'] is None or active['queued_used']:
                                                    emit_block(
                                                        active['start'],
                                                        active['pairs'],
                                                        end_before_abs = start_byte,
                                                        tag_reason = "fallback:next-afaf",
                                                        dump_flag = binary_checker_flag,
                                                        event_data_check_flag = event_check_flag
                                                    )
                                                    active = {
                                                        'start': start_byte,
                                                        'align': s,
                                                        'pairs': [],
                                                        'first2': None,
                                                        'footer_abs': None,
                                                        'queued_head': None,
                                                        'queued_used': False,
                                                    }
                                                else:
                                                    pass

                        prev_shifted[s] = shifted
                        have_prev_shifted[s] = True

                    prev_raw = b
                    prev_pos = curr_pos
                    total_bytes += 1

    finally:
        if logger1 is not None:
            logger1.close()
        if logger2 is not None:
            logger2.close()

    return 


def common_options(func):
    @click.option("--limit"  , "-l", type=int, default=None, help="maximum data size to be analyzed")
    @click.option('--binary' , '-b', is_flag=True, help='binary dump flag')
    @click.option('--event'  , '-e', is_flag=True, help='plot event by event flag')
    @click.option('--decode' , '-d', is_flag=True, help='decode flag')
    @click.option('--file'   , type=str, default=None, help='file path')
    @click.option('--save'   , is_flag=True, help='output file generation flag')

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@common_options
def main(limit, binary, event, decode, file, save):

    fileinfo = get_fileinfo()

    HEADER_MARKER = 0xafaf
    FOOTER_MARKER = 0xfafa
    T1_MARKER = 0xfffa
    T2_MARKER = 0xfaaf
    T3_MARKER = 0xaffa

    input_list = check_input_file(fileinfo, file)

    print(f"found input file: {input_list['found']}")
    
    DATA = input_list['found']
    BASEOUTPUT = fileinfo["base_output_path"]  + "/" + pathlib.Path(input_list['found']).stem
    OUTPUT1 = BASEOUTPUT + "_raw.parquet" if save else None
    OUTPUT2 = BASEOUTPUT + "_event.parquet" if save else None

    print(BASEOUTPUT)

    if save:
        if not pathlib.Path(OUTPUT1).parent.exists():
            raise FileNotFoundError(f"output directory does not exist: {pathlib.Path(OUTPUT1).parent}")
        
        if not pathlib.Path(OUTPUT2).parent.exists():
            raise FileExistsError(f"output file already exists: {OUTPUT2}") 

        print(f"output file path: {OUTPUT1}")
        print(f"output file path: {OUTPUT2}")

    chunk = 1 << 10 

    if decode:
        scan_stream(
            DATA, HEADER_MARKER, 
            FOOTER_MARKER, T1_MARKER , T2_MARKER, T3_MARKER, 
            chunk, limit, output1=OUTPUT1, output2=OUTPUT2,
            binary_checker_flag=binary, event_check_flag=event
        )

        print(f"completed. file path: {DATA}")

if __name__ == '__main__':
    main()
