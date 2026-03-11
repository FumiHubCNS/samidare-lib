from pathlib import Path
from typing import Optional, Dict


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
    search_limit: Optional[int] = 256,
) -> Optional[Dict]:
    """
    offset 以降から最初の afaf xxxx ... fafa xxxx を1つ探す。

    Args:
        data: 対象バイト列
        offset: 探索開始位置
        bit_shift: fafa xxxx を探すときに使うビットシフト量
        search_limit: afaf xxxx の後ろを何バイトまで見るか。Noneなら末尾まで

    Returns:
        見つかった場合:
            {
                "start": afaf の開始位置,
                "end": fafa xxxx の終端の次の位置,
                "xxxx": xxxx,
                "footer_start": fafa の開始位置(論理位置ベースの近似),
                "bit_shift": bit_shift,
            }
        見つからなければ None
    """
    afaf = b"\xaf\xaf"
    fafa = b"\xfa\xfa"

    pos = data.find(afaf, offset)
    if pos == -1:
        return None

    if pos + 4 > len(data):
        return None

    xxxx = data[pos + 2:pos + 4]

    tail_raw = data[pos + 2:]
    if search_limit is not None:
        tail_raw = tail_raw[:search_limit]

    tail_shifted = shifted_view(tail_raw, bit_shift)
    target = fafa + xxxx

    rel = tail_shifted.find(target)
    if rel == -1:
        return None

    footer_start = pos + 2 + rel
    end = footer_start + 4

    return {
        "start": pos,
        "end": end,
        "xxxx": xxxx,
        "footer_start": footer_start,
        "bit_shift": bit_shift,
    }

path = "/Users/fendo/Work/Data/root/sampa-minitpc-test/data2025/catmini/10hz.bin"
data = Path(path).read_bytes()

hit = find_one_block(data, offset=0, bit_shift=0, search_limit=256)
print(hit)