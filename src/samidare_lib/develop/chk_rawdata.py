from pathlib import Path

path = "/Users/fendo/Work/Data/root/sampa-minitpc-test/data2025/catmini/10hz.bin"


def hex_ascii_dump(data: bytes, base_offset: int = 0, width: int = 16):
    for i in range(0, len(data), width):
        row = data[i:i+width]
        hex_part = " ".join(f"{b:02x}" for b in row)
        ascii_part = "".join(chr(b) if 32 <= b <= 126 else "." for b in row)
        print(f"0x{base_offset + i:08x}: {hex_part:<{width*3}} {ascii_part}")

def hex_dump(data: bytes, base_offset: int = 0, width: int = 16):
    for i in range(0, len(data), width):
        row = data[i:i+width]
        hex_part = row.hex(" ")
        print(f"0x{base_offset + i:08x}: {hex_part}")


chunk_size = 64
max_chunks = 5

with open(path, "rb") as f:
    for i in range(max_chunks):
        offset = f.tell()
        buf = f.read(chunk_size)
        if not buf:
            print("EOF")
            break

        print(f"--- chunk {i} ---")
        hex_dump(buf, base_offset=offset, width=16)
        print()