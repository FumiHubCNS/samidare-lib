from pathlib import Path
import pathlib
import sys
import toml

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

def check_input_file(fileinfo: dict, file: str | None = None):
    base = Path(fileinfo["base_input_path"])
    input_name = fileinfo["input_file_name"]

    file_candidates = []
    if file is not None:
        file_candidates = [
            Path(file),
            Path(f"{file}.bin"),
            base / file,
            base / f"{file}.bin",
        ]

    fileinfo_candidates = [
        base / input_name,
        base / f"{input_name}.bin",
    ]

    # file 側を優先して候補化
    candidates = file_candidates + fileinfo_candidates

    checked = []
    seen = set()

    for cand in candidates:
        cand = cand.resolve(strict=False)
        cand_str = str(cand)

        if cand_str in seen:
            continue
        seen.add(cand_str)

        exists = cand.is_file()
        checked.append({
            "path": cand_str,
            "exists": exists,
        })

        if exists:
            return {
                "found": cand_str,
                "checked": checked,
            }

    raise FileNotFoundError(
        "No input file found. Tried:\n" +
        "\n".join(item["path"] for item in checked)
    )

def get_fileinfo():
    toml_file_path = this_file_path  / "../../../parameters.toml"

    if toml_file_path.is_file():
        print(f"loading config from toml file., path: {toml_file_path}")
        with open(toml_file_path, "r") as f:
            config = toml.load(f)

        fileinfo = config["fileinfo"]

    else:
        print(f"{toml_file_path} does not exist. skipping config loading.")
        
        fileinfo = {
            "base_input_path": "./rawdata",
            "base_output_path": "./output",
            "input_file_name": "sample",
        }

    return fileinfo
