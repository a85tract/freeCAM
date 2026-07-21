#!/usr/bin/env python3
"""Generate the model-kernel export map and expected symbol list."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: prepare_abi_exports.py ABI_JSON LINKER_MAP SYMBOL_LIST"
        )
    source, linker_map, symbol_list = map(Path, sys.argv[1:])
    payload = json.loads(source.read_text())
    version = int(payload["abi_version"])
    symbols = sorted(payload["exports"])
    if not symbols or any(not name.startswith("pycam_sima_") for name in symbols):
        raise SystemExit("ABI exports must be nonempty pycam_sima_* symbols")
    linker_map.parent.mkdir(parents=True, exist_ok=True)
    linker_map.write_text(
        f"PYCAM_SIMA_{version}.0 {{\n"
        "  global:\n"
        + "".join(f"    {name};\n" for name in symbols)
        + "  local:\n"
        "    *;\n"
        "};\n"
    )
    symbol_list.parent.mkdir(parents=True, exist_ok=True)
    symbol_list.write_text("".join(f"{name}\n" for name in symbols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
