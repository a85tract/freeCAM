#!/usr/bin/env python3
"""Write a function contract that serves a hook, from the draft the source gives.

    tools/draft_pi_cam_hook_contract.py trb_mtn_stress::compute_tms [--dimensions nbndlw=16 ...]
        [--draft native/pi_cam/functions/drafts/x.draft.yaml] [--out native/pi_cam/functions/x.yaml]

A hook (native/pi_cam/hooks.yaml) needs the callee's own argument list --
names, kinds, ranks, intents and extents -- and nothing of what a standalone
image needs: no parameter table, no module-state audit, no archive members.
tools/draft_pi_cam_function_spec.py already reads the argument list off the
source and marks everything else REVIEW; this tool resolves those marks the
only way a hook needs them: the routine's extent dummies (pcols, pver, ...)
become structural with the configuration's values, every other scalar an
input or output by its intent, every array's public shape its native shape
less the column axis, and the ranges, tunables and image blocks are left
empty with a header that says so.  The written contract is checked by
loading it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_DIMENSIONS = {"pcols": 16, "pver": 30, "pverp": 31, "pcnst": 57}
STRUCTURAL = {"pcols", "pver", "pverp", "pcnst", "mix", "mkx", "ncnst"}


def draft(qualified_name: str, path: Path) -> None:
    subprocess.run([sys.executable, str(REPO / "tools/draft_pi_cam_function_spec.py"), qualified_name,
                    "--out", str(path)], check=True, cwd=REPO, capture_output=True, text=True)


def hook_contract(document: dict, dimensions: dict[str, int], shapes: dict[str, list[str]] | None = None) -> dict:
    arguments = []
    for a in document["arguments"]:
        item = {k: a[k] for k in ("name", "fortran_type", "dtype", "rank", "intent", "native_shape")}
        name = a["name"]
        if a["fortran_type"] in ("logical", "character"):
            item["dtype"] = "int32"                   # the contract's carrier for a non-numeric kind
        if name in (shapes or {}):
            item["native_shape"] = list(shapes[name])  # an assumed-shape dummy, given the extent its caller passes
        if a.get("pointer") is True:
            item["intent"] = "in"                     # a pointer dummy declares no intent; the hook passes the pointer on
            item["role"] = "workspace"
            item["public_shape"] = [x for x in item["native_shape"] if x != "pcols"]
        elif str(item["intent"]) in ("None", "", "none"):
            item["intent"] = "inout"                  # declared without an intent: the hook passes it both ways
            item["role"] = "inout"
            item["public_shape"] = [x for x in item["native_shape"] if x != "pcols"] if a["rank"] else []
        elif name in STRUCTURAL and a["rank"] == 0:
            item["role"] = "structural"
            item["value"] = dimensions.get(name, dimensions.get({"mix": "pcols", "mkx": "pver", "ncnst": "pcnst"}.get(name, name)))
            if item["value"] is None:
                raise SystemExit(f"structural dummy {name!r} needs a value: pass --dimensions {name}=N")
            dimensions[name] = item["value"]           # the extent is then known under the dummy's own name
        else:
            item["role"] = {"in": "input", "out": "output", "inout": "inout"}[item["intent"]]
            item["public_shape"] = [x for x in a["native_shape"] if x != "pcols"] if a["rank"] else []
        for flag in ("pointer", "optional"):
            if a.get(flag) is True:
                item[flag] = True
        if a["fortran_type"] in ("logical", "character"):
            item["carrier"] = a["fortran_type"]
        if a.get("units") and "REVIEW" not in str(a["units"]):
            item["units"] = a["units"]
        text = str(a.get("description", "")).split(" ---")[0].strip()
        if text and "REVIEW" not in text:
            item["description"] = text
        arguments.append(item)
    return {
        "schema_version": 1, "function": document["function"], "qualified_name": document["qualified_name"],
        "routine": document["routine"], "source": document["source"], "module": document["module"],
        "binding": "module", "layout": "column",
        "dimensions": dict(dimensions), "public_axes": {"pver": "lev", "pverp": "ilev"},
        "arguments": arguments, "parameters": {}, "module_state": [],
        "image": {"archive_members": [f"{document['module']}.o"], "stubs": {"inert": [], "fail_closed": [], "abort": []},
                  "base_address": 0x50000000},
    }


HEADER = """# Function contract for {routine} ({module}), serving the hook the image puts on
# the routine (native/pi_cam/hooks.yaml).  Written by
# tools/draft_pi_cam_hook_contract.py from the argument list the source
# declares: names, kinds, ranks, intents and extents are the routine's own, the
# extents named by its integer dummies where it has them.  The routine's
# tunables and module state stay in their own storage, untouched by the hook,
# and no standalone image is built from this contract, so the parameter,
# module-state and image blocks are minimal.
#
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("qualified_name")
    parser.add_argument("--dimensions", nargs="*", default=[], metavar="NAME=VALUE")
    parser.add_argument("--shape", nargs="*", default=[], metavar="NAME=EXTENT,EXTENT",
                        help="the native shape of an assumed-shape (pointer) dummy, as the caller passes it")
    parser.add_argument("--draft", type=Path, help="an existing draft to resolve instead of drafting anew")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    routine = args.qualified_name.split("::")[-1]
    dimensions = dict(DEFAULT_DIMENSIONS)
    for item in args.dimensions:
        name, value = item.split("=")
        dimensions[name] = int(value)
    draft_path = args.draft
    if draft_path is None:
        draft_path = REPO / "build" / f"{routine}.draft.yaml"
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft(args.qualified_name, draft_path)
    document = yaml.safe_load(draft_path.read_text())
    shapes = {item.split("=", 1)[0]: item.split("=", 1)[1].split(",") for item in args.shape}
    contract = hook_contract(document, dimensions, shapes)
    out = args.out or (REPO / "native/pi_cam/functions" / f"{routine}.yaml")
    out.write_text(HEADER.format(routine=routine, module=document["module"])
                   + yaml.safe_dump(contract, sort_keys=False, default_flow_style=None, width=100))
    from freecam.physics.spec import load_function_spec
    spec = load_function_spec(str(out))
    awkward = [a.name for a in spec.arguments if a.pointer or a.carrier or a.optional]
    print(f"wrote {out.relative_to(REPO)}: {len(spec.arguments)} arguments; awkward: {awkward or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
