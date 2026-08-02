#!/usr/bin/env python3
"""Capture immutable provenance for one completed PI-atm CESM run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable


_ABSOLUTE_PATH = re.compile(r"/(?:[^\s'\",]+/)*[^\s'\",]+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _files(paths: Iterable[Path]) -> list[Path]:
    return sorted({path.resolve() for path in paths if path.is_file()})


def _referenced_inputs(files: Iterable[Path]) -> list[Path]:
    result: set[Path] = set()
    for source in files:
        text = source.read_text(errors="replace")
        for value in _ABSOLUTE_PATH.findall(text):
            candidate = Path(value.rstrip(")"))
            if candidate.is_file():
                result.add(candidate.resolve())
    return sorted(result)


def _xml_value(case: Path, name: str) -> str:
    result = subprocess.run(
        [str(case / "xmlquery"), name, "--value"],
        cwd=case,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pbs-job-id")
    args = parser.parse_args()

    case = args.case.resolve()
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else Path(_xml_value(case, "RUNDIR")).resolve()
    )
    executable = Path(_xml_value(case, "EXEROOT")) / "cesm.exe"
    case_xml = _files(case.glob("env_*.xml"))
    case_docs = _files((case / "CaseDocs").iterdir())
    run_configuration = _files(
        path
        for path in run_dir.iterdir()
        if path.name.endswith(("_in", ".nml", ".rc", ".txt.prescribed"))
        or path.name in {"drv_flds_in"}
    )
    referenced = _referenced_inputs((*case_docs, *run_configuration))
    numerical_outputs = _files(run_dir.glob("*.nc"))
    logs = _files(run_dir.glob("*.log.*")) + _files(run_dir.glob("*.log"))

    payload = {
        "schema_version": 1,
        "case": str(case),
        "run_dir": str(run_dir),
        "pbs_job_id": args.pbs_job_id,
        "executable": _record(executable),
        "pe_layout": {
            name: int(_xml_value(case, name))
            for name in (
                "TOTALPES",
                "NTASKS_CPL",
                "NTASKS_ATM",
                "NTASKS_LND",
                "ROOTPE_LND",
                "NTASKS_ICE",
                "ROOTPE_ICE",
                "NTASKS_OCN",
                "ROOTPE_OCN",
                "NTASKS_ROF",
                "ROOTPE_ROF",
            )
        },
        "case_xml": [_record(path) for path in case_xml],
        "case_docs": [_record(path) for path in case_docs],
        "run_configuration": [_record(path) for path in run_configuration],
        "referenced_inputs": [_record(path) for path in referenced],
        "numerical_outputs": [_record(path) for path in numerical_outputs],
        "logs": [_record(path) for path in logs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
