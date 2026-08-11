#!/usr/bin/env python3
"""dlopen every generated source device beside the real PI-CAM image."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path

from freecam.pi_cam.native import _prepare_fortran_runtime
from freecam.pi_cam.process_devices import PICAMProcessDeviceCatalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    manifest = json.loads(arguments.native_manifest.read_text())
    main_library = Path(str(manifest["library"]))
    _prepare_fortran_runtime()
    ctypes.CDLL(str(main_library), mode=ctypes.RTLD_LOCAL)
    ctypes.CDLL(str(main_library), mode=os.RTLD_NOLOAD | ctypes.RTLD_GLOBAL)
    catalog = PICAMProcessDeviceCatalog.from_reports(
        arguments.generation, arguments.validation
    )
    loaded: dict[Path, ctypes.CDLL] = {}
    records = []
    for qualified_name in catalog.names:
        device = catalog._devices[qualified_name]
        error = None
        try:
            library = loaded.get(device.library)
            if library is None:
                library = ctypes.CDLL(
                    str(device.library),
                    mode=ctypes.RTLD_LOCAL | getattr(os, "RTLD_LAZY", 1),
                )
                loaded[device.library] = library
            getattr(library, device.symbol)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "name": qualified_name,
                "library": str(device.library),
                "symbol": device.symbol,
                "loaded": error is None,
                "error": error,
            }
        )
    report = {
        "schema_version": 1,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "native_manifest": str(arguments.native_manifest.resolve()),
        "processes": len(records),
        "libraries": len({record["library"] for record in records}),
        "loaded_processes": sum(record["loaded"] for record in records),
        "loaded_libraries": len(loaded),
        "passed": all(record["loaded"] for record in records),
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"loaded {report['loaded_processes']}/{report['processes']} process symbols "
        f"from {report['loaded_libraries']}/{report['libraries']} source devices"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
