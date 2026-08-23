#!/usr/bin/env python3
"""Convert physics-function capture streams into per-function replay bundles.

Every rank's stream is decoded, before/after records are paired per call,
and one .npz per function is written holding, for every argument tag, the
stacked 'before' and 'after' arrays with the record axis last, plus the
per-record metadata (timestep, chunk id, ncol, MPI rank, dt).  A summary
under validation/ records counts, the ncol histogram, pointer association,
and a content hash per argument, so a replay can state exactly what it
reproduced.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from freecam.physics.capture import (
    KIND_POINTER,
    array_sha256,
    iter_records,
    pair_records,
    stream_paths,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, default=Path("validation"))
    parser.add_argument("--executable-sha256", default=None)
    args = parser.parse_args()

    by_function: dict[str, list] = {}
    paths = stream_paths(args.capture_prefix)
    for path in paths:
        records = list(iter_records(path))
        for function in {item.function for item in records}:
            by_function.setdefault(function, []).extend(
                pair_records([item for item in records if item.function == function])
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for function, pairs in sorted(by_function.items()):
        tags = list(pairs[0][0].entries)
        bundle: dict[str, np.ndarray] = {
            "nstep": np.array([b.nstep for b, _ in pairs], dtype=np.int32),
            "lchnk": np.array([b.lchnk for b, _ in pairs], dtype=np.int32),
            "ncol": np.array([b.ncol for b, _ in pairs], dtype=np.int32),
            "mpi_rank": np.array([b.mpi_rank for b, _ in pairs], dtype=np.int32),
            "dt": np.array([b.dt for b, _ in pairs], dtype=np.float64),
        }
        hashes: dict[str, dict[str, str]] = {}
        associated: dict[str, int] = {}
        for tag in tags:
            for phase, index in (("before", 0), ("after", 1)):
                entries = [pair[index].entries[tag] for pair in pairs]
                kind = entries[0].kind
                if kind == KIND_POINTER:
                    associated[tag] = sum(int(entry.associated) for entry in entries)
                    bundle[f"{phase}__{tag}__associated"] = np.array(
                        [int(entry.associated) for entry in entries], dtype=np.int32
                    )
                    shapes = {entry.values.shape for entry in entries if entry.associated}
                    if not shapes:
                        continue
                    shape = shapes.pop()
                    stacked = np.zeros((*shape, len(entries)), dtype=np.float64, order="F")
                    for position, entry in enumerate(entries):
                        if entry.associated:
                            stacked[..., position] = entry.values
                else:
                    stacked = np.stack([entry.values for entry in entries], axis=-1)
                    stacked = np.asarray(stacked, order="F")
                bundle[f"{phase}__{tag}"] = stacked
                hashes.setdefault(tag, {})[phase] = array_sha256(stacked)
        output = args.output_dir / f"{function}_capture.npz"
        np.savez(output, **bundle)
        summary = {
            "schema_version": 1,
            "function": function,
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "executable_sha256": args.executable_sha256,
            "capture_prefix": str(args.capture_prefix),
            "bundle": str(output),
            "rank_files": len(paths),
            "records": len(pairs),
            "steps": sorted({b.nstep for b, _ in pairs}),
            "ranks": len({b.mpi_rank for b, _ in pairs}),
            "ncol_histogram": dict(sorted(Counter(int(b.ncol) for b, _ in pairs).items())),
            "dt": sorted({float(b.dt) for b, _ in pairs}),
            "arguments": tags,
            "pointer_associated_counts": associated,
            "sha256": hashes,
        }
        args.summary_dir.mkdir(parents=True, exist_ok=True)
        (args.summary_dir / f"pi_cam_{function}_capture_50step.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(f"{function}: {len(pairs)} records from {summary['ranks']} ranks -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
