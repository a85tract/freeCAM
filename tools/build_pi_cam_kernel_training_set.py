#!/usr/bin/env python3
"""Turn a captured physics kernel into a training set: X in, Y out.

A capture bundle holds every argument of every call the model made, with
padding lanes and all, keyed by the routine's own dummy names.  A surrogate
needs something narrower and flatter: for each live column, the numbers the
routine reads, and the numbers it answers.

What this writes:

* ``X`` ``(samples, features)`` -- the reviewed spec's inputs and in/out
  arguments, in spec order, each profile flattened over levels and each
  scalar one column, taken from the capture's ``before`` side;
* ``Y`` ``(samples, targets)`` -- what the routine returned, from the
  ``after`` side, in spec order;
* the column names for both, so a model can be read back by argument;
* the metadata each sample came from (timestep, chunk, rank), which is what
  lets a split be by time or by chunk rather than at random -- neighbouring
  columns of one chunk are not independent draws.

Arguments the reviewed configuration leaves inert are dropped by name and
the reason recorded, so a model does not learn a dependence that cannot
exist.

    tools/build_pi_cam_kernel_training_set.py \\
        --function mmacro_pcond --bundle <capture>.npz --output <training>.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from freecam.physics.spec import load_function_spec  # noqa: E402

#: Arguments this configuration never lets the routine read, and why.  They
#: are in the capture because they are in the call; they are not in X because
#: a model that learned from them would be learning noise.
INERT: dict[str, dict[str, str]] = {
    "mmacro_pcond": {
        "clrw_old": "read only under i_rhminl > 0, which is 0 here",
        "clri_old": "read only under i_rhmini > 0, which is 0 here",
        "tke": "workspace: read only under rhminl_opt/rhmini_opt",
        "qtl_flx": "workspace: read only under rhminl_opt",
        "qti_flx": "workspace: read only under rhmini_opt",
        "cmfr_det": "workspace: read only under rhminl_opt/rhmini_opt",
        "qlr_det": "workspace: read only under rhminl_opt",
        "qir_det": "workspace: read only under rhmini_opt",
    },
}


def live_columns(bundle, name: str, rank: int, ncol: np.ndarray) -> np.ndarray:
    """One row per live column, padding lanes dropped.

    The capture stores an argument with the record axis last.  What the
    leading axes mean is the argument's own rank, which the reviewed spec
    states: a scalar is one value per call and belongs to every column of
    it; a ``(pcols)`` argument is one value per column; a ``(pcols, pver)``
    argument is a profile.  A record's lanes past its own ``ncol`` are
    whatever CAM's storage held and are not samples.
    """

    array = np.asarray(bundle[name])
    rows = []
    for record, columns in enumerate(ncol):
        columns = int(columns)
        if rank == 0:                                     # one value per call
            value = array.reshape(-1)[record] if array.ndim == 1 else array[0, record]
            rows.append(np.full((columns, 1), value, dtype=np.float64))
        elif rank == 1:                                   # one value per column
            rows.append(array[:columns, record].reshape(columns, 1))
        else:                                             # a profile per column
            rows.append(array[:columns, ..., record].reshape(columns, -1))
    return np.concatenate(rows, axis=0)


def build(function: str, bundle_path: Path, output: Path) -> dict[str, object]:
    """Fill X and Y on disk, one argument at a time.

    A month of captured columns is tens of gigabytes and every argument in
    the bundle is a few hundred megabytes of its own, so nothing here holds
    more than one argument and one block: X and Y are memory-mapped and
    written in place, column block by column block.
    """

    spec = load_function_spec(function)
    bundle = np.load(bundle_path, allow_pickle=True)
    ncol = np.asarray(bundle["ncol"]).reshape(-1)
    inert = INERT.get(function, {})
    samples = int(ncol.sum())

    inputs = [item for item in spec.arguments
              if item.role in ("input", "inout") and item.name not in inert]
    targets = [item for item in spec.arguments if item.role in ("output", "inout")]
    width = lambda item: 1 if item.rank < 2 else int(spec.dimensions["pver"])  # noqa: E731
    features = sum(width(item) for item in inputs)
    answers = sum(width(item) for item in targets)

    output.mkdir(parents=True, exist_ok=True)
    matrices = {}
    for name, count in (("X", features), ("Y", answers)):
        matrices[name] = np.lib.format.open_memmap(
            output / f"{name}.npy", mode="w+", dtype=np.float32, shape=(samples, count))

    names = {"X": [], "Y": []}
    for key, items, side in (("X", inputs, "before"), ("Y", targets, "after")):
        at = 0
        for item in items:
            block = live_columns(bundle, f"{side}__{item.name}", item.rank, ncol)
            matrices[key][:, at:at + block.shape[1]] = block.astype(np.float32)
            names[key] += ([item.name] if block.shape[1] == 1
                           else [f"{item.name}[{k}]" for k in range(block.shape[1])])
            at += block.shape[1]
            del block
        matrices[key].flush()
        print(f"  {key}: {samples} x {at}")

    per_record = {key: np.asarray(bundle[key]).reshape(-1)
                  for key in ("nstep", "ncol", "mpi_rank", "dt")}
    per_record["lchnk"] = np.asarray(bundle["lchnk"]).reshape(-1)
    expanded = {key: np.concatenate([np.full(int(n), value[r])
                                     for r, n in enumerate(ncol)])
                for key, value in per_record.items()}

    meta = {
        "x_names": np.array(names["X"]), "y_names": np.array(names["Y"]),
        "x_arguments": np.array([item.name for item in inputs]),
        "y_arguments": np.array([item.name for item in targets]),
        "levels": np.int64(spec.dimensions["pver"]),
        **{f"meta_{key}": value for key, value in expanded.items()},
        "provenance": np.array(json.dumps({
            "function": function, "qualified_name": spec.qualified_name,
            "bundle": str(bundle_path), "dropped_as_inert": inert,
            "samples": samples, "records": int(ncol.size),
        })),
    }
    np.savez(output / "meta.npz", **meta)
    return {"samples": samples, "features": features, "answers": answers,
            "inputs": len(inputs), "targets": len(targets)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="a directory: X.npy, Y.npy and meta.npz")
    arguments = parser.parse_args()

    report = build(arguments.function, arguments.bundle, arguments.output)
    print(f"{arguments.function}: {report['samples']} columns")
    print(f"  inputs  {report['inputs']} arguments -> {report['features']} features")
    print(f"  outputs {report['targets']} arguments -> {report['answers']} targets")
    print(f"  dropped as inert: {list(INERT.get(arguments.function, {}))}")
    print(f"  wrote {arguments.output}/X.npy, Y.npy, meta.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
