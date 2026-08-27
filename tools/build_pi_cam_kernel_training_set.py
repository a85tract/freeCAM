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

Two sources, one output contract.  ``--bundle`` reads a capture: the model's
own states, but one namelist, so the nine tunable parameters are constant and
carry no signal.  ``--dataset`` reads a generated dataset
(``examples/generate_mmacro_pcond_dataset.py``), where the parameters are
drawn per sample and so become columns of X -- continuous ones as they are, a
categorical one as an indicator per value, because ``cldfrc_iceopt`` selects
between code paths rather than sliding along a response.

    tools/build_pi_cam_kernel_training_set.py \\
        --function mmacro_pcond --bundle <capture>.npz --output <training>.npz

    tools/build_pi_cam_kernel_training_set.py \\
        --function mmacro_pcond --dataset <generated>.nc --output <training>.npz
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


def build_from_dataset(function: str, dataset_paths: list[Path], output: Path) -> dict:
    """X and Y from a generated dataset, parameters included as features.

    A capture answers what the routine does to the model's states under one
    namelist.  A dataset answers what it does to states under a namelist that
    moves, which is the axis a capture cannot hold and the axis a surrogate is
    wanted for.  X therefore carries the parameters alongside the inputs.
    """

    import netCDF4

    spec = load_function_spec(function)
    inert = INERT.get(function, {})
    # Several files are one set: generating half a million samples in one
    # process holds every one of them in memory at once, which is how the
    # first attempt was killed at sample 325000.  Chunks are generated apart,
    # under different seeds, and joined here.
    datasets = [netCDF4.Dataset(path) for path in dataset_paths]
    data = datasets[0]

    # A sample whose call did not return is in the file with its status, not
    # silently absent; only the ones that ran are training data.
    keep = []
    for one in datasets:
        if "status" in one.variables:
            status = np.asarray(one.variables["status"][:]).astype(str)
            good = status == "ok"
            if not good.all():
                print(f"  dropping {int((~good).sum())} samples that did not return: "
                      f"{sorted(set(status[~good]))}")
            keep.append(good)
        else:
            keep.append(None)

    def read(name: str) -> np.ndarray:
        blocks = []
        for one, good in zip(datasets, keep):
            block = np.asarray(one.variables[name][:], dtype=np.float64)
            block = block.reshape(block.shape[0], -1)
            blocks.append(block if good is None else block[good])
        return np.concatenate(blocks, axis=0)

    inputs = [item for item in spec.arguments
              if item.role in ("input", "inout") and item.name not in inert]
    targets = [item for item in spec.arguments if item.role in ("output", "inout")]

    x_blocks, x_names = [], []
    for item in inputs:
        block = read(f"input__{item.name}")
        x_blocks.append(block)
        x_names += ([item.name] if block.shape[1] == 1
                    else [f"{item.name}[{k}]" for k in range(block.shape[1])])
    for name, parameter in spec.parameters.items():
        column = read(f"parameter__{name}")
        if parameter.values:
            # A selector, not a response: one indicator per admitted value, so
            # the model never interpolates between two different code paths.
            for value in parameter.values:
                x_blocks.append((np.round(column) == value).astype(np.float64))
                x_names.append(f"parameter:{name}=={value}")
        else:
            x_blocks.append(column)
            x_names.append(f"parameter:{name}")

    y_blocks, y_names = [], []
    for item in targets:
        prefix = "updated__" if item.role == "inout" else "output__"
        block = read(f"{prefix}{item.name}")
        y_blocks.append(block)
        y_names += ([item.name] if block.shape[1] == 1
                    else [f"{item.name}[{k}]" for k in range(block.shape[1])])

    output.mkdir(parents=True, exist_ok=True)
    matrices = {"X": np.concatenate(x_blocks, axis=1).astype(np.float32),
                "Y": np.concatenate(y_blocks, axis=1).astype(np.float32)}
    for name, matrix in matrices.items():
        np.save(output / f"{name}.npy", matrix)
        print(f"  {name}: {matrix.shape[0]} x {matrix.shape[1]}")

    samples = matrices["X"].shape[0]
    attributes = {key: str(getattr(data, key)) for key in data.ncattrs()}
    attributes["seeds"] = [str(getattr(one, "seed", "")) for one in datasets]
    meta = {
        "x_names": np.array(x_names), "y_names": np.array(y_names),
        "x_arguments": np.array([item.name for item in inputs] + list(spec.parameters)),
        "y_arguments": np.array([item.name for item in targets]),
        "levels": np.int64(spec.dimensions["pver"]),
        # A dataset's samples are independent draws, so a split may be random;
        # the sample id is kept so one can still be traced back to its draw.
        "meta_sample_id": np.arange(samples),
        "meta_chunk": np.concatenate([
            np.full(int(good.sum()) if good is not None else one.dimensions["sample"].size, index)
            for index, (one, good) in enumerate(zip(datasets, keep))]),
        "provenance": np.array(json.dumps({
            "function": function, "qualified_name": spec.qualified_name,
            "datasets": [str(path) for path in dataset_paths], "dropped_as_inert": inert,
            "samples": samples, "source": "dataset",
            "parameters_in_x": list(spec.parameters),
            "dataset_attributes": attributes,
        })),
    }
    np.savez(output / "meta.npz", **meta)
    return {"samples": samples, "features": matrices["X"].shape[1],
            "answers": matrices["Y"].shape[1],
            "inputs": len(inputs), "targets": len(targets)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path, help="a capture bundle: the model's own states, one namelist")
    source.add_argument("--dataset", type=Path, nargs="+",
                        help="one or more generated datasets, joined: parameters drawn per sample")
    parser.add_argument("--output", type=Path, required=True,
                        help="a directory: X.npy, Y.npy and meta.npz")
    arguments = parser.parse_args()

    report = (build(arguments.function, arguments.bundle, arguments.output)
              if arguments.bundle is not None
              else build_from_dataset(arguments.function, arguments.dataset, arguments.output))
    print(f"{arguments.function}: {report['samples']} columns")
    print(f"  inputs  {report['inputs']} arguments -> {report['features']} features")
    print(f"  outputs {report['targets']} arguments -> {report['answers']} targets")
    print(f"  dropped as inert: {list(INERT.get(arguments.function, {}))}")
    print(f"  wrote {arguments.output}/X.npy, Y.npy, meta.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
