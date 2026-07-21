#!/usr/bin/env python3
"""Print a stable digest of Python-generated FVM geometry."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from pycam_sima.model.fvm_geometry import generate_fvm_geometry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertical-coordinate", type=Path)
    parser.add_argument("--load-library", type=Path)
    args = parser.parse_args()
    if args.load_library is not None:
        from pycam_sima.model.backend import KernelBackend

        KernelBackend(args.load_library)
    if args.vertical_coordinate is None:
        interfaces = np.linspace(0.0, 1.0, 31, dtype=np.float64)
        hybrid_b = np.zeros_like(interfaces)
        reference_pressure = 100000.0
    else:
        with Dataset(args.vertical_coordinate) as dataset:
            interfaces = np.asarray(dataset.variables["hyai"][:], dtype=np.float64)
            hybrid_b = np.asarray(dataset.variables["hybi"][:], dtype=np.float64)
            reference_pressure = float(dataset.variables["P0"].getValue())
    geometry = generate_fvm_geometry(interfaces, hybrid_b, reference_pressure)
    digest = hashlib.sha256()
    for name in sorted(geometry):
        value = np.ascontiguousarray(geometry[name])
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        results = MPI.COMM_WORLD.gather(digest.hexdigest(), root=0)
        if rank == 0:
            print(results)
    except ImportError:
        print(digest.hexdigest())


if __name__ == "__main__":
    main()
