from __future__ import annotations

import os
import sys
from pathlib import Path


_REEXEC_MARKER = "PYCAM_SIMA_MPI_ENV_READY"


def ensure_mpi_loader_environment() -> None:
    """Re-exec once with Derecho's MPICH ABI libraries on the loader path."""
    try:
        from mpi4py import MPI  # noqa: F401
    except ImportError as exc:
        if "libmpi.so" not in str(exc) or os.environ.get(_REEXEC_MARKER):
            raise RuntimeError(f"mpi4py cannot load the MPI runtime: {exc}") from exc
    else:
        return

    candidates: list[Path] = []
    active_mpi = os.environ.get("CRAY_MPICH_DIR")
    if active_mpi:
        candidates.append(Path(active_mpi) / "lib-abi-mpich" / "libmpi.so.12")
    candidates.extend(
        sorted(
            Path("/opt/cray/pe/mpich").glob("*/ofi/*/*/lib-abi-mpich/libmpi.so.12"),
            reverse=True,
        )
    )
    candidates = [candidate for candidate in candidates if candidate.is_file()]
    if not candidates:
        raise RuntimeError("mpi4py needs libmpi.so.12 and no Cray MPICH ABI library was found")
    abi_dir = candidates[0].parent
    native_dir = abi_dir.parent / "lib"
    env = os.environ.copy()
    previous = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(abi_dir), str(native_dir), previous) if value
    )
    env[_REEXEC_MARKER] = "1"
    os.execve(
        sys.executable,
        [sys.executable, "-m", "pycam_sima.cli", *sys.argv[1:]],
        env,
    )
