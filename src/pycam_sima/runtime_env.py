from __future__ import annotations

import os
import sys
from pathlib import Path


_REEXEC_MARKER = "PYCAM_SIMA_MPI_ENV_READY"


def mpi_loader_environment(environment: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment containing Derecho's MPICH ABI loader paths."""
    env = (environment or os.environ).copy()
    candidates: list[Path] = []
    active_mpi = env.get("CRAY_MPICH_DIR")
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
    previous = env.get("LD_LIBRARY_PATH", "")
    current = previous.split(":") if previous else []
    paths = [str(abi_dir), str(native_dir), *current]
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(path for path in paths if path))
    return env


def ensure_mpi_loader_environment() -> None:
    """Re-exec once with Derecho's MPICH ABI libraries on the loader path."""
    try:
        from mpi4py import MPI  # noqa: F401
    except ImportError as exc:
        if "libmpi.so" not in str(exc) or os.environ.get(_REEXEC_MARKER):
            raise RuntimeError(f"mpi4py cannot load the MPI runtime: {exc}") from exc
    else:
        return

    env = mpi_loader_environment()
    env[_REEXEC_MARKER] = "1"
    os.execve(
        sys.executable,
        [sys.executable, "-m", "pycam_sima.cli", *sys.argv[1:]],
        env,
    )
