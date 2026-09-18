"""A model the image runs itself: a TorchScript file bound at a hooked kernel.

Put in a stage's kernel slot, a :class:`NativeModel` is not called from Python
at all.  The stage binds the file at the kernel's hook (``pycam_hooks_bind_model_v1``)
and runs the original Fortran stage whole; the hook, reached inside the compiled
routine, hands the kernel's arrays to the model through FTorch and writes the
answer back, so a step has one Python/Fortran crossing whatever is replaced.
The Python replacements -- a callable answering the frame at a pause -- remain
for validation, frame capture and quick experiments.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

from .errors import PhysicsError


class NativeModel:
    """A TorchScript model by path, for a kernel slot; the image loads it, not Python."""

    #: the segment runner must never see this in a slot: it is not a frame callable
    takes_frame = False

    def __init__(self, path: str | Path, *, shadow: bool = False, device: str = "cpu", device_index: int | None = None) -> None:
        #: run the model on every call but let the original answer: bit-for-bit, cost measured
        self.shadow = bool(shadow)
        #: where the image runs the model: ``cpu``, or ``cuda`` on ``device_index`` (None: this
        #: rank's share of the node's GPUs, from its node-local rank)
        if device not in ("cpu", "cuda"):
            raise PhysicsError(f"a native model runs on 'cpu' or 'cuda', not {device!r}")
        self.device = str(device)
        self.device_index = None if device_index is None else int(device_index)
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise PhysicsError(f"native model {self.path} is not a file")
        if not self.is_torchscript(self.path):
            raise PhysicsError(f"native model {self.path} is not a TorchScript archive")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    @staticmethod
    def is_torchscript(path: str | Path) -> bool:
        """Whether ``path`` is a TorchScript archive (a zip carrying the module's code and constants)."""

        path = Path(path)
        if not path.is_file() or not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        return any(name.endswith("constants.pkl") for name in names) and any("/code/" in name for name in names)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise PhysicsError(
            f"{self.path.name} is a native model: the image answers the kernel with it; "
            f"it is not called from Python")

    def resolved_device_index(self) -> int:
        """The GPU this rank uses: the given index, or the node-local rank spread over the visible GPUs."""

        if self.device == "cpu":
            return -1
        if self.device_index is not None:
            return self.device_index
        return local_gpu_index()

    @property
    def key(self) -> str:
        """What identifies this binding to the stage: the file, the device and the mode."""

        where = self.device if self.device == "cpu" else f"cuda:{self.resolved_device_index()}"
        return f"{self.sha256}:{where}{':shadow' if self.shadow else ''}"

    def describe(self) -> dict[str, Any]:
        record = {"file": self.path.name, "sha256": self.sha256, "binding": "torchscript", "shadow": self.shadow, "device": self.device}
        if self.device != "cpu":
            record["device_index"] = self.resolved_device_index()
        return record

    def __repr__(self) -> str:
        device = f", device={self.device!r}" if self.device != "cpu" else ""
        return f"NativeModel({str(self.path)!r}{device}{', shadow=True' if self.shadow else ''})"


def local_gpu_index() -> int:
    """This rank's GPU on its node: the node-local rank modulo the GPUs visible to it.

    Cray MPICH publishes the node-local rank as ``PMI_LOCAL_RANK`` (Slurm as
    ``SLURM_LOCALID``); the visible GPUs are ``CUDA_VISIBLE_DEVICES`` when set, else four,
    a Derecho GPU node's complement.
    """

    import os

    local = os.environ.get("PMI_LOCAL_RANK") or os.environ.get("SLURM_LOCALID") or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK") or "0"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = len([d for d in visible.split(",") if d.strip()]) if visible else 4
    return int(local) % max(count, 1)


#: the compiled code of every plugin made in this process, by identity: kept alive for the
#: life of the process, and how a plugin unpickled from the process registry finds its address
_PLUGIN_CODE: dict[str, Any] = {}


class NativePlugin:
    """A compiled kernel (a Numba cfunc of the hook's plugin interface) for a kernel slot.

    Built by :func:`freecam.physics.numba_kernel.compile_kernel`; the stage binds its
    address at the kernel's hook (``pycam_hooks_bind_plugin_v1``) and the image calls it
    directly on every call, inside Fortran, with the model block's arrays.

    ``identity`` names the compiled function the same way on every rank (the hook, the
    source file, the function, the mode): a stage is cloudpickled into each rank's process
    registry and the payload must hash the same on all of them (7402200), so the pickle
    carries the identity and never the address, which is this process's own.
    """

    takes_frame = False

    def __init__(self, adapter: Any, *, label: str, kernel: str, identity: str, shadow: bool = False,
                 inputs: list[str] | None = None, outputs: list[str] | None = None) -> None:
        self._adapter = adapter
        self.address = int(adapter.address)
        self.identity = str(identity)
        _PLUGIN_CODE[self.identity] = adapter
        self.label = str(label)
        self.kernel = str(kernel)
        self.shadow = bool(shadow)
        self.inputs = list(inputs or ())
        self.outputs = list(outputs or ())

    @property
    def key(self) -> str:
        """What identifies this binding to the stage: the code's address and the mode."""

        return f"numba:{self.address:#x}{':shadow' if self.shadow else ''}"

    def __getstate__(self) -> dict[str, Any]:
        # the compiled code is not picklable and the address is this process's: the pickle
        # carries the identity, identical on every rank, and the code is found again here
        state = dict(self.__dict__)
        state.pop("_adapter", None)
        state.pop("address", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        adapter = _PLUGIN_CODE.get(self.identity)
        if adapter is None:
            raise PhysicsError(
                f"{self.label}: a compiled plugin cannot cross processes by pickle; its code lives in "
                f"the process that compiled it (compile it on every rank with compile_kernel)")
        self._adapter = adapter
        self.address = int(adapter.address)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise PhysicsError(
            f"{self.label} is a compiled plugin: the image calls it at the hook; it is not called from Python")

    def describe(self) -> dict[str, Any]:
        return {"function": self.label, "binding": "numba", "kernel": self.kernel, "shadow": self.shadow,
                "inputs": len(self.inputs), "outputs": len(self.outputs)}

    def __repr__(self) -> str:
        return f"NativePlugin({self.label!r}{', shadow=True' if self.shadow else ''})"


__all__ = ["NativeModel", "NativePlugin"]
