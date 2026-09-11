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

    def __init__(self, path: str | Path, *, shadow: bool = False) -> None:
        #: run the model on every call but let the original answer: bit-for-bit, cost measured
        self.shadow = bool(shadow)
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

    def describe(self) -> dict[str, Any]:
        return {"file": self.path.name, "sha256": self.sha256, "binding": "torchscript", "shadow": self.shadow}

    def __repr__(self) -> str:
        return f"NativeModel({str(self.path)!r}{', shadow=True' if self.shadow else ''})"


class NativePlugin:
    """A compiled kernel (a Numba cfunc of the hook's plugin interface) for a kernel slot.

    Built by :func:`freecam.physics.numba_kernel.compile_kernel`; the stage binds its
    address at the kernel's hook (``pycam_hooks_bind_plugin_v1``) and the image calls it
    directly on every call, inside Fortran, with the model block's arrays.
    """

    takes_frame = False

    def __init__(self, adapter: Any, *, label: str, kernel: str, shadow: bool = False,
                 inputs: list[str] | None = None, outputs: list[str] | None = None) -> None:
        self._adapter = adapter                      # keeps the compiled code alive
        self.address = int(adapter.address)
        self.label = str(label)
        self.kernel = str(kernel)
        self.shadow = bool(shadow)
        self.inputs = list(inputs or ())
        self.outputs = list(outputs or ())

    @property
    def key(self) -> str:
        """What identifies this binding to the stage: the code's address and the mode."""

        return f"numba:{self.address:#x}{':shadow' if self.shadow else ''}"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise PhysicsError(
            f"{self.label} is a compiled plugin: the image calls it at the hook; it is not called from Python")

    def describe(self) -> dict[str, Any]:
        return {"function": self.label, "binding": "numba", "kernel": self.kernel, "shadow": self.shadow,
                "inputs": len(self.inputs), "outputs": len(self.outputs)}

    def __repr__(self) -> str:
        return f"NativePlugin({self.label!r}{', shadow=True' if self.shadow else ''})"


__all__ = ["NativeModel", "NativePlugin"]
