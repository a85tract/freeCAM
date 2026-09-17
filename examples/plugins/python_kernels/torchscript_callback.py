"""A Python function at the hook of compute_uwshcu_inv: the interpreter runs a TorchScript model.

    PYCAM_CALLBACK_MODEL=/path/to/model.pt  freecam ... --kernel-function compute_uwshcu_inv=examples/plugins/python_kernels/torchscript_callback.py:answer

The hook hands ``answer`` the model block's inputs by name as Fortran-ordered views of the
kernel's arrays; the function wraps them as tensors without a copy, runs the model the image
would otherwise run through FTorch, and returns the outputs by name, cut from the model's one
packed tensor along the hook's layout.  It measures what the Python detour costs against the
same model bound as a NativeModel: same weights, same answers, the interpreter in the loop.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

KERNEL = "compute_uwshcu_inv"
_STATE: dict[str, Any] = {}


def _load() -> None:
    import torch

    from freecam.physics.native_model import model_block, packed_layout

    torch.set_num_threads(1)
    path = os.environ.get("PYCAM_CALLBACK_MODEL")
    if not path:
        raise RuntimeError("PYCAM_CALLBACK_MODEL names the TorchScript model the callback runs")
    hook, spec, inputs, outputs = model_block(KERNEL)
    _STATE["model"] = torch.jit.load(path).eval()
    _STATE["inputs"] = [item.name for item in inputs]
    _STATE["layout"] = packed_layout(hook, spec, outputs)
    _STATE["torch"] = torch


def answer(batch: dict[str, Any]) -> dict[str, np.ndarray]:
    """The model's outputs by name for one call of the kernel."""

    if not _STATE:
        _load()
    torch = _STATE["torch"]
    tensors = [torch.tensor([batch[name]], dtype=torch.float64) if np.ndim(batch[name]) == 0 else torch.from_numpy(batch[name])
               for name in _STATE["inputs"]]
    with torch.no_grad():
        packed = _STATE["model"](*tensors).numpy()
    columns = packed.shape[0]
    result: dict[str, np.ndarray] = {}
    for entry in _STATE["layout"]:
        piece = packed[:, entry["offset"]:entry["offset"] + entry["width"]]
        dims = list(entry["dims"])
        if entry["subset"] is not None:
            dims[-1] = len(entry["subset"])                # the compact array over the subset's indices
        result[entry["name"]] = piece.reshape([columns] + dims) if entry["rank"] > 1 else piece.reshape(columns)
    return result
