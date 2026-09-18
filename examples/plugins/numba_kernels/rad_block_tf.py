"""The block-contract transformer (train_rad_tf.py --contract block) as the Python driver's model of the compute block.

Bound with ``--radiation-block-model <this file>:radiation_block_tf`` (PYCAM_RAD_BLOCK_MODEL) and the TorchScript file
named by FREECAM_RAD_BLOCK_TF.  The driver hands the block's inputs by name; this hands them to the saved module as
tensors in BLOCK_INPUTS order (scalars as one-element tensors, the arrays with their pcols rows) and returns the
twelve outputs as Fortran-ordered arrays.  One thread per rank: the ranks share the cores.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from freecam.physics.radiation_process import BLOCK_INPUTS, OUTPUTS  # noqa: E402

SCALARS = frozenset(("nstep", "lchnk", "ncol", "dt", "calday", "dosw", "dolw"))
torch.set_num_threads(1)
MODEL = torch.jit.load(os.environ["FREECAM_RAD_BLOCK_TF"])
MODEL.eval()


def radiation_block_tf(inputs: dict) -> dict:
    """The block contract: inputs by name -> the twelve outputs, padded to pcols rows."""
    ncol = int(inputs["ncol"])
    pcols = int(np.asarray(inputs["clat"]).shape[0])
    tensors = []
    for name in BLOCK_INPUTS:
        if name in SCALARS:
            tensors.append(torch.tensor([float(inputs[name])], dtype=torch.float64))
        else:
            v = np.ascontiguousarray(np.asarray(inputs[name], dtype=np.float64))
            v[ncol:] = 0.0                       # rows past ncol are unset storage: keep them finite
            tensors.append(torch.from_numpy(v))
    with torch.no_grad():
        outs = MODEL(*tensors)
    answer = {}
    for name, value in zip(OUTPUTS, outs):
        v = value.numpy()
        if v.ndim == 2:
            out = np.zeros((pcols, v.shape[1]), np.float64, order="F")
            out[:ncol, :] = v[:ncol]
        else:
            out = np.zeros(pcols, np.float64)
            out[:ncol] = v[:ncol]
        answer[name] = out
    return answer
