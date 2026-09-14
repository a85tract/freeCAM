"""A trained emulator of radiation_tend's whole compute block, as a Python process model called by the Python driver.

Loads the weights and layout train_rad_block.py wrote, named by FREECAM_RAD_BLOCK=<prefix>.  The model takes
the block contract's inputs by name (freecam.physics.radiation_process.BLOCK_INPUTS: the state, the buffer's
fields, the surface, the calendar day and the column's latitude and longitude -- no zenith angle, no RRTMG
state) and returns the two heating rates and the ten fluxes, padded to the chunk's pcols rows.  The features
are built and the forward run in NumPy: three matrix products through the single-threaded BLAS the ranks share
their cores with (OMP_NUM_THREADS=1 in the jobs), nothing compiled, nothing to warm up.  The Python driver
calls any Python callable; a compiled kernel is the plugin path's need, not this one's.  The network's
lit-column logit gates the shortwave outputs.
"""
import json, os
import numpy as np
_PREFIX = os.environ["FREECAM_RAD_BLOCK"]
_W = np.load(f"{_PREFIX}_weights.npz")
_L = json.load(open(f"{_PREFIX}_layout.json"))
assert _L.get("contract") == "block", f"{_PREFIX}_layout.json is not a block-contract layout"
W1T = np.ascontiguousarray(_W["W1T"], dtype=np.float32); B1 = np.ascontiguousarray(_W["b1"], dtype=np.float32)
W2T = np.ascontiguousarray(_W["W2T"], dtype=np.float32); B2 = np.ascontiguousarray(_W["b2"], dtype=np.float32)
W3T = np.ascontiguousarray(_W["W3T"], dtype=np.float32); B3 = np.ascontiguousarray(_W["b3"], dtype=np.float32)
X_LO = np.ascontiguousarray(_W["x_lo"], dtype=np.float32); X_HI = np.ascontiguousarray(_W["x_hi"], dtype=np.float32)
Y_MIN = np.ascontiguousarray(_W["y_min"], dtype=np.float64); Y_MAX = np.ascontiguousarray(_W["y_max"], dtype=np.float64)
SW_COLS = np.ascontiguousarray(_W["sw_cols"], dtype=np.bool_)
FEATURES = [tuple(row) for row in _L["features"]]; TARGETS = [tuple(row) for row in _L["targets"]]
NF, NT, LIT = int(_L["NF"]), int(_L["NT"]), int(_L["lit_index"]); LOG_FLOOR = float(_L["log_floor"])
_SCALAR_SOURCES = {"calday"}


def features(inputs, ncol):
    """The feature matrix of one chunk's live columns, as train_rad_block.py built it: one slice per feature group."""
    x = np.empty((ncol, NF), np.float64)
    for name, src, idx, kind, lo, hi in FEATURES:
        if src in _SCALAR_SOURCES:
            x[:, lo] = float(inputs[src])
            continue
        a = np.asarray(inputs[src], dtype=np.float64)[:ncol]
        if idx is not None:
            a = a[..., idx]
        a = a.reshape(ncol, -1)
        x[:, lo:hi] = np.log10(np.maximum(a, 0.0) + LOG_FLOOR) if kind == "log" else a
    return x


def forward(x64):
    """Clip the features to the training range, three dense layers, clip the targets, gate the shortwave by the lit logit."""
    x = np.nan_to_num(x64.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(x, X_LO, X_HI, out=x)
    h1 = np.maximum(x @ W1T + B1, 0.0, dtype=np.float32)
    h2 = np.maximum(h1 @ W2T + B2, 0.0, dtype=np.float32)
    y32 = h2 @ W3T + B3
    y = np.clip(y32[:, :LIT].astype(np.float64), Y_MIN, Y_MAX)
    lit = y32[:, LIT] > 0.0
    y[:, SW_COLS] *= lit[:, None]
    return y


def radiation_block(inputs):
    """The block contract: inputs by name -> the twelve outputs, padded to pcols rows."""
    ncol = int(inputs["ncol"])
    y = forward(features(inputs, ncol))
    pcols = int(np.asarray(inputs["clat"]).shape[0])
    answer = {}
    for name, lo, hi in TARGETS:
        if hi - lo > 1:
            out = np.zeros((pcols, hi - lo), np.float64, order="F"); out[:ncol, :] = y[:, lo:hi]
        else:
            out = np.zeros(pcols, np.float64); out[:ncol] = y[:, lo]
        answer[name] = out
    return answer
