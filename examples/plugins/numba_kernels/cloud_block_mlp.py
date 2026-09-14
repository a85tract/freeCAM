"""A trained emulator of one compute block of the cloud macro/microphysics stage, for the stage's Python driver.

Bound with ``--cloud-block-model <this file>:macro_block`` (or ``--cloud-micro-block-model ...:micro_block``) and the
weights and layout named by FREECAM_CLOUD_BLOCK (a prefix: ``<prefix>_weights.npz`` and ``<prefix>_layout.json`` from
train_cloud_block.py).  Called once per chunk with the block contract's inputs by name; returns the tendency object
(``ptend_s``, the full ``ptend_q`` with the flagged constituents filled, the flags), the detrainment integrals for
the macrophysics block, and the buffer fields the block was seen to change -- the driver writes those where the
original driver leaves them and keeps the rest.  Features as the trainer built them, the forward in Numba.
"""
import json
import os

import numpy as np
from numba import njit

_PREFIX = os.environ["FREECAM_CLOUD_BLOCK"]
_W = np.load(f"{_PREFIX}_weights.npz")
_L = json.load(open(f"{_PREFIX}_layout.json"))
W1T = np.ascontiguousarray(_W["W1T"], dtype=np.float32); B1 = np.ascontiguousarray(_W["b1"], dtype=np.float32)
W2T = np.ascontiguousarray(_W["W2T"], dtype=np.float32); B2 = np.ascontiguousarray(_W["b2"], dtype=np.float32)
W3 = np.ascontiguousarray(_W["W3T"].T, dtype=np.float32); B3 = np.ascontiguousarray(_W["b3"], dtype=np.float32)
X_MEAN = np.ascontiguousarray(_W["x_mean"], dtype=np.float64); X_STD = np.ascontiguousarray(_W["x_std"], dtype=np.float64)
Y_MEAN = np.ascontiguousarray(_W["y_mean"], dtype=np.float64); Y_STD = np.ascontiguousarray(_W["y_std"], dtype=np.float64)
Y_MIN = np.ascontiguousarray(_W["y_min"], dtype=np.float64); Y_MAX = np.ascontiguousarray(_W["y_max"], dtype=np.float64)
KINDS = np.ascontiguousarray(_W["kinds"], dtype=np.int8); LQ = np.ascontiguousarray(_W["lq"], dtype=np.int32)
NF, NT, PVER, PCNST = int(_L["NF"]), int(_L["NT"]), int(_L["pver"]), int(_L["pcnst"])
BLOCK = str(_L["block"])
LOG_FLOOR = float(_L["log_floor"])
FEATURES = [(name, src, idx, int(lo), int(hi)) for name, src, idx, lo, hi in _L["features"]]
TARGETS = [(name, int(lo), int(hi)) for name, lo, hi in _L["targets"]]
FLAGGED = np.asarray(_L["flagged"], dtype=np.int64)


@njit(fastmath=True)
def _dense_axpy(x, WT, b, out):
    n, k = x.shape
    m = WT.shape[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = b[j]
        for l in range(k):
            xi = x[i, l]
            for j in range(m):
                out[i, j] += xi * WT[l, j]


@njit(fastmath=True)
def _dense_wide(h, W, b, out):
    n, k = h.shape
    m = W.shape[0]
    for j in range(m):
        bj = b[j]
        for i in range(n):
            s = np.float32(0.0)
            for l in range(k):
                s += h[i, l] * W[j, l]
            out[i, j] = s + bj


@njit
def _forward(x64, kinds, x_mean, x_std, W1T, B1, W2T, B2, W3, B3, y_mean, y_std, y_min, y_max):
    n, nf = x64.shape
    x = np.empty((n, nf), np.float32)
    for i in range(n):
        for f in range(nf):
            v = x64[i, f]
            if kinds[f] != 0:
                v = np.log10(max(v, 0.0) + 1e-30)
            v = (v - x_mean[f]) / x_std[f]
            if not (v == v):
                v = 0.0
            if v < -20.0:
                v = -20.0
            elif v > 20.0:
                v = 20.0
            x[i, f] = v
    h1 = np.empty((n, W1T.shape[1]), np.float32); _dense_axpy(x, W1T, B1, h1)
    for i in range(n):
        for j in range(h1.shape[1]):
            if h1[i, j] < 0.0:
                h1[i, j] = 0.0
    h2 = np.empty((n, W2T.shape[1]), np.float32); _dense_axpy(h1, W2T, B2, h2)
    for i in range(n):
        for j in range(h2.shape[1]):
            if h2[i, j] < 0.0:
                h2[i, j] = 0.0
    y32 = np.empty((n, W3.shape[0]), np.float32); _dense_wide(h2, W3, B3, y32)
    y = np.empty((n, W3.shape[0]), np.float64)
    for i in range(n):
        for t in range(y.shape[1]):
            v = np.float64(y32[i, t]) * y_std[t] + y_mean[t]
            if v < y_min[t]:
                v = y_min[t]
            elif v > y_max[t]:
                v = y_max[t]
            y[i, t] = v
    return y


def features(inputs, ncol):
    """The feature matrix of one chunk's live columns, as train_cloud_block.py built it (raw; the forward transforms)."""

    x = np.empty((ncol, NF), np.float64)
    for name, src, idx, lo, hi in FEATURES:
        a = np.asarray(inputs[src], dtype=np.float64)[:ncol]
        if idx is not None:
            a = a[..., idx]
        x[:, lo:hi] = a.reshape(ncol, -1)
    return x


def block_answer(inputs):
    ncol = int(inputs["ncol"])
    pcols = int(np.asarray(inputs["state_t"]).shape[0])
    y = _forward(features(inputs, ncol), KINDS, X_MEAN, X_STD, W1T, B1, W2T, B2, W3, B3, Y_MEAN, Y_STD, Y_MIN, Y_MAX)
    answer = {"ptend_ls": 1, "ptend_lq": LQ}
    for name, lo, hi in TARGETS:
        block = y[:, lo:hi]
        if name == "ptend_q":
            full = np.zeros((pcols, PVER, PCNST), np.float64, order="F")
            full[:ncol, :, FLAGGED] = block.reshape(ncol, PVER, len(FLAGGED))
            answer[name] = full
        elif hi - lo == 1:
            out = np.zeros(pcols, np.float64); out[:ncol] = block[:, 0]; answer[name] = out
        else:
            shape = np.asarray(inputs[name]).shape if name in inputs else (pcols, hi - lo)
            out = np.zeros(shape, np.float64, order="F"); out[:ncol] = block.reshape((ncol,) + tuple(shape[1:])); answer[name] = out
    return answer


def macro_block(inputs):
    """The macrophysics block (macrop_driver_tend) answered by the network."""

    return block_answer(inputs)


def micro_block(inputs):
    """The microphysics block (activation, driver, tendency sum) answered by the network."""

    return block_answer(inputs)
