"""A trained emulator of radiation_tend's computing branch as a process model.

Bound with --radiation-model <this file>:radiation_emulator (PYCAM_RAD_MODEL) and the weights and
layout named by FREECAM_RAD_MLP (a prefix: <prefix>_weights.npz and <prefix>_layout.json written by
train_rad.py).  Called once per chunk on radiative steps with the process contract's inputs by name;
returns the two heating rates and the ten fluxes.  The features are built as train_rad.py built them
(log10 with a floor for the positive heavy-tailed fields), the forward runs in Numba, and the shortwave
outputs are set to zero where the sun is down, as the physics has them.
"""
import json
import os

import numpy as np
from numba import njit

_PREFIX = os.environ["FREECAM_RAD_MLP"]
_W = np.load(f"{_PREFIX}_weights.npz")
_L = json.load(open(f"{_PREFIX}_layout.json"))
W1T = np.ascontiguousarray(_W["W1T"], dtype=np.float32); B1 = np.ascontiguousarray(_W["b1"], dtype=np.float32)
W2T = np.ascontiguousarray(_W["W2T"], dtype=np.float32); B2 = np.ascontiguousarray(_W["b2"], dtype=np.float32)
W3 = np.ascontiguousarray(_W["W3T"].T, dtype=np.float32); B3 = np.ascontiguousarray(_W["b3"], dtype=np.float32)
X_LO = np.ascontiguousarray(_W["x_lo"], dtype=np.float32); X_HI = np.ascontiguousarray(_W["x_hi"], dtype=np.float32)
Y_MIN = np.ascontiguousarray(_W["y_min"], dtype=np.float64); Y_MAX = np.ascontiguousarray(_W["y_max"], dtype=np.float64)
SW_COLS = np.ascontiguousarray(_W["sw_cols"], dtype=np.bool_)
NF, NT, H = int(_L["NF"]), int(_L["NT"]), W1T.shape[1]
LOG_FLOOR = float(_L["log_floor"])
FEATURES = [(name, src, idx, kind, int(lo), int(hi)) for name, src, idx, kind, lo, hi in _L["features"]]
TARGETS = [(name, int(lo), int(hi)) for name, lo, hi in _L["targets"]]
COSZ_COL = next(lo for name, _, _, _, lo, _ in FEATURES if name == "coszrs")


@njit
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
def _forward(x, W1T, B1, W2T, B2, W3, B3, X_LO, X_HI, Y_MIN, Y_MAX, SW_COLS, cosz_col):
    n = x.shape[0]
    for i in range(n):
        for f in range(x.shape[1]):
            v = x[i, f]
            if not (v == v):
                v = np.float32(0.0)
            if v < X_LO[f]:
                v = X_LO[f]
            elif v > X_HI[f]:
                v = X_HI[f]
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
        lit = x[i, cosz_col] > 0.0
        for t in range(y.shape[1]):
            v = np.float64(y32[i, t])
            if v < Y_MIN[t]:
                v = Y_MIN[t]
            elif v > Y_MAX[t]:
                v = Y_MAX[t]
            if SW_COLS[t] and not lit:
                v = 0.0
            y[i, t] = v
    return y


def features(inputs, ncol):
    """The feature matrix of one chunk's live columns, as train_rad.py built it."""

    x = np.empty((ncol, NF), np.float64)
    for name, src, idx, kind, lo, hi in FEATURES:
        value = inputs[src]
        if np.ndim(value) == 0:
            x[:, lo:hi] = float(value)
            continue
        a = np.asarray(value)[:ncol]
        if idx is not None:
            a = a[..., idx]
        a = a.reshape(ncol, -1)
        x[:, lo:hi] = np.log10(np.maximum(a, 0.0) + LOG_FLOOR) if kind == "log" else a
    return x


def radiation_emulator(inputs):
    ncol = int(inputs["ncol"])
    # coszrs must be read raw for the day/night mask: it is a 'lin' feature, so its column holds it as is
    x = features(inputs, ncol).astype(np.float32)
    y = _forward(x, W1T, B1, W2T, B2, W3, B3, X_LO, X_HI, Y_MIN, Y_MAX, SW_COLS, COSZ_COL)
    pcols = int(np.asarray(inputs["coszrs"]).shape[0])
    answer = {}
    for name, lo, hi in TARGETS:
        if hi - lo > 1:
            out = np.zeros((pcols, hi - lo), np.float64, order="F"); out[:ncol, :] = y[:, lo:hi]
        else:
            out = np.zeros(pcols, np.float64); out[:ncol] = y[:, lo]
        answer[name] = out
    return answer
