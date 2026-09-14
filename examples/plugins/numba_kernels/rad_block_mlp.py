"""A trained emulator of radiation_tend's whole compute block, as a Python process model called by the Python driver.

Loads the weights and layout train_rad_block.py wrote, named by FREECAM_RAD_BLOCK=<prefix>.  The model takes
the block contract's inputs by name (freecam.physics.radiation_process.BLOCK_INPUTS: the state, the buffer's
fields, the surface, the calendar day and the column's latitude and longitude -- no zenith angle, no RRTMG
state) and returns the two heating rates and the ten fluxes, padded to the chunk's pcols rows.  The features
are built and the forward run by Numba; the network's lit-column logit gates the shortwave outputs.
"""
import json, os
import numpy as np
from numba import njit

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


@njit(cache=False)
def _dense_axpy(x, WT, b, out):
    n, k = x.shape; m = WT.shape[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = b[j]
        for f in range(k):
            v = x[i, f]
            if v != 0.0:
                for j in range(m):
                    out[i, j] += v * WT[f, j]


@njit(cache=False)
def _forward(x, W1T, B1, W2T, B2, W3T, B3, X_LO, X_HI, Y_MIN, Y_MAX, SW_COLS, lit_index):
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
    y32 = np.empty((n, W3T.shape[1]), np.float32); _dense_axpy(h2, W3T, B3, y32)
    y = np.empty((n, lit_index), np.float64)
    for i in range(n):
        lit = y32[i, lit_index] > 0.0
        for t in range(lit_index):
            v = np.float64(y32[i, t])
            if v < Y_MIN[t]:
                v = Y_MIN[t]
            elif v > Y_MAX[t]:
                v = Y_MAX[t]
            if SW_COLS[t] and not lit:
                v = 0.0
            y[i, t] = v
    return y


_SOURCES = []
for _name, _src, _idx, _kind, _lo, _hi in FEATURES:
    if _src not in _SOURCES:
        _SOURCES.append(_src)
_SCALAR_SOURCES = {"calday"}


def _generate_feature_builder():
    lines = ["def _build(ncol, x, " + ", ".join(f"a_{src}" for src in _SOURCES) + "):",
             "    for i in range(ncol):"]
    for name, src, idx, kind, lo, hi in FEATURES:
        width = hi - lo
        if src in _SCALAR_SOURCES:
            lines.append(f"        x[i, {lo}] = a_{src}")
            continue
        if width == 1:
            lines.append(f"        v = a_{src}[i]")
            lines.append(f"        x[i, {lo}] = {'np.log10(max(v, 0.0) + LOG_FLOOR)' if kind == 'log' else 'v'}")
            continue
        index = f"a_{src}[i, k, {idx}]" if idx is not None else f"a_{src}[i, k]"
        lines.append(f"        for k in range({width}):")
        lines.append(f"            v = {index}")
        lines.append(f"            x[i, {lo} + k] = {'np.log10(max(v, 0.0) + LOG_FLOOR)' if kind == 'log' else 'v'}")
    return "\n".join(lines) + "\n"


_namespace = {"np": np, "LOG_FLOOR": LOG_FLOOR}
exec(compile(_generate_feature_builder(), "<radiation block feature builder>", "exec"), _namespace)
_build_features = njit(_namespace["_build"])


def features(inputs, ncol):
    x = np.empty((ncol, NF), np.float64)
    arrays = [float(inputs[src]) if src in _SCALAR_SOURCES else np.asarray(inputs[src], dtype=np.float64) for src in _SOURCES]
    _build_features(ncol, x, *arrays)
    return x


def radiation_block(inputs):
    """The block contract: inputs by name -> the twelve outputs, padded to pcols rows."""
    ncol = int(inputs["ncol"])
    x = features(inputs, ncol).astype(np.float32)
    y = _forward(x, W1T, B1, W2T, B2, W3T, B3, X_LO, X_HI, Y_MIN, Y_MAX, SW_COLS, LIT)
    pcols = int(np.asarray(inputs["clat"]).shape[0])
    answer = {}
    for name, lo, hi in TARGETS:
        if hi - lo > 1:
            out = np.zeros((pcols, hi - lo), np.float64, order="F"); out[:ncol, :] = y[:, lo:hi]
        else:
            out = np.zeros(pcols, np.float64); out[:ncol] = y[:, lo]
        answer[name] = out
    return answer
