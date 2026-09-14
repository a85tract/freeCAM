"""A trained emulator of one compute block of the cloud macro/microphysics stage, for the stage's Python driver.

Bound with ``--cloud-block-model <this file>:macro_block`` (or ``--cloud-micro-block-model ...:micro_block``) and the
weights and layout named by FREECAM_CLOUD_BLOCK (a prefix: ``<prefix>_weights.npz`` and ``<prefix>_layout.json`` from
train_cloud_block.py).  Called once per chunk with the block contract's inputs by name; returns the tendency object
(``ptend_s``, the full ``ptend_q`` with the flagged constituents filled, the flags), the detrainment integrals for
the macrophysics block, and the buffer fields the block was seen to change -- the driver writes those where the
original driver leaves them and keeps the rest.  Features as the trainer built them; the forward is three
matrix products in NumPy through the single-threaded BLAS (OMP_NUM_THREADS=1 in the jobs) -- the Python driver
calls any Python callable, and a compiled kernel is the plugin path's need, not this one's.
"""
import json
import os

import numpy as np

_PREFIX = os.environ["FREECAM_CLOUD_BLOCK"]
_W = np.load(f"{_PREFIX}_weights.npz")
_L = json.load(open(f"{_PREFIX}_layout.json"))
W1T = np.ascontiguousarray(_W["W1T"], dtype=np.float32); B1 = np.ascontiguousarray(_W["b1"], dtype=np.float32)
W2T = np.ascontiguousarray(_W["W2T"], dtype=np.float32); B2 = np.ascontiguousarray(_W["b2"], dtype=np.float32)
W3T = np.ascontiguousarray(_W["W3T"], dtype=np.float32); B3 = np.ascontiguousarray(_W["b3"], dtype=np.float32)
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
#: the constituents the network's ptend_q covers (all flagged ones, or the five bulk ones of a --core network)
Q_OUT = np.asarray(_L.get("q_out", _L["flagged"]), dtype=np.int64)
#: fields a --core network leaves to arithmetic: AST from the stratus fractions, the old-time-sample copies from the updated state
DERIVED = tuple(_L.get("derived", ()))
CPAIR = 1004.64            # physconst: the driver's T update is its dry static energy update over cpair


def forward(x64):
    """The transform of every feature (log10 where the trainer decided), standardise and clamp, three dense layers,
    de-standardise, clip to the training range."""

    x = x64.astype(np.float64, copy=True)
    log = KINDS != 0
    x[:, log] = np.log10(np.maximum(x[:, log], 0.0) + LOG_FLOOR)
    x = np.nan_to_num((x - X_MEAN) / X_STD, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(x, -20.0, 20.0, out=x)
    x = x.astype(np.float32)
    h1 = np.maximum(x @ W1T + B1, 0.0, dtype=np.float32)
    h2 = np.maximum(h1 @ W2T + B2, 0.0, dtype=np.float32)
    y32 = h2 @ W3T + B3
    return np.clip(y32.astype(np.float64) * Y_STD + Y_MEAN, Y_MIN, Y_MAX)


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
    y = forward(features(inputs, ncol))
    answer = {"ptend_ls": 1, "ptend_lq": LQ}
    for name, lo, hi in TARGETS:
        block = y[:, lo:hi]
        if name == "ptend_q":
            full = np.zeros((pcols, PVER, PCNST), np.float64, order="F")
            full[:ncol, :, Q_OUT] = block.reshape(ncol, PVER, len(Q_OUT))
            answer[name] = full
        elif hi - lo == 1:
            out = np.zeros(pcols, np.float64); out[:ncol] = block[:, 0]; answer[name] = out
        else:
            shape = np.asarray(inputs[name]).shape if name in inputs else (pcols, hi - lo)
            out = np.zeros(shape, np.float64, order="F"); out[:ncol] = block.reshape((ncol,) + tuple(shape[1:])); answer[name] = out
    _clamp_water(inputs, answer, ncol)
    if DERIVED:
        _derive(inputs, answer, ncol, pcols)
    return answer


def _clamp_water(inputs, answer, ncol):
    """No constituent the network tends may go negative over the step: dq >= -q/dt.  The physics clips such values
    afterwards (qneg3) with a warning a line each; a model that respects the floor keeps the log quiet and the mass
    where the clip would have put it anyway."""

    dt = float(inputs["dt"])
    q = np.asarray(inputs["state_q"], dtype=np.float64)[:ncol]
    dq = answer["ptend_q"]
    for m in Q_OUT:
        np.maximum(dq[:ncol, :, m], -q[:, :, m] / dt, out=dq[:ncol, :, m])


def _derive(inputs, answer, ncol, pcols):
    """What the driver forms from its own results, formed here: AST = max(ALST, AIST), and the old-time-sample copies
    of the state as the driver's two updates leave it (macrop_driver.F90:1210-1216): T from the static energy
    tendency, the water constituents from theirs."""

    dt = float(inputs["dt"])
    q = np.asarray(inputs["state_q"], dtype=np.float64); t = np.asarray(inputs["state_t"], dtype=np.float64)
    dq = answer["ptend_q"]; ds = answer["ptend_s"]
    def field(values):
        out = np.zeros((pcols, PVER), np.float64, order="F"); out[:ncol] = values[:ncol]; return out
    if "AST" in DERIVED and "ALST" in answer and "AIST" in answer:
        answer["AST"] = field(np.maximum(answer["ALST"], answer["AIST"]))
    if "CLDO" in DERIVED and "AST" in inputs:
        answer["CLDO"] = field(np.asarray(inputs["AST"], dtype=np.float64))     # microp_aero_run: cldo = cldn
    updated = {m: q[:, :, m] + dq[:, :, m] * dt for m in (0, 1, 2, 3, 4)}
    copies = {"TCWAT": t + ds * dt / CPAIR, "QCWAT": updated[0], "LCWAT": updated[1] + updated[2], "ICCWAT": updated[2],
              "NLWAT": updated[3], "NIWAT": updated[4]}
    for name, values in copies.items():
        if name in DERIVED:
            answer[name] = field(values)


def macro_block(inputs):
    """The macrophysics block (macrop_driver_tend) answered by the network."""

    return block_answer(inputs)


def micro_block(inputs):
    """The microphysics block (activation, driver, tendency sum) answered by the network."""

    return block_answer(inputs)
