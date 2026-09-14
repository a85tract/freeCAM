"""Trained emulators of the cloud macro/microphysics stage's compute blocks, for the stage's Python driver.

Bound with ``--cloud-block-model <this file>:macro_block`` and ``--cloud-micro-block-model <this file>:micro_block``.
Each block's network is named by its own prefix -- FREECAM_CLOUD_BLOCK_MACRO and FREECAM_CLOUD_BLOCK_MICRO
(``<prefix>_weights.npz`` and ``<prefix>_layout.json`` from train_cloud_block.py), FREECAM_CLOUD_BLOCK serving
either that is not set -- and loaded when first asked for.  Called once per chunk with the block contract's inputs by
name; returns the tendency object (``ptend_s``, the full ``ptend_q`` with the constituents the network covers
filled and floored so none goes negative over the step, the flags), the detrainment integrals for the macrophysics
block, the buffer fields the block was seen to change, and the fields a --core network leaves to arithmetic (AST from
the stratus fractions, the old-time-sample copies of the updated state, CLDO from the cloud fraction).  The driver
writes those where the original driver leaves them and keeps the rest.  Features as the trainer built them; the
forward is three matrix products in NumPy through the single-threaded BLAS (OMP_NUM_THREADS=1 in the jobs) -- the
Python driver calls any Python callable, and a compiled kernel is the plugin path's need, not this one's.
"""
import json
import os

import numpy as np

CPAIR = 1004.64            # physconst: the driver's T update is its dry static energy update over cpair


class BlockNetwork:
    """One block's network: weights, layout and the arithmetic around the forward."""

    def __init__(self, prefix: str) -> None:
        W = np.load(f"{prefix}_weights.npz")
        L = json.load(open(f"{prefix}_layout.json"))
        self.prefix = prefix
        self.W1T = np.ascontiguousarray(W["W1T"], dtype=np.float32); self.B1 = np.ascontiguousarray(W["b1"], dtype=np.float32)
        self.W2T = np.ascontiguousarray(W["W2T"], dtype=np.float32); self.B2 = np.ascontiguousarray(W["b2"], dtype=np.float32)
        self.W3T = np.ascontiguousarray(W["W3T"], dtype=np.float32); self.B3 = np.ascontiguousarray(W["b3"], dtype=np.float32)
        self.X_MEAN = np.ascontiguousarray(W["x_mean"], dtype=np.float64); self.X_STD = np.ascontiguousarray(W["x_std"], dtype=np.float64)
        self.Y_MEAN = np.ascontiguousarray(W["y_mean"], dtype=np.float64); self.Y_STD = np.ascontiguousarray(W["y_std"], dtype=np.float64)
        self.Y_MIN = np.ascontiguousarray(W["y_min"], dtype=np.float64); self.Y_MAX = np.ascontiguousarray(W["y_max"], dtype=np.float64)
        self.KINDS = np.ascontiguousarray(W["kinds"], dtype=np.int8); self.LQ = np.ascontiguousarray(W["lq"], dtype=np.int32)
        self.NF, self.NT, self.PVER, self.PCNST = int(L["NF"]), int(L["NT"]), int(L["pver"]), int(L["pcnst"])
        self.BLOCK = str(L["block"])
        self.LOG_FLOOR = float(L["log_floor"])
        self.FEATURES = [(name, src, idx, int(lo), int(hi)) for name, src, idx, lo, hi in L["features"]]
        self.TARGETS = [(name, int(lo), int(hi)) for name, lo, hi in L["targets"]]
        #: the constituents the network's ptend_q covers (all flagged ones, or the five bulk ones of a --core network)
        self.Q_OUT = np.asarray(L.get("q_out", L["flagged"]), dtype=np.int64)
        #: fields a --core network leaves to arithmetic
        self.DERIVED = tuple(L.get("derived", ()))

    def features(self, inputs, ncol):
        """The feature matrix of one chunk's live columns, as train_cloud_block.py built it (raw; forward transforms)."""
        x = np.empty((ncol, self.NF), np.float64)
        for name, src, idx, lo, hi in self.FEATURES:
            a = np.asarray(inputs[src], dtype=np.float64)[:ncol]
            if idx is not None:
                a = a[..., idx]
            x[:, lo:hi] = a.reshape(ncol, -1)
        return x

    def forward(self, x64):
        """The transform of every feature (log10 where the trainer decided), standardise and clamp, three dense
        layers, de-standardise, clip to the training range."""
        x = x64.astype(np.float64, copy=True)
        log = self.KINDS != 0
        x[:, log] = np.log10(np.maximum(x[:, log], 0.0) + self.LOG_FLOOR)
        x = np.nan_to_num((x - self.X_MEAN) / self.X_STD, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(x, -20.0, 20.0, out=x)
        x = x.astype(np.float32)
        h1 = np.maximum(x @ self.W1T + self.B1, 0.0, dtype=np.float32)
        h2 = np.maximum(h1 @ self.W2T + self.B2, 0.0, dtype=np.float32)
        y32 = h2 @ self.W3T + self.B3
        return np.clip(y32.astype(np.float64) * self.Y_STD + self.Y_MEAN, self.Y_MIN, self.Y_MAX)

    def answer(self, inputs):
        ncol = int(inputs["ncol"])
        pcols = int(np.asarray(inputs["state_t"]).shape[0])
        y = self.forward(self.features(inputs, ncol))
        answer = {"ptend_ls": 1, "ptend_lq": self.LQ}
        for name, lo, hi in self.TARGETS:
            block = y[:, lo:hi]
            if name == "ptend_q":
                full = np.zeros((pcols, self.PVER, self.PCNST), np.float64, order="F")
                full[:ncol, :, self.Q_OUT] = block.reshape(ncol, self.PVER, len(self.Q_OUT))
                answer[name] = full
            elif hi - lo == 1:
                out = np.zeros(pcols, np.float64); out[:ncol] = block[:, 0]; answer[name] = out
            else:
                shape = np.asarray(inputs[name]).shape if name in inputs else (pcols, hi - lo)
                out = np.zeros(shape, np.float64, order="F"); out[:ncol] = block.reshape((ncol,) + tuple(shape[1:])); answer[name] = out
        self._clamp_water(inputs, answer, ncol)
        if self.DERIVED:
            self._derive(inputs, answer, ncol, pcols)
        return answer

    def _clamp_water(self, inputs, answer, ncol):
        """No constituent the network tends may go negative over the step: dq >= -q/dt.  The physics clips such
        values afterwards (qneg3) with a warning a line each; a model that respects the floor keeps the log quiet and
        the mass where the clip would have put it anyway."""
        dt = float(inputs["dt"])
        q = np.asarray(inputs["state_q"], dtype=np.float64)[:ncol]
        dq = answer["ptend_q"]
        for m in self.Q_OUT:
            np.maximum(dq[:ncol, :, m], -q[:, :, m] / dt, out=dq[:ncol, :, m])

    def _derive(self, inputs, answer, ncol, pcols):
        """What the driver forms from its own results, formed here: AST = max(ALST, AIST); the old-time-sample copies
        of the state as the driver's two updates leave it (macrop_driver.F90:1210-1216): T from the static energy
        tendency, the water constituents from theirs; CLDO from the cloud fraction the activation was given."""
        dt = float(inputs["dt"])
        q = np.asarray(inputs["state_q"], dtype=np.float64); t = np.asarray(inputs["state_t"], dtype=np.float64)
        dq = answer["ptend_q"]; ds = answer["ptend_s"]
        def field(values):
            out = np.zeros((pcols, self.PVER), np.float64, order="F"); out[:ncol] = values[:ncol]; return out
        if "AST" in self.DERIVED and "ALST" in answer and "AIST" in answer:
            answer["AST"] = field(np.maximum(answer["ALST"], answer["AIST"]))
        if "CLDO" in self.DERIVED and "AST" in inputs:
            answer["CLDO"] = field(np.asarray(inputs["AST"], dtype=np.float64))     # microp_aero_run: cldo = cldn
        updated = {m: q[:, :, m] + dq[:, :, m] * dt for m in (0, 1, 2, 3, 4)}
        copies = {"TCWAT": t + ds * dt / CPAIR, "QCWAT": updated[0], "LCWAT": updated[1] + updated[2], "ICCWAT": updated[2],
                  "NLWAT": updated[3], "NIWAT": updated[4]}
        for name, values in copies.items():
            if name in self.DERIVED:
                answer[name] = field(values)


_NETWORKS: dict[str, BlockNetwork] = {}


def network(block: str) -> BlockNetwork:
    """The block's network, loaded on first use from FREECAM_CLOUD_BLOCK_<BLOCK> (or FREECAM_CLOUD_BLOCK)."""
    if block not in _NETWORKS:
        prefix = os.environ.get(f"FREECAM_CLOUD_BLOCK_{block.upper()}") or os.environ["FREECAM_CLOUD_BLOCK"]
        net = BlockNetwork(prefix)
        if net.BLOCK != block:
            raise ValueError(f"{prefix} was trained for the {net.BLOCK} block, not {block}")
        _NETWORKS[block] = net
    return _NETWORKS[block]


def features(inputs, ncol, block="macro"):
    return network(block).features(inputs, ncol)


def block_answer(inputs, block="macro"):
    return network(block).answer(inputs)


def macro_block(inputs):
    """The macrophysics block (macrop_driver_tend) answered by its network."""
    return network("macro").answer(inputs)


def micro_block(inputs):
    """The microphysics block (activation, driver, tendency sum) answered by its network."""
    return network("micro").answer(inputs)
