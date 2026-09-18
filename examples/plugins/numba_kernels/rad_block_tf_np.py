"""The block-contract transformer (train_rad_tf.py --contract block) as the Python driver's model, in NumPy.

Bound with ``--radiation-block-model <this file>:radiation_block_tf`` (PYCAM_RAD_BLOCK_MODEL) and the weights named
by FREECAM_RAD_TF_NP: a prefix whose ``<prefix>_tfweights.npz`` rad_tf_plugin.export_weights writes from the
trainer's checkpoint.  The same network as the TorchScript file, written out: the tokens as the trainer's wrapper
builds them, the embedding and position, the pre-norm encoder layers with four-head attention and an exact-erf
GELU, the two heads, the de-standardisation, clamps and the lit-logit gate -- as matrix products through the
single-threaded BLAS and NumPy's softmax and layer norm.  No libtorch on the ranks, no operator dispatch, no copy
of the inputs into tensors: the Python driver calls any Python callable.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from freecam.physics.radiation_process import OUTPUTS  # noqa: E402

PVER, NLEV, NSCAL, NHEADS = 30, 34, 8, 4
FLOOR = 1e-30
_Z = np.load(f"{os.environ['FREECAM_RAD_TF_NP']}_tfweights.npz")
assert str(_Z["contract"]) == "block", "the weights are the slot contract's; this module answers the block"
D = int(_Z["d"]); LAYERS = int(_Z["layers"])
f32 = lambda k: np.ascontiguousarray(_Z[k], dtype=np.float32)
W_E, B_E, POS = f32("W_e").T, f32("b_e"), f32("pos")
W_LEV, B_LEV, W_COL, B_COL = f32("W_lev").T, f32("b_lev"), f32("W_col").T, f32("b_col")
LAYER = [dict(Win=f32(f"Win{L}").T, bin=f32(f"bin{L}"), Wout=f32(f"Wout{L}").T, bout=f32(f"bout{L}"), W1=f32(f"W1_{L}").T, b1=f32(f"b1_{L}"),
              W2=f32(f"W2_{L}").T, b2=f32(f"b2_{L}"), g1=f32(f"g1_{L}"), be1=f32(f"be1_{L}"), g2=f32(f"g2_{L}"), be2=f32(f"be2_{L}")) for L in range(LAYERS)]
S = {k[2:]: np.asarray(_Z[k], dtype=np.float64) for k in _Z.files if k.startswith("s_")}
X_MEAN, X_STD, S_MEAN, S_STD = (S[k].astype(np.float32) for k in ("x_mean", "x_std", "s_mean", "s_std"))
SW_COL = S["sw_col"] != 0


def _log10(a):
    return np.log10(np.maximum(a, 0.0) + FLOOR)


_A = tuple(np.float32(c) for c in (0.3275911, 1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592))


def _erf(x):
    # Abramowitz and Stegun 7.1.26, |error| < 1.5e-7: the GELU's erf without SciPy, in single precision
    z = np.abs(x)
    t = np.float32(1.0) / (np.float32(1.0) + _A[0] * z)
    poly = ((((_A[1] * t + _A[2]) * t + _A[3]) * t + _A[4]) * t + _A[5]) * t
    return np.copysign(np.float32(1.0) - poly * np.exp(-z * z), x)


def _gelu(x):
    return np.float32(0.5) * x * (np.float32(1.0) + _erf(x * np.float32(1.0 / np.sqrt(2.0))))


def _layernorm(x, g, b):
    mean = x.mean(-1, keepdims=True, dtype=np.float32)
    c = x - mean
    var = np.mean(c * c, -1, keepdims=True, dtype=np.float32)
    return c / np.sqrt(var + np.float32(1e-5)) * g + b


def _attention(x, p):
    # batched matrix products over (column, head): q k^T, softmax, and the weighted values; no einsum, which NumPy
    # evaluates as loops on shapes this small
    n, T, d = x.shape; dh = d // NHEADS
    qkv = (x.reshape(n * T, d) @ p["Win"] + p["bin"]).reshape(n, T, 3, NHEADS, dh).transpose(2, 0, 3, 1, 4)   # (3, n, H, T, dh)
    q, k, v = qkv[0], qkv[1], qkv[2]
    scores = (q @ k.transpose(0, 1, 3, 2)) * np.float32(1.0 / np.sqrt(dh))                                  # (n, H, T, T)
    scores -= scores.max(-1, keepdims=True)
    w = np.exp(scores); w /= w.sum(-1, keepdims=True)
    ctx = (w @ v).transpose(0, 2, 1, 3).reshape(n * T, d)                                                     # (n*T, d)
    return (ctx @ p["Wout"] + p["bout"]).reshape(n, T, d)


def network(tok):
    """tok (n, 30, 42) float32 standardised -> y_lev (n, 30, 2), y_col (n, 11) in standardised units."""
    n, T, _ = tok.shape
    h = (tok.reshape(n * T, tok.shape[2]) @ W_E + B_E).reshape(n, T, D) + POS
    for p in LAYER:
        h = h + _attention(_layernorm(h, p["g1"], p["be1"]), p)
        ff = _layernorm(h, p["g2"], p["be2"]).reshape(n * T, D)
        h = h + (_gelu(ff @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]).reshape(n, T, D)
    return h.reshape(n * T, D) @ W_LEV + B_LEV, h.mean(1) @ W_COL + B_COL


def tokens(inputs, ncol):
    """The wrapper's 34 level features and 8 column scalars, standardised and clamped, as one token per level."""
    g = lambda name: np.asarray(inputs[name], dtype=np.float64)[:ncol]
    q = g("state_q")
    lev = np.stack([g("state_t"), _log10(g("state_pmid")), _log10(q[:, :, 0]), g("cld"), _log10(g("iclwp")), _log10(g("iciwp")), g("dei"), g("mu"),
                    _log10(g("lambdac")), g("cldfsnow"), _log10(g("icswp")), g("des"), _log10(g("ozone"))], axis=2)
    x = np.concatenate([lev, _log10(q[:, :, 42:57]), _log10(g("dgnumwet")), _log10(g("qaerwat"))], axis=2).astype(np.float32)
    x = np.clip(np.nan_to_num((x - X_MEAN) / X_STD, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
    s = np.stack([g("cam_in_asdir"), g("cam_in_asdif"), g("cam_in_aldir"), g("cam_in_aldif"), g("cam_in_lwup"), g("clat"), g("clon"),
                  np.full(ncol, float(inputs["calday"]))], axis=1).astype(np.float32)
    s = np.clip(np.nan_to_num((s - S_MEAN) / S_STD, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
    return np.concatenate([x, np.broadcast_to(s[:, None, :], (ncol, PVER, NSCAL))], axis=2).astype(np.float32)


def radiation_block_tf(inputs: dict) -> dict:
    """The block contract: inputs by name -> the twelve outputs, padded to pcols rows."""
    ncol = int(inputs["ncol"]); pcols = int(np.asarray(inputs["clat"]).shape[0])
    y_lev, y_col = network(tokens(inputs, ncol))
    y_lev = y_lev.reshape(ncol, PVER, 2)
    lev = np.clip(y_lev.astype(np.float64) * S["ys_lev"] + S["ym_lev"], S["ylo_lev"], S["yhi_lev"])
    col = np.clip(y_col[:, :10].astype(np.float64) * S["ys_col"] + S["ym_col"], S["ylo_col"], S["yhi_col"])
    lit = y_col[:, 10] > 0.0
    lev[:, :, 0] *= lit[:, None]
    col[:, SW_COL] *= lit[:, None]
    answer = {}
    for j, name in enumerate(OUTPUTS):
        if j < 2:
            out = np.zeros((pcols, PVER), np.float64, order="F"); out[:ncol] = lev[:, :, j]
        else:
            out = np.zeros(pcols, np.float64); out[:ncol] = col[:, j - 2]
        answer[name] = out
    return answer
