"""The trained level-token transformer as a compiled plugin at the radiation process slot (the slot contract).

Bound with ``--radiation-plugin <this file>:radiation_table_kernel`` (PYCAM_RAD_PLUGIN, or the shadow flag) and
the weights named by FREECAM_RAD_TF: a prefix whose ``<prefix>_tfweights.npz`` :func:`export_weights` writes once
from the trainer's ``<prefix>.ckpt.pt``.  The whole network runs in Numba, frozen into the compiled code the image
calls inside Fortran: the 42-feature tokens of the 30 levels (train_rad_tf's feature order), the embedding with a
learned position, the pre-norm encoder layers (4 heads, GELU feed-forward), the per-level and the mean-token heads,
the de-standardisation, the clamps and the day/night gate of the shortwave.  There is no BLAS in this environment:
every product is a loop (about 20 GMAC/s on the linear layers, less in the attention), which is what the plugin
path costs for a transformer without a tensor library behind it.
"""
import math
import os
import numpy as np
from numba import njit

PVER, NLEV, NSCAL = 30, 34, 8
NHEADS = 4
LOG_FLOOR = 1e-30
FM = {"fastmath": True}


def export_weights(prefix: str) -> str:
    """Write ``<prefix>_tfweights.npz`` from the trainer's ``<prefix>.ckpt.pt`` (needs torch; run once, not on the ranks)."""
    import torch

    ck = torch.load(f"{prefix}.ckpt.pt", weights_only=False)
    net, stats, layers = ck["net"], ck["stats"], int(ck["layers"])
    f32 = lambda k: np.ascontiguousarray(net[k].detach().numpy(), dtype=np.float32)
    payload = {"d": np.int64(ck["d"]), "layers": np.int64(layers), "contract": np.array(ck.get("contract", "slot")),
               "W_e": f32("embed.weight"), "b_e": f32("embed.bias"), "pos": f32("pos")[0],
               "W_lev": f32("head_lev.weight"), "b_lev": f32("head_lev.bias"), "W_col": f32("head_col.weight"), "b_col": f32("head_col.bias")}
    for L in range(layers):
        p = f"enc.layers.{L}."
        payload.update({f"Win{L}": f32(p + "self_attn.in_proj_weight"), f"bin{L}": f32(p + "self_attn.in_proj_bias"),
                        f"Wout{L}": f32(p + "self_attn.out_proj.weight"), f"bout{L}": f32(p + "self_attn.out_proj.bias"),
                        f"W1_{L}": f32(p + "linear1.weight"), f"b1_{L}": f32(p + "linear1.bias"),
                        f"W2_{L}": f32(p + "linear2.weight"), f"b2_{L}": f32(p + "linear2.bias"),
                        f"g1_{L}": f32(p + "norm1.weight"), f"be1_{L}": f32(p + "norm1.bias"),
                        f"g2_{L}": f32(p + "norm2.weight"), f"be2_{L}": f32(p + "norm2.bias")})
    for k, v in stats.items():
        payload["s_" + k] = np.asarray(v, dtype=np.float64)
    out = f"{prefix}_tfweights.npz"
    np.savez(out, **payload)
    return out


# -- the network, in Numba ---------------------------------------------------------------------------------------
@njit(**FM)
def _linear(x, W, b, out):
    # out[n, m] = x[n, k] . W[m, k] + b[m]   (PyTorch's weight layout; the inner product runs along both rows)
    n, k = x.shape
    m = W.shape[0]
    for i in range(n):
        for j in range(m):
            acc = b[j]
            for f in range(k):
                acc += x[i, f] * W[j, f]
            out[i, j] = acc


@njit(**FM)
def _layernorm(x, g, b, out):
    n, d = x.shape
    for i in range(n):
        mean = np.float32(0.0)
        for j in range(d):
            mean += x[i, j]
        mean /= d
        var = np.float32(0.0)
        for j in range(d):
            dv = x[i, j] - mean
            var += dv * dv
        var /= d
        inv = np.float32(1.0) / np.sqrt(var + np.float32(1e-5))
        for j in range(d):
            out[i, j] = (x[i, j] - mean) * inv * g[j] + b[j]


@njit(**FM)
def _attention(qkv, ctx, ncols, d):
    # rows of qkv are the T levels of each column in turn; per column and head the q, k, v tiles are copied contiguous
    T = PVER
    dh = d // NHEADS
    scale = np.float32(1.0 / math.sqrt(dh))
    q = np.empty((T, dh), np.float32)
    k = np.empty((T, dh), np.float32)
    v = np.empty((T, dh), np.float32)
    s = np.empty((T, T), np.float32)
    for c in range(ncols):
        r0 = c * T
        for hd in range(NHEADS):
            o = hd * dh
            for t in range(T):
                for e in range(dh):
                    q[t, e] = qkv[r0 + t, o + e] * scale
                    k[t, e] = qkv[r0 + t, d + o + e]
                    v[t, e] = qkv[r0 + t, 2 * d + o + e]
            for t in range(T):
                m = np.float32(-1e30)
                for u in range(T):
                    acc = np.float32(0.0)
                    for e in range(dh):
                        acc += q[t, e] * k[u, e]
                    s[t, u] = acc
                    if acc > m:
                        m = acc
                ssum = np.float32(0.0)
                for u in range(T):
                    w = np.exp(s[t, u] - m)
                    s[t, u] = w
                    ssum += w
                inv = np.float32(1.0) / ssum
                for e in range(dh):
                    ctx[r0 + t, o + e] = np.float32(0.0)
                for u in range(T):
                    w = s[t, u] * inv
                    for e in range(dh):
                        ctx[r0 + t, o + e] += w * v[u, e]


@njit(**FM)
def _gelu(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            v = x[i, j]
            x[i, j] = np.float32(0.5) * v * (np.float32(1.0) + np.float32(math.erf(v / math.sqrt(2.0))))


@njit(**FM)
def _add(h, p):
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            h[i, j] += p[i, j]


@njit(**FM)
def _network(tok2, ncols, W_e, b_e, pos, layers, W_lev, b_lev, W_col, b_col, y_lev2, y_col):
    # tok2 (ncols*30, 42) standardised float32 -> y_lev2 (ncols*30, 2) and y_col (ncols, heads) in standardised units
    T = PVER
    rows = ncols * T
    d = W_e.shape[0]
    h = np.empty((rows, d), np.float32)
    _linear(tok2, W_e, b_e, h)
    for c in range(ncols):
        for t in range(T):
            for j in range(d):
                h[c * T + t, j] += pos[t, j]
    ln = np.empty((rows, d), np.float32)
    qkv = np.empty((rows, 3 * d), np.float32)
    ctx = np.empty((rows, d), np.float32)
    proj = np.empty((rows, d), np.float32)
    for L in range(len(layers)):
        Win, bin_, Wout, bout, W1, b1, W2, b2, g1, be1, g2, be2 = layers[L]
        _layernorm(h, g1, be1, ln)
        _linear(ln, Win, bin_, qkv)
        _attention(qkv, ctx, ncols, d)
        _linear(ctx, Wout, bout, proj)
        _add(h, proj)
        _layernorm(h, g2, be2, ln)
        ff = np.empty((rows, W1.shape[0]), np.float32)
        _linear(ln, W1, b1, ff)
        _gelu(ff)
        _linear(ff, W2, b2, proj)
        _add(h, proj)
    _linear(h, W_lev, b_lev, y_lev2)
    mean = np.empty((ncols, d), np.float32)
    for c in range(ncols):
        for j in range(d):
            acc = np.float32(0.0)
            for t in range(T):
                acc += h[c * T + t, j]
            mean[c, j] = acc / T
    _linear(mean, W_col, b_col, y_col)


# -- the features, as train_rad_tf's wrapper builds them ---------------------------------------------------------
@njit
def _log10f(v):
    return np.log10(max(v, 0.0) + LOG_FLOOR)


@njit
def _standardise(v, mean, std):
    v = (v - mean) / std
    if not (v == v) or v == np.inf or v == -np.inf:
        v = 0.0
    return np.float32(min(max(v, -20.0), 20.0))


@njit
def _tokens(ncol, tok2, coszrs, clat, calday, state_t, state_pmid, state_q, cld, cldfsnow, dei, mu, lambdac, iciwp, iclwp, des, icswp,
            dgnumwet, qaerwat, lwup, asdir, asdif, aldir, aldif, o3vmr, x_mean, x_std, s_mean, s_std):
    raw = np.empty(NLEV, np.float64)
    for i in range(ncol):
        for k in range(PVER):
            raw[0] = state_t[i, k]
            raw[1] = _log10f(state_pmid[i, k])
            raw[2] = _log10f(state_q[i, k, 0])
            raw[3] = cld[i, k]
            raw[4] = _log10f(iclwp[i, k])
            raw[5] = _log10f(iciwp[i, k])
            raw[6] = dei[i, k]
            raw[7] = mu[i, k]
            raw[8] = _log10f(lambdac[i, k])
            raw[9] = cldfsnow[i, k]
            raw[10] = _log10f(icswp[i, k])
            raw[11] = des[i, k]
            raw[12] = _log10f(o3vmr[i, k])
            for c in range(15):
                raw[13 + c] = _log10f(state_q[i, k, 42 + c])
            for m in range(3):
                raw[28 + m] = _log10f(dgnumwet[i, k, m])
                raw[31 + m] = _log10f(qaerwat[i, k, m])
            r = i * PVER + k
            for f in range(NLEV):
                tok2[r, f] = _standardise(raw[f], x_mean[f], x_std[f])
        sc = (coszrs[i], asdir[i], asdif[i], aldir[i], aldif[i], lwup[i], clat[i], calday)
        for j in range(NSCAL):
            v = _standardise(sc[j], s_mean[j], s_std[j])
            for k in range(PVER):
                tok2[i * PVER + k, NLEV + j] = v


@njit
def _finish(ncol, y_lev2, y_col, coszrs, ym_lev, ys_lev, ylo_lev, yhi_lev, ym_col, ys_col, ylo_col, yhi_col, sw_col,
            o_qrs, o_qrl, o_fsns, o_fsnt, o_flns, o_flnt, o_fsds, o_sols, o_soll, o_solsd, o_solld, o_flwds):
    for i in range(ncol):
        lit = coszrs[i] > 0.0
        for k in range(PVER):
            r = i * PVER + k
            qrs = min(max(np.float64(y_lev2[r, 0]) * ys_lev[k, 0] + ym_lev[k, 0], ylo_lev[k, 0]), yhi_lev[k, 0])
            qrl = min(max(np.float64(y_lev2[r, 1]) * ys_lev[k, 1] + ym_lev[k, 1], ylo_lev[k, 1]), yhi_lev[k, 1])
            o_qrs[i, k] = qrs if lit else 0.0
            o_qrl[i, k] = qrl
        for j in range(10):
            v = min(max(np.float64(y_col[i, j]) * ys_col[j] + ym_col[j], ylo_col[j]), yhi_col[j])
            if sw_col[j] and not lit:
                v = 0.0
            if j == 0:
                o_fsns[i] = v
            elif j == 1:
                o_fsnt[i] = v
            elif j == 2:
                o_flns[i] = v
            elif j == 3:
                o_flnt[i] = v
            elif j == 4:
                o_fsds[i] = v
            elif j == 5:
                o_sols[i] = v
            elif j == 6:
                o_soll[i] = v
            elif j == 7:
                o_solsd[i] = v
            elif j == 8:
                o_solld[i] = v
            else:
                o_flwds[i] = v


# -- the weights: module globals, frozen into the compiled kernel ---------------------------------------------------
def _load():
    prefix = os.environ["FREECAM_RAD_TF"]
    z = np.load(f"{prefix}_tfweights.npz")
    if str(z["contract"]) != "slot":
        raise ValueError(f"{prefix} was trained for the {z['contract']} contract; this plugin answers the slot")
    keys = ("Win", "bin", "Wout", "bout", "W1_", "b1_", "W2_", "b2_", "g1_", "be1_", "g2_", "be2_")
    layers = tuple(tuple(np.ascontiguousarray(z[f"{k}{L}"]) for k in keys) for L in range(int(z["layers"])))
    stats = {k[2:]: np.ascontiguousarray(z[k]) for k in z.files if k.startswith("s_")}
    return z, layers, stats


_Z, LAYERS, _STATS = _load()
W_E, B_E, POS = _Z["W_e"], _Z["b_e"], _Z["pos"]
W_LEV, B_LEV, W_COL, B_COL = _Z["W_lev"], _Z["b_lev"], _Z["W_col"], _Z["b_col"]
X_MEAN, X_STD, S_MEAN, S_STD = (_STATS[k] for k in ("x_mean", "x_std", "s_mean", "s_std"))
YM_LEV, YS_LEV, YLO_LEV, YHI_LEV = (_STATS[k] for k in ("ym_lev", "ys_lev", "ylo_lev", "yhi_lev"))
YM_COL, YS_COL, YLO_COL, YHI_COL = (_STATS[k] for k in ("ym_col", "ys_col", "ylo_col", "yhi_col"))
SW_COL = np.ascontiguousarray(_STATS["sw_col"] != 0)
D_MODEL = int(_Z["d"])


def radiation_table_kernel(nstep, lchnk, ncol, calday, dosw, dolw, coszrs, clat, clon, state_t, state_pmid, state_pint, state_pdel,
                           state_lnpint, state_lnpmid, state_q, cld, cldfsnow, dei, mu, lambdac, iciwp, iclwp, des, icswp, dgnumwet, qaerwat,
                           cam_in_lwup, cam_in_asdir, cam_in_asdif, cam_in_aldir, cam_in_aldif, rstate_h2ovmr, rstate_o3vmr, rstate_co2vmr,
                           rstate_ch4vmr, rstate_o2vmr, rstate_n2ovmr, rstate_cfc11vmr, rstate_cfc12vmr, rstate_cfc22vmr, rstate_ccl4vmr,
                           rstate_pmidmb, rstate_pintmb, rstate_tlay, rstate_tlev,
                           o_qrs, o_qrl, o_fsns, o_fsnt, o_flns, o_flnt, o_fsds, o_sols, o_soll, o_solsd, o_solld, o_flwds):
    """The slot's table (radiation_process.TABLE_INPUTS then TABLE_OUTPUTS); compiled by compile_table_kernel on every rank."""
    n = int(ncol)
    tok2 = np.empty((n * PVER, NLEV + NSCAL), np.float32)
    _tokens(n, tok2, coszrs, clat, calday, state_t, state_pmid, state_q, cld, cldfsnow, dei, mu, lambdac, iciwp, iclwp, des, icswp,
            dgnumwet, qaerwat, cam_in_lwup, cam_in_asdir, cam_in_asdif, cam_in_aldir, cam_in_aldif, rstate_o3vmr, X_MEAN, X_STD, S_MEAN, S_STD)
    y_lev2 = np.empty((n * PVER, 2), np.float32)
    y_col = np.empty((n, W_COL.shape[0]), np.float32)
    _network(tok2, n, W_E, B_E, POS, LAYERS, W_LEV, B_LEV, W_COL, B_COL, y_lev2, y_col)
    _finish(n, y_lev2, y_col, coszrs, YM_LEV, YS_LEV, YLO_LEV, YHI_LEV, YM_COL, YS_COL, YLO_COL, YHI_COL, SW_COL,
            o_qrs, o_qrl, o_fsns, o_fsnt, o_flns, o_flnt, o_fsds, o_sols, o_soll, o_solsd, o_solld, o_flwds)
