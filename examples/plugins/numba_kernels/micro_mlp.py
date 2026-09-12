"""The trained micro_mg_tend surrogate's forward pass as a compiled plugin kernel: no libtorch.

Weights come from FREECAM_MICRO_MLP_WEIGHTS (an .npz written from the training checkpoint with the
input standardisation folded into the first layer and the output de-standardisation into the last);
the kernel is the same arithmetic the TorchScript module runs, written as loops Numba compiles.
Generated from micro_mlp_layout.json; the argument order is the hook's model block, inputs then outputs.
"""
import os
import numpy as np
from numba import njit

_W = np.load(os.environ["FREECAM_MICRO_MLP_WEIGHTS"])
W1T = np.ascontiguousarray(_W["W1T"], dtype=np.float32); B1 = np.ascontiguousarray(_W["b1"], dtype=np.float32)
W2T = np.ascontiguousarray(_W["W2T"], dtype=np.float32); B2 = np.ascontiguousarray(_W["b2"], dtype=np.float32)
W3T = np.ascontiguousarray(_W["W3T"], dtype=np.float32); B3 = np.ascontiguousarray(_W["b3"], dtype=np.float32)
X_LO = np.ascontiguousarray(_W["x_lo"], dtype=np.float32); X_HI = np.ascontiguousarray(_W["x_hi"], dtype=np.float32)
Y_MIN = np.ascontiguousarray(_W["y_min"], dtype=np.float64); Y_MAX = np.ascontiguousarray(_W["y_max"], dtype=np.float64)
NF, NT, H = W1T.shape[0], W3T.shape[1], W1T.shape[1]


@njit
def _dense(x, WT, b, out):
    n, k = x.shape
    m = WT.shape[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = b[j]
        for l in range(k):
            xi = x[i, l]
            for j in range(m):
                out[i, j] += xi * WT[l, j]


def micro_mlp(deltatin, tn, qn, qc, qi, nc, ni, p, pdel, cldn, liqcldf, relvar, accre_enhan, icecldf, naai, npccnin, rndst, nacon, reff_rain, reff_snow, tnd_qsnow, tnd_nsnow, re_ice, frzimm, frzcnt, frzdep, o_qc, o_qi, o_nc, o_ni, o_rate1ord_cw2pr_st, o_tlat, o_qvlat, o_qctend, o_qitend, o_nctend, o_nitend, o_effc, o_effc_fn, o_effi, o_prect, o_preci, o_nevapr, o_evapsnow, o_am_evp_st, o_prain, o_prodsnow, o_cmeout, o_deffi, o_pgamrad, o_lamcrad, o_qsout, o_dsout, o_rflx, o_sflx, o_qrout, o_reff_rain, o_reff_snow, o_qcsevap, o_qisevap, o_qvres, o_cmeiout, o_vtrmc, o_vtrmi, o_qcsedten, o_qisedten, o_prao, o_prco, o_mnuccco, o_mnuccto, o_msacwio, o_psacwso, o_bergso, o_bergo, o_melto, o_homoo, o_qcreso, o_prcio, o_praio, o_qireso, o_mnuccro, o_pracso, o_meltsdt, o_frzrdt, o_mnuccdo, o_nrout, o_nsout, o_refl, o_arefl, o_areflz, o_frefl, o_csrfl, o_acsrfl, o_fcsrfl, o_rercld, o_ncai, o_ncal, o_qrout2, o_qsout2, o_nrout2, o_nsout2, o_drout2, o_dsout2, o_freqs, o_freqr, o_nfice, o_prer_evap, o_preo, o_prdso, o_frzro, o_meltso, o_wtfc, o_wtfi, o_wtprelat, o_wtpostlat):
    n = tn.shape[0]
    dt = deltatin
    x = np.empty((n, 630), np.float32)
    for i in range(n):
        for k in range(30):
            x[i, 0 + k] = tn[i, k]
        for k in range(30):
            x[i, 30 + k] = qn[i, k]
        for k in range(30):
            x[i, 60 + k] = qc[i, k]
        for k in range(30):
            x[i, 90 + k] = qi[i, k]
        for k in range(30):
            x[i, 120 + k] = nc[i, k]
        for k in range(30):
            x[i, 150 + k] = ni[i, k]
        for k in range(30):
            x[i, 180 + k] = p[i, k]
        for k in range(30):
            x[i, 210 + k] = pdel[i, k]
        for k in range(30):
            x[i, 240 + k] = cldn[i, k]
        for k in range(30):
            x[i, 270 + k] = liqcldf[i, k]
        for k in range(30):
            x[i, 300 + k] = icecldf[i, k]
        for k in range(30):
            x[i, 330 + k] = naai[i, k]
        for k in range(30):
            x[i, 360 + k] = npccnin[i, k]
        for k in range(30):
            for d in range(4):
                x[i, 390 + k * 4 + d] = rndst[i, k, d]
        for k in range(30):
            for d in range(4):
                x[i, 510 + k * 4 + d] = nacon[i, k, d]
    for i in range(n):
        for f in range(630):
            v = x[i, f]
            if not (v == v):
                v = np.float32(0.0)
            if v < X_LO[f]:
                v = X_LO[f]
            elif v > X_HI[f]:
                v = X_HI[f]
            x[i, f] = v
    h1 = np.empty((n, H), np.float32)
    _dense(x, W1T, B1, h1)
    for i in range(n):
        for j in range(H):
            if h1[i, j] < 0.0:
                h1[i, j] = 0.0
    h2 = np.empty((n, H), np.float32)
    _dense(h1, W2T, B2, h2)
    for i in range(n):
        for j in range(H):
            if h2[i, j] < 0.0:
                h2[i, j] = 0.0
    y32 = np.empty((n, 2494), np.float32)
    _dense(h2, W3T, B3, y32)
    y = np.empty((n, 2494), np.float64)
    for i in range(n):
        for t in range(2494):
            v = np.float64(y32[i, t])
            if v < Y_MIN[t]:
                v = Y_MIN[t]
            elif v > Y_MAX[t]:
                v = Y_MAX[t]
            y[i, t] = v
    for i in range(n):
        for k in range(30):
            v = -qn[i, k] / dt
            if y[i, 60 + k] < v:
                y[i, 60 + k] = v
        for k in range(30):
            v = -qc[i, k] / dt
            if y[i, 90 + k] < v:
                y[i, 90 + k] = v
        for k in range(30):
            v = -qi[i, k] / dt
            if y[i, 120 + k] < v:
                y[i, 120 + k] = v
        for k in range(30):
            v = -nc[i, k] / dt
            if y[i, 150 + k] < v:
                y[i, 150 + k] = v
        for k in range(30):
            v = -ni[i, k] / dt
            if y[i, 180 + k] < v:
                y[i, 180 + k] = v
    for i in range(n):
        for k in range(30):
            o_rate1ord_cw2pr_st[i, k] = y[i, 0 + k]
        for k in range(30):
            o_tlat[i, k] = y[i, 30 + k]
        for k in range(30):
            o_qvlat[i, k] = y[i, 60 + k]
        for k in range(30):
            o_qctend[i, k] = y[i, 90 + k]
        for k in range(30):
            o_qitend[i, k] = y[i, 120 + k]
        for k in range(30):
            o_nctend[i, k] = y[i, 150 + k]
        for k in range(30):
            o_nitend[i, k] = y[i, 180 + k]
        for k in range(30):
            o_effc[i, k] = y[i, 210 + k]
        for k in range(30):
            o_effc_fn[i, k] = y[i, 240 + k]
        for k in range(30):
            o_effi[i, k] = y[i, 270 + k]
        o_prect[i] = y[i, 300]
        o_preci[i] = y[i, 301]
        for k in range(30):
            o_nevapr[i, k] = y[i, 302 + k]
        for k in range(30):
            o_evapsnow[i, k] = y[i, 332 + k]
        for k in range(30):
            o_am_evp_st[i, k] = y[i, 362 + k]
        for k in range(30):
            o_prain[i, k] = y[i, 392 + k]
        for k in range(30):
            o_prodsnow[i, k] = y[i, 422 + k]
        for k in range(30):
            o_cmeout[i, k] = y[i, 452 + k]
        for k in range(30):
            o_deffi[i, k] = y[i, 482 + k]
        for k in range(30):
            o_pgamrad[i, k] = y[i, 512 + k]
        for k in range(30):
            o_lamcrad[i, k] = y[i, 542 + k]
        for k in range(30):
            o_qsout[i, k] = y[i, 572 + k]
        for k in range(30):
            o_dsout[i, k] = y[i, 602 + k]
        for k in range(31):
            o_rflx[i, k] = y[i, 632 + k]
        for k in range(31):
            o_sflx[i, k] = y[i, 663 + k]
        for k in range(30):
            o_qrout[i, k] = y[i, 694 + k]
        for k in range(30):
            o_reff_rain[i, k] = y[i, 724 + k]
        for k in range(30):
            o_reff_snow[i, k] = y[i, 754 + k]
        for k in range(30):
            o_qcsevap[i, k] = y[i, 784 + k]
        for k in range(30):
            o_qisevap[i, k] = y[i, 814 + k]
        for k in range(30):
            o_qvres[i, k] = y[i, 844 + k]
        for k in range(30):
            o_cmeiout[i, k] = y[i, 874 + k]
        for k in range(30):
            o_vtrmc[i, k] = y[i, 904 + k]
        for k in range(30):
            o_vtrmi[i, k] = y[i, 934 + k]
        for k in range(30):
            o_qcsedten[i, k] = y[i, 964 + k]
        for k in range(30):
            o_qisedten[i, k] = y[i, 994 + k]
        for k in range(30):
            o_prao[i, k] = y[i, 1024 + k]
        for k in range(30):
            o_prco[i, k] = y[i, 1054 + k]
        for k in range(30):
            o_mnuccco[i, k] = y[i, 1084 + k]
        for k in range(30):
            o_mnuccto[i, k] = y[i, 1114 + k]
        for k in range(30):
            o_msacwio[i, k] = y[i, 1144 + k]
        for k in range(30):
            o_psacwso[i, k] = y[i, 1174 + k]
        for k in range(30):
            o_bergso[i, k] = y[i, 1204 + k]
        for k in range(30):
            o_bergo[i, k] = y[i, 1234 + k]
        for k in range(30):
            o_melto[i, k] = y[i, 1264 + k]
        for k in range(30):
            o_homoo[i, k] = y[i, 1294 + k]
        for k in range(30):
            o_qcreso[i, k] = y[i, 1324 + k]
        for k in range(30):
            o_prcio[i, k] = y[i, 1354 + k]
        for k in range(30):
            o_praio[i, k] = y[i, 1384 + k]
        for k in range(30):
            o_qireso[i, k] = y[i, 1414 + k]
        for k in range(30):
            o_mnuccro[i, k] = y[i, 1444 + k]
        for k in range(30):
            o_pracso[i, k] = y[i, 1474 + k]
        for k in range(30):
            o_meltsdt[i, k] = y[i, 1504 + k]
        for k in range(30):
            o_frzrdt[i, k] = y[i, 1534 + k]
        for k in range(30):
            o_mnuccdo[i, k] = y[i, 1564 + k]
        for k in range(30):
            o_nrout[i, k] = y[i, 1594 + k]
        for k in range(30):
            o_nsout[i, k] = y[i, 1624 + k]
        for k in range(30):
            o_refl[i, k] = y[i, 1654 + k]
        for k in range(30):
            o_arefl[i, k] = y[i, 1684 + k]
        for k in range(30):
            o_areflz[i, k] = y[i, 1714 + k]
        for k in range(30):
            o_frefl[i, k] = y[i, 1744 + k]
        for k in range(30):
            o_csrfl[i, k] = y[i, 1774 + k]
        for k in range(30):
            o_acsrfl[i, k] = y[i, 1804 + k]
        for k in range(30):
            o_fcsrfl[i, k] = y[i, 1834 + k]
        for k in range(30):
            o_rercld[i, k] = y[i, 1864 + k]
        for k in range(30):
            o_ncai[i, k] = y[i, 1894 + k]
        for k in range(30):
            o_ncal[i, k] = y[i, 1924 + k]
        for k in range(30):
            o_qrout2[i, k] = y[i, 1954 + k]
        for k in range(30):
            o_qsout2[i, k] = y[i, 1984 + k]
        for k in range(30):
            o_nrout2[i, k] = y[i, 2014 + k]
        for k in range(30):
            o_nsout2[i, k] = y[i, 2044 + k]
        for k in range(30):
            o_drout2[i, k] = y[i, 2074 + k]
        for k in range(30):
            o_dsout2[i, k] = y[i, 2104 + k]
        for k in range(30):
            o_freqs[i, k] = y[i, 2134 + k]
        for k in range(30):
            o_freqr[i, k] = y[i, 2164 + k]
        for k in range(30):
            o_nfice[i, k] = y[i, 2194 + k]
        for k in range(30):
            o_prer_evap[i, k] = y[i, 2224 + k]
        for k in range(30):
            o_preo[i, k] = y[i, 2254 + k]
        for k in range(30):
            o_prdso[i, k] = y[i, 2284 + k]
        for k in range(30):
            o_frzro[i, k] = y[i, 2314 + k]
        for k in range(30):
            o_meltso[i, k] = y[i, 2344 + k]
        for k in range(30):
            o_wtfc[i, k] = y[i, 2374 + k]
        for k in range(30):
            o_wtfi[i, k] = y[i, 2404 + k]
        for k in range(30):
            o_wtprelat[i, k] = y[i, 2434 + k]
        for k in range(30):
            o_wtpostlat[i, k] = y[i, 2464 + k]
        for k in range(30):
            o_qc[i, k] = qc[i, k]
        for k in range(30):
            o_qi[i, k] = qi[i, k]
        for k in range(30):
            o_nc[i, k] = nc[i, k]
        for k in range(30):
            o_ni[i, k] = ni[i, k]
