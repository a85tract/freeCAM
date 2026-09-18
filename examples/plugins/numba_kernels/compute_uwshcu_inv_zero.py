"""A kernel with compute_uwshcu_inv's arguments that writes zeros: for shadow runs, the plugin
mechanism's cost alone -- the hook's pointer tables, the Numba adapter, the call -- with the
original answering.  Generated from the hook's model block; not physics.
"""


def compute_uwshcu_inv_zero(dt, ps0_inv, zs0_inv, p0_inv, z0_inv, dp0_inv, u0_inv, v0_inv, qv0_inv, ql0_inv, qi0_inv, t0_inv, s0_inv, tr0_inv, tke_inv, cldfrct_inv, concldfrct_inv, pblh, cush, dpdry0_inv, cush_out, umf_inv, slflx_inv, qtflx_inv, flxprc1_inv, flxsnow1_inv, qvten_inv, qlten_inv, qiten_inv, sten_inv, uten_inv, vten_inv, trten_inv, qrten_inv, qsten_inv, precip, snow, evapc_inv, cufrc_inv, qcu_inv, qlu_inv, qiu_inv, cbmf, qc_inv, rliq, cnt_inv, cnb_inv, wtprec, wtsnow, wtqc_inv):
    cush_out[:] = 0.0
    umf_inv[:] = 0.0
    slflx_inv[:] = 0.0
    qtflx_inv[:] = 0.0
    flxprc1_inv[:] = 0.0
    flxsnow1_inv[:] = 0.0
    qvten_inv[:] = 0.0
    qlten_inv[:] = 0.0
    qiten_inv[:] = 0.0
    sten_inv[:] = 0.0
    uten_inv[:] = 0.0
    vten_inv[:] = 0.0
    trten_inv[:] = 0.0
    qrten_inv[:] = 0.0
    qsten_inv[:] = 0.0
    precip[:] = 0.0
    snow[:] = 0.0
    evapc_inv[:] = 0.0
    cufrc_inv[:] = 0.0
    qcu_inv[:] = 0.0
    qlu_inv[:] = 0.0
    qiu_inv[:] = 0.0
    cbmf[:] = 0.0
    qc_inv[:] = 0.0
    rliq[:] = 0.0
    cnt_inv[:] = 0.0
    cnb_inv[:] = 0.0
    wtprec[:] = 0.0
    wtsnow[:] = 0.0
    wtqc_inv[:] = 0.0
