"""A kernel with compute_eddy_diff's arguments that writes zeros: for shadow runs, the plugin
mechanism's cost alone -- the hook's pointer tables, the Numba adapter, the call -- with the
original answering.  Generated from the hook's model block; not physics.
"""


def compute_eddy_diff_zero(lchnk, ncol, t, qv, ztodt, ql, qi, s, pdel, rpdel, cldn, qrl, wsedl, z, zi, pmid, pi, u, v, taux, tauy, shflx, qflx, nturb, kvm_in, kvh_in, tauresx, tauresy, ksrftms, rrho, ustar, pblh, kvm_out, kvh_out, kvq, cgh, cgs, tpert, qpert, wpert, tke, bprod, sprod, sfi, tauresx_out, tauresy_out, wstarpbl, tkes, went, sm_aw):
    rrho[:] = 0.0
    ustar[:] = 0.0
    pblh[:] = 0.0
    kvm_out[:] = 0.0
    kvh_out[:] = 0.0
    kvq[:] = 0.0
    cgh[:] = 0.0
    cgs[:] = 0.0
    tpert[:] = 0.0
    qpert[:] = 0.0
    wpert[:] = 0.0
    tke[:] = 0.0
    bprod[:] = 0.0
    sprod[:] = 0.0
    sfi[:] = 0.0
    tauresx_out[:] = 0.0
    tauresy_out[:] = 0.0
    wstarpbl[:] = 0.0
    tkes[:] = 0.0
    went[:] = 0.0
    sm_aw[:] = 0.0
