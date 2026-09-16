"""A kernel with zm_conv_evap's arguments that writes zeros: for shadow runs, the plugin
mechanism's cost alone -- the hook's pointer tables, the Numba adapter, the call -- with the
original answering.  Generated from the hook's model block; not physics.
"""


def zm_conv_evap_zero(ncol, lchnk, t, pmid, pdel, q, landfrac, tend_s, tend_q, prdprec, cldfrc, deltat, prec, tend_s_out, tend_s_snwprd, tend_s_snwevmlt, tend_q_out, prec_out, snow, evpstore, substore, ntprprd, ntsnprd, flxprec, flxsnow):
    tend_s_out[:] = 0.0
    tend_s_snwprd[:] = 0.0
    tend_s_snwevmlt[:] = 0.0
    tend_q_out[:] = 0.0
    prec_out[:] = 0.0
    snow[:] = 0.0
    evpstore[:] = 0.0
    substore[:] = 0.0
    ntprprd[:] = 0.0
    ntsnprd[:] = 0.0
    flxprec[:] = 0.0
    flxsnow[:] = 0.0
