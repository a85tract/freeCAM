"""A kernel with compute_tms's arguments that writes zeros: for shadow runs, the plugin
mechanism's cost alone -- the hook's pointer tables, the Numba adapter, the call -- with the
original answering.  Generated from the hook's model block; not physics.
"""


def compute_tms_zero(ncol, u, v, t, pmid, exner, zm, sgh, landfrac, ksrf, taux, tauy):
    ksrf[:] = 0.0
    taux[:] = 0.0
    tauy[:] = 0.0
