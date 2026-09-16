"""A kernel with momtran's arguments that writes zeros: for shadow runs, the plugin
mechanism's cost alone -- the hook's pointer tables, the Numba adapter, the call -- with the
original answering.  Generated from the hook's model block; not physics.
"""


def momtran_zero(lchnk, ncol, q, mu, md, du, eu, ed, dp, dsubcld, il1g, il2g, nstep, dt, dqdt, pguall, pgdall, icwu, icwd, seten):
    dqdt[:] = 0.0
    pguall[:] = 0.0
    pgdall[:] = 0.0
    icwu[:] = 0.0
    icwd[:] = 0.0
    seten[:] = 0.0
