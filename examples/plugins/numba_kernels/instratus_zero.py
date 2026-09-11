"""A kernel with instratus_condensate's arguments that writes zeros: for shadow runs, the plugin
mechanism's cost alone -- the hook's pointer tables, the Numba adapter, the call -- with the
original answering.  Generated from the hook's model block; not physics.
"""


def instratus_zero(k, p_in, t0_in, qv0_in, ql0_in, qi0_in, ni0_in, a_dc_in, ql_dc_in, qi_dc_in, a_sc_in, ql_sc_in, qi_sc_in, landfrac, snowh, rhmini_in, rhminl_in, rhminl_adj_land_in, rhminh_in, t_out, qv_out, ql_out, qi_out, al_st_out, ai_st_out, ql_st_out, qi_st_out):
    t_out[:] = 0.0
    qv_out[:] = 0.0
    ql_out[:] = 0.0
    qi_out[:] = 0.0
    al_st_out[:] = 0.0
    ai_st_out[:] = 0.0
    ql_st_out[:] = 0.0
    qi_st_out[:] = 0.0
