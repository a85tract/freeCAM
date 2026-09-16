"""A kernel with mmacro_pcond's arguments that writes zeros: for shadow runs, the plugin
mechanism's cost alone -- the hook's pointer tables, the Numba adapter, the call -- with the
original answering.  Generated from the hook's model block; not physics.
"""


def mmacro_pcond_zero(dt, p, dp, t0, qv0, ql0, qi0, nl0, ni0, a_t, a_qv, a_ql, a_qi, a_nl, a_ni, c_t, c_qv, c_ql, c_qi, c_nl, c_ni, c_qlst, d_t, d_qv, d_ql, d_qi, d_nl, d_ni, a_cud, a_cu0, clrw_old, clri_old, landfrac, snowh, t0_out, qv0_out, ql0_out, qi0_out, nl0_out, ni0_out, s_tendout, qv_tendout, ql_tendout, qi_tendout, nl_tendout, ni_tendout, qme, qvadj, qladj, qiadj, qllim, qilim, cld, al_st_star, ai_st_star, ql_st_star, qi_st_star):
    t0_out[:] = 0.0
    qv0_out[:] = 0.0
    ql0_out[:] = 0.0
    qi0_out[:] = 0.0
    nl0_out[:] = 0.0
    ni0_out[:] = 0.0
    s_tendout[:] = 0.0
    qv_tendout[:] = 0.0
    ql_tendout[:] = 0.0
    qi_tendout[:] = 0.0
    nl_tendout[:] = 0.0
    ni_tendout[:] = 0.0
    qme[:] = 0.0
    qvadj[:] = 0.0
    qladj[:] = 0.0
    qiadj[:] = 0.0
    qllim[:] = 0.0
    qilim[:] = 0.0
    cld[:] = 0.0
    al_st_star[:] = 0.0
    ai_st_star[:] = 0.0
    ql_st_star[:] = 0.0
    qi_st_star[:] = 0.0
