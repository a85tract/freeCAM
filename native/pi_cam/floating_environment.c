#include <xmmintrin.h>

/*
 * Intel Fortran's executable startup installs this MXCSR control word when
 * CAM is compiled with -ftz.  Python is the process main in PyCAM, so that
 * startup hook never runs.  Restore the source executable's floating-point
 * environment before any CAM numerical routine executes.
 */
void pycam_pi_cam_set_fp_environment_v1(void) {
    _mm_setcsr(0x9fc0u);
}

unsigned int pycam_pi_cam_get_mxcsr_v1(void) {
    return _mm_getcsr();
}
