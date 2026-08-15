/*
 * Link anchor for the original non-PIC CESM component/coupler objects.
 *
 * The resulting executable header is converted to ET_DYN and loaded only via
 * ctypes.CDLL.  This main function is therefore never used during FreeCAM,
 * but keeping an inert native entry point lets the original link layout stay
 * unchanged without embedding CPython or introducing any C/Fortran-to-Python
 * control path.
 */

int main(void) { return 64; }
