# cython: language_level=3
"""Direct calls into the image for the walk's hottest crossings -- a trial.

Each class holds a C function pointer taken from a ctypes function object
and the addresses of the ctypes out-arguments the Python code already
keeps, and makes the call without ctypes' per-call argument marshalling.
No arithmetic; the Fortran entries are the same ones the ctypes path calls.
"""
from libc.stdint cimport int32_t, int64_t, uintptr_t

ctypedef int (*kernel_fn)(int32_t, int32_t, void**, int32_t*, int64_t*, int32_t, int32_t, char*, int32_t) noexcept nogil
ctypedef int (*probe2_fn)(int32_t, int32_t, void**, int32_t*, int64_t*) noexcept nogil
ctypedef int (*pbuf1_fn)(int32_t, int32_t, int32_t, void**, int64_t*) noexcept nogil
ctypedef int (*pbuf2_fn)(int32_t, int32_t, int32_t, int32_t, int32_t, void**, int32_t*, int64_t*) noexcept nogil
ctypedef int (*outfld_fn)(char*, int32_t, double*, int32_t, int32_t) noexcept nogil


cdef class KernelCall:
    """A bound direct-kernel call: BoundCall's tables, invoked directly."""
    cdef kernel_fn fn
    cdef int32_t action_id, count, max_rank, fcomm, errlen
    cdef void** pointers
    cdef int32_t* ndims
    cdef int64_t* shapes
    cdef char* errmsg
    cdef object keep

    def __init__(self, uintptr_t fn, int action_id, int count, uintptr_t pointers, uintptr_t ndims,
                 uintptr_t shapes, int max_rank, int fcomm, uintptr_t errmsg, int errlen, keep):
        self.fn = <kernel_fn>fn
        self.action_id = action_id; self.count = count; self.max_rank = max_rank
        self.fcomm = fcomm; self.errlen = errlen
        self.pointers = <void**>pointers; self.ndims = <int32_t*>ndims; self.shapes = <int64_t*>shapes
        self.errmsg = <char*>errmsg
        self.keep = keep

    def __call__(self):
        self.errmsg[0] = 0
        return self.fn(self.action_id, self.count, self.pointers, self.ndims, self.shapes,
                       self.max_rank, self.fcomm, self.errmsg, self.errlen)


cdef class Probe2:
    """``entry(a, b, &ptr, &ndims, extents)``: the address, rank and shape it answers."""
    cdef probe2_fn fn
    cdef void* ptr
    cdef int32_t nd
    cdef int64_t extents[5]

    def __init__(self, uintptr_t fn):
        self.fn = <probe2_fn>fn

    def __call__(self, int a, int b):
        cdef int status, i
        self.ptr = NULL; self.nd = 0
        status = self.fn(a, b, &self.ptr, &self.nd, self.extents)
        if status != 0:
            return status, 0, 0, ()
        return 0, <uintptr_t>self.ptr, self.nd, tuple([self.extents[i] for i in range(self.nd)])


cdef class PBufProbe:
    """The physics buffer's two accessors, answering (status, address, shape)."""
    cdef pbuf1_fn fn1
    cdef pbuf2_fn fn2
    cdef void* ptr
    cdef int32_t nd
    cdef int64_t extents[3]

    def __init__(self, uintptr_t fn1, uintptr_t fn2):
        self.fn1 = <pbuf1_fn>fn1
        self.fn2 = <pbuf2_fn>fn2

    def plane(self, int chunk, int index, int time_sliced):
        cdef int status
        self.ptr = NULL
        status = self.fn1(chunk, index, time_sliced, &self.ptr, self.extents)
        if status != 0:
            return status, 0, ()
        return 0, <uintptr_t>self.ptr, (self.extents[0], self.extents[1])

    def any(self, int chunk, int index, int time_sliced, int rank, int is_integer):
        cdef int status, i
        self.ptr = NULL; self.nd = 0
        status = self.fn2(chunk, index, time_sliced, rank, is_integer, &self.ptr, &self.nd, self.extents)
        if status != 0:
            return status, 0, 0, ()
        return 0, <uintptr_t>self.ptr, self.nd, tuple([self.extents[i] for i in range(self.nd)])


cdef class Outfld:
    """``outfld(name, len, array, idim, lchnk)`` by address."""
    cdef outfld_fn fn

    def __init__(self, uintptr_t fn):
        self.fn = <outfld_fn>fn

    def __call__(self, bytes name, int n, uintptr_t data, int idim, int lchnk):
        return self.fn(<char*>name, n, <double*>data, idim, lchnk)


def address_of(function):
    """The C address of a ctypes function object."""
    import ctypes
    return ctypes.cast(function, ctypes.c_void_p).value
