"""The part of a Python-driven CAM stage that is not about any one stage.

``Macrophysics`` proved that a CAM physics stage can run with its driver
layer in Python and every floating-point number still computed by the
oracle's Fortran.  Most of what that took is not about macrophysics at all:
binding a handles module's ``bind(C)`` entries, calling CAM's host services
(``physics_state_copy``, ``physics_ptend_init``, ``physics_update``,
``outfld``), auditing the image's direct kernels against the reviewed
descriptors, allocating one scratch array per kernel field, moving doubles
into and out of those arrays without arithmetic, and installing the whole
thing between the two halves of a split stage.

That is what lives here.  A stage subclasses :class:`NativeStage`, declares
the four names that locate it (its workflow process, the two halves it runs
between, and its Fortran prefix), says which kernels it runs and which
module constants it needs, and writes the one method that is genuinely its
own: the transliteration of its Fortran driver, statement for statement.

The contract every stage inherits is the one Gate B2 tests: **Python
computes no floating-point number.**  Nothing in this module does
arithmetic on a model value.  The copies are bit-exact moves of doubles;
the calls pass addresses.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..pi_cam.errors import PICAMConfigurationError
from ..pi_cam.facade import Physics
from ..pi_cam.kernel_codegen import load_direct_kernels
from .capture import lane_sha256
from .errors import PhysicsError

REPO = Path(__file__).resolve().parents[3]

#: The reviewed descriptors for every kernel promoted into the model image.
DESCRIPTORS = REPO / "native/pi_cam/direct_kernels_promoted.yaml"

_c = ctypes
_INT, _DBL, _STR = _c.c_int, _c.c_double, _c.c_char_p
_P_VOID = _c.POINTER(_c.c_void_p)
_P_INT = _c.POINTER(_c.c_int)
_P_I32 = _c.POINTER(_c.c_int32)
_P_I64 = _c.POINTER(_c.c_int64)
_P_DBL = _c.POINTER(_c.c_double)

#: The entries every ``pycam_<prefix>_handles`` module exposes, as
#: ``attribute -> (name, argtypes, optional)``.  ``{prefix}`` is the stage's
#: Fortran prefix; a name without it is shared by every stage.  ``None``
#: argtypes leaves the call undeclared, for wrappers whose pointers are
#: passed positionally.  ``optional`` entries are absent from images built
#: before they existed, and the stage falls back.
HOST_ENTRIES: dict[str, tuple[str, list | None, bool]] = {
    "set_owner": ("pycam_{prefix}_set_owner_v1", [_INT], False),
    "bind_hosts": ("pycam_{prefix}_bind_hosts_v1", [], False),
    "state_copy": ("pycam_{prefix}_state_copy_v1", [_INT], False),
    "state_dealloc": ("pycam_{prefix}_state_dealloc_v1", [_INT], False),
    "ptend_init": ("pycam_{prefix}_ptend_init_v1",
                   [_INT, _INT, _STR, _INT, _INT, _INT, _P_I32], False),
    "ptend_sum": ("pycam_{prefix}_ptend_sum_v1", [_INT, _INT], False),
    "update": ("pycam_{prefix}_update_v1", [_INT, _DBL], False),
    "view": ("pycam_{prefix}_view_v1", [_INT, _INT, _P_VOID, _P_INT, _P_I64], False),
    "outfld": ("pycam_outfld_v1", [_STR, _INT, _P_DBL, _INT, _INT], False),
    "nstep": ("pycam_{prefix}_nstep_v1", [], True),
    "dt": ("pycam_{prefix}_dt_v1", [], True),
}


def check(status: int, what: str) -> None:
    """A non-zero status from a handle entry is a refusal, never a warning."""

    if status != 0:
        raise PICAMConfigurationError(f"{what} refused with status {status}")


def as_view(pointer: ctypes.c_void_p, ndims: int, extents: Sequence[int]) -> np.ndarray:
    """A zero-copy F-ordered view of Fortran storage at ``pointer``."""

    shape = tuple(int(extents[i]) for i in range(ndims))
    count = int(np.prod(shape)) if shape else 1
    buffer = (ctypes.c_double * count).from_address(pointer.value)
    return np.ndarray(shape, dtype=np.float64, buffer=buffer, order="F")


def fortran(array: np.ndarray) -> np.ndarray:
    """An F-contiguous float64 array, itself if already so."""

    return np.asfortranarray(array, dtype=np.float64)


def pointer_of(array: np.ndarray):
    """The address of an F-contiguous double array, for a Fortran dummy."""

    assert array.flags.f_contiguous and array.dtype == np.float64
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class HostEntries:
    """One stage's handles module, bound through ctypes.

    Subclasses extend :attr:`TABLE` with the entries their own handles
    module adds beyond the shared set -- a wrapper around a routine that
    takes ``pbuf``, say, which cannot be promoted as a direct kernel.
    """

    TABLE: dict[str, tuple[str, list | None, bool]] = HOST_ENTRIES

    def __init__(self, library: Any, prefix: str) -> None:
        self.library = library
        self.prefix = prefix
        for attribute, (template, argtypes, optional) in self.TABLE.items():
            setattr(self, attribute,
                    self._bind(template.format(prefix=prefix), argtypes, optional=optional))

    def _bind(self, name: str, argtypes: list | None, *, optional: bool = False):
        try:
            function = getattr(self.library, name)
        except AttributeError as error:
            if optional:
                return None
            raise PICAMConfigurationError(
                f"the image exposes no {name}; it predates this stage's boundary"
            ) from error
        function.restype = ctypes.c_int
        if argtypes is not None:
            function.argtypes = argtypes
        return function


class HostServices:
    """CAM's host services for one stage, as calls that take plain arrays.

    Every method here is a call into the model image.  None of them
    computes anything: they copy a state, allocate or accumulate a ptend,
    apply one, hand a field to history, or return a view of storage the
    Fortran owns.
    """

    def __init__(self, entries: HostEntries, pcnst: int) -> None:
        self.e = entries
        self.pcnst = pcnst

    # -- storage -------------------------------------------------------------

    def _deref(self, entry, what: str, *arguments, ndims_max: int = 5) -> np.ndarray:
        pointer = ctypes.c_void_p()
        ndims = ctypes.c_int()
        extents = (ctypes.c_int64 * ndims_max)()
        check(entry(*arguments, ctypes.byref(pointer), ctypes.byref(ndims), extents), what)
        return as_view(pointer, ndims.value, extents)

    def view(self, lchnk: int, code: int) -> np.ndarray:
        """A zero-copy view of one component of the stage's held derived types."""

        return self._deref(self.e.view, f"{self.e.prefix} view(chunk {lchnk}, code {code})",
                           lchnk, code)

    # -- derived types -------------------------------------------------------

    def state_copy(self, lchnk: int) -> None:
        check(self.e.state_copy(lchnk), "physics_state_copy")

    def state_dealloc(self, lchnk: int) -> None:
        check(self.e.state_dealloc(lchnk), "physics_state_dealloc")

    def ptend_init(self, lchnk: int, which: int, name: str, *, ls: bool | None = None,
                   lq: np.ndarray | None = None) -> None:
        with_flags = lq is not None
        flags = np.zeros(self.pcnst, dtype=np.int32) if lq is None else np.asarray(lq, dtype=np.int32)
        assert flags.shape == (self.pcnst,)
        check(self.e.ptend_init(
            lchnk, which, name.encode("ascii"), len(name), int(with_flags),
            int(bool(ls)), flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ), f"physics_ptend_init({name!r})")

    def ptend_sum(self, lchnk: int, ncol: int) -> None:
        check(self.e.ptend_sum(lchnk, ncol), "physics_ptend_sum")

    def update(self, lchnk: int, dt: float) -> None:
        check(self.e.update(lchnk, float(dt)), "physics_update")

    # -- history -------------------------------------------------------------

    def outfld(self, name: str, array: np.ndarray, idim: int, lchnk: int) -> None:
        array = fortran(array)
        check(self.e.outfld(name.encode("ascii"), len(name), pointer_of(array), idim, lchnk),
              f"outfld({name!r})")


class Local(Mapping[str, np.ndarray]):
    """``scratch[name][..., 0]`` on every access, so late allocations are seen."""

    def __init__(self, scratch: dict[str, np.ndarray]) -> None:
        self._scratch = scratch

    def __getitem__(self, name: str) -> np.ndarray:
        return self._scratch[name][..., 0]

    def __iter__(self):
        return iter(self._scratch)

    def __len__(self) -> int:
        return len(self._scratch)


class StageRuntime:
    """What one stage needs on one rank, built once and kept on the stage.

    It binds the handles module, reads the stage's module constants, checks
    that the image's direct kernels take the fields the reviewed descriptors
    say they take, and allocates one scratch array per kernel field.  From
    then on it does one thing: run a kernel on one chunk, copying views in
    before the call and out after it.
    """

    def __init__(self, native: Any, stage: "NativeStage") -> None:
        self.native = native
        self.stage = stage
        self.rank = 0
        self.nstep = 0
        # <stage>.TRACE_ENV=<dir>: one JSON line per chunk per step with the
        # live-lane hash of every swappable-kernel argument before and after
        # the call, comparable record for record with a capture of the oracle.
        directory = os.environ.get(stage.TRACE_ENV)
        self.trace = None
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.trace = open(Path(directory) / f"{stage.PREFIX}_trace.rank-{os.getpid()}.jsonl", "a")
        library = native.library
        self.entries = stage.entries_class(library, stage.PREFIX)
        check(self.entries.bind_hosts(), f"pycam_{stage.PREFIX}_bind_hosts_v1")
        dims = native.pool.dimensions
        self.pcols, self.pver, self.pverp, self.pcnst = (
            int(dims[k]) for k in ("pcols", "pver", "pverp", "pcnst"))
        self.constants = stage.read_constants(library)
        stage.refuse_unsupported(self.constants)
        self.handles = stage.services_class(self.entries, self.pcnst)
        self.pbuf = stage.build_pbuf(library, self)
        # The image's manifest names each kernel's fields but not their
        # extents; the reviewed descriptors carry both.  Shapes come from the
        # descriptors, and the image is asked only to confirm it declares the
        # same fields in the same order.
        self.descriptors = {k.name: k for k in load_direct_kernels(stage.DESCRIPTORS)}
        for name in stage.KERNELS:
            declared = [a["field"] for a in native.kernel_arguments(name)]
            described = [a.field for a in self.descriptors[name].arguments]
            if declared != described:
                raise PICAMConfigurationError(
                    f"the image's {name} takes {declared}; the descriptors say {described}")
        self.scratch = self._allocate()
        stage.after_runtime(self)
        check(self.entries.set_owner(1), f"pycam_{stage.PREFIX}_set_owner_v1")

    # -- scratch -------------------------------------------------------------

    @property
    def extents(self) -> dict[str, int]:
        """Every extent name a descriptor may use, to its value on this rank."""

        sizes = {"pcols": self.pcols, "pver": self.pver, "pverp": self.pverp,
                 "pcnst": self.pcnst, "chunks": 1}
        sizes.update(self.stage.extra_extents(self.constants))
        return sizes

    def _allocate(self) -> dict[str, np.ndarray]:
        """One (..., 1) F-ordered array per kernel field, plus the driver's locals."""

        sizes = self.extents
        scratch: dict[str, np.ndarray] = {}
        for name in self.stage.KERNELS:
            if name in self.stage.UNSCRATCHED:
                continue
            for argument in self.descriptors[name].arguments:
                key = argument.field.removeprefix(f"{self.stage.PREFIX}.")
                if key in scratch:
                    continue
                shape = tuple(sizes[e] if e in sizes else int(e) for e in argument.extents)
                scratch[key] = np.zeros(shape, dtype=np.dtype(argument.dtype), order="F")
        for name, shape in self.stage.EXTRA_SCRATCH:
            scratch.setdefault(name, np.zeros(
                tuple(sizes[e] if e in sizes else int(e) for e in shape),
                dtype=np.float64, order="F"))
        return scratch

    @property
    def local(self) -> Local:
        """The scratch arrays with the chunk axis dropped: live views, never copies."""

        return Local(self.scratch)

    def _scratch_for(self, argument: Any, key: str) -> np.ndarray:
        sizes = self.extents
        shape_names = argument.extents or self.stage.FALLBACK_EXTENTS.get(key)
        if not shape_names or len(shape_names) != argument.rank:
            raise PICAMConfigurationError(
                f"no extents for kernel field {argument.field!r} (rank {argument.rank})")
        shape = tuple(sizes[e] if e in sizes else int(e) for e in shape_names)
        self.scratch[key] = np.zeros(shape, dtype=np.dtype(argument.dtype), order="F")
        return self.scratch[key]

    # -- the pool ------------------------------------------------------------

    def column(self, field: str, index: int) -> np.ndarray:
        """One chunk's lane of a StatePool field, F-ordered."""

        return fortran(np.asarray(self.native.pool[field])[:, index])

    def cam_in(self, index: int) -> dict[str, np.ndarray]:
        """The surface fields the stage reads, one chunk's lane each."""

        return {name: self.column(f"cam_in.{name}", index) for name in self.stage.CAM_IN}

    # -- running a kernel ----------------------------------------------------

    def kernel_on_chunk(self, name: str, inputs: Mapping[str, Any], *,
                        outputs: Mapping[str, Any], fields: Mapping[str, str] | None = None,
                        ncol: int | None = None) -> None:
        """Run one direct kernel on one chunk, copying views in and out exactly.

        Inputs that are handle or buffer views are copied into the kernel's
        scratch slice before the call; outputs mapped to a view are copied
        back after it -- live lanes only, since the Fortran driver never
        writes a padding lane and a view's padding must stay what CAM left
        there.  ``None`` means "the scratch array of the same name".  Every
        copy is a bit-exact move of doubles; no arithmetic happens here.
        """

        arrays: dict[str, np.ndarray] = {}
        inverse = {} if fields is None else {field: local for local, field in fields.items()}
        prefix = f"{self.stage.PREFIX}."
        for argument in self.descriptors[name].arguments:
            field = argument.field
            local = field.removeprefix(prefix) if fields is None else inverse[field]
            scratch = self.scratch[local] if local in self.scratch else self._scratch_for(argument, local)
            value = inputs.get(local)
            if value is not None:
                self._copy_in(scratch, value)
            arrays[field] = scratch
        self.native.run_kernel(name, arrays)
        for local, target in outputs.items():
            if target is None:
                continue
            self._copy_out(target, self.scratch[local], ncol)

    def swappable_kernel(self, name: str, inputs: Mapping[str, Any], *,
                         outputs: Mapping[str, np.ndarray], ncol: int, lchnk: int, dt: float,
                         kernel: Callable[..., Mapping[str, np.ndarray]] | None = None,
                         fields: Mapping[str, str] | None = None) -> None:
        """The one kernel of the stage a model may replace.

        With ``kernel`` unset the original Fortran runs through its direct
        kernel and the result is bit-for-bit.  With a callable in its place
        the live columns go in as ``(ncol, ...)`` arrays and the returned
        values are written back to the same lanes -- a full replacement, no
        per-column fallback.  Either way, if the stage is tracing, the
        live-lane hash of every argument is recorded before and after, so a
        run can be compared with a capture of the oracle argument by
        argument.
        """

        trace = self.trace
        before = ({key: lane_sha256(np.asarray(value), ncol) for key, value in inputs.items()}
                  if trace is not None else None)
        if kernel is None:
            self.kernel_on_chunk(name, inputs, outputs=outputs, fields=fields, ncol=ncol)
        else:
            batch = {}
            for key, value in inputs.items():
                array = np.asarray(value)
                batch[key] = array[:ncol].copy() if array.ndim else array
            answer = kernel(batch)
            missing = [key for key in outputs if key not in answer]
            if missing:
                raise PhysicsError(
                    f"kernel returned {len(answer)} of {len(outputs)} values; missing {missing}")
            for key, target in outputs.items():
                target[:ncol, ...] = np.asarray(answer[key], dtype=np.float64)
        if trace is not None:
            after = {key: lane_sha256(np.asarray(value), ncol) for key, value in outputs.items()}
            trace.write(json.dumps({
                "mpi_rank": self.rank, "lchnk": lchnk, "nstep": self.nstep, "ncol": ncol,
                "dt": dt, "kernel": name, "replaced": kernel is not None,
                "before": before, "after": after}) + "\n")
            trace.flush()

    # -- exact moves ---------------------------------------------------------

    @staticmethod
    def _copy_in(scratch: np.ndarray, value: Any) -> None:
        array = np.asarray(value)
        if array.ndim == 0:
            scratch[...] = array.astype(scratch.dtype)
        else:
            scratch[..., 0] = array.astype(scratch.dtype, copy=False)

    def _copy_out(self, target: np.ndarray, scratch: np.ndarray, ncol: int | None) -> None:
        source = scratch if target.ndim == 0 or target.ndim == scratch.ndim else scratch[..., 0]
        if ncol is None or target.ndim == 0 or target.shape[0] != self.pcols:
            target[...] = source
        else:
            target[:ncol, ...] = source[:ncol, ...]


class _StageProcess(Physics):
    """The installed process: no pool fields, native access, transactional off.

    A transliterated stage reads and writes nothing through the StatePool --
    every array it touches is CAM's own storage reached by handle -- so
    there is nothing for the transactional snapshot to protect, and the
    process must say so.
    """

    reads = ()
    writes = ()
    native = True
    transactional = False

    def __init__(self, stage: "NativeStage") -> None:
        self.stage = stage
        self.name = stage.PROCESS_NAME

    def run(self, state: Any, context: Any) -> None:
        self.stage.tend(state, context)


class NativeStage:
    """A CAM physics stage whose driver layer is Python.

    A subclass declares where the stage sits in the workflow, what its
    Fortran prefix is, which kernels it runs, and how to read its module
    constants -- then writes :meth:`tend_chunk`, the transliteration of its
    Fortran driver.  Everything else is inherited.
    """

    #: The whole stage as one workflow process, and the two halves that
    #: bracket the point where the Fortran driver used to be called.
    STAGE = ""
    FIRST_HALF = ""
    SECOND_HALF = ""

    #: The stage's Fortran prefix: ``pycam_<PREFIX>_*`` entries, ``<PREFIX>.``
    #: kernel field names.
    PREFIX = ""
    PROCESS_NAME = ""

    #: The direct kernels :meth:`tend_chunk` runs; every one must be in the
    #: image and in the reviewed descriptors with the same field list.
    KERNELS: tuple[str, ...] = ()
    #: Kernels whose fields are named after StatePool entries rather than the
    #: stage's own locals, so they get no scratch of their own up front.
    UNSCRATCHED: tuple[str, ...] = ()
    #: Extra scratch the driver keeps that no kernel declares.
    EXTRA_SCRATCH: tuple[tuple[str, tuple[str, ...]], ...] = ()
    #: Shapes for fields whose descriptor carries no extents.
    FALLBACK_EXTENTS: dict[str, tuple[str, ...]] = {}
    #: The ``cam_in`` surface fields the stage reads, by StatePool name.
    CAM_IN: tuple[str, ...] = ()

    DESCRIPTORS = DESCRIPTORS
    TRACE_ENV = "FREECAM_STAGE_TRACE"
    entries_class = HostEntries
    services_class = HostServices

    def __init__(self, *, kernel: Callable[..., Mapping[str, np.ndarray]] | None = None) -> None:
        #: What computes the stage's one replaceable kernel.  ``None`` runs
        #: the original Fortran and the stage is bit-for-bit; a callable --
        #: a ``torch.nn.Module`` wrapped to take and return the batch by
        #: name -- takes its place entirely.
        self.kernel = kernel
        self.calls: list[str] = []      # what tend() did last, for the sequence test
        self._process: Any = None
        self._runtimes: dict[int, StageRuntime] = {}

    # -- what a subclass supplies ------------------------------------------

    def read_constants(self, library: Any) -> Any:
        """The stage's module state, read once from the image."""

        raise NotImplementedError

    def refuse_unsupported(self, constants: Any) -> None:
        """Fail closed on any path the admitted configuration never takes."""

    def build_pbuf(self, library: Any, runtime: StageRuntime) -> Any:
        """The physics-buffer fields the stage reads, verified against the image."""

        return None

    def extra_extents(self, constants: Any) -> Mapping[str, int]:
        """Extent names beyond the pool's dimensions, from the module state."""

        return {}

    def after_runtime(self, runtime: StageRuntime) -> None:
        """A hook for whatever the stage must resolve once the scratch exists."""

    def tend_chunk(self, runtime: StageRuntime, lchnk: int, ncol: int, index: int,
                   dt: float, nstep: int) -> None:
        """The Fortran driver, statement for statement, for one chunk."""

        raise NotImplementedError

    # -- installing --------------------------------------------------------

    def attach(self, run: Any) -> Any:
        """Put :meth:`tend` between the two halves of this stage."""

        run.workflow.process(self.STAGE)     # fail early if this is not a PI-CAM workflow
        process = _StageProcess(self)
        run.workflow.process(self.STAGE).disable()
        for name in (self.FIRST_HALF, self.SECOND_HALF):
            run.workflow.process(name).enable()
        self._process = run.workflow.insert_after(self.FIRST_HALF, process)
        return self._process

    def runtime(self, native: Any) -> StageRuntime:
        """This rank's runtime, built on first use and kept per pool."""

        key = id(native.pool)
        try:
            return self._runtimes[key]
        except KeyError:
            built = StageRuntime(native, self)
            self._runtimes[key] = built
            return built

    # -- running -----------------------------------------------------------

    def tend(self, fields: Any, context: Any) -> None:
        """The Fortran driver, for every chunk this rank owns."""

        native = context.native
        if native is None:
            raise PhysicsError(f"{type(self).__name__}.tend must run as a native process")
        runtime = self.runtime(native)
        entries = runtime.entries
        dt = float(entries.dt()) if entries.dt is not None else float(context.timestep_seconds)
        nstep = int(entries.nstep()) if entries.nstep is not None else int(context.step)
        runtime.rank = int(context.rank)
        runtime.nstep = nstep
        self.calls = []
        for index, (lchnk, ncol) in enumerate(zip(*native.chunks)):
            self.tend_chunk(runtime, int(lchnk), int(ncol), index, dt, nstep)


__all__ = [
    "DESCRIPTORS", "HOST_ENTRIES", "HostEntries", "HostServices", "Local",
    "NativeStage", "StageRuntime", "as_view", "check", "fortran", "pointer_of",
]
