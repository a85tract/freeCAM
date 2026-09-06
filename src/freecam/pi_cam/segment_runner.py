"""The image's segment runners, spoken from Python.

A segment runner is one stage's original Fortran made pausable at named
kernel call sites: ``pycam_<prefix>_start_v1`` runs from the top of the
stage and returns when a replaced kernel would have been called,
``_frame_v1`` describes that call's arguments as views of the Fortran
storage, ``_resume_v1`` continues past it.  Which stages have one, and under
what prefix, is the manifest ``native/pi_cam/segment_runners.yaml`` -- the
backend consults it rather than knowing any stage by name -- and each is
presented as the :class:`~freecam.physics.segments.SegmentRunner` a
:class:`~freecam.physics.segments.SegmentedStage` drives.  A frame's
argument names and intents come from the same reviewed descriptor the
kernel's direct form is built from, in the call's order.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from freecam.physics.segments import FrameArgument, KernelFrame, SegmentEvent
from .errors import NativeCAMError
from .kernel_codegen import load_direct_kernels
from .tables import load_table

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "native/pi_cam/segment_runners.yaml"

DTYPES = {1: np.float64, 2: np.int32}
INTENTS = {0: "in", 1: "out", 2: "inout"}
#: The entries every runner exports, as ``<prefix>_<suffix>_v1``.
ENTRY_SUFFIXES = ("create", "start", "frame", "resume", "error", "reset", "destroy")


@dataclass(frozen=True, slots=True)
class RunnerKernel:
    """One kernel a runner can pause at."""

    name: str
    owner: str
    validated_by: tuple[str, ...] = ()
    #: a reviewed function contract whose argument list is the frame's, in
    #: order; None when the frame follows the kernel's direct-kernel descriptor
    contract: str | None = None
    #: a frame descriptor table (segment_frames.yaml) the pausable generator wrote
    frame: str | None = None

    @property
    def validated(self) -> bool:
        """Whether every gate record named exists in this checkout."""

        return bool(self.validated_by) and all((REPO / path).is_file() for path in self.validated_by)


@dataclass(frozen=True, slots=True)
class RunnerSpec:
    """A runner as the manifest describes it."""

    stage: str
    prefix: str
    module: str
    generator: str
    descriptors: str
    kernels: tuple[RunnerKernel, ...] = field(default_factory=tuple)
    #: whether the runner exports `<prefix>_original_v1`, running the paused call itself
    original: bool = False

    @property
    def kernel_names(self) -> tuple[str, ...]:
        return tuple(kernel.name for kernel in self.kernels)

    @property
    def entries(self) -> tuple[str, ...]:
        return tuple(f"{self.prefix}_{suffix}_v1" for suffix in ENTRY_SUFFIXES)

    def kernel_id(self, name: str) -> int:
        """The module's number for ``name``: its position in the manifest plus one."""

        return self.kernel_names.index(name) + 1

    def kernel(self, name: str) -> RunnerKernel:
        for kernel in self.kernels:
            if kernel.name == name:
                return kernel
        raise KeyError(name)


def load_manifest(path: str | Path | None = None) -> tuple[RunnerSpec, ...]:
    """Every runner the manifest declares, in its order."""

    source = Path(path) if path is not None else MANIFEST
    payload = load_table(source)
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", 0)) != 1:
        raise NativeCAMError(f"{source}: segment runner manifest requires schema_version: 1")
    runners = payload.get("runners")
    if not isinstance(runners, list):
        raise NativeCAMError(f"{source}: 'runners' must be a list")
    specs: list[RunnerSpec] = []
    seen_stages: set[str] = set()
    seen_kernels: set[str] = set()
    for record in runners:
        if not isinstance(record, Mapping):
            raise NativeCAMError(f"{source}: every runner must be a mapping")
        kernels = []
        for item in record.get("kernels") or ():
            name = str(item["name"])
            if name in seen_kernels:
                raise NativeCAMError(f"{source}: kernel {name!r} is claimed by two runners")
            seen_kernels.add(name)
            kernels.append(RunnerKernel(
                name=name, owner=str(item.get("owner", "")),
                validated_by=tuple(str(p) for p in item.get("validated_by") or ()),
                contract=None if item.get("contract") is None else str(item["contract"]),
                frame=None if item.get("frame") is None else str(item["frame"]),
            ))
        stage = str(record["stage"])
        if stage in seen_stages:
            raise NativeCAMError(f"{source}: stage {stage!r} has two runners")
        seen_stages.add(stage)
        if not kernels:
            raise NativeCAMError(f"{source}: runner for {stage!r} pauses at no kernel")
        specs.append(RunnerSpec(
            stage=stage, prefix=str(record["prefix"]), module=str(record["module"]),
            generator=str(record["generator"]), descriptors=str(record["descriptors"]),
            kernels=tuple(kernels), original=bool(record.get("original", False)),
        ))
    return tuple(specs)


def runner_spec(stage: str, path: str | Path | None = None) -> RunnerSpec | None:
    """The manifest's runner for ``stage``, or None when it has none."""

    for spec in load_manifest(path):
        if spec.stage == stage:
            return spec
    return None


def runner_kernels(path: str | Path | None = None) -> dict[str, tuple[str, ...]]:
    """stage -> the kernels its runner pauses at, for every runner in the manifest."""

    return {spec.stage: spec.kernel_names for spec in load_manifest(path)}


def bindable_kernels(path: str | Path | None = None) -> tuple[str, ...]:
    """Every kernel some runner pauses at."""

    return tuple(name for spec in load_manifest(path) for name in spec.kernel_names)


def frame_names_from_descriptor(path: Path, kernel: str) -> tuple[str, ...]:
    """The slots of ``kernel`` in a pausable runner's frame table, in order."""

    payload = load_table(path)
    slots = (payload.get("kernels") or {}).get(kernel)
    if not slots:
        raise NativeCAMError(f"{path} describes no frame for kernel {kernel!r}")
    return tuple(str(slot["name"]) for slot in slots)


def frame_names_from_contract(path: Path) -> tuple[str, ...]:
    """The routine's arguments in order, less the character ones no model answers."""

    from freecam.physics.spec import load_function_spec

    spec = load_function_spec(path)
    return tuple(a.name for a in spec.arguments if not a.fortran_type.lower().startswith("character"))


def image_offers_runner(library: Any, spec: RunnerSpec | None = None) -> bool:
    """Whether ``library`` exports the runner's entries."""

    if spec is None:
        spec = runner_spec(STAGE)
        if spec is None:
            return False
    return all(hasattr(library, name) for name in spec.entries)


class ImageSegmentRunner:
    """A runner's ``pycam_<prefix>_*`` entries as a SegmentRunner."""

    def __init__(self, library: Any, spec: RunnerSpec, descriptors: str | Path | None = None) -> None:
        self.library = library
        self.spec = spec
        #: the kernels this runner can pause at; a stage chooses segmented
        #: execution under ``auto`` only when every replaced kernel is one
        self.kernels = spec.kernel_names
        path = Path(descriptors) if descriptors is not None else REPO / spec.descriptors
        described = {k.name: k for k in load_direct_kernels(path)}
        #: kernel -> the frame's argument names in the call's order, without the stage prefix
        self.names: dict[str, tuple[str, ...]] = {}
        for kernel in spec.kernels:
            if kernel.frame is not None:
                self.names[kernel.name] = frame_names_from_descriptor(REPO / kernel.frame, kernel.name)
            elif kernel.contract is not None:
                self.names[kernel.name] = frame_names_from_contract(REPO / kernel.contract)
            elif kernel.name in described:
                self.names[kernel.name] = tuple(a.field.split(".", 1)[1] for a in described[kernel.name].arguments)
            else:
                raise NativeCAMError(
                    f"{spec.descriptors} describes no kernel named {kernel.name!r} and the manifest "
                    f"names no contract; the runner for {spec.stage!r} cannot decode its frames")
        self.slots = max(len(names) for names in self.names.values())
        self._entry = {suffix: getattr(library, f"{spec.prefix}_{suffix}_v1") for suffix in ENTRY_SUFFIXES}
        self._original = getattr(library, f"{spec.prefix}_original_v1", None) if spec.original else None
        if spec.original and self._original is None:
            raise NativeCAMError(f"the manifest says the {spec.prefix} runner runs the original at a pause, "
                                 f"but the image exports no {spec.prefix}_original_v1")
        self._bind()

    def _bind(self) -> None:
        c_int, c_int64, c_ptr = ctypes.c_int, ctypes.c_int64, ctypes.c_void_p
        pi = ctypes.POINTER(c_int)
        for entry in self._entry.values():
            entry.restype = c_int
        self._entry["create"].argtypes = [pi]
        self._entry["start"].argtypes = [c_int, c_int, pi, pi]
        self._entry["frame"].argtypes = [c_int, pi, pi, pi, pi, pi, pi, c_int,
                                         ctypes.POINTER(c_ptr), pi, ctypes.POINTER(c_int64), pi, pi]
        self._entry["resume"].argtypes = [c_int, c_int, c_int, pi]
        self._entry["error"].argtypes = [c_int, ctypes.c_char_p, c_int]
        self._entry["reset"].argtypes = [c_int]
        self._entry["destroy"].argtypes = [c_int]
        if self._original is not None:
            self._original.restype = c_int
            self._original.argtypes = [c_int, c_int]

    @property
    def runs_original(self) -> bool:
        """Whether this runner can run the paused call itself (see run_original)."""

        return self._original is not None

    def run_original(self, context: int, kernel: str) -> None:
        """Run the original call on the paused frame's storage, in the runner."""

        if self._original is None:
            raise NativeCAMError(f"the {self.spec.prefix} runner does not run the original at a pause")
        status = self._original(context, self.spec.kernel_id(kernel))
        if status:
            raise NativeCAMError(f"{self.spec.prefix} runner refused to run the original ({status}): {self.error(context)}")

    # -- the SegmentRunner protocol -------------------------------------------

    def create(self, stage: str) -> int:
        if stage != self.spec.stage:
            raise NativeCAMError(f"the image's {self.spec.prefix} runner does not run {stage!r}")
        context = ctypes.c_int(0)
        status = self._entry["create"](ctypes.byref(context))
        if status:
            raise NativeCAMError(f"{self.spec.prefix} runner refused to create a context ({status}): {self.error(0)}")
        return int(context.value)

    def start(self, context: int, mask: Mapping[str, bool]) -> SegmentEvent:
        unknown = sorted(name for name, replaced in mask.items() if replaced and name not in self.kernels)
        if unknown:
            raise NativeCAMError(f"the {self.spec.prefix} runner cannot pause at {unknown}; it pauses at {list(self.kernels)}")
        flags = (ctypes.c_int * len(self.kernels))(*(1 if mask.get(name) else 0 for name in self.kernels))
        event = ctypes.c_int(2)
        status = self._entry["start"](context, len(self.kernels), flags, ctypes.byref(event))
        if status:
            raise NativeCAMError(f"{self.spec.prefix} runner refused to start ({status}): {self.error(context)}")
        return SegmentEvent(event.value)

    def frame(self, context: int) -> KernelFrame:
        n = self.slots
        kernel, index, lchnk, ncol, substep, token = (ctypes.c_int(0) for _ in range(6))
        pointers = (ctypes.c_void_p * n)()
        ndims = (ctypes.c_int * n)()
        shapes = (ctypes.c_int64 * (3 * n))()
        dtypes = (ctypes.c_int * n)()
        intents = (ctypes.c_int * n)()
        status = self._entry["frame"](
            context, ctypes.byref(kernel), ctypes.byref(index), ctypes.byref(lchnk), ctypes.byref(ncol),
            ctypes.byref(substep), ctypes.byref(token), n, pointers, ndims, shapes, dtypes, intents)
        if status:
            raise NativeCAMError(f"{self.spec.prefix} runner has no frame ({status}): {self.error(context)}")
        if not 1 <= kernel.value <= len(self.kernels):
            raise NativeCAMError(f"{self.spec.prefix} runner paused on kernel id {kernel.value}, which it does not declare")
        name = self.kernels[kernel.value - 1]
        arguments = []
        for i, argument in enumerate(self.names[name]):
            rank = ndims[i]
            shape = tuple(int(shapes[3 * i + axis]) for axis in range(rank))
            dtype = np.dtype(DTYPES[dtypes[i]])
            count = int(np.prod(shape)) if shape else 1
            if count == 0 or not pointers[i]:
                # no storage behind this argument in this call (a field the
                # configuration never registered, or no packed column)
                array = np.zeros(shape, dtype=dtype, order="F")
            else:
                buffer = (ctypes.c_byte * (count * dtype.itemsize)).from_address(pointers[i])
                array = np.ndarray(shape, dtype=dtype, buffer=buffer, order="F")
            arguments.append(FrameArgument(argument, array, INTENTS[intents[i]]))
        return KernelFrame(kernel=name, call_index=index.value, lchnk=lchnk.value,
                           ncol=ncol.value, substep=substep.value, arguments=tuple(arguments),
                           token=token.value)

    def resume(self, context: int, kernel: str, token: int) -> SegmentEvent:
        event = ctypes.c_int(2)
        status = self._entry["resume"](context, self.spec.kernel_id(kernel), int(token), ctypes.byref(event))
        if status:
            raise NativeCAMError(f"{self.spec.prefix} runner refused to resume ({status}): {self.error(context)}")
        return SegmentEvent(event.value)

    def error(self, context: int) -> str:
        buffer = ctypes.create_string_buffer(256)
        self._entry["error"](context, buffer, len(buffer))
        return buffer.value.decode(errors="replace")

    def reset(self, context: int) -> None:
        self._entry["reset"](context)

    def destroy(self, context: int) -> None:
        self._entry["destroy"](context)


def runner_for(library: Any, stage: str, manifest: str | Path | None = None) -> ImageSegmentRunner | None:
    """The runner ``library`` offers for ``stage``, or None when the manifest or the image has none."""

    spec = runner_spec(stage, manifest)
    if spec is None or not image_offers_runner(library, spec):
        return None
    return ImageSegmentRunner(library, spec)


# -- the stage-7 runner by its historical names ------------------------------------

STAGE = "cam_run1.cloud_macro_microphysics"
_STAGE7 = runner_spec(STAGE)
#: the stage-7 runner's kernels and entries, as its tests and the builder first knew them
KERNELS: tuple[str, ...] = _STAGE7.kernel_names if _STAGE7 is not None else ()
ENTRIES: tuple[str, ...] = _STAGE7.entries if _STAGE7 is not None else ()


class StageSevenRunner(ImageSegmentRunner):
    """The ``pycam_stage7_*`` entries as a SegmentRunner: the first runner, by its old name."""

    def __init__(self, library: Any, descriptors: str | Path | None = None) -> None:
        spec = runner_spec(STAGE)
        if spec is None:
            raise NativeCAMError(f"the manifest declares no runner for {STAGE!r}")
        super().__init__(library, spec, descriptors)


__all__ = ["ENTRIES", "ENTRY_SUFFIXES", "KERNELS", "MANIFEST", "STAGE", "ImageSegmentRunner",
           "RunnerKernel", "RunnerSpec", "StageSevenRunner", "bindable_kernels", "frame_names_from_contract",
           "frame_names_from_descriptor",
           "image_offers_runner", "load_manifest", "runner_for", "runner_kernels", "runner_spec"]
