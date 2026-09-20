"""The hook table: kernels reached inside compiled routines by symbol redirection.

``native/pi_cam/hooks.yaml`` names each hooked kernel, its reviewed function
contract (the argument list the hook carries and the frame Python sees), how
the original is reached, and which caller objects have their references
redirected.  The generator writes ``pycam_hooks.F90`` from it, the device
build performs and audits the redirections, and the runtime reads the call
counts every hook keeps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import PICAMConfigurationError

REPO = Path(__file__).resolve().parents[3]
HOOKS = REPO / "native/pi_cam/hooks.yaml"
HOOK_MODULE = REPO / "native/pi_cam/support/pycam_hooks.F90"
REDIRECTS = ("rename-references", "weaken-definition")
#: how the hook procedure is bound: ``c`` (bind(C), interoperable dummies, can pause for Python)
#: or ``fortran`` (a plain module procedure with the callee's own logical, character and
#: pointer dummies; it can answer with the original or a bound model, never pause)
BINDINGS = ("c", "fortran")


@dataclass(frozen=True, slots=True)
class HookCaller:
    object: str
    routine: str


@dataclass(frozen=True, slots=True)
class Hook:
    kernel: str
    contract: str
    callee_symbol: str
    redirect: str
    original_module: str | None
    original_routine: str | None
    original_symbol: str | None
    callers: tuple[HookCaller, ...] = field(default_factory=tuple)
    #: contract arguments a bound TorchScript model takes, in its forward's order;
    #: empty when the hook cannot be given a model
    model_inputs: tuple[str, ...] = ()
    #: the contract outputs the model returns, in order
    model_outputs: tuple[str, ...] = ()
    #: whether the model returns its outputs as one (columns, width) tensor, the outputs
    #: laid side by side in ``model_outputs`` order, each flattened per column
    model_packed: bool = False
    #: contract outputs the hook zeroes itself instead of taking from the model
    model_zero_outputs: tuple[str, ...] = ()
    #: outputs the model returns only at some 1-based indices of their last axis, in that order
    #: inside the packed tensor; the hook zeroes the rest of the array.  Packed blocks only
    model_subsets: tuple[tuple[str, tuple[int, ...]], ...] = ()
    #: ``c`` or ``fortran``, see :data:`BINDINGS`
    binding: str = "c"

    @property
    def takes_model(self) -> bool:
        """Whether the image can run a TorchScript model at this hook."""

        return bool(self.model_inputs) and bool(self.model_outputs)

    def model_subset(self, name: str) -> tuple[int, ...] | None:
        """The last-axis indices the model returns for output ``name``; None for the whole array."""

        for output, indices in self.model_subsets:
            if output == name:
                return indices
        return None

    @property
    def symbol(self) -> str:
        """The external name the redirected callers call."""

        if self.redirect == "weaken-definition":
            return self.callee_symbol
        if self.binding == "fortran":
            return f"pycam_hooks_mp_hook_{self.kernel}_"       # the module procedure's own name
        return f"pycam_hook_{self.kernel}_"

    @property
    def pausable(self) -> bool:
        """Whether the hook can hand Python a frame: only a bind(C) hook has one."""

        return self.binding == "c"

    @property
    def id(self) -> int:  # noqa: A003 - the hook's number in the table, 1-based
        return HOOK_IDS[self.kernel]


HOOK_IDS: dict[str, int] = {}


@dataclass(frozen=True, slots=True)
class HookTable:
    hooks: tuple[Hook, ...]
    fiber_stack_bytes: int
    sha256: str

    def hook(self, kernel: str) -> Hook:
        for hook in self.hooks:
            if hook.kernel == kernel:
                return hook
        raise KeyError(kernel)

    @property
    def kernel_names(self) -> tuple[str, ...]:
        return tuple(hook.kernel for hook in self.hooks)


def load_hooks(path: str | Path | None = None) -> HookTable:
    source = Path(path) if path is not None else HOOKS
    text = source.read_text()
    payload = yaml.safe_load(text) or {}
    if int(payload.get("schema_version", 0)) != 1:
        raise PICAMConfigurationError(f"{source}: hooks require schema_version: 1")
    hooks: list[Hook] = []
    seen: set[str] = set()
    for index, record in enumerate(payload.get("hooks") or (), start=1):
        kernel = str(record["kernel"])
        if kernel in seen:
            raise PICAMConfigurationError(f"{source}: hook {kernel!r} is listed twice")
        seen.add(kernel)
        redirect = str(record.get("redirect", "rename-references"))
        if redirect not in REDIRECTS:
            raise PICAMConfigurationError(f"{source}: hook {kernel!r} redirect must be one of {REDIRECTS}")
        original = dict(record.get("original") or {})
        if "symbol" in original and ("module" in original or "routine" in original):
            raise PICAMConfigurationError(f"{source}: hook {kernel!r}: original is a symbol or a module routine, not both")
        if "symbol" not in original and not ("module" in original and "routine" in original):
            raise PICAMConfigurationError(f"{source}: hook {kernel!r}: original needs module+routine or symbol")
        if redirect == "weaken-definition" and "symbol" not in original:
            raise PICAMConfigurationError(f"{source}: hook {kernel!r}: a weakened definition is reached by its second name")
        callers = tuple(HookCaller(object=str(c["object"]), routine=str(c["routine"])) for c in record.get("callers") or ())
        if not callers:
            raise PICAMConfigurationError(f"{source}: hook {kernel!r} names no callers")
        model = dict(record.get("model") or {})
        model_inputs = tuple(str(name) for name in model.get("inputs") or ())
        model_outputs = tuple(str(name) for name in model.get("outputs") or ())
        if model and (not model_inputs or not model_outputs):
            raise PICAMConfigurationError(f"{source}: hook {kernel!r}: a model block names inputs and outputs")
        if len(set(model_inputs)) != len(model_inputs) or len(set(model_outputs)) != len(model_outputs):
            raise PICAMConfigurationError(f"{source}: hook {kernel!r}: a model argument is listed twice")
        model_packed = bool(model.get("packed", False))
        model_zero_outputs = tuple(str(name) for name in model.get("zero_outputs") or ())
        if set(model_zero_outputs) & set(model_outputs) or len(set(model_zero_outputs)) != len(model_zero_outputs):
            raise PICAMConfigurationError(f"{source}: hook {kernel!r}: a zeroed output is listed twice or also returned by the model")
        if (model_packed or model_zero_outputs) and not model_outputs:
            raise PICAMConfigurationError(f"{source}: hook {kernel!r}: packed or zeroed outputs need a model block")
        subsets = []
        for name, indices in dict(model.get("subset") or {}).items():
            if str(name) not in model_outputs:
                raise PICAMConfigurationError(f"{source}: hook {kernel!r}: subset output {name!r} is not returned by the model")
            values = tuple(int(i) for i in (indices or ()))
            if not values or len(set(values)) != len(values) or min(values) < 1:
                raise PICAMConfigurationError(f"{source}: hook {kernel!r}: the subset of {name!r} needs distinct 1-based indices")
            subsets.append((str(name), values))
        if subsets and not model_packed:
            raise PICAMConfigurationError(f"{source}: hook {kernel!r}: a subset is returned inside a packed tensor only")
        binding = str(record.get("binding", "c"))
        if binding not in BINDINGS:
            raise PICAMConfigurationError(f"{source}: hook {kernel!r} binding must be one of {BINDINGS}")
        if binding == "fortran" and redirect != "rename-references":
            raise PICAMConfigurationError(f"{source}: hook {kernel!r}: a Fortran-bound hook is reached by renamed references")
        HOOK_IDS[kernel] = index
        hooks.append(Hook(
            kernel=kernel, contract=str(record["contract"]), callee_symbol=str(record["callee_symbol"]),
            redirect=redirect, original_module=original.get("module"), original_routine=original.get("routine"),
            original_symbol=original.get("symbol"), callers=callers,
            model_inputs=model_inputs, model_outputs=model_outputs, binding=binding,
            model_packed=model_packed, model_zero_outputs=model_zero_outputs, model_subsets=tuple(subsets),
        ))
    import hashlib

    return HookTable(hooks=tuple(hooks), fiber_stack_bytes=int(payload.get("fiber_stack_bytes", 512 << 20)),
                     sha256=hashlib.sha256(text.encode()).hexdigest())


__all__ = ["BINDINGS", "HOOKS", "HOOK_MODULE", "Hook", "HookCaller", "HookTable", "hooked_model_kernels", "load_hooks"]


def read_hook_counts(library: Any) -> dict[str, dict[str, int]]:
    """Every hook's call and pause counts from a loaded image; empty when the image has no hooks."""

    import ctypes

    if library is None:
        return {}
    try:
        count_entry = library.pycam_hooks_count_v1
        name_entry = library.pycam_hooks_name_v1
        counts_entry = library.pycam_hooks_counts_v1
    except AttributeError:
        return {}
    count_entry.restype = ctypes.c_int32
    name_entry.restype = ctypes.c_int32
    name_entry.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
    counts_entry.restype = ctypes.c_int32
    counts_entry.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)]
    missed_entry = getattr(library, "pycam_hooks_missed_v1", None)     # armed calls made off the fiber
    if missed_entry is not None:
        missed_entry.restype = ctypes.c_int32
        missed_entry.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int64)]
    modeled_entry = getattr(library, "pycam_hooks_modeled_v1", None)   # calls a bound model answered in the image
    if modeled_entry is not None:
        modeled_entry.restype = ctypes.c_int32
        modeled_entry.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int64)]
    seconds_entry = getattr(library, "pycam_hooks_model_seconds_v1", None)  # wall time in the model branch / forward
    if seconds_entry is not None:
        seconds_entry.restype = ctypes.c_int32
        seconds_entry.argtypes = [ctypes.c_int32] + [ctypes.POINTER(ctypes.c_double)] * 6
    result: dict[str, dict[str, int]] = {}
    for hook in range(1, int(count_entry()) + 1):
        buffer = ctypes.create_string_buffer(64)
        if name_entry(hook, buffer, 64) != 0:
            continue
        calls, paused = ctypes.c_int64(0), ctypes.c_int64(0)
        if counts_entry(hook, ctypes.byref(calls), ctypes.byref(paused)) != 0:
            continue
        record = {"calls": int(calls.value), "paused": int(paused.value)}
        if missed_entry is not None:
            missed = ctypes.c_int64(0)
            if missed_entry(hook, ctypes.byref(missed)) == 0:
                record["missed"] = int(missed.value)
        if modeled_entry is not None:
            modeled = ctypes.c_int64(0)
            if modeled_entry(hook, ctypes.byref(modeled)) == 0:
                record["modeled"] = int(modeled.value)
        if seconds_entry is not None:
            branch, forward, whole, first, warm, original = (ctypes.c_double(0.0) for _ in range(6))
            if seconds_entry(hook, ctypes.byref(branch), ctypes.byref(forward), ctypes.byref(whole),
                             ctypes.byref(first), ctypes.byref(warm), ctypes.byref(original)) == 0 and record.get("modeled"):
                record["model_seconds"] = float(branch.value)          # the hook's model branch
                record["forward_seconds"] = float(forward.value)       # the model's forward alone
                record["call_seconds"] = float(whole.value)            # the whole model call, as the hook sees it
                record["first_call_seconds"] = float(first.value)      # the first modeled call alone
                record["warm_seconds"] = float(warm.value)             # the warm-up forward at bind
                record["original_seconds"] = float(original.value)     # the original on the calls a shadow model also answered
        result[buffer.value.decode("ascii", errors="replace")] = record
    return result


BIND_STATUS = {1: "no such hook", 2: "the hook takes no model (hooks.yaml has no model block for it)",
               3: "the hook is armed for a Python replacement", 4: "the path is empty or too long",
               5: "the image has no model entry: it was built without FTorch",
               6: "the image has no plugin entry: it was built before plugins",
               7: "the device is neither the host nor CUDA"}
#: what pycam_hooks_arm_v1 answers when a hook cannot be armed
ARM_STATUS = {1: "no such hook", 3: "a model is bound at the hook", 5: "the hook has no frame (a Fortran-bound hook cannot pause)"}


def hooked_model_kernels() -> frozenset[str]:
    """The kernels whose hook has a model block: where a function or a model can stand inside the image.

    Read once from the committed table; a stage asks every step.
    """

    global _HOOKED_MODEL_KERNELS
    if _HOOKED_MODEL_KERNELS is None:
        _HOOKED_MODEL_KERNELS = frozenset(hook.kernel for hook in load_hooks().hooks if hook.takes_model)
    return _HOOKED_MODEL_KERNELS


_HOOKED_MODEL_KERNELS: frozenset[str] | None = None


#: FTorch's device codes: torch_kCPU and torch_kCUDA
MODEL_DEVICES = {"cpu": 0, "cuda": 1}


def bind_hook_model(library: Any, hook_id: int, path: str | Path, *, shadow: bool = False,
                    device: str = "cpu", device_index: int = -1) -> None:
    """Load a TorchScript model into the image at hook ``hook_id``: from then on the hook
    answers its calls with the model, inside Fortran, until :func:`unbind_hook_model`.

    With ``shadow`` the model runs on every call and its answer is discarded while the
    original keeps answering: the run stays bit-for-bit and the model path's cost is
    measured in situ.  ``device`` is where the model lives and runs, ``cpu`` or ``cuda``
    with this rank's ``device_index``; the hook makes its input tensors there and takes
    the answer back on the host.
    """

    import ctypes

    if device not in MODEL_DEVICES:
        raise PICAMConfigurationError(f"model device must be one of {sorted(MODEL_DEVICES)}, not {device!r}")
    if device != "cpu":
        entry = getattr(library, "pycam_hooks_bind_model_v2", None)
        if entry is None:
            raise PICAMConfigurationError(f"cannot bind a model on {device!r} at hook {hook_id}: the image has no "
                                          f"device-aware bind entry (pycam_hooks_bind_model_v2); rebuild it")
        _refuse_shadowing_torch(device, hook_id)
        count, report = _cuda_device_report()
        if count <= 0 or (device_index >= 0 and device_index >= count):
            raise PICAMConfigurationError(f"cannot bind a model on {device}:{device_index} at hook {hook_id}: the CUDA "
                                          f"runtime the image loaded sees {max(count, 0)} device(s); {report}")
        encoded = str(Path(path)).encode()
        entry.restype = ctypes.c_int32
        entry.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        status = int(entry(int(hook_id), encoded, len(encoded), 1 if shadow else 0, MODEL_DEVICES[device], int(device_index)))
        if status != 0:
            raise PICAMConfigurationError(f"binding a model on {device}:{device_index} at hook {hook_id} failed: "
                                          f"{BIND_STATUS.get(status, status)}")
        return
    entry = getattr(library, "pycam_hooks_bind_model_v1", None)
    if entry is None:
        raise PICAMConfigurationError(f"cannot bind a model at hook {hook_id}: {BIND_STATUS[5]}")
    entry.restype = ctypes.c_int32
    entry.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32]
    encoded = str(Path(path)).encode()
    status = int(entry(int(hook_id), encoded, len(encoded), 1 if shadow else 0))
    if status != 0:
        raise PICAMConfigurationError(
            f"cannot bind {path} at hook {hook_id}: {BIND_STATUS.get(status, f'status {status}')}")


def _cuda_device_report() -> tuple[int, str]:
    """How many devices the CUDA runtime the image loaded can see, and the facts behind the
    number (runtime status, driver and runtime versions, the visible-devices variable, the CUDA
    libraries mapped): the bind fails closed on zero instead of letting FTorch exit the process."""

    import ctypes
    import os

    try:
        cudart = ctypes.CDLL("libcudart.so.12")
    except OSError as exc:
        return -1, f"no CUDA runtime is loaded in this process ({exc})"
    count = ctypes.c_int(0)
    status = int(cudart.cudaGetDeviceCount(ctypes.byref(count)))
    cudart.cudaGetErrorString.restype = ctypes.c_char_p
    message = cudart.cudaGetErrorString(status).decode(errors="replace")
    driver, runtime = ctypes.c_int(0), ctypes.c_int(0)
    cudart.cudaDriverGetVersion(ctypes.byref(driver))
    cudart.cudaRuntimeGetVersion(ctypes.byref(runtime))
    report = (f"cudaGetDeviceCount status {status} ({message}), driver {driver.value}, runtime {runtime.value}, "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}, {_process_state()}")
    return (int(count.value) if status == 0 else 0), report


def _process_state() -> str:
    """The process facts a failing CUDA driver load turns on: the CUDA libraries mapped, the
    number of memory mappings against the kernel's limit, resident and virtual size, open
    descriptors against their limit."""

    import os
    import resource

    facts = []
    try:
        with open("/proc/self/maps") as maps:
            lines = maps.readlines()
        mapped = sorted({line.split()[-1] for line in lines if "libcuda" in line})
        limit = Path("/proc/sys/vm/max_map_count").read_text().strip()
        facts.append(f"mapped {mapped}, {len(lines)} mappings of {limit}")
    except OSError:
        pass
    try:
        status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
        facts.append(f"VmRSS {status.get('VmRSS', '?').strip()}, VmSize {status.get('VmSize', '?').strip()}")
    except OSError:
        pass
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        facts.append(f"{len(os.listdir('/proc/self/fd'))} open descriptors of {soft}")
    except OSError:
        pass
    return ", ".join(facts)


def _refuse_shadowing_torch(device: str, hook_id: int) -> None:
    """A device model needs the image's own libtorch.  A ``torch`` already imported into this
    process loaded its libraries under the same names first, and the loader hands those to the
    image's FTorch (first loaded wins): a CPU-only wheel then answers ``device_count`` with zero
    and FTorch exits the process from inside Fortran.  Refuse here, with the cause named."""

    import sys

    loaded = sys.modules.get("torch")
    if loaded is None or getattr(getattr(loaded, "version", None), "cuda", None) is not None:
        return
    raise PICAMConfigurationError(
        f"cannot bind a model on {device} at hook {hook_id}: this process has imported a CPU-only torch "
        f"({getattr(loaded, '__version__', '?')} from {getattr(loaded, '__file__', '?')}); its libtorch was loaded "
        f"first under the names the image's CUDA libtorch uses, so the image would see no device.  Keep torch out "
        f"of the rank processes, or install the CUDA torch the image's FTorch was built against")


def bind_hook_plugin(library: Any, hook_id: int, address: int, *, shadow: bool = False) -> None:
    """Bind a compiled plugin (the address of a C function of the hook's plugin interface, e.g. a
    Numba cfunc) at hook ``hook_id``: the hook hands it the model block's arrays as pointer and
    extent tables and takes its outputs, inside Fortran, until :func:`unbind_hook_model`.
    """

    import ctypes

    entry = getattr(library, "pycam_hooks_bind_plugin_v1", None)
    if entry is None:
        raise PICAMConfigurationError(f"cannot bind a plugin at hook {hook_id}: {BIND_STATUS[6]}")
    entry.restype = ctypes.c_int32
    entry.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32]
    status = int(entry(int(hook_id), ctypes.c_void_p(int(address)), 1 if shadow else 0))
    if status != 0:
        raise PICAMConfigurationError(
            f"cannot bind a plugin at hook {hook_id}: {BIND_STATUS.get(status, f'status {status}')}")


def unbind_hook_model(library: Any, hook_id: int) -> None:
    """Release the model bound at ``hook_id``; the hook answers with the original again."""

    import ctypes

    entry = getattr(library, "pycam_hooks_unbind_model_v1", None)
    if entry is None:
        return
    entry.restype = ctypes.c_int32
    entry.argtypes = [ctypes.c_int32]
    entry(int(hook_id))


__all__ += ["read_hook_counts", "bind_hook_model", "bind_hook_plugin", "unbind_hook_model", "BIND_STATUS", "ARM_STATUS"]
