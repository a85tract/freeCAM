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

    @property
    def symbol(self) -> str:
        """The external name the redirected callers call."""

        if self.redirect == "weaken-definition":
            return self.callee_symbol
        return f"pycam_hook_{self.kernel}_"

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
        HOOK_IDS[kernel] = index
        hooks.append(Hook(
            kernel=kernel, contract=str(record["contract"]), callee_symbol=str(record["callee_symbol"]),
            redirect=redirect, original_module=original.get("module"), original_routine=original.get("routine"),
            original_symbol=original.get("symbol"), callers=callers,
        ))
    import hashlib

    return HookTable(hooks=tuple(hooks), fiber_stack_bytes=int(payload.get("fiber_stack_bytes", 512 << 20)),
                     sha256=hashlib.sha256(text.encode()).hexdigest())


__all__ = ["HOOKS", "HOOK_MODULE", "Hook", "HookCaller", "HookTable", "load_hooks"]


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
    result: dict[str, dict[str, int]] = {}
    for hook in range(1, int(count_entry()) + 1):
        buffer = ctypes.create_string_buffer(64)
        if name_entry(hook, buffer, 64) != 0:
            continue
        calls, paused = ctypes.c_int64(0), ctypes.c_int64(0)
        if counts_entry(hook, ctypes.byref(calls), ctypes.byref(paused)) != 0:
            continue
        result[buffer.value.decode("ascii", errors="replace")] = {"calls": int(calls.value), "paused": int(paused.value)}
    return result


__all__ += ["read_hook_counts"]
