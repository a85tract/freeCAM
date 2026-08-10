"""Compilation and ABI validation for generated legacy PI-CAM adapters.

Catalog generation is intentionally not treated as runtime proof.  This
module records four separate gates:

* the generated adapter can be parsed as Fortran;
* it can be compiled against the module files of a real CAM build;
* its unresolved procedure symbol exists in that build's ``libatm.a``;
* the generic pointer ABI works for every generated dtype/rank family.

The final scientific/BFB gate remains explicit because an unbound candidate
does not yet have enough information to call a real CAM routine safely.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import threading
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from freecam.core.fortran_adapter import PointerTableAdapter

from .errors import PICAMConfigurationError
from .source_catalog import (
    FortranArgument,
    FortranProcedure,
    _adapter_manifest,
    _generate_pointer_adapter,
)


VALIDATION_SCHEMA_VERSION = 2
_FPARSER_LOCK = threading.Lock()


def _portable_runtime_path(value: str | Path) -> str:
    """Hide the validating user's name while retaining machine path layout."""

    text = str(value)
    user = os.environ.get("USER", "").strip()
    if not user:
        return text
    for root in ("/glade/work", "/glade/derecho/scratch"):
        text = text.replace(f"{root}/{user}", f"{root}/$USER")
    return text


@dataclass(frozen=True, slots=True)
class AdapterBuildContext:
    """One real CAM build used to compile source-matched adapters."""

    name: str
    module_dirs: tuple[Path, ...]
    original_libraries: tuple[Path, ...]
    selected_sources: frozenset[str] | None = None
    build_root: Path | None = None
    case_root: Path | None = None

    def accepts_source(self, source: str) -> bool:
        return self.selected_sources is None or source in self.selected_sources

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "build_root": (
                None
                if self.build_root is None
                else _portable_runtime_path(self.build_root)
            ),
            "case_root": (
                None
                if self.case_root is None
                else _portable_runtime_path(self.case_root)
            ),
            "module_dirs": [
                _portable_runtime_path(path) for path in self.module_dirs
            ],
            "original_libraries": [
                _portable_runtime_path(path) for path in self.original_libraries
            ],
            "selected_sources": (
                None
                if self.selected_sources is None
                else len(self.selected_sources)
            ),
        }


@dataclass(frozen=True, slots=True)
class AdapterCompileAttempt:
    """Compilation evidence for one adapter in one source-matched build."""

    context: str
    compile_status: str
    archive_symbol_status: str
    resolved_archive_symbols: tuple[str, ...]
    object_sha256: str | None
    failure_kind: str | None
    compile_diagnostic: str | None

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["resolved_archive_symbols"] = list(self.resolved_archive_symbols)
        return payload


@dataclass(frozen=True, slots=True)
class AdapterCompileResult:
    """One generated adapter's real-build compilation evidence."""

    name: str
    adapter_source: str
    original_source: str
    original_module: str | None
    original_routine: str
    active_plan_actions: tuple[str, ...]
    parse_status: str
    selected_context: str | None
    compile_status: str
    archive_symbol_status: str
    resolved_archive_symbols: tuple[str, ...]
    object_sha256: str | None
    failure_kind: str | None
    parse_diagnostic: str | None
    compile_diagnostic: str | None
    attempts: tuple[AdapterCompileAttempt, ...] = ()

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["active_plan_actions"] = list(self.active_plan_actions)
        payload["resolved_archive_symbols"] = list(self.resolved_archive_symbols)
        payload["attempts"] = [attempt.as_dict() for attempt in self.attempts]
        return payload


@dataclass(frozen=True, slots=True)
class ABISmokeResult:
    """Synthetic execution proof for one dtype/rank pointer family."""

    dtype: str
    rank: int
    status: str
    compiler_command: tuple[str, ...]
    diagnostic: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["compiler_command"] = [
            _portable_runtime_path(value) for value in self.compiler_command
        ]
        return payload


class PICAMAdapterValidator:
    """Validate generated adapters without claiming unbound kernels are safe."""

    def __init__(
        self,
        descriptor_root: str | Path,
        *,
        compiler: str | Path,
        module_dirs: Sequence[str | Path] = (),
        original_library: str | Path | None = None,
        build_contexts: Sequence[AdapterBuildContext] | None = None,
        work_root: str | Path,
        workers: int = 1,
    ) -> None:
        self.descriptor_root = Path(descriptor_root).resolve()
        compiler_text = str(compiler)
        resolved_compiler = (
            shutil.which(compiler_text)
            if "/" not in compiler_text
            # Keep compiler-driver symlinks intact.  Cray PE drivers use the
            # invoked filename to select the real backend; resolving the
            # symlink to ``redirect`` breaks that dispatch.
            else str(Path(compiler_text).absolute())
        )
        self.compiler = resolved_compiler or compiler_text
        legacy_module_dirs = tuple(Path(path).resolve() for path in module_dirs)
        legacy_library = (
            None if original_library is None else Path(original_library).resolve()
        )
        if build_contexts is not None and (legacy_module_dirs or legacy_library):
            raise PICAMConfigurationError(
                "build_contexts cannot be combined with legacy module_dirs or "
                "original_library"
            )
        if build_contexts is None:
            contexts = (
                AdapterBuildContext(
                    name="default",
                    module_dirs=legacy_module_dirs,
                    original_libraries=(
                        () if legacy_library is None else (legacy_library,)
                    ),
                ),
            )
        else:
            contexts = tuple(build_contexts)
        if not contexts:
            raise PICAMConfigurationError("at least one adapter build context is required")
        if len({context.name for context in contexts}) != len(contexts):
            raise PICAMConfigurationError("adapter build context names must be unique")
        self.build_contexts = contexts
        # Preserve the legacy attributes for callers that inspect them.
        self.module_dirs = legacy_module_dirs
        self.original_library = legacy_library
        self.work_root = Path(work_root).resolve()
        self.workers = max(1, int(workers))
        if not self.descriptor_root.is_dir():
            raise PICAMConfigurationError(
                f"adapter descriptor root does not exist: {self.descriptor_root}"
            )
        if not Path(self.compiler).is_file() and shutil.which(self.compiler) is None:
            raise PICAMConfigurationError(f"Fortran compiler not found: {compiler}")
        for context in self.build_contexts:
            for path in context.module_dirs:
                if not path.is_dir():
                    raise PICAMConfigurationError(
                        f"module directory not found for {context.name}: {path}"
                    )
            for path in context.original_libraries:
                if not path.is_file():
                    raise PICAMConfigurationError(
                        f"original CAM library not found for {context.name}: {path}"
                    )

    @classmethod
    def from_context_file(
        cls,
        descriptor_root: str | Path,
        *,
        compiler: str | Path,
        context_file: str | Path,
        work_root: str | Path,
        workers: int = 1,
    ) -> "PICAMAdapterValidator":
        """Construct a validator from a real CAM build-context matrix."""

        return cls(
            descriptor_root,
            compiler=compiler,
            build_contexts=load_adapter_build_contexts(context_file),
            work_root=work_root,
            workers=workers,
        )

    def validate(self) -> dict[str, object]:
        """Run all mechanical gates and return a machine-readable report."""

        manifests = tuple(sorted(self.descriptor_root.rglob("adapter.json")))
        if not manifests:
            raise PICAMConfigurationError(
                f"no generated adapter.json files below {self.descriptor_root}"
            )
        self.work_root.mkdir(parents=True, exist_ok=True)
        archive_symbols = {
            context.name: self._archive_symbols(context)
            for context in self.build_contexts
        }
        arguments = tuple((path, archive_symbols) for path in manifests)
        if self.workers == 1:
            results = tuple(self._validate_one(item) for item in arguments)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                results = tuple(executor.map(self._validate_one, arguments))

        families = sorted(
            {
                (str(argument["dtype"]), int(argument["rank"]))
                for manifest in manifests
                for argument in json.loads(manifest.read_text())["arguments"]
            }
        )
        smoke = tuple(self._run_abi_smoke(dtype, rank) for dtype, rank in families)
        active = self._active_plan_records()
        reachable_generated_keys = {
            (str(item["name"]), str(item["source"]))
            for item in active["reachable_records"]
            if bool(item["generated_adapter"])
        }
        reachable_generated_results = tuple(
            item
            for item in results
            if (item.name, item.original_source) in reachable_generated_keys
        )
        active["reachable_generated_build"] = {
            "procedures": len(reachable_generated_results),
            "compile_status_counts": _counts(
                item.compile_status for item in reachable_generated_results
            ),
            "archive_symbol_status_counts": _counts(
                item.archive_symbol_status for item in reachable_generated_results
            ),
            "failure_kind_counts": _counts(
                item.failure_kind
                for item in reachable_generated_results
                if item.failure_kind is not None
            ),
            "build_ready": [
                {
                    "name": item.name,
                    "source": item.original_source,
                    "object_sha256": item.object_sha256,
                }
                for item in reachable_generated_results
                if item.compile_status == "passed"
                and item.archive_symbol_status in {"passed", "not_requested"}
            ],
        }
        bfb_records = [
            {
                "name": item.name,
                "source": item.original_source,
                "pi_plan_reachability": (
                    "potentially_exercised"
                    if (item.name, item.original_source) in reachable_generated_keys
                    else "not_exercised"
                ),
                "bfb_status": (
                    "required_pending_runtime_trace"
                    if (item.name, item.original_source) in reachable_generated_keys
                    else "not_exercised"
                ),
            }
            for item in results
        ]
        compile_failures = tuple(
            item for item in results if item.compile_status != "passed"
        )
        archive_failures = tuple(
            item for item in results if item.archive_symbol_status == "failed"
        )
        compiler_version = subprocess.run(
            [self.compiler, "--version"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "validator": "freecam.pi_cam.adapter_validation",
            "validator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "host": socket.gethostname(),
            "descriptor_root": _portable_runtime_path(self.descriptor_root),
            "descriptor_tree_sha256": _descriptor_tree_hash(manifests),
            "compiler": self.compiler,
            "compiler_version": compiler_version[0] if compiler_version else None,
            "module_dirs": [
                _portable_runtime_path(path) for path in self.module_dirs
            ],
            "original_library": (
                None
                if self.original_library is None
                else _portable_runtime_path(self.original_library)
            ),
            "build_contexts": [
                context.as_dict() for context in self.build_contexts
            ],
            "generated_adapters": len(results),
            "parse_status_counts": _counts(item.parse_status for item in results),
            "compile_status_counts": _counts(item.compile_status for item in results),
            "archive_symbol_status_counts": _counts(
                item.archive_symbol_status for item in results
            ),
            "failure_kind_counts": _counts(
                item.failure_kind for item in results if item.failure_kind is not None
            ),
            "full_compile_gate": {
                "compiled_and_symbol_resolved": sum(
                    item.compile_status == "passed"
                    and item.archive_symbol_status in {"passed", "not_requested"}
                    for item in results
                ),
                "compile_failures": len(compile_failures),
                "archive_symbol_failures": len(archive_failures),
                "passed": not compile_failures and not archive_failures,
            },
            # Compatibility alias.  Unlike schema v1, no unavailable-source
            # exemption is permitted: every emitted adapter must match at
            # least one declared real build context.
            "case_build_gate": {
                "compiled_and_symbol_resolved": sum(
                    item.compile_status == "passed"
                    and item.archive_symbol_status in {"passed", "not_requested"}
                    for item in results
                ),
                "not_present_or_different_in_case_build": 0,
                "generator_compile_failures": len(compile_failures),
                "archive_symbol_failures": len(archive_failures),
                "passed": not compile_failures and not archive_failures,
            },
            "abi_signature_families": len(smoke),
            "abi_smoke_status_counts": _counts(item.status for item in smoke),
            "runtime_policy": (
                "Real generated candidates remain uncalled until explicit StatePool, "
                "scalar, dimension, and context bindings exist."
            ),
            "scientific_policy": (
                "Compilation and synthetic ABI execution are not BFB evidence. "
                "A process exercised by the PI case requires native capture/replay "
                "and a 512-rank 50-step CAM gate; an inactive process is recorded "
                "as not_exercised rather than treated as a BFB failure."
            ),
            "scientific_bfb_scope": {
                "status_counts": _counts(
                    str(item["bfb_status"]) for item in bfb_records
                ),
                "basis": (
                    "Static reachability from the current Python PI-CAM plan is "
                    "conservative. Reachable adapters require a runtime execution "
                    "trace before capture/replay and 50-step BFB; unreachable "
                    "adapters are skipped as not_exercised for this case."
                ),
                "records": bfb_records,
            },
            "active_plan": active,
            "abi_smoke": [item.as_dict() for item in smoke],
            "adapters": [item.as_dict() for item in results],
        }

    def write_report(self, path: str | Path) -> Path:
        output = Path(path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.validate(), indent=2, sort_keys=True) + "\n")
        return output

    def _validate_one(
        self,
        item: tuple[Path, Mapping[str, frozenset[str] | None]],
    ) -> AdapterCompileResult:
        manifest_path, symbols_by_context = item
        manifest = json.loads(manifest_path.read_text())
        descriptor = yaml.safe_load((manifest_path.parent / "kernel.yaml").read_text())
        adapter = manifest_path.parent / str(manifest["adapter_source"])
        relative_adapter = str(adapter.relative_to(self.descriptor_root))
        parse_status = "passed"
        parse_diagnostic: str | None = None
        try:
            _parse_fortran(adapter)
        except Exception as exc:  # fail closed and preserve parser evidence
            # fparser's symbol-table walk rejects some valid bind(C)
            # functions.  Record that tool limitation, but let the real CAM
            # compiler decide whether the generated Fortran is valid.
            parse_status = "parser_tool_error"
            parse_diagnostic = f"{type(exc).__name__}: {exc}"

        original_source = str(manifest["original_source"])
        attempts: list[AdapterCompileAttempt] = []
        selected: AdapterCompileAttempt | None = None
        for context in self.build_contexts:
            if not context.accepts_source(original_source):
                continue
            output_dir = (
                self.work_root
                / _safe_slug(context.name)
                / manifest_path.parent.name
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            object_path = output_dir / "adapter.o"
            command = self._compile_command(
                adapter, object_path, output_dir, context=context
            )
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                cwd=output_dir,
            )
            if completed.returncode != 0:
                diagnostic = _short_diagnostic(completed.stderr or completed.stdout)
                attempts.append(
                    AdapterCompileAttempt(
                        context=context.name,
                        compile_status="failed",
                        archive_symbol_status="skipped_compile_failure",
                        resolved_archive_symbols=(),
                        object_sha256=None,
                        failure_kind=_classify_compile_failure(diagnostic),
                        compile_diagnostic=diagnostic,
                    )
                )
                continue

            undefined = _nm_symbols(object_path, undefined=True)
            archive_symbols = symbols_by_context[context.name]
            if archive_symbols is None:
                symbol_status = "not_requested"
                resolved: tuple[str, ...] = ()
            else:
                resolved = tuple(sorted(undefined.intersection(archive_symbols)))
                symbol_status = "passed" if resolved else "failed"
            attempt = AdapterCompileAttempt(
                context=context.name,
                compile_status="passed",
                archive_symbol_status=symbol_status,
                resolved_archive_symbols=resolved,
                object_sha256=sha256(object_path.read_bytes()).hexdigest(),
                failure_kind=(
                    "original_symbol_missing_from_archive"
                    if symbol_status == "failed"
                    else None
                ),
                compile_diagnostic=None,
            )
            attempts.append(attempt)
            if symbol_status in {"passed", "not_requested"}:
                selected = attempt
                break

        if selected is None:
            if attempts:
                selected = next(
                    (
                        attempt
                        for attempt in attempts
                        if attempt.compile_status == "passed"
                    ),
                    attempts[-1],
                )
            else:
                selected = AdapterCompileAttempt(
                    context="none",
                    compile_status="failed",
                    archive_symbol_status="skipped_compile_failure",
                    resolved_archive_symbols=(),
                    object_sha256=None,
                    failure_kind="source_not_in_build_matrix",
                    compile_diagnostic=(
                        f"{original_source} is not selected by any declared "
                        "real CAM build context"
                    ),
                )
        return AdapterCompileResult(
            name=str(manifest["name"]),
            adapter_source=relative_adapter,
            original_source=original_source,
            original_module=manifest.get("original_module"),
            original_routine=str(manifest["original_routine"]),
            active_plan_actions=tuple(descriptor.get("active_plan_actions", ())),
            parse_status=parse_status,
            selected_context=(
                None if selected.context == "none" else selected.context
            ),
            compile_status=selected.compile_status,
            archive_symbol_status=selected.archive_symbol_status,
            resolved_archive_symbols=selected.resolved_archive_symbols,
            object_sha256=selected.object_sha256,
            failure_kind=selected.failure_kind,
            parse_diagnostic=parse_diagnostic,
            compile_diagnostic=selected.compile_diagnostic,
            attempts=tuple(attempts),
        )

    def _compile_command(
        self,
        adapter: Path,
        object_path: Path,
        module_output: Path,
        *,
        context: AdapterBuildContext,
    ) -> list[str]:
        command = [self.compiler, "-c", "-fPIC"]
        compiler_name = Path(self.compiler).name.lower()
        if compiler_name in {"ifort", "ifx", "ftn"}:
            command.extend(("-free", "-module", str(module_output)))
        else:
            command.extend(("-ffree-line-length-none", f"-J{module_output}"))
        for path in context.module_dirs:
            command.extend(("-I", str(path)))
        command.extend((str(adapter), "-o", str(object_path)))
        return command

    def _archive_symbols(
        self, context: AdapterBuildContext
    ) -> frozenset[str] | None:
        if not context.original_libraries:
            return None
        symbols: set[str] = set()
        for library in context.original_libraries:
            symbols.update(_nm_symbols(library, undefined=False))
        return frozenset(symbols)

    def _active_plan_records(self) -> dict[str, object]:
        descriptors: list[tuple[Path, Mapping[str, Any]]] = []
        for descriptor_path in sorted(self.descriptor_root.rglob("kernel.yaml")):
            descriptor = yaml.safe_load(descriptor_path.read_text())
            descriptors.append((descriptor_path, descriptor))
        by_name: dict[str, list[int]] = {}
        for index, (_, descriptor) in enumerate(descriptors):
            by_name.setdefault(str(descriptor["qualified_name"]).lower(), []).append(index)
        direct = {
            index
            for index, (_, descriptor) in enumerate(descriptors)
            if descriptor.get("active_plan_actions")
        }
        reachable = set(direct)
        queue = list(sorted(direct))
        while queue:
            index = queue.pop(0)
            descriptor = descriptors[index][1]
            for callee in descriptor.get("resolved_calls", ()):
                for target in by_name.get(str(callee).lower(), ()):
                    if target not in reachable:
                        reachable.add(target)
                        queue.append(target)

        def record(index: int) -> dict[str, object]:
            descriptor_path, descriptor = descriptors[index]
            return {
                "name": str(descriptor["qualified_name"]),
                "source": str(descriptor["source"]),
                "actions": list(descriptor.get("active_plan_actions", ())),
                "adapter_status": str(descriptor["adapter_status"]),
                "blockers": list(descriptor.get("blockers", ())),
                "generated_adapter": (descriptor_path.parent / "adapter.F90").is_file(),
                "unresolved_calls": list(descriptor.get("unresolved_calls", ())),
            }

        records = [record(index) for index in sorted(direct)]
        reachable_records = [record(index) for index in sorted(reachable)]
        return {
            "procedures": len(records),
            "status_counts": _counts(str(item["adapter_status"]) for item in records),
            "generated_adapter_count": sum(
                bool(item["generated_adapter"]) for item in records
            ),
            "records": records,
            "reachable_procedures": len(reachable_records),
            "reachable_status_counts": _counts(
                str(item["adapter_status"]) for item in reachable_records
            ),
            "reachable_generated_adapter_count": sum(
                bool(item["generated_adapter"]) for item in reachable_records
            ),
            "reachable_unresolved_call_names": len(
                {
                    str(name)
                    for item in reachable_records
                    for name in item["unresolved_calls"]
                }
            ),
            "reachable_records": reachable_records,
        }

    def _run_abi_smoke(self, dtype: str, rank: int) -> ABISmokeResult:
        family = f"{dtype}_r{rank}"
        with tempfile.TemporaryDirectory(prefix=f"freecam-{family}-") as temporary:
            root = Path(temporary)
            source = root / "kernel.F90"
            adapter_source = root / "adapter.F90"
            library = root / "kernel.so"
            procedure = _smoke_procedure(dtype, rank)
            source.write_text(_smoke_kernel_source(dtype, rank))
            adapter_source.write_text(_generate_pointer_adapter(procedure))
            command = [self.compiler, "-shared", "-fPIC"]
            if Path(self.compiler).name.lower() in {"ifort", "ifx", "ftn"}:
                command.append("-free")
            else:
                command.append("-ffree-line-length-none")
            command.extend((str(source), str(adapter_source), "-o", str(library)))
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                cwd=root,
            )
            if completed.returncode != 0:
                return ABISmokeResult(
                    dtype=dtype,
                    rank=rank,
                    status="compile_failed",
                    compiler_command=tuple(command),
                    diagnostic=_short_diagnostic(completed.stderr or completed.stdout),
                )
            values = _smoke_array(dtype, rank)
            before = values.copy()
            manifest = _adapter_manifest(procedure, adapter_source.name)
            runtime = PointerTableAdapter(
                ctypes.CDLL(str(library)),
                {
                    "smoke": {
                        "symbol": manifest["symbol"],
                        "action_id": 0,
                        "arguments": (
                            {"field": "value", "dtype": dtype, "rank": rank},
                        ),
                    }
                },
                library_name=str(library),
            )
            try:
                runtime.call("smoke", {"value": values}, fcomm=0)
            except Exception as exc:
                return ABISmokeResult(
                    dtype=dtype,
                    rank=rank,
                    status="runtime_failed",
                    compiler_command=tuple(command),
                    diagnostic=f"{type(exc).__name__}: {exc}",
                )
            expected = before + np.asarray(1, dtype=values.dtype)
            status = "passed" if np.array_equal(values, expected) else "value_mismatch"
            return ABISmokeResult(
                dtype=dtype,
                rank=rank,
                status=status,
                compiler_command=tuple(command),
                diagnostic=None if status == "passed" else "kernel output did not add one",
            )


def load_adapter_build_contexts(
    path: str | Path,
) -> tuple[AdapterBuildContext, ...]:
    """Load real CAM build contexts and derive modules, archives, and sources."""

    matrix_path = Path(path).resolve()
    if not matrix_path.is_file():
        raise PICAMConfigurationError(
            f"adapter build-context file does not exist: {matrix_path}"
        )
    payload = yaml.safe_load(matrix_path.read_text())
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise PICAMConfigurationError(
            f"unsupported adapter build-context schema in {matrix_path}"
        )
    records = payload.get("contexts")
    if not isinstance(records, list) or not records:
        raise PICAMConfigurationError(
            f"adapter build-context file has no contexts: {matrix_path}"
        )
    contexts: list[AdapterBuildContext] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise PICAMConfigurationError("each adapter build context must be a map")
        name = str(record.get("name", "")).strip()
        if not name:
            raise PICAMConfigurationError("adapter build context is missing name")
        build_root = _context_path(record.get("build_root"), matrix_path.parent)
        if build_root is None or not build_root.is_dir():
            raise PICAMConfigurationError(
                f"build_root for adapter context {name} does not exist: {build_root}"
            )
        case_root = _context_path(record.get("case_root"), matrix_path.parent)
        if case_root is not None and not case_root.is_dir():
            raise PICAMConfigurationError(
                f"case_root for adapter context {name} does not exist: {case_root}"
            )
        module_values = record.get("module_dirs")
        if module_values is None:
            module_dirs = _module_dirs_below(build_root)
        else:
            if not isinstance(module_values, list):
                raise PICAMConfigurationError(
                    f"module_dirs for adapter context {name} must be a list"
                )
            module_dirs = tuple(
                _required_context_path(value, matrix_path.parent, "module directory")
                for value in module_values
            )
        library_values = record.get("original_libraries")
        if library_values is None:
            libraries = tuple(
                sorted(
                    {
                        *build_root.rglob("libatm.a"),
                        *build_root.rglob("libcosp.a"),
                    }
                )
            )
        else:
            if not isinstance(library_values, list):
                raise PICAMConfigurationError(
                    f"original_libraries for adapter context {name} must be a list"
                )
            libraries = tuple(
                _required_context_path(value, matrix_path.parent, "original library")
                for value in library_values
            )
        if not module_dirs:
            raise PICAMConfigurationError(
                f"adapter context {name} contains no compiled Fortran modules"
            )
        if not libraries:
            raise PICAMConfigurationError(
                f"adapter context {name} contains no CAM/COSP archive"
            )
        selected_sources = (
            None
            if case_root is None
            else _selected_cam_sources(case_root, build_root)
        )
        contexts.append(
            AdapterBuildContext(
                name=name,
                module_dirs=module_dirs,
                original_libraries=libraries,
                selected_sources=selected_sources,
                build_root=build_root,
                case_root=case_root,
            )
        )
    return tuple(contexts)


def _context_path(value: object, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _required_context_path(value: object, base: Path, label: str) -> Path:
    path = _context_path(value, base)
    if path is None or not path.exists():
        raise PICAMConfigurationError(f"{label} does not exist: {path}")
    return path


def _module_dirs_below(build_root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted({path.parent.resolve() for path in build_root.rglob("*.mod")})
    )


def _selected_cam_sources(case_root: Path, build_root: Path) -> frozenset[str]:
    filepath = case_root / "Buildconf/camconf/Filepath"
    srcfiles = build_root / "atm/obj/Srcfiles"
    if not filepath.is_file() or not srcfiles.is_file():
        raise PICAMConfigurationError(
            "a source-matched adapter context requires both "
            f"{filepath} and {srcfiles}"
        )
    source_dirs = tuple(
        Path(line.strip()).resolve()
        for line in filepath.read_text().splitlines()
        if line.strip()
    )
    selected: set[str] = set()
    for line in srcfiles.read_text().splitlines():
        source_name = line.strip()
        if not source_name:
            continue
        for directory in source_dirs:
            candidate = directory / source_name
            if not candidate.is_file():
                continue
            portable = _portable_cam_source(candidate)
            if portable is not None:
                selected.add(portable)
            break
    selected.update(_selected_auxiliary_sources(build_root))
    if not selected:
        raise PICAMConfigurationError(
            f"adapter context did not resolve any original CAM sources: {case_root}"
        )
    return frozenset(selected)


def _selected_auxiliary_sources(build_root: Path) -> set[str]:
    """Resolve source members of separately built CAM physics archives."""

    selected: set[str] = set()
    for archive in build_root.rglob("libcosp.a"):
        makefile = archive.parent / "Makefile"
        if not makefile.is_file():
            continue
        source_dirs = []
        for line in makefile.read_text().splitlines():
            match = re.match(r"^[A-Za-z0-9_]*PATH\s*:=\s*(\S+)\s*$", line)
            if match:
                directory = Path(match.group(1)).resolve()
                if directory.is_dir():
                    source_dirs.append(directory)
        # COSP_PATH is a parent of several more-specific *_PATH entries, so
        # the same source can be discovered more than once.  De-duplicate by
        # resolved path before deciding whether an archive member has one
        # unambiguous source.
        candidates: dict[str, set[Path]] = {}
        for directory in source_dirs:
            for source in directory.rglob("*"):
                if source.is_file() and source.suffix.lower() in {
                    ".f",
                    ".f90",
                    ".f95",
                }:
                    candidates.setdefault(source.stem.lower(), set()).add(
                        source.resolve()
                    )
        completed = subprocess.run(
            ["ar", "t", str(archive)],
            check=True,
            capture_output=True,
            text=True,
        )
        for member in completed.stdout.splitlines():
            matches = tuple(candidates.get(Path(member).stem.lower(), ()))
            if len(matches) != 1:
                continue
            portable = _portable_cam_source(matches[0])
            if portable is not None:
                selected.add(portable)
    return selected


def _portable_cam_source(path: Path) -> str | None:
    parts = path.resolve().parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ("components", "cam"):
            return "/".join(parts[index:])
    return None


def _parse_fortran(path: Path) -> None:
    from fparser.common.readfortran import FortranFileReader
    from fparser.two.parser import ParserFactory
    from fparser.two.symbol_table import SYMBOL_TABLES

    # fparser keeps process-global symbol tables and is not thread-safe.  The
    # real adapter compilation is parallel, but this advisory parse is
    # serialized and reset so its result is deterministic.
    with _FPARSER_LOCK:
        SYMBOL_TABLES.clear()
        try:
            ParserFactory().create(std="f2008")(
                FortranFileReader(str(path), ignore_comments=False)
            )
        finally:
            SYMBOL_TABLES.clear()


def _nm_symbols(path: Path, *, undefined: bool) -> set[str]:
    command = ["nm", "-u" if undefined else "--defined-only", str(path)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    symbols: set[str] = set()
    for line in completed.stdout.splitlines():
        fields = line.split()
        if fields:
            symbols.add(fields[-1])
    return symbols


def _classify_compile_failure(diagnostic: str) -> str:
    lowered = diagnostic.lower()
    if "compiled module file" in lowered or "cannot open module file" in lowered:
        return "module_not_in_case_build"
    if "only-list does not exist" in lowered or "not accessible" in lowered:
        return "procedure_not_public_or_not_in_build"
    if "number of actual arguments" in lowered and "dummy arguments" in lowered:
        return "case_build_signature_variant"
    if "rank mismatch" in lowered or "type mismatch" in lowered:
        return "generated_interface_mismatch"
    return "compiler_rejected_adapter"


def _short_diagnostic(value: str, limit: int = 4000) -> str:
    text = value.strip()
    return text if len(text) <= limit else text[:limit] + "\n... truncated ..."


def _safe_slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _descriptor_tree_hash(manifests: Sequence[Path]) -> str:
    digest = sha256()
    for manifest in manifests:
        for path in (
            manifest.parent / "kernel.yaml",
            manifest.parent / "adapter.F90",
            manifest,
        ):
            digest.update(str(path.relative_to(manifest.parents[1])).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _smoke_procedure(dtype: str, rank: int) -> FortranProcedure:
    return FortranProcedure(
        qualified_name=f"freecam_smoke_{dtype}_r{rank}::smoke",
        name="smoke",
        module=f"freecam_smoke_{dtype}_r{rank}",
        procedure_kind="subroutine",
        source="synthetic/kernel.F90",
        parser="synthetic",
        line_start=1,
        line_end=1,
        arguments=(
            FortranArgument(
                name="value",
                fortran_type=dtype,
                kind=None,
                dtype=dtype,
                intent="inout",
                rank=rank,
                dimensions=tuple(":" for _ in range(rank)),
            ),
        ),
        uses=(),
        wildcard_uses=(),
        calls=(),
        possible_function_calls=(),
        role="numeric_kernel",
        signature_status="complete",
        adapter_status="candidate",
        blockers=(),
    )


def _smoke_kernel_source(dtype: str, rank: int) -> str:
    declaration = {
        "complex64": "complex(c_float_complex)",
        "complex128": "complex(c_double_complex)",
        "float32": "real(c_float)",
        "float64": "real(c_double)",
        "int32": "integer(c_int32_t)",
        "int64": "integer(c_int64_t)",
    }[dtype]
    dimensions = "" if rank == 0 else "(" + ",".join(":" for _ in range(rank)) + ")"
    return (
        f"module freecam_smoke_{dtype}_r{rank}\n"
        "  use, intrinsic :: iso_c_binding\n"
        "  implicit none\n"
        "contains\n"
        "  subroutine smoke(value)\n"
        f"    {declaration}, intent(inout) :: value{dimensions}\n"
        "    value = value + 1\n"
        "  end subroutine smoke\n"
        f"end module freecam_smoke_{dtype}_r{rank}\n"
    )


def _smoke_array(dtype: str, rank: int) -> np.ndarray:
    shape = () if rank == 0 else tuple(2 for _ in range(rank))
    values = np.ones(shape, dtype=np.dtype(dtype))
    return values if rank == 0 else np.asfortranarray(values)
