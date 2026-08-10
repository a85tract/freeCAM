"""Deterministic whole-tree inventory for legacy iCESM CAM physics code.

The old CAM source predates CCPP metadata.  This module uses fparser's Fortran
AST to recover syntax facts, then applies small, reviewable rules to classify
every procedure.  Generated descriptors are evidence and adapter inputs; they
never replace the original numerical implementation.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import re
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .errors import PICAMConfigurationError
from .plan import PICAMStepPlan


FORTRAN_SUFFIXES = frozenset({".f", ".f90"})
CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FortranArgument:
    name: str
    fortran_type: str
    kind: str | None
    dtype: str | None
    intent: str | None
    rank: int
    dimensions: tuple[str, ...]
    optional: bool = False
    pointer: bool = False
    allocatable: bool = False
    value: bool = False

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["dimensions"] = list(self.dimensions)
        return payload


@dataclass(frozen=True, slots=True)
class FortranProcedure:
    qualified_name: str
    name: str
    module: str | None
    procedure_kind: str
    source: str
    parser: str
    line_start: int
    line_end: int
    arguments: tuple[FortranArgument, ...]
    uses: tuple[str, ...]
    wildcard_uses: tuple[str, ...]
    calls: tuple[str, ...]
    possible_function_calls: tuple[str, ...]
    role: str
    signature_status: str
    adapter_status: str
    blockers: tuple[str, ...]
    matched_rules: tuple[str, ...] = ()
    active_plan_actions: tuple[str, ...] = ()
    resolved_calls: tuple[str, ...] = ()
    unresolved_calls: tuple[str, ...] = ()
    source_sha256: str = ""

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["arguments"] = [item.as_dict() for item in self.arguments]
        for name in (
            "uses",
            "wildcard_uses",
            "calls",
            "possible_function_calls",
            "blockers",
            "matched_rules",
            "active_plan_actions",
            "resolved_calls",
            "unresolved_calls",
        ):
            payload[name] = list(getattr(self, name))
        return payload


@dataclass(frozen=True, slots=True)
class FortranParseFailure:
    source: str
    error_type: str
    message: str
    fallback_procedures: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["fallback_procedures"] = list(self.fallback_procedures)
        return payload


@dataclass(frozen=True, slots=True)
class KernelRule:
    rule_id: str
    match: Mapping[str, Any]
    set_values: Mapping[str, Any]

    def matches(self, procedure: FortranProcedure) -> bool:
        scalar_patterns = {
            "name_regex": procedure.name,
            "module_regex": procedure.module or "",
            "source_regex": procedure.source,
            "qualified_name_regex": procedure.qualified_name,
            "role": procedure.role,
            "procedure_kind": procedure.procedure_kind,
        }
        for key, value in self.match.items():
            if key in scalar_patterns:
                candidate = scalar_patterns[key]
                if key.endswith("_regex"):
                    if re.search(str(value), candidate, re.IGNORECASE) is None:
                        return False
                elif str(value).lower() != candidate.lower():
                    return False
            elif key == "uses_any":
                expected = {str(item).lower() for item in value}
                if not expected.intersection(procedure.uses):
                    return False
            elif key == "uses_regex":
                if not any(
                    re.search(str(value), module, re.IGNORECASE)
                    for module in procedure.uses
                ):
                    return False
            elif key == "calls_any":
                expected = {str(item).lower() for item in value}
                if not expected.intersection(procedure.calls):
                    return False
            elif key == "calls_regex":
                if not any(
                    re.search(str(value), call, re.IGNORECASE)
                    for call in procedure.calls
                ):
                    return False
            elif key == "has_blockers":
                expected = {str(item) for item in value}
                if not expected.intersection(procedure.blockers):
                    return False
            else:
                raise PICAMConfigurationError(
                    f"kernel rule {self.rule_id!r} has unknown matcher {key!r}"
                )
        return True


@dataclass(frozen=True, slots=True)
class PICAMKernelRules:
    kind_map: Mapping[str, str]
    dimension_aliases: Mapping[str, str]
    rules: tuple[KernelRule, ...]
    overrides: Mapping[str, Mapping[str, Any]]
    source: str

    @classmethod
    def load(cls, path: str | Path) -> "PICAMKernelRules":
        source = Path(path).resolve()
        payload = yaml.safe_load(source.read_text())
        if not isinstance(payload, Mapping):
            raise PICAMConfigurationError(f"kernel rules must be a mapping: {source}")
        if int(payload.get("schema_version", 0)) != 1:
            raise PICAMConfigurationError(
                f"kernel rules require schema_version: 1: {source}"
            )
        records = payload.get("rules", ())
        if not isinstance(records, Sequence):
            raise PICAMConfigurationError("kernel rules.rules must be a sequence")
        rules: list[KernelRule] = []
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, Mapping):
                raise PICAMConfigurationError("every kernel rule must be a mapping")
            rule_id = str(record.get("id", "")).strip()
            if not rule_id or rule_id in seen:
                raise PICAMConfigurationError(
                    f"kernel rule id is empty or duplicated: {rule_id!r}"
                )
            seen.add(rule_id)
            match = record.get("match", {})
            values = record.get("set", {})
            if not isinstance(match, Mapping) or not isinstance(values, Mapping):
                raise PICAMConfigurationError(
                    f"kernel rule {rule_id!r} match/set must be mappings"
                )
            rules.append(KernelRule(rule_id, dict(match), dict(values)))
        overrides = payload.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise PICAMConfigurationError("kernel rules.overrides must be a mapping")
        return cls(
            kind_map={
                str(key).lower(): str(value)
                for key, value in dict(payload.get("kind_map", {})).items()
            },
            dimension_aliases={
                str(key).lower(): str(value)
                for key, value in dict(payload.get("dimension_aliases", {})).items()
            },
            rules=tuple(rules),
            overrides={str(key).lower(): dict(value) for key, value in overrides.items()},
            source=str(source),
        )

    def apply(self, procedure: FortranProcedure) -> FortranProcedure:
        values: dict[str, Any] = {}
        matched: list[str] = []
        for rule in self.rules:
            if rule.matches(procedure):
                matched.append(rule.rule_id)
                values.update(rule.set_values)
        for key in (procedure.qualified_name.lower(), procedure.name.lower()):
            if key in self.overrides:
                matched.append(f"override:{key}")
                values.update(self.overrides[key])
                break
        allowed = {"role", "signature_status", "adapter_status"}
        unknown = set(values).difference(allowed | {"remove_blockers", "add_blockers"})
        if unknown:
            raise PICAMConfigurationError(
                f"rules for {procedure.qualified_name} set unknown values: "
                + ", ".join(sorted(unknown))
            )
        blockers = set(procedure.blockers)
        blockers.difference_update(str(item) for item in values.get("remove_blockers", ()))
        blockers.update(str(item) for item in values.get("add_blockers", ()))
        return replace(
            procedure,
            role=str(values.get("role", procedure.role)),
            signature_status=str(
                values.get("signature_status", procedure.signature_status)
            ),
            adapter_status=str(values.get("adapter_status", procedure.adapter_status)),
            blockers=tuple(sorted(blockers)),
            matched_rules=tuple(matched),
        )


class PICAMSourceCatalog:
    """All physical procedures found in one pinned CAM source tree."""

    def __init__(
        self,
        *,
        project_root: Path,
        source_root: Path,
        scan_roots: Sequence[Path],
        rules: PICAMKernelRules,
        source_files: Sequence[Path],
        parsed_files: Sequence[Path],
        procedures: Sequence[FortranProcedure],
        failures: Sequence[FortranParseFailure],
    ) -> None:
        self.project_root = project_root.resolve()
        self.source_root = source_root.resolve()
        self.scan_roots = tuple(path.resolve() for path in scan_roots)
        self.rules = rules
        self.source_files = tuple(path.resolve() for path in source_files)
        self.parsed_files = tuple(path.resolve() for path in parsed_files)
        self.procedures = tuple(sorted(procedures, key=lambda item: item.qualified_name))
        self.failures = tuple(sorted(failures, key=lambda item: item.source))
        self.source_revision = _git_revision(self.source_root)
        cam_root = self.source_root / "components/cam"
        self.cam_source_revision = _git_revision(
            cam_root if cam_root.is_dir() else self.source_root
        )
        self.source_tree_sha256 = _source_tree_hash(self.source_files, self.source_root)

    @classmethod
    def discover(
        cls,
        project_root: str | Path,
        *,
        source_root: str | Path | None = None,
        rules_path: str | Path | None = None,
        scan_roots: Sequence[str | Path] | None = None,
        workers: int = 1,
    ) -> "PICAMSourceCatalog":
        project = Path(project_root).resolve()
        source = Path(
            source_root or project / "external/iCESM1.3.1_fzhu"
        ).resolve()
        rules = PICAMKernelRules.load(
            rules_path or project / "native/pi_cam/kernel_rules.yaml"
        )
        roots = tuple(
            Path(item).resolve()
            for item in (
                scan_roots
                or (source / "components/cam/src/physics",)
            )
        )
        files = tuple(
            sorted(
                {
                    path.resolve()
                    for root in roots
                    for path in root.rglob("*")
                    if path.is_file() and path.suffix.lower() in FORTRAN_SUFFIXES
                }
            )
        )
        arguments = [(str(path), str(source), rules.kind_map) for path in files]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                scanned = tuple(executor.map(_scan_file_worker, arguments))
        else:
            scanned = tuple(_scan_file_worker(item) for item in arguments)
        procedures: list[FortranProcedure] = []
        failures: list[FortranParseFailure] = []
        parsed_files: list[Path] = []
        for path, (records, failure) in zip(files, scanned, strict=True):
            procedures.extend(
                _procedure_from_payload(record, rules.dimension_aliases)
                for record in records
            )
            if failure is not None:
                failures.append(
                    FortranParseFailure(
                        source=str(failure["source"]),
                        error_type=str(failure["error_type"]),
                        message=str(failure["message"]),
                        fallback_procedures=tuple(failure["fallback_procedures"]),
                    )
                )
            else:
                parsed_files.append(path)
        procedures = _resolve_catalog_calls(procedures)
        plan = PICAMStepPlan.default()
        plan_actions = {
            candidate.lower(): action.qualified_name
            for action in plan.actions
            for candidate in (action.name, action.operation)
        }
        final: list[FortranProcedure] = []
        for procedure in procedures:
            active = tuple(
                sorted(
                    {
                        plan_actions[name]
                        for name in (procedure.name.lower(),)
                        if name in plan_actions
                    }
                )
            )
            final.append(rules.apply(replace(procedure, active_plan_actions=active)))
        return cls(
            project_root=project,
            source_root=source,
            scan_roots=roots,
            rules=rules,
            source_files=files,
            parsed_files=parsed_files,
            procedures=final,
            failures=failures,
        )

    def summary(self) -> dict[str, object]:
        role_counts = _counts(item.role for item in self.procedures)
        parser_counts = _counts(item.parser for item in self.procedures)
        signature_counts = _counts(item.signature_status for item in self.procedures)
        adapter_counts = _counts(item.adapter_status for item in self.procedures)
        physics = tuple(
            item
            for item in self.procedures
            if item.role in {"process", "numeric_kernel"}
        )
        blockers = _counts(
            blocker
            for procedure in self.procedures
            for blocker in procedure.blockers
        )
        physics_blockers = _counts(
            blocker for procedure in physics for blocker in procedure.blockers
        )
        cataloged_fallback = sum(len(item.fallback_procedures) for item in self.failures)
        fallback_sources = {
            item.source for item in self.procedures if item.parser == "fallback-regex"
        }
        cpp_sources = {
            item.source for item in self.procedures if item.parser == "fparser-cpp"
        }
        raw_file_coverage = (
            len(self.parsed_files) / len(self.source_files)
            if self.source_files
            else 1.0
        )
        signature_coverage = (
            (len(self.procedures) - parser_counts.get("fallback-regex", 0))
            / len(self.procedures)
            if self.procedures
            else 1.0
        )
        return {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "generator": "freecam.pi_cam.source_catalog",
            "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_root": _portable(self.source_root, self.project_root),
            "source_revision": self.source_revision,
            "cam_source_revision": self.cam_source_revision,
            "source_tree_sha256": self.source_tree_sha256,
            "rules": _portable(Path(self.rules.source), self.project_root),
            "rules_sha256": sha256(Path(self.rules.source).read_bytes()).hexdigest(),
            "scan_roots": [
                _portable(path, self.project_root) for path in self.scan_roots
            ],
            "source_files": len(self.source_files),
            "parsed_files": len(self.parsed_files),
            "parse_failed_files": len(self.failures),
            "cpp_recovered_files": len(cpp_sources),
            "unresolved_signature_files": len(fallback_sources),
            "procedures": len(self.procedures),
            "physics_kernel_procedures": len(physics),
            "physics_kernel_adapter_status_counts": _counts(
                item.adapter_status for item in physics
            ),
            "blocker_counts": blockers,
            "physics_kernel_blocker_counts": physics_blockers,
            "parser_counts": parser_counts,
            "ast_parsed_procedures": parser_counts.get("fparser", 0),
            "fallback_procedures": parser_counts.get("fallback-regex", 0),
            "fallback_procedure_names": cataloged_fallback,
            "descriptors": len(self.procedures),
            "generated_pointer_adapters": sum(
                _can_generate_pointer_adapter(item) for item in self.procedures
            ),
            "role_counts": role_counts,
            "signature_status_counts": signature_counts,
            "adapter_status_counts": adapter_counts,
            "active_plan_procedures": sum(
                bool(item.active_plan_actions) for item in self.procedures
            ),
            "resolved_call_edges": sum(
                len(item.resolved_calls) for item in self.procedures
            ),
            "unresolved_call_names": sum(
                len(item.unresolved_calls) for item in self.procedures
            ),
            "raw_ast_file_coverage": raw_file_coverage,
            "procedure_signature_coverage": signature_coverage,
            "catalog_coverage": signature_coverage,
        }

    def select(
        self,
        *,
        role: str | None = None,
        adapter_status: str | None = None,
        parser: str | None = None,
    ) -> tuple[FortranProcedure, ...]:
        """Return a deterministic subset without changing catalog semantics."""

        return tuple(
            procedure
            for procedure in self.procedures
            if (role is None or procedure.role == role)
            and (
                adapter_status is None
                or procedure.adapter_status == adapter_status
            )
            and (parser is None or procedure.parser == parser)
        )

    def procedure(self, name: str) -> FortranProcedure:
        """Resolve one qualified name, rejecting ambiguous bare names."""

        token = str(name).strip().lower()
        matches = tuple(
            procedure
            for procedure in self.procedures
            if procedure.qualified_name.lower() == token
            or procedure.name.lower() == token
        )
        if not matches:
            raise PICAMConfigurationError(f"unknown PI-CAM procedure {name!r}")
        if len(matches) > 1:
            choices = ", ".join(item.qualified_name for item in matches[:8])
            raise PICAMConfigurationError(
                f"ambiguous PI-CAM procedure {name!r}; use one of: {choices}"
            )
        return matches[0]

    def machine_record(self) -> dict[str, object]:
        summary = self.summary()
        summary["procedure_count"] = summary.pop("procedures")
        parsed = {path.resolve() for path in self.parsed_files}
        cpp_sources = {
            procedure.source
            for procedure in self.procedures
            if procedure.parser == "fparser-cpp"
        }
        fallback_sources = {
            procedure.source
            for procedure in self.procedures
            if procedure.parser == "fallback-regex"
        }
        return {
            **summary,
            "source_file_inventory": [
                {
                    "source": _portable(path, self.source_root),
                    "raw_ast_parsed": path in parsed,
                    "cpp_recovered": _portable(path, self.source_root) in cpp_sources,
                    "fallback_required": (
                        _portable(path, self.source_root) in fallback_sources
                    ),
                }
                for path in self.source_files
            ],
            "procedures": [item.as_dict() for item in self.procedures],
            "parse_failures": [item.as_dict() for item in self.failures],
        }

    def write_report(self, path: str | Path) -> Path:
        output = Path(path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.machine_record(), indent=2, sort_keys=True) + "\n")
        return output

    def write_descriptors(
        self, output_root: str | Path, *, clean: bool = False
    ) -> tuple[Path, ...]:
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        expected: set[Path] = set()
        outputs: list[Path] = []
        name_counts = _counts(item.qualified_name for item in self.procedures)
        for procedure in self.procedures:
            slug = _descriptor_slug(procedure)
            if name_counts[procedure.qualified_name] > 1:
                identity = sha256(
                    (
                        f"{procedure.source}\0{procedure.qualified_name}\0"
                        f"{procedure.line_start}"
                    ).encode("utf-8")
                ).hexdigest()[:12]
                slug += f"__{identity}"
            child = root / slug
            path = child / "kernel.yaml"
            expected.add(path)
            child.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": CATALOG_SCHEMA_VERSION,
                "generated": {
                    "source_catalog": "freecam.pi_cam.source_catalog",
                    "rules": _portable(Path(self.rules.source), self.project_root),
                    "source_sha256": procedure.source_sha256,
                    "matched_rules": list(procedure.matched_rules),
                },
                **procedure.as_dict(),
            }
            path.write_text(yaml.safe_dump(payload, sort_keys=False))
            if _can_generate_pointer_adapter(procedure):
                adapter = child / "adapter.F90"
                adapter.write_text(_generate_pointer_adapter(procedure))
                manifest = child / "adapter.json"
                manifest.write_text(
                    json.dumps(
                        _adapter_manifest(procedure, adapter.name),
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                expected.update((adapter, manifest))
            outputs.append(path)
        index = root / "catalog.json"
        expected.add(index)
        index.write_text(json.dumps(self.summary(), indent=2, sort_keys=True) + "\n")
        if clean:
            for pattern in ("kernel.yaml", "adapter.F90", "adapter.json"):
                for path in root.rglob(pattern):
                    if path not in expected:
                        path.unlink()
            for directory in sorted(
                (path for path in root.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                try:
                    directory.rmdir()
                except OSError:
                    pass
        return tuple(outputs)


def _scan_file_worker(
    arguments: tuple[str, str, Mapping[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    path_text, source_root_text, kind_map = arguments
    path = Path(path_text)
    source_root = Path(source_root_text)
    text = path.read_text(errors="replace")
    portable = _portable(path, source_root)
    digest = sha256(path.read_bytes()).hexdigest()
    try:
        from fparser.common.readfortran import FortranFileReader
        from fparser.two.parser import ParserFactory

        tree = ParserFactory().create(std="f2008")(
            FortranFileReader(str(path), ignore_comments=False)
        )
        return _records_from_tree(
            tree,
            text=text,
            portable=portable,
            digest=digest,
            kind_map=kind_map,
            parser_name="fparser",
        ), None
    except Exception as raw_exc:
        parsed_records: list[dict[str, Any]] = []
        cpp_error: str | None = None
        try:
            from fparser.common.readfortran import FortranStringReader
            from fparser.two.parser import ParserFactory

            cpp = subprocess.run(
                [
                    "cpp",
                    "-P",
                    "-traditional-cpp",
                    "-DUSE_CONTIGUOUS=",
                    "-I",
                    str(path.parent),
                    "-I",
                    str(source_root / "cime/src/share/include"),
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            preprocessed = cpp.stdout
            tree = ParserFactory().create(std="f2008")(
                FortranStringReader(preprocessed, ignore_comments=False)
            )
            parsed_records = _records_from_tree(
                tree,
                text=preprocessed,
                portable=portable,
                digest=digest,
                kind_map=kind_map,
                parser_name="fparser-cpp",
            )
        except Exception as exc:
            cpp_error = str(exc).splitlines()[-1] if str(exc) else repr(exc)
        fallback_records = _fallback_procedure_records(text, portable, digest)
        parsed_names = {record["name"] for record in parsed_records}
        missing_records = [
            record for record in fallback_records if record["name"] not in parsed_names
        ]
        records = [*parsed_records, *missing_records]
        fallback = tuple(sorted(record["name"] for record in missing_records))
        raw_message = (
            str(raw_exc).splitlines()[-1] if str(raw_exc) else repr(raw_exc)
        )
        message = raw_message if cpp_error is None else f"{raw_message}; cpp: {cpp_error}"
        return records, FortranParseFailure(
            source=portable,
            error_type=type(raw_exc).__name__,
            message=message,
            fallback_procedures=fallback,
        ).as_dict()


def _records_from_tree(
    tree: Any,
    *,
    text: str,
    portable: str,
    digest: str,
    kind_map: Mapping[str, str],
    parser_name: str,
) -> list[dict[str, Any]]:
    from fparser.two.Fortran2003 import (
        Call_Stmt,
        Function_Stmt,
        Function_Subprogram,
        Module,
        Module_Stmt,
        Subroutine_Stmt,
        Subroutine_Subprogram,
        Type_Declaration_Stmt,
        Use_Stmt,
    )
    from fparser.two.utils import walk

    procedures: list[dict[str, Any]] = []
    classes = (Subroutine_Subprogram, Function_Subprogram)
    for node in walk(tree, classes):
        if _nearest_procedure(node.parent, classes) is not None:
            continue
        stmt_class = (
            Subroutine_Stmt
            if isinstance(node, Subroutine_Subprogram)
            else Function_Stmt
        )
        statements = walk(node, stmt_class)
        if not statements:
            continue
        statement = statements[0]
        name = str(statement.items[1]).lower()
        argument_list = statement.items[2]
        argument_names = tuple(
            str(item).lower()
            for item in (() if argument_list is None else argument_list.items)
        )
        module = _module_name(node, Module, Module_Stmt)
        declarations: dict[str, FortranArgument] = {}
        for declaration in walk(node, Type_Declaration_Stmt):
            if _nearest_procedure(declaration, classes) is not node:
                continue
            for argument in _declaration_arguments(declaration, kind_map):
                declarations[argument.name] = argument
        arguments_out: list[FortranArgument] = []
        blockers: set[str] = set()
        for argument_name in argument_names:
            if argument_name not in declarations:
                blockers.add("undeclared_argument")
                arguments_out.append(
                    FortranArgument(
                        argument_name, "unknown", None, None, None, 0, ()
                    )
                )
            else:
                argument = declarations[argument_name]
                arguments_out.append(argument)
                blockers.update(_argument_blockers(argument))
        uses: set[str] = set()
        wildcard_uses: set[str] = set()
        for use in walk(node, Use_Stmt):
            if _nearest_procedure(use, classes) is not node:
                continue
            module_name = str(use.items[2]).lower()
            uses.add(module_name)
            if "ONLY" not in str(use).upper():
                wildcard_uses.add(module_name)
        calls = {
            str(call.items[0]).lower()
            for call in walk(node, Call_Stmt)
            if _nearest_procedure(call, classes) is node
        }
        line_start, line_end = _node_span(node, statement, text)
        body = "\n".join(text.splitlines()[line_start - 1 : line_end])
        if wildcard_uses:
            blockers.add("wildcard_module_import")
        if re.search(r"\b(common|equivalence)\b", body, re.IGNORECASE):
            blockers.add("legacy_shared_storage")
        if re.search(
            r"\b(read|write|open|close|flush|inquire)\s*(?:\(|\*)",
            body,
            re.IGNORECASE,
        ):
            blockers.add("fortran_io")
        if any(
            module_name.startswith(("mpi", "pio", "mct", "spmd"))
            for module_name in uses
        ):
            blockers.add("parallel_runtime_dependency")
        if isinstance(node, Function_Subprogram):
            blockers.add("function_result_abi")
        signature_status = "explicit" if not blockers else "needs_rule"
        adapter_status = "candidate" if not blockers else "blocked"
        role = _initial_role(name, body, uses, calls)
        candidates = _possible_function_names(body, declarations, name)
        qualified = f"{module or Path(portable).stem.lower()}::{name}"
        procedures.append(
            FortranProcedure(
                qualified_name=qualified,
                name=name,
                module=module,
                procedure_kind=(
                    "subroutine"
                    if isinstance(node, Subroutine_Subprogram)
                    else "function"
                ),
                source=portable,
                parser=parser_name,
                line_start=line_start,
                line_end=line_end,
                arguments=tuple(arguments_out),
                uses=tuple(sorted(uses)),
                wildcard_uses=tuple(sorted(wildcard_uses)),
                calls=tuple(sorted(calls)),
                possible_function_calls=tuple(sorted(candidates)),
                role=role,
                signature_status=signature_status,
                adapter_status=adapter_status,
                blockers=tuple(sorted(blockers)),
                source_sha256=digest,
            ).as_dict()
        )
    return procedures


def _procedure_from_payload(
    payload: Mapping[str, Any],
    dimension_aliases: Mapping[str, str] | None = None,
) -> FortranProcedure:
    sequence_fields = {
        "uses",
        "wildcard_uses",
        "calls",
        "possible_function_calls",
        "blockers",
        "matched_rules",
        "active_plan_actions",
        "resolved_calls",
        "unresolved_calls",
    }
    values = dict(payload)
    aliases = {
        str(name).lower(): str(value)
        for name, value in (dimension_aliases or {}).items()
    }
    arguments: list[FortranArgument] = []
    for record in payload.get("arguments", ()):
        dimensions = tuple(
            aliases.get(str(value).lower(), str(value).lower())
            for value in record.get("dimensions", ())
        )
        arguments.append(
            FortranArgument(**{**record, "dimensions": dimensions})
        )
    values["arguments"] = tuple(arguments)
    for name in sequence_fields:
        values[name] = tuple(payload.get(name, ()))
    return FortranProcedure(**values)


def _nearest_procedure(node: Any, classes: tuple[type, ...]) -> Any | None:
    current = node
    while current is not None:
        if isinstance(current, classes):
            return current
        current = getattr(current, "parent", None)
    return None


def _module_name(node: Any, module_class: type, statement_class: type) -> str | None:
    current = getattr(node, "parent", None)
    while current is not None:
        if isinstance(current, module_class):
            from fparser.two.utils import walk

            statements = walk(current, statement_class)
            return str(statements[0].items[1]).lower() if statements else None
        current = getattr(current, "parent", None)
    return None


def _declaration_arguments(
    declaration: Any, kind_map: Mapping[str, str]
) -> tuple[FortranArgument, ...]:
    type_text = str(declaration.items[0]).lower()
    attributes = "" if declaration.items[1] is None else str(declaration.items[1])
    upper_attributes = attributes.upper()
    intent_match = re.search(r"INTENT\s*\(\s*(INOUT|IN|OUT)\s*\)", upper_attributes)
    intent = None if intent_match is None else intent_match.group(1).lower()
    dimension_match = re.search(r"DIMENSION\s*\((.*)\)", attributes, re.IGNORECASE)
    attr_dimensions = (
        ()
        if dimension_match is None
        else tuple(_split_top_level(dimension_match.group(1)))
    )
    kind_match = re.search(r"kind\s*=\s*([a-zA-Z0-9_]+)", type_text)
    kind = None if kind_match is None else kind_match.group(1).lower()
    base = type_text.split("(", 1)[0].strip()
    derived_match = re.match(r"(?:type|class)\s*\(\s*([a-zA-Z0-9_]+)", type_text)
    if derived_match:
        base = f"type:{derived_match.group(1).lower()}"
    dtype = _dtype_for(base, kind, kind_map)
    results: list[FortranArgument] = []
    entities = declaration.items[2]
    for entity in (() if entities is None else entities.items):
        name = str(entity.items[0]).lower()
        shape = entity.items[1]
        dimensions = (
            tuple(_split_top_level(str(shape))) if shape is not None else attr_dimensions
        )
        results.append(
            FortranArgument(
                name=name,
                fortran_type=base,
                kind=kind,
                dtype=dtype,
                intent=intent,
                rank=len(dimensions),
                dimensions=tuple(item.strip().lower() for item in dimensions),
                optional="OPTIONAL" in upper_attributes,
                pointer="POINTER" in upper_attributes,
                allocatable="ALLOCATABLE" in upper_attributes,
                value="VALUE" in upper_attributes,
            )
        )
    return tuple(results)


def _dtype_for(base: str, kind: str | None, kind_map: Mapping[str, str]) -> str | None:
    if kind and kind.lower() in kind_map:
        mapped = str(kind_map[kind.lower()])
        if base == "complex":
            return {"float32": "complex64", "float64": "complex128"}.get(
                mapped, mapped
            )
        return mapped
    defaults = {
        "real": "float32",
        "double precision": "float64",
        "integer": "int32",
        "logical": "logical",
        "complex": "complex64",
        "character": "character",
    }
    return defaults.get(base)


def _argument_blockers(argument: FortranArgument) -> set[str]:
    blockers: set[str] = set()
    if argument.intent is None:
        blockers.add("missing_intent")
    if argument.dtype is None or argument.fortran_type.startswith("type:"):
        blockers.add("derived_or_unknown_type")
    if argument.dtype in {"logical", "character"}:
        blockers.add("nontrivial_c_representation")
    if argument.optional:
        blockers.add("optional_argument")
    if argument.pointer:
        blockers.add("pointer_argument")
    if argument.allocatable:
        blockers.add("allocatable_argument")
    if any(item == "*" or ".." in item for item in argument.dimensions):
        blockers.add("unsupported_extent")
    return blockers


def _initial_role(
    name: str, body: str, uses: Iterable[str], calls: Iterable[str]
) -> str:
    if re.search(
        r"(?:^|_)(?:init|initialize|register|final|finalize|readnl|restart)(?:_|$)",
        name,
    ):
        return "lifecycle"
    if any(item.startswith(("mpi", "pio", "mct")) for item in uses):
        return "host_service"
    if name.endswith(("_tend", "_run", "_calc", "_driver", "_update")):
        return "process"
    if re.search(
        r"(?:^|_)(?:read|write|open|close|history|restart|output|input)(?:_|$)",
        name,
    ) and re.search(
        r"\b(?:read|write|open|close|flush)\b", body, re.IGNORECASE
    ):
        return "host_service"
    if name in calls:
        return "helper"
    return "numeric_kernel"


def _possible_function_names(
    body: str, declarations: Mapping[str, FortranArgument], own_name: str
) -> set[str]:
    excluded = set(declarations) | {own_name}
    excluded.update(
        {
            "abs", "acos", "aimag", "aint", "allocated", "associated",
            "cos", "dble", "dot_product", "exp", "int", "kind", "len",
            "log", "max", "maxval", "merge", "min", "minval", "mod",
            "nint", "present", "real", "reshape", "sign", "sin", "size",
            "sqrt", "sum", "tiny", "trim", "where", "write", "read",
        }
    )
    calls = {
        match.group(1).lower()
        for match in re.finditer(r"\b([a-zA-Z][a-zA-Z0-9_]*)\s*\(", body)
    }
    return calls.difference(excluded)


def _resolve_catalog_calls(
    procedures: Sequence[FortranProcedure],
) -> list[FortranProcedure]:
    by_name: dict[str, list[FortranProcedure]] = {}
    by_module_name: dict[tuple[str, str], FortranProcedure] = {}
    for procedure in procedures:
        by_name.setdefault(procedure.name, []).append(procedure)
        if procedure.module:
            by_module_name[(procedure.module, procedure.name)] = procedure
    output: list[FortranProcedure] = []
    for procedure in procedures:
        resolved: set[str] = set()
        unresolved: set[str] = set()
        for call_name in set(procedure.calls).union(procedure.possible_function_calls):
            local = (
                by_module_name.get((procedure.module, call_name))
                if procedure.module
                else None
            )
            candidates = by_name.get(call_name, ())
            if local is not None:
                resolved.add(local.qualified_name)
            elif len(candidates) == 1:
                resolved.add(candidates[0].qualified_name)
            elif candidates or call_name in procedure.calls:
                unresolved.add(call_name)
        output.append(
            replace(
                procedure,
                resolved_calls=tuple(sorted(resolved)),
                unresolved_calls=tuple(sorted(unresolved)),
            )
        )
    return output


def _node_span(node: Any, statement: Any, text: str) -> tuple[int, int]:
    start = int(getattr(getattr(statement, "item", None), "span", (1, 1))[0])
    end = start
    content = getattr(node, "content", ())
    if content:
        item = getattr(content[-1], "item", None)
        if item is not None and getattr(item, "span", None):
            end = int(item.span[-1])
    return max(1, start), min(max(start, end), len(text.splitlines()))


def _split_top_level(value: str) -> list[str]:
    output: list[str] = []
    current: list[str] = []
    depth = 0
    for character in value:
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        if character == "," and depth == 0:
            output.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if current:
        output.append("".join(current).strip())
    return [item for item in output if item]


def _fallback_procedure_names(text: str) -> set[str]:
    pattern = re.compile(
        r"(?im)^\s*(?!end\b)(?:[a-z0-9_()=*,:]+\s+)*"
        r"(?:subroutine|function)\s+([a-z][a-z0-9_]*)"
    )
    return {match.group(1).lower() for match in pattern.finditer(text)}


def _fallback_procedure_records(
    text: str,
    portable_source: str,
    digest: str,
) -> list[dict[str, Any]]:
    """Recover a fail-closed descriptor when legacy CPP defeats fparser.

    The fallback intentionally does not claim a usable ABI.  It preserves the
    procedure name, source span and direct CALL edges so every source routine
    remains visible in the catalog while its signature is marked unavailable.
    """

    header = re.compile(
        r"(?im)^\s*(?!end\b)(?:[a-z0-9_()=*,:]+\s+)*"
        r"(?P<kind>subroutine|function)\s+(?P<name>[a-z][a-z0-9_]*)"
        r"\s*(?:\((?P<args>[^)]*)\))?"
    )
    matches = tuple(header.finditer(text))
    module_match = re.search(
        r"(?im)^\s*module\s+(?!procedure\b)([a-z][a-z0-9_]*)", text
    )
    module = None if module_match is None else module_match.group(1).lower()
    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, match in enumerate(matches):
        name = match.group("name").lower()
        if name in seen_names:
            continue
        seen_names.add(name)
        kind = match.group("kind").lower()
        line_start = text.count("\n", 0, match.start()) + 1
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        end_match = re.search(
            rf"(?im)^\s*end\s+(?:{kind})(?:\s+{re.escape(name)})?\s*$",
            text[match.end() : next_start],
        )
        end_offset = (
            next_start
            if end_match is None
            else match.end() + end_match.end()
        )
        line_end = text.count("\n", 0, end_offset) + 1
        body = text[match.start() : end_offset]
        raw_arguments = (match.group("args") or "").replace("&", " ")
        argument_names = tuple(
            item.strip().lower()
            for item in _split_top_level(raw_arguments)
            if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", item.strip())
        )
        arguments = tuple(
            FortranArgument(
                name=argument,
                fortran_type="unknown",
                kind=None,
                dtype=None,
                intent=None,
                rank=0,
                dimensions=(),
            )
            for argument in argument_names
        )
        uses = {
            item.lower()
            for item in re.findall(
                r"(?im)^\s*use(?:\s*,[^:]*)?(?:::)?\s*([a-z][a-z0-9_]*)",
                body,
            )
        }
        calls = {
            item.lower()
            for item in re.findall(
                r"(?i)\bcall\s+([a-z][a-z0-9_]*)", body
            )
        }
        qualified = f"{module or Path(portable_source).stem.lower()}::{name}"
        records.append(
            FortranProcedure(
                qualified_name=qualified,
                name=name,
                module=module,
                procedure_kind=kind,
                source=portable_source,
                parser="fallback-regex",
                line_start=line_start,
                line_end=line_end,
                arguments=arguments,
                uses=tuple(sorted(uses)),
                wildcard_uses=(),
                calls=tuple(sorted(calls)),
                possible_function_calls=(),
                role=_initial_role(name, body, uses, calls),
                signature_status="unparsed",
                adapter_status="parse_blocked",
                blockers=("parse_failed_file", "unparsed_signature"),
                source_sha256=digest,
            ).as_dict()
        )
    return records


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _descriptor_slug(procedure: FortranProcedure) -> str:
    base = procedure.qualified_name.replace("::", "__")
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", base).lower()


_C_INTEROP_TYPES = {
    "complex64": "complex(c_float_complex)",
    "complex128": "complex(c_double_complex)",
    "float32": "real(c_float)",
    "float64": "real(c_double)",
    "int32": "integer(c_int32_t)",
    "int64": "integer(c_int64_t)",
}


def _can_generate_pointer_adapter(procedure: FortranProcedure) -> bool:
    return (
        procedure.procedure_kind == "subroutine"
        and procedure.adapter_status == "candidate"
        and all(argument.dtype in _C_INTEROP_TYPES for argument in procedure.arguments)
    )


def _adapter_symbol(procedure: FortranProcedure) -> str:
    identity = sha256(
        (
            f"{procedure.source}\0{procedure.qualified_name}\0"
            f"{procedure.line_start}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    # Fortran 2003 identifiers are limited to 63 characters.  The digest keeps
    # truncated legacy names unambiguous while the manifest retains the full
    # qualified routine name.
    return f"freecam_pi_cam_{procedure.name[:28]}_{identity}_v1"


def _adapter_manifest(
    procedure: FortranProcedure, adapter_source: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": procedure.qualified_name,
        "original_source": procedure.source,
        "original_module": procedure.module,
        "original_routine": procedure.name,
        "adapter_source": adapter_source,
        "symbol": _adapter_symbol(procedure),
        "action_id": 0,
        "status": "generated-unbound",
        "binding_policy": (
            "Each argument requires an explicit StatePool field, dimension, or "
            "scalar binding before this adapter may be installed."
        ),
        "arguments": [argument.as_dict() for argument in procedure.arguments],
    }


def _generate_pointer_adapter(procedure: FortranProcedure) -> str:
    """Generate a generic ABI shim without copying numerical source code."""

    if not _can_generate_pointer_adapter(procedure):
        raise PICAMConfigurationError(
            f"procedure {procedure.qualified_name} is not pointer-adapter ready"
        )
    symbol = _adapter_symbol(procedure)
    module_name = f"freecam_adapter_{symbol.rsplit('_', 2)[-2]}"
    lines = [
        "! Generated by freecam.pi_cam.source_catalog; do not edit.",
        f"module {module_name}",
        "  use, intrinsic :: iso_c_binding, only: c_char, c_double, c_float, &",
        "       c_double_complex, c_float_complex, c_int, c_int32_t, &",
        "       c_int64_t, c_null_char, c_ptr, c_f_pointer",
    ]
    if procedure.module is not None:
        lines.append(
            f"  use {procedure.module}, only: {procedure.name}"
        )
    lines.extend(("  implicit none", "  private", f"  public :: {symbol}"))
    if procedure.module is None:
        names = ", ".join(argument.name for argument in procedure.arguments)
        lines.extend(("  interface", f"    subroutine {procedure.name}({names})"))
        lines.append(
            "      import :: c_double, c_double_complex, c_float, &"
        )
        lines.append(
            "           c_float_complex, c_int32_t, c_int64_t"
        )
        for argument in procedure.arguments:
            declaration = _C_INTEROP_TYPES[str(argument.dtype)]
            intent = argument.intent or "inout"
            if argument.rank:
                if any(dimension.strip() == ":" for dimension in argument.dimensions):
                    dimensions = ",".join(":" for _ in range(argument.rank))
                else:
                    # Explicit-shape external procedures use sequence
                    # association and receive the first-element address.
                    dimensions = "*"
                declaration += f", intent({intent}) :: {argument.name}({dimensions})"
            else:
                declaration += f", intent({intent}) :: {argument.name}"
            lines.append(f"      {declaration}")
        lines.extend((f"    end subroutine {procedure.name}", "  end interface"))
    lines.extend(
        (
            "contains",
            "",
            f"  integer(c_int) function {symbol}(action_id, nargs, pointers, &",
            "       ndims, shapes, max_rank, fortran_comm, errmsg, errmsg_len) &",
            "       bind(C, &",
            f"       name='{symbol}') result(status)",
            "    integer(c_int), value, intent(in) :: action_id, nargs, max_rank",
            "    integer(c_int), value, intent(in) :: fortran_comm, errmsg_len",
            "    type(c_ptr), intent(in) :: pointers(*)",
            "    integer(c_int32_t), intent(in) :: ndims(*)",
            "    integer(c_int64_t), intent(in) :: shapes(*)",
            "    character(kind=c_char), intent(out) :: errmsg(*)",
            "    integer :: index",
        )
    )
    for argument in procedure.arguments:
        fortran_type = _C_INTEROP_TYPES[str(argument.dtype)]
        if argument.rank:
            dimensions = ",".join(":" for _ in range(argument.rank))
            lines.append(
                f"    {fortran_type}, pointer :: arg_{argument.name}({dimensions})"
            )
        else:
            lines.append(f"    {fortran_type}, pointer :: arg_{argument.name}")
    lines.extend(
        (
            "",
            "    do index = 1, max(1, int(errmsg_len))",
            "      errmsg(index) = c_null_char",
            "    end do",
            "    status = 0_c_int",
            f"    if (action_id /= 0 .or. nargs /= {len(procedure.arguments)}) then",
            "      status = 1_c_int",
            "      return",
            "    end if",
        )
    )
    maximum_rank = max((item.rank for item in procedure.arguments), default=0)
    if maximum_rank:
        lines.extend(
            (
                f"    if (max_rank < {maximum_rank}) then",
                "      status = 2_c_int",
                "      return",
                "    end if",
            )
        )
    for field_index, argument in enumerate(procedure.arguments, start=1):
        lines.extend(
            (
                f"    if (ndims({field_index}) /= {argument.rank}) then",
                f"      status = {10 + field_index}_c_int",
                "      return",
                "    end if",
            )
        )
        if argument.rank:
            shape = ", ".join(
                f"int(shapes({(field_index - 1)} * max_rank + {axis}))"
                for axis in range(1, argument.rank + 1)
            )
            lines.append(
                f"    call c_f_pointer(pointers({field_index}), "
                f"arg_{argument.name}, (/ {shape} /))"
            )
        else:
            lines.append(
                f"    call c_f_pointer(pointers({field_index}), arg_{argument.name})"
            )
    call_arguments = ", ".join(
        f"arg_{argument.name}" for argument in procedure.arguments
    )
    lines.append(f"    call {procedure.name}({call_arguments})")
    lines.extend(
        (
            f"  end function {symbol}",
            "",
            f"end module {module_name}",
            "",
        )
    )
    return "\n".join(lines)


def _portable(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _source_tree_hash(files: Sequence[Path], source_root: Path) -> str:
    digest = sha256()
    for path in files:
        digest.update(_portable(path, source_root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
