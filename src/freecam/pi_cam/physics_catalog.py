"""Flat, case-reachable PI-CAM physics process catalog.

The legacy CAM source exposes procedures, not CCPP-style process metadata.  The
catalog built here joins the pinned AST inventory with the active PI-atm call
graph and gives every reachable scientific routine one stable, flat Python
name.  Runtime capability is deliberately kept separate from discovery: a
procedure is not called independently until its required StatePool/context
bindings have been admitted by the native runtime.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .errors import PICAMConfigurationError


PHYSICS_CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PICAMPhysicsRules:
    """Small reviewed semantic layer on top of source-derived AST facts."""

    include_roles: frozenset[str]
    exclude_roles: frozenset[str]
    names: Mapping[str, str]
    levels: Mapping[str, str]
    source: str = "built-in"

    @classmethod
    def default(cls) -> "PICAMPhysicsRules":
        return cls(
            include_roles=frozenset({"process", "numeric_kernel"}),
            exclude_roles=frozenset({"lifecycle", "host_service"}),
            names={},
            levels={},
        )

    @classmethod
    def load(cls, path: str | Path) -> "PICAMPhysicsRules":
        source = Path(path).resolve()
        payload = yaml.safe_load(source.read_text())
        if not isinstance(payload, Mapping) or int(payload.get("schema_version", 0)) != 1:
            raise PICAMConfigurationError(
                f"physics process rules require schema_version: 1: {source}"
            )
        return cls(
            include_roles=frozenset(
                str(item) for item in payload.get("include_roles", ())
            ),
            exclude_roles=frozenset(
                str(item) for item in payload.get("exclude_roles", ())
            ),
            names={
                str(key).lower(): str(value)
                for key, value in dict(payload.get("names", {})).items()
            },
            levels={
                str(key).lower(): str(value)
                for key, value in dict(payload.get("levels", {})).items()
            },
            source=str(source),
        )

    def include(self, procedure: Mapping[str, Any]) -> bool:
        role = str(procedure.get("role", "numeric_kernel"))
        return role in self.include_roles and role not in self.exclude_roles

    def name(self, procedure: Mapping[str, Any], default: str) -> str:
        for key in _rule_keys(procedure):
            if key in self.names:
                return self.names[key]
        return default

    def level(self, procedure: Mapping[str, Any], default: str) -> str:
        for key in _rule_keys(procedure):
            if key in self.levels:
                level = self.levels[key]
                if level not in {"process", "helper"}:
                    raise PICAMConfigurationError(
                        f"invalid physics process level {level!r} for {key}"
                    )
                return level
        return default


@dataclass(frozen=True, slots=True)
class PICAMPhysicsProcess:
    """One flat Python-facing process backed by one original CAM procedure."""

    name: str
    routine: str
    qualified_name: str
    source: str
    role: str
    level: str
    adapter_status: str
    generated_adapter: bool
    blockers: tuple[str, ...]
    call_depth: int
    workflow_actions: tuple[str, ...]
    parent_actions: tuple[str, ...]
    parent_processes: tuple[str, ...]
    aliases: tuple[str, ...]
    arguments: tuple[Mapping[str, Any], ...]
    result: Mapping[str, Any] | None = None

    @property
    def independently_runnable(self) -> bool:
        """Whether this raw source procedure has a complete runtime binding."""

        return self.adapter_status == "validated"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "blockers",
            "workflow_actions",
            "parent_actions",
            "parent_processes",
            "aliases",
            "arguments",
        ):
            payload[key] = list(payload[key])
        payload["result"] = None if self.result is None else dict(self.result)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PICAMPhysicsProcess":
        return cls(
            name=str(payload["name"]),
            routine=str(payload["routine"]),
            qualified_name=str(payload["qualified_name"]),
            source=str(payload["source"]),
            role=str(payload["role"]),
            level=str(payload["level"]),
            adapter_status=str(payload["adapter_status"]),
            generated_adapter=bool(payload.get("generated_adapter", False)),
            blockers=tuple(str(item) for item in payload.get("blockers", ())),
            call_depth=int(payload.get("call_depth", 0)),
            workflow_actions=tuple(
                str(item) for item in payload.get("workflow_actions", ())
            ),
            parent_actions=tuple(
                str(item) for item in payload.get("parent_actions", ())
            ),
            parent_processes=tuple(
                str(item) for item in payload.get("parent_processes", ())
            ),
            aliases=tuple(str(item) for item in payload.get("aliases", ())),
            arguments=tuple(dict(item) for item in payload.get("arguments", ())),
            result=(
                None
                if not isinstance(payload.get("result"), Mapping)
                else dict(payload["result"])
            ),
        )


class PICAMPhysicsCatalog:
    """The deterministic flat catalog for the admitted PI-atm execution path."""

    def __init__(
        self,
        processes: Iterable[PICAMPhysicsProcess],
        *,
        reachable_procedures: int,
        excluded_lifecycle: int,
        source_revision: str,
        rules_sha256: str = "",
    ) -> None:
        self.processes = tuple(sorted(processes, key=lambda item: item.name))
        self.reachable_procedures = int(reachable_procedures)
        self.excluded_lifecycle = int(excluded_lifecycle)
        self.source_revision = str(source_revision)
        self.rules_sha256 = str(rules_sha256)
        names = [item.name for item in self.processes]
        if len(names) != len(set(names)):
            duplicates = sorted(name for name in set(names) if names.count(name) > 1)
            raise PICAMConfigurationError(
                "flat PI-CAM process names are not unique: " + ", ".join(duplicates)
            )

    @property
    def physics_processes(self) -> tuple[PICAMPhysicsProcess, ...]:
        return tuple(item for item in self.processes if item.level == "process")

    @property
    def helpers(self) -> tuple[PICAMPhysicsProcess, ...]:
        return tuple(item for item in self.processes if item.level == "helper")

    @classmethod
    def load_default(cls) -> "PICAMPhysicsCatalog":
        resource = files("freecam.pi_cam.data").joinpath("pi_cam_physics_catalog.json")
        return cls.from_record(json.loads(resource.read_text()))

    @classmethod
    def from_record(cls, payload: Mapping[str, Any]) -> "PICAMPhysicsCatalog":
        if int(payload.get("schema_version", 0)) != PHYSICS_CATALOG_SCHEMA_VERSION:
            raise PICAMConfigurationError("unsupported PI-CAM physics catalog schema")
        return cls(
            (PICAMPhysicsProcess.from_dict(item) for item in payload["processes"]),
            reachable_procedures=int(payload["reachable_procedures"]),
            excluded_lifecycle=int(payload.get("excluded_lifecycle", 0)),
            source_revision=str(payload.get("source_revision", "unknown")),
            rules_sha256=str(payload.get("rules_sha256", "")),
        )

    def machine_record(self) -> dict[str, Any]:
        return {
            "schema_version": PHYSICS_CATALOG_SCHEMA_VERSION,
            "generator": "freecam.pi_cam.physics_catalog",
            "source_revision": self.source_revision,
            "rules_sha256": self.rules_sha256,
            "reachable_procedures": self.reachable_procedures,
            "catalog_entries": len(self.processes),
            "physics_interfaces": len(self.physics_processes),
            "helper_routines": len(self.helpers),
            "excluded_lifecycle": self.excluded_lifecycle,
            "level_counts": _counts(item.level for item in self.processes),
            "adapter_status_counts": _counts(
                item.adapter_status for item in self.processes
            ),
            "processes": [item.as_dict() for item in self.processes],
        }

    def write(self, path: str | Path) -> Path:
        output = Path(path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.machine_record(), indent=2, sort_keys=True) + "\n"
        )
        return output

    def process(self, name: str) -> PICAMPhysicsProcess:
        token = str(name)
        matches = tuple(
            item
            for item in self.processes
            if token == item.name
            or token == item.routine
            or token == item.qualified_name
            or token in item.aliases
        )
        if len(matches) != 1:
            raise KeyError(f"physics process {name!r} is unknown or ambiguous")
        return matches[0]

    def for_workflow_action(self, action: str) -> tuple[PICAMPhysicsProcess, ...]:
        return tuple(
            item for item in self.processes if str(action) in item.workflow_actions
        )


def build_physics_catalog(
    inventory: Mapping[str, Any],
    *,
    generated_adapters: Mapping[str, Any] | None = None,
    rules: PICAMPhysicsRules | None = None,
) -> PICAMPhysicsCatalog:
    """Build the flat catalog from a pinned source inventory and call graph."""

    rules = rules or PICAMPhysicsRules.default()
    procedures = tuple(dict(item) for item in inventory.get("procedures", ()))
    if not procedures:
        raise PICAMConfigurationError("PI-CAM inventory has no procedures")
    by_qualified: dict[str, list[int]] = defaultdict(list)
    for index, procedure in enumerate(procedures):
        by_qualified[str(procedure["qualified_name"]).lower()].append(index)

    roots = {
        index: tuple(str(item) for item in procedure.get("active_plan_actions", ()))
        for index, procedure in enumerate(procedures)
        if procedure.get("active_plan_actions")
    }
    if not roots:
        raise PICAMConfigurationError("PI-CAM inventory has no active-plan roots")
    distance = {index: 0 for index in roots}
    parent_actions: dict[int, set[str]] = {
        index: set(actions) for index, actions in roots.items()
    }
    queue = deque(sorted(roots))
    queued = set(queue)
    while queue:
        index = queue.popleft()
        queued.discard(index)
        procedure = procedures[index]
        for qualified in procedure.get("resolved_calls", ()):
            for target in by_qualified.get(str(qualified).lower(), ()):
                changed = False
                candidate_depth = distance[index] + 1
                if target not in distance or candidate_depth < distance[target]:
                    distance[target] = candidate_depth
                    changed = True
                before = len(parent_actions.get(target, ()))
                parent_actions.setdefault(target, set()).update(parent_actions[index])
                changed = changed or len(parent_actions[target]) != before
                if changed and target not in queued:
                    queue.append(target)
                    queued.add(target)

    generated = _generated_adapter_index(generated_adapters)
    candidates: list[dict[str, Any]] = []
    excluded_lifecycle = 0
    for index in sorted(distance):
        procedure = procedures[index]
        role = str(procedure.get("role", "numeric_kernel"))
        if not rules.include(procedure):
            excluded_lifecycle += 1
            continue
        qualified = str(procedure["qualified_name"])
        routine = str(procedure["name"])
        actions = tuple(sorted(parent_actions[index]))
        direct_actions = tuple(
            str(item) for item in procedure.get("active_plan_actions", ())
        )
        preferred = rules.name(
            procedure,
            _workflow_name(direct_actions) or routine,
        )
        writes_state = any(
            str(argument.get("intent", "")).lower() in {"out", "inout"}
            for argument in procedure.get("arguments", ())
        )
        default_level = (
            "process"
            if role == "process" or direct_actions or writes_state
            else "helper"
        )
        candidates.append(
            {
                "index": index,
                "preferred": _identifier(preferred),
                "routine": routine,
                "qualified_name": qualified,
                "source": str(procedure["source"]),
                "role": role,
                "level": rules.level(procedure, default_level),
                "adapter_status": str(procedure.get("adapter_status", "blocked")),
                "generated_adapter": _has_generated_adapter(
                    qualified, str(procedure["source"]), generated
                ),
                "blockers": tuple(str(item) for item in procedure.get("blockers", ())),
                "call_depth": distance[index],
                "workflow_actions": direct_actions,
                "parent_actions": actions,
                "parent_processes": tuple(
                    sorted({_workflow_name((action,)) for action in actions})
                ),
                "arguments": tuple(dict(item) for item in procedure.get("arguments", ())),
                "result": (
                    None
                    if not isinstance(procedure.get("result"), Mapping)
                    else dict(procedure["result"])
                ),
            }
        )

    preferred_counts = _counts(item["preferred"] for item in candidates)
    assigned: set[str] = set()
    processes: list[PICAMPhysicsProcess] = []
    for candidate in candidates:
        name = str(candidate["preferred"])
        if preferred_counts[name] > 1:
            name = _collision_name(candidate)
        base = name
        suffix = 2
        while name in assigned:
            name = f"{base}_{suffix}"
            suffix += 1
        assigned.add(name)
        aliases = tuple(
            dict.fromkeys(
                token
                for token in (
                    str(candidate["routine"]),
                    str(candidate["qualified_name"]),
                    str(candidate["preferred"]),
                )
                if token != name
            )
        )
        processes.append(
            PICAMPhysicsProcess(
                name=name,
                routine=str(candidate["routine"]),
                qualified_name=str(candidate["qualified_name"]),
                source=str(candidate["source"]),
                role=str(candidate["role"]),
                level=str(candidate["level"]),
                adapter_status=str(candidate["adapter_status"]),
                generated_adapter=bool(candidate["generated_adapter"]),
                blockers=tuple(candidate["blockers"]),
                call_depth=int(candidate["call_depth"]),
                workflow_actions=tuple(candidate["workflow_actions"]),
                parent_actions=tuple(candidate["parent_actions"]),
                parent_processes=tuple(candidate["parent_processes"]),
                aliases=aliases,
                arguments=tuple(candidate["arguments"]),
                result=candidate["result"],
            )
        )
    return PICAMPhysicsCatalog(
        processes,
        reachable_procedures=len(distance),
        excluded_lifecycle=excluded_lifecycle,
        source_revision=str(
            inventory.get("cam_source_revision")
            or inventory.get("source_revision")
            or "unknown"
        ),
        rules_sha256=(
            sha256(Path(rules.source).read_bytes()).hexdigest()
            if rules.source != "built-in" and Path(rules.source).is_file()
            else ""
        ),
    )


def merge_runtime_process_records(
    runtime_records: Sequence[Mapping[str, Any]],
    catalog: PICAMPhysicsCatalog,
) -> tuple[dict[str, Any], ...]:
    """Overlay runnable workflow boundaries on the flat source catalog."""

    merged: list[dict[str, Any]] = []
    consumed: set[tuple[str, str]] = set()
    for source_record in runtime_records:
        record = dict(source_record)
        action = f"{record['phase']}.{record['name']}"
        descriptors = catalog.for_workflow_action(action)
        consumed.update((item.qualified_name, item.source) for item in descriptors)
        aliases = tuple(
            dict.fromkeys(
                str(item)
                for item in (
                    record.get("operation"),
                    *(descriptor.routine for descriptor in descriptors),
                    *(alias for descriptor in descriptors for alias in descriptor.aliases),
                )
                if item
            )
        )
        record.update(
            {
                "api_name": str(record["name"]),
                "aliases": aliases,
                "runnable": True,
                "capability": "runtime",
                "runtime_template": False,
                "requires_binding": False,
                "level": "process",
                "source_procedures": tuple(
                    item.qualified_name for item in descriptors
                ),
                "sources": tuple(item.source for item in descriptors),
                "blockers": (),
                "parent_actions": (action,),
                "parent_processes": (str(record["name"]),),
            }
        )
        merged.append(record)

    for process in catalog.physics_processes:
        if (process.qualified_name, process.source) in consumed:
            continue
        phases = tuple(
            dict.fromkeys(action.split(".", 1)[0] for action in process.parent_actions)
        )
        merged.append(
            {
                "phase": phases[0] if len(phases) == 1 else "+".join(phases),
                "name": process.name,
                "api_name": process.name,
                "operation": process.routine,
                "kind": "catalog_process",
                "granularity": process.level,
                "level": process.level,
                "native_id": None,
                "enabled": None,
                "runnable": False,
                "capability": (
                    "adapter_generated"
                    if process.generated_adapter
                    else process.adapter_status
                ),
                "runtime_template": process.generated_adapter,
                "requires_binding": True,
                "adapter_status": process.adapter_status,
                "generated_adapter": process.generated_adapter,
                "qualified_name": process.qualified_name,
                "source": process.source,
                "source_procedures": (process.qualified_name,),
                "sources": (process.source,),
                "aliases": process.aliases,
                "blockers": process.blockers,
                "call_depth": process.call_depth,
                "workflow_actions": process.workflow_actions,
                "parent_actions": process.parent_actions,
                "parent_processes": process.parent_processes,
                "arguments": process.arguments,
            }
        )
    return tuple(merged)


def _generated_adapter_index(
    payload: Mapping[str, Any] | None,
) -> frozenset[tuple[str, str]]:
    if payload is None:
        return frozenset()
    generated = payload.get("generated")
    if isinstance(generated, list):
        return frozenset(
            (
                str(item.get("name", "")).lower(),
                str(item.get("original_source", "")),
            )
            for item in generated
            if isinstance(item, Mapping)
        )
    active = payload.get("active_plan", {})
    records = active.get("reachable_records", ()) if isinstance(active, Mapping) else ()
    return frozenset(
        (str(item.get("name", "")).lower(), str(item.get("source", "")))
        for item in records
        if bool(item.get("generated_adapter", False))
    )


def _rule_keys(procedure: Mapping[str, Any]) -> tuple[str, ...]:
    qualified = str(procedure.get("qualified_name", "")).lower()
    source = str(procedure.get("source", ""))
    routine = str(procedure.get("name", "")).lower()
    return (f"{qualified}@{source}".lower(), qualified, routine)


def _has_generated_adapter(
    qualified_name: str,
    source: str,
    generated: frozenset[tuple[str, str]],
) -> bool:
    return (qualified_name.lower(), source) in generated


def _workflow_name(actions: Sequence[str]) -> str:
    names = tuple(str(action).split(".", 1)[-1] for action in actions)
    return names[0] if names else ""


def _collision_name(candidate: Mapping[str, Any]) -> str:
    source = Path(str(candidate["source"]))
    parts = list(source.parts)
    prefix = ""
    try:
        physics = parts.index("physics")
        if physics + 1 < len(parts) - 1:
            prefix = parts[physics + 1]
    except ValueError:
        pass
    if not prefix:
        prefix = str(candidate["qualified_name"]).split("::", 1)[0]
    return _identifier(f"{prefix}_{candidate['preferred']}")


def _identifier(value: str) -> str:
    result = re.sub(r"[^0-9a-zA-Z_]", "_", str(value).strip().lower())
    result = re.sub(r"_+", "_", result).strip("_")
    if not result:
        raise PICAMConfigurationError(f"cannot create a Python process name from {value!r}")
    if result[0].isdigit():
        result = "physics_" + result
    return result


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
