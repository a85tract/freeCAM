"""Generic, editable execution plans compiled from CCPP suite XML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import xml.etree.ElementTree as ET

from .errors import MissingKernelError


@dataclass(slots=True)
class SuiteScheme:
    """One source-qualified occurrence of a scheme in a CCPP suite."""

    name: str
    source_group: str
    occurrence: int
    enabled: bool = True

    @property
    def key(self) -> str:
        return f"{self.source_group}.{self.name}@{self.occurrence}"


@dataclass(slots=True)
class SuiteNode:
    """A group, subcycle, or scheme node in the preserved XML control tree."""

    kind: str
    name: str
    children: list["SuiteNode"] = field(default_factory=list)
    scheme: SuiteScheme | None = None

    def clone(self) -> "SuiteNode":
        return SuiteNode(
            kind=self.kind,
            name=self.name,
            children=[child.clone() for child in self.children],
            scheme=(
                None
                if self.scheme is None
                else SuiteScheme(
                    self.scheme.name,
                    self.scheme.source_group,
                    self.scheme.occurrence,
                    self.scheme.enabled,
                )
            ),
        )


class CCPPSuitePlan:
    """An XML-derived suite tree with explicit loop and scheme boundaries."""

    def __init__(
        self,
        name: str,
        groups: Mapping[str, SuiteNode],
        *,
        source: str | Path | None = None,
        sequence_safe: bool = True,
    ) -> None:
        self.name = str(name)
        self.source = None if source is None else Path(source).resolve()
        self._groups = {
            group: node.clone() for group, node in groups.items()
        }
        self._sequence_safe = bool(sequence_safe)
        self._reindex()

    @classmethod
    def from_xml(cls, path: str | Path) -> "CCPPSuitePlan":
        source = Path(path).resolve()
        root = ET.parse(source).getroot()
        if root.tag.lower() != "suite":
            raise ValueError(f"{source}: root element must be <suite>")
        suite_name = root.attrib.get("name", source.stem)
        occurrence = 0

        def convert(element: ET.Element, group: str) -> SuiteNode:
            nonlocal occurrence
            tag = element.tag.lower()
            if tag == "scheme":
                scheme_name = (element.text or "").strip().lower()
                if not scheme_name:
                    raise ValueError(f"{source}: empty <scheme> element")
                occurrence += 1
                scheme = SuiteScheme(scheme_name, group, occurrence)
                return SuiteNode("scheme", scheme_name, scheme=scheme)
            if tag == "subcycle":
                loop = element.attrib.get("loop", "").strip().lower()
                if not loop:
                    raise ValueError(f"{source}: <subcycle> requires loop=")
                node = SuiteNode("subcycle", loop)
            elif tag == "group":
                node = SuiteNode("group", group)
            else:
                raise ValueError(
                    f"{source}: unsupported suite control element <{tag}>"
                )
            node.children = [convert(child, group) for child in element]
            return node

        groups: dict[str, SuiteNode] = {}
        for element in root:
            if element.tag.lower() != "group":
                raise ValueError(
                    f"{source}: suite children must be <group>, got "
                    f"<{element.tag}>"
                )
            group = element.attrib.get("name", "").strip().lower()
            if not group:
                raise ValueError(f"{source}: <group> requires name=")
            if group in groups:
                raise ValueError(f"{source}: duplicate group {group!r}")
            groups[group] = convert(element, group)
        return cls(suite_name, groups, source=source)

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(self._groups)

    @property
    def sequence_safe(self) -> bool:
        return self._sequence_safe

    @property
    def schemes(self) -> tuple[SuiteScheme, ...]:
        return tuple(self._schemes.values())

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._schemes)

    def copy(self) -> "CCPPSuitePlan":
        return CCPPSuitePlan(
            self.name,
            self._groups,
            source=self.source,
            sequence_safe=self.sequence_safe,
        )

    def scheme(
        self, selector: str, *, group: str | None = None
    ) -> SuiteScheme:
        if selector in self._schemes:
            scheme = self._schemes[selector]
            if group is not None and self.execution_group(scheme.key) != group:
                raise ValueError(
                    f"{selector!r} is not currently in group {group!r}"
                )
            return scheme
        matches = [
            scheme
            for scheme in self._schemes.values()
            if scheme.name == selector
            and (
                group is None
                or self.execution_group(scheme.key) == group
            )
        ]
        if not matches and "." in selector:
            source_group, _, name = selector.partition(".")
            matches = [
                scheme
                for scheme in self._schemes.values()
                if scheme.source_group == source_group
                and scheme.name == name
                and (
                    group is None
                    or self.execution_group(scheme.key) == group
                )
            ]
        if not matches:
            raise ValueError(
                f"unknown scheme {selector!r} in suite {self.name!r}"
            )
        if len(matches) != 1:
            raise ValueError(
                f"scheme {selector!r} is ambiguous; use one of "
                f"{tuple(item.key for item in matches)}"
            )
        return matches[0]

    def execution_group(self, key: str) -> str:
        try:
            return self._locations[key][0]
        except KeyError as exc:
            raise ValueError(f"unknown suite scheme key {key!r}") from exc

    def enable(self, selector: str, *, group: str | None = None) -> None:
        self.scheme(selector, group=group).enabled = True
        self._refresh_safety()

    def disable(
        self,
        selector: str,
        *,
        group: str | None = None,
        unsafe: bool = False,
    ) -> None:
        if not unsafe:
            raise ValueError(
                "disabling a pinned-suite scheme requires unsafe=True"
            )
        self.scheme(selector, group=group).enabled = False
        self._sequence_safe = False

    def move(
        self,
        selector: str,
        *,
        before: str | None = None,
        after: str | None = None,
        group: str | None = None,
        to_group: str | None = None,
        unsafe: bool = False,
    ) -> None:
        if not unsafe:
            raise ValueError(
                "changing a pinned-suite order requires unsafe=True"
            )
        if before is not None and after is not None:
            raise ValueError("provide at most one of before= or after=")
        if before is None and after is None and to_group is None:
            raise ValueError("provide before=, after=, or to_group=")
        moving = self.scheme(selector, group=group)
        _moving_group, parent, node = self._locations[moving.key]
        anchor_selector = before if before is not None else after
        if anchor_selector is None:
            assert to_group is not None
            if to_group not in self._groups:
                raise ValueError(f"unknown suite group {to_group!r}")
            destination = self._groups[to_group]
            insert_at = len(destination.children)
        else:
            anchor = self.scheme(anchor_selector, group=to_group)
            anchor_group, destination, anchor_node = self._locations[
                anchor.key
            ]
            if anchor.key == moving.key:
                raise ValueError("a scheme cannot move relative to itself")
            if to_group is not None and anchor_group != to_group:
                raise ValueError(
                    f"anchor is not in requested group {to_group!r}"
                )
            insert_at = destination.children.index(anchor_node)
            if after is not None:
                insert_at += 1
        old_index = parent.children.index(node)
        parent.children.pop(old_index)
        if destination is parent and old_index < insert_at:
            insert_at -= 1
        destination.children.insert(insert_at, node)
        self._sequence_safe = False
        self._reindex()

    def expanded(
        self,
        group: str,
        dimensions: Mapping[str, int] | Callable[[str], int],
    ) -> tuple[SuiteScheme, ...]:
        """Expand subcycles and return the exact run sequence for one group."""

        if group not in self._groups:
            raise ValueError(f"unknown suite group {group!r}")

        def dimension(name: str) -> int:
            value = (
                dimensions(name)
                if callable(dimensions)
                else dimensions[name]
            )
            count = int(value)
            if count < 0:
                raise ValueError(
                    f"subcycle {name!r} has negative count {count}"
                )
            return count

        def walk(node: SuiteNode) -> Iterable[SuiteScheme]:
            if node.kind == "scheme":
                assert node.scheme is not None
                if node.scheme.enabled:
                    yield node.scheme
                return
            repeat = dimension(node.name) if node.kind == "subcycle" else 1
            for _ in range(repeat):
                for child in node.children:
                    yield from walk(child)

        return tuple(walk(self._groups[group]))

    def describe(self, group: str | None = None) -> list[dict[str, Any]]:
        if group is not None and group not in self._groups:
            raise ValueError(f"unknown suite group {group!r}")
        rows: list[dict[str, Any]] = []

        def walk(
            node: SuiteNode,
            execution_group: str,
            controls: tuple[str, ...],
        ) -> None:
            if node.kind == "scheme":
                assert node.scheme is not None
                rows.append(
                    {
                        "key": node.scheme.key,
                        "name": node.scheme.name,
                        "source_group": node.scheme.source_group,
                        "execution_group": execution_group,
                        "enabled": node.scheme.enabled,
                        "controls": controls,
                    }
                )
                return
            next_controls = controls
            if node.kind == "subcycle":
                next_controls = (*controls, f"subcycle:{node.name}")
            for child in node.children:
                walk(child, execution_group, next_controls)

        selected = self.group_names if group is None else (group,)
        for group_name in selected:
            walk(self._groups[group_name], group_name, ())
        return rows

    def _reindex(self) -> None:
        schemes: dict[str, SuiteScheme] = {}
        locations: dict[str, tuple[str, SuiteNode, SuiteNode]] = {}

        def walk(group: str, parent: SuiteNode) -> None:
            for child in parent.children:
                if child.kind == "scheme":
                    assert child.scheme is not None
                    if child.scheme.key in schemes:
                        raise ValueError(
                            f"duplicate scheme identity {child.scheme.key!r}"
                        )
                    schemes[child.scheme.key] = child.scheme
                    locations[child.scheme.key] = (group, parent, child)
                else:
                    walk(group, child)

        for group, root in self._groups.items():
            walk(group, root)
        self._schemes = schemes
        self._locations = locations

    def _refresh_safety(self) -> None:
        if not all(item.enabled for item in self._schemes.values()):
            self._sequence_safe = False


class CCPPDeviceHost:
    """Run any XML suite by routing standard-name StatePool data to devices."""

    def __init__(
        self,
        pool: Any,
        registry: Any,
        plan: CCPPSuitePlan,
        *,
        strict: bool = True,
        host_services: Any | None = None,
    ) -> None:
        self.pool = pool
        self.registry = registry
        self.plan = plan.copy()
        self.strict = bool(strict)
        self.host_services = host_services
        self.last_scheme: str | None = None
        self.last_lifecycle: str | None = None

    def _invoke(self, process: str) -> bool:
        before = self.pool.pointer_records()
        if process in self.registry.process_names:
            self.registry.invoke(process, self.pool)
        elif (
            self.host_services is not None
            and process in self.host_services.process_names
        ):
            self.host_services.invoke(process, self.pool)
        else:
            if self.strict:
                available = set(self.registry.process_names)
                if self.host_services is not None:
                    available.update(self.host_services.process_names)
                raise MissingKernelError(
                    f"suite {self.plan.name!r} requires process {process!r}, "
                    "but no built device or Python host service provides it; "
                    f"available process count={len(available)}"
                )
            return False
        self.pool.assert_pointer_stability(before)
        return True

    def run_scheme(self, selector: str, *, group: str | None = None) -> bool:
        scheme = self.plan.scheme(selector, group=group)
        invoked = self._invoke(scheme.name)
        if invoked:
            self.last_scheme = scheme.key
        return invoked

    def run_group(
        self,
        group: str,
        *,
        callback: Callable[[str, "CCPPDeviceHost"], None] | None = None,
    ) -> tuple[str, ...]:
        executed: list[str] = []
        for scheme in self.plan.expanded(group, self.pool.dimensions):
            if self._invoke(scheme.name):
                self.last_scheme = scheme.key
                executed.append(scheme.key)
                if callback is not None:
                    callback(scheme.key, self)
        return tuple(executed)

    def run_lifecycle(self, phase: str) -> tuple[str, ...]:
        """Run an explicit initialize/timestep/finalize lifecycle boundary."""

        valid = {
            "register",
            "initialize",
            "timestep_initial",
            "timestep_final",
            "finalize",
        }
        if phase not in valid:
            raise ValueError(
                f"unknown CCPP lifecycle {phase!r}; choose from {sorted(valid)}"
            )
        executed: list[str] = []
        seen: set[str] = set()
        for scheme in self.plan.schemes:
            process = f"{scheme.name}:{phase}"
            if process in seen:
                continue
            seen.add(process)
            if self._invoke(process):
                executed.append(process)
        self.last_lifecycle = phase
        return tuple(executed)
