#!/usr/bin/env python3
"""Make a CAM driver pausable: the original Fortran, hoisted verbatim, cut at its kernel calls.

A *pausable runner* is what the stage-7 runner is by hand: a process's Fortran
run continuously by a state machine that returns to Python only where a
replaced kernel would have been called.  This module generates one from a
spec, so the eleventh process costs a spec and not a transcription:

* every routine the spec names (the tphysbc glue of the action, the driver
  the glue calls, a routine the driver calls) becomes one Fortran module --
  the routine's locals hoisted to module state, its body copied statement
  for statement into *pieces* (subroutines holding balanced ranges of the
  pinned text), its dummies as module variables the caller binds;
* the control flow the pieces are cut at -- the `if`, `do` and `select`
  statements that enclose a pause -- is the *skeleton*, re-expressed by the
  runner's state machine exactly as the source has it;
* each *pause* is one `call` of the source: the runner stops before it with
  a *frame* describing the call's arguments where they live (the actuals of
  the pinned call, in the callee's argument order, with the callee's intents),
  and continues past it on resume; `original` runs the very call statement
  on the paused frame for the validation gate.

Nothing numerical is written here.  Every arithmetic statement is the pinned
source's, every routine called is the original, and the ranges are pinned by
hash so a source that moves fails --check rather than being transcribed from
the wrong place.

Spec (YAML, one per process under native/pi_cam/pausable/):

    schema_version: 1
    prefix: dadadj                     # modules pycam_dadadj_*, entries pycam_dadadj_*_v1
    stage: cam_run1.dry_adjustment
    refuse:                            # Fortran conditions refused at create, with the message
      - {when: "trim(shallow_scheme) /= 'UW'", message: "written for the UW shallow scheme"}
    getopts: [shallow_scheme]          # phys_getopts(<name>_out=<name>) at create
    units:
      glue:                            # the action's tphysbc block; always named glue
        source: components/cam/src/physics/cam/physpkg.F90
        routine: tphysbc
        declarations: [1674, 1874]
        module_header: [1, 60]         # the module's own use statements
        dummies: {state: "host_state(lchnk)", tend: "host_tend(lchnk)", ...}
        preamble: ["lchnk = state%lchnk", "ncol = state%ncol", ...]
        carries: {dlf: 4, rliq: 6}     # tphysbc locals that are pycesm carries: name -> 0039 code
        pbuf_indices: {prec_sh_idx: PREC_SH}   # module-private indices the block reads
        body: [...]                    # the tree, see below
      driver:
        source: components/cam/src/physics/cam/convect_shallow.F90
        routine: convect_shallow_tend
        declarations: [416, 575]
        module_header: [1, 80]
        pbuf_indices: auto             # every <x>_idx the module registers or looks up
        body: [...]
    kernels:
      compute_uwshcu_inv:
        source: components/cam/src/physics/cam/uwshcu.F90
        routine: compute_uwshcu_inv

Body tree nodes:
    - piece: [first, last]                       verbatim range
    - pause: {kernel: NAME, call: [first, last]} the call statement; pause before it
    - unit: {name: driver, call: [first, last]}  the call statement that enters another unit
    - if: LINE  then: [...]  else: [...]         the `if (...) then` at LINE (else optional)
    - do: LINE  body: [...]                      the `do` header at LINE
    - select: LINE cases: {"'UW'": [...]} refused: ["'Hack'", ...]
                                                 the `select case` at LINE; other cases refused
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REPO = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO / "external/iCESM1.3.1_fzhu"
SPECS = REPO / "native/pi_cam/pausable"
SUPPORT = REPO / "native/pi_cam/support"
FRAMES = REPO / "native/pi_cam/segment_frames.yaml"

INTENT_CODE = {"in": 0, "out": 1, "inout": 2}
DTYPE_CODE = {"real": 1, "integer": 2, "logical": 2}
KEYWORDS = set("""if then else elseif endif end do enddo select case default call return continue
    cycle exit where elsewhere forall allocate deallocate nullify associated allocated present size
    shape lbound ubound min max sum abs sqrt exp log any all count merge trim len len_trim adjustl
    real int nint dble mod modulo sign huge tiny epsilon maxval minval matmul transpose reshape
    true false and or not eqv neqv gt ge lt le eq ne write read print iulog r8 in out inout intent
    optional pointer target allocatable dimension parameter save private public use only implicit
    none subroutine function module contains type integer logical character""".split())


# ---------------------------------------------------------------------------
# reading the pinned source
# ---------------------------------------------------------------------------


def source_path(relative: str) -> Path:
    path = SOURCE_ROOT / relative
    if not path.is_file():
        raise SystemExit(f"pinned source not found: {path}")
    return path


def read_lines(relative: str) -> list[str]:
    return source_path(relative).read_text().splitlines()


def range_digest(lines: list[str], first: int, last: int) -> str:
    return hashlib.sha256("\n".join(lines[first - 1:last]).encode()).hexdigest()[:16]


def strip_comment(text: str) -> str:
    """The code of a line: everything before a `!` that is not inside quotes."""

    out, quote = [], None
    for character in text:
        if quote:
            out.append(character)
            if character == quote:
                quote = None
        elif character in ("'", '"'):
            quote = character
            out.append(character)
        elif character == "!":
            break
        else:
            out.append(character)
    return "".join(out).rstrip()


def statements(lines: list[str], first: int, last: int) -> list[tuple[int, int, str]]:
    """(first line, last line, text) of every statement in the range, continuations joined."""

    out, buffer, start = [], "", None
    for number in range(first, last + 1):
        text = strip_comment(lines[number - 1])
        if not text.strip():
            continue
        if start is None:
            start = number
        piece = text.strip()
        if piece.startswith("&"):
            piece = piece[1:].strip()
        text = (buffer + " " + piece) if buffer else piece
        if text.rstrip().endswith("&"):
            buffer = text.rstrip()[:-1].rstrip()
            continue
        out.append((start, number, re.sub(r"\s+", " ", text.strip())))
        buffer, start = "", None
    if buffer:
        raise SystemExit(f"unterminated continuation ending at line {last}")
    return out


def is_preprocessor(line: str) -> bool:
    return line.lstrip().startswith("#")


# ---------------------------------------------------------------------------
# declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decl:
    name: str
    kind: str                 # real(r8) | integer | logical | character(len=16) | type(physics_state)
    attributes: str           # "pointer, dimension(:,:)" etc., intent removed
    dims: str                 # "pcols,pver" or ""
    init: str                 # "= 0._r8" or "=> null()" or ""
    intent: str | None        # in | out | inout | None

    @property
    def rank(self) -> int:
        return self.dims.count(",") + 1 if self.dims else 0

    @property
    def is_pointer(self) -> bool:
        return "pointer" in self.attributes

    @property
    def is_allocatable(self) -> bool:
        return "allocatable" in self.attributes

    @property
    def is_parameter(self) -> bool:
        return "parameter" in self.attributes

    @property
    def is_derived(self) -> bool:
        return self.kind.lower().startswith("type(")

    @property
    def base_type(self) -> str:
        return self.kind.split("(")[0].lower()


_DECL = re.compile(
    r"^(real(?:\s*\([^)]*\))?|integer(?:\s*\([^)]*\))?|logical|character\s*\([^)]*\)|type\s*\(\s*\w+\s*\))"
    r"(.*?)::\s*(.*)$", re.I)
#: the older form without `::` -- `integer lchnk`, `real(r8) zero(pcols)` -- one or more names
_DECL_OLD = re.compile(
    r"^(real(?:\s*\([^)]*\))?|integer(?:\s*\([^)]*\))?|logical|character\s*\([^)]*\)|type\s*\(\s*\w+\s*\))"
    r"\s+([A-Za-z_][\w ,()*:+-]*)$", re.I)


def _split_top(text: str) -> list[str]:
    out, depth, current = [], 0, []
    for character in text:
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        if character == "," and depth == 0:
            out.append("".join(current).strip()); current = []
        else:
            current.append(character)
    if "".join(current).strip():
        out.append("".join(current).strip())
    return out


def _procedure_dummies(lines: list[str], first: int, last: int) -> set[str]:
    """Names declared as procedures in the range: the specifics of an `interface` block, or a
    bare `optional ::` / `external` naming something no type declaration covers."""

    found: set[str] = set()
    depth = 0
    for _, _, text in statements(lines, first, last):
        low = text.strip().lower()
        if re.match(r"^interface\b", low):
            depth += 1
            continue
        if re.match(r"^end\s*interface\b", low):
            depth = max(depth - 1, 0)
            continue
        if depth:
            match = re.match(r"^(?:[\w() ,=]*?\s)?(?:subroutine|function)\s+(\w+)\s*\(", low)
            if match:
                found.add(match.group(1))
        match = re.match(r"^(?:optional|external)\s*::\s*(.+)$", low)
        if match:
            found |= {n.strip() for n in match.group(1).split(",")}
    return found


def parse_declarations(lines: list[str], first: int, last: int) -> dict[str, Decl]:
    """name -> Decl for every entity declared in the range (dummies included, with their intent)."""

    out: dict[str, Decl] = {}
    for _, _, text in statements(lines, first, last):
        match = _DECL.match(text)
        if match:
            kind, attributes, names = match.groups()
        else:
            old_form = _DECL_OLD.match(text)
            if not old_form or re.match(r"^\w[\w()]*\s+(function|subroutine|parameter)\b", text, re.I):
                continue
            kind, names = old_form.groups()
            attributes = ""
        kind = re.sub(r"\s+", "", kind)
        intent = None
        intent_match = re.search(r"intent\s*\(\s*(in|out|inout)\s*\)", attributes, re.I)
        if intent_match:
            intent = intent_match.group(1).lower()
        attributes = re.sub(r"intent\s*\(\s*\w+\s*\)", "", attributes, flags=re.I)
        dimension = re.search(r"dimension\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", attributes, re.I)
        common_dims = dimension.group(1) if dimension else ""
        attributes = re.sub(r"dimension\s*\([^()]*(?:\([^()]*\)[^()]*)*\)", "", attributes, flags=re.I)
        attributes = re.sub(r"\s+", " ", attributes.strip(" ,")).replace(" ,", ",")
        attributes = ", ".join(a.strip() for a in attributes.split(",") if a.strip())
        for item in _split_top(names):
            match = re.match(r"(\w+)\s*(\(([^()]*(?:\([^()]*\)[^()]*)*)\))?\s*(=>?\s*.+)?$", item)
            if not match:
                continue
            name, _, dims, init = match.groups()
            dims = (dims or common_dims or "").replace(" ", "")
            dims = dims.replace("state%psetcols", "pcols").replace("psetcols", "pcols")
            out[name.lower()] = Decl(name.lower(), kind, attributes.lower(), dims, (init or "").strip(), intent)
    return out


def signature(lines: list[str], first: int, last: int, routine: str) -> list[str]:
    """The routine's dummy names in order, from its `subroutine` statement."""

    for _, _, text in statements(lines, first, last):
        match = re.match(rf"^\s*(?:recursive\s+)?subroutine\s+{routine}\s*\((.*)\)\s*$", text, re.I)
        if match:
            return [a.strip().lower() for a in _split_top(match.group(1)) if a.strip()]
    raise SystemExit(f"no `subroutine {routine}(...)` in lines {first}-{last}")


def use_statements(lines: list[str], ranges: Iterable[tuple[int, int]]) -> list[str]:
    """The `use` statements of the given ranges, one line each, in order, deduplicated."""

    seen: set[str] = set()
    out: list[str] = []
    for first, last in ranges:
        for _, _, text in statements(lines, first, last):
            if re.match(r"^\s*use\s+\w+", text, re.I):
                key = re.sub(r"\s+", " ", text.lower())
                if key not in seen:
                    seen.add(key)
                    out.append(text)
    return out


def identifiers(text: str) -> set[str]:
    code = strip_comment(text)
    code = re.sub(r"'[^']*'|\"[^\"]*\"", " ", code)
    return {w.lower() for w in re.findall(r"\b([A-Za-z_]\w*)\b", code)
            if not re.fullmatch(r"\d+", w)}


def buffer_indices(lines: list[str], first: int, last: int) -> list[tuple[str, str]]:
    """(index variable, field name) for every pbuf index the range registers or looks up."""

    found: dict[str, str] = {}
    for _, _, text in statements(lines, first, last):
        match = re.match(r"call pbuf_add_field\(\s*'([^']+)'.*,\s*(\w+_idx)\s*\)$", text, re.I)
        if match:
            found.setdefault(match.group(2).lower(), match.group(1))
            continue
        match = re.match(r"(\w+_idx)\s*=\s*pbuf_get_index\(\s*'([^']+)'", text, re.I)
        if match:
            found.setdefault(match.group(1).lower(), match.group(2))
    return sorted(found.items())


# ---------------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------------


@dataclass
class Pause:
    kernel: str
    first: int
    last: int
    unit: "Unit" = None            # type: ignore[assignment]
    statement: str = ""
    pc_at: str = ""
    pc_after: str = ""
    site: int = 0                  # 0 for a kernel's only site, else its number in source order

    @property
    def tag(self) -> str:
        """The kernel's name, suffixed by the site when the kernel pauses at several."""

        return self.kernel if not self.site else f"{self.kernel}_{self.site}"


@dataclass
class UnitCall:
    unit: str
    first: int
    last: int
    statement: str = ""


@dataclass
class Node:
    kind: str                       # piece | pause | unit | if | do | select
    first: int = 0
    last: int = 0
    line: int = 0
    children: list = field(default_factory=list)          # if.then / do.body
    orelse: list = field(default_factory=list)            # if.else
    elifs: list = field(default_factory=list)             # if.elif: (line, body) pairs
    flows: set = field(default_factory=set)               # piece: the flow codes its statements set
    cases: dict = field(default_factory=dict)             # select
    refused: list = field(default_factory=list)           # select
    pause: Pause | None = None
    call: UnitCall | None = None
    name: str = ""                                        # piece name


@dataclass
class Unit:
    key: str
    prefix: str
    source: str
    routine: str
    declarations: tuple[int, int]
    module_header: tuple[int, int] | None
    body: list[Node]
    dummies: dict[str, str]
    preamble: list[str]
    carries: dict[str, int]
    pbuf_indices: Any
    uses: list[str] = field(default_factory=list)
    records: dict[str, str] = field(default_factory=dict)      # local -> accessor entry handing its address
    getopts: list[str] = field(default_factory=list)           # module-private options read by phys_getopts
    locals: dict[str, str] = field(default_factory=dict)       # local -> storage it is a pointer to (glue)
    module_private: list[tuple[int, int]] = field(default_factory=list)   # the module's own declarations, verbatim
    helpers: list[tuple[int, int]] = field(default_factory=list)          # the module's private procedures, verbatim
    postamble: list[str] = field(default_factory=list)                    # statements after the body, every chunk (glue)
    automatic: list[str] = field(default_factory=list)                    # locals with run-time extents, allocated once
    carries_entry: str = "pycam_macro_forcing_v1"                         # the accessor handing a carry's chunk address
    elided: list[tuple[int, int]] = field(default_factory=list)           # body lines another action's leaf performs
    lines: list[str] = field(default_factory=list)
    decls: dict[str, Decl] = field(default_factory=dict)
    dummy_names: list[str] = field(default_factory=list)
    body_range: tuple[int, int] = (0, 0)
    pieces: list[Node] = field(default_factory=list)
    pauses: list[Pause] = field(default_factory=list)
    unit_calls: list[UnitCall] = field(default_factory=list)

    @property
    def module(self) -> str:
        return f"pycam_{self.prefix}_{self.key}"


@dataclass
class Kernel:
    name: str
    source: str
    routine: str
    lines: list[str] = field(default_factory=list)
    dummies: list[str] = field(default_factory=list)
    decls: dict[str, Decl] = field(default_factory=dict)
    body_range: tuple[int, int] = (0, 0)


@dataclass
class Spec:
    path: Path
    prefix: str
    stage: str
    refuse: list[dict]
    getopts: list[str]
    units: dict[str, Unit]
    kernels: dict[str, Kernel]
    hosts: str = "pycam_stage_hosts"
    runner_uses: list[str] = field(default_factory=list)      # modules the runner's skeleton and refusals need

    @property
    def runner_module(self) -> str:
        return f"pycam_{self.prefix}_runner"

    @property
    def entry_prefix(self) -> str:
        return f"pycam_{self.prefix}"


def _parse_body(items: list, unit: Unit) -> list[Node]:
    nodes: list[Node] = []
    for item in items:
        if "piece" in item:
            first, last = item["piece"]
            nodes.append(Node("piece", first=first, last=last))
        elif "pause" in item:
            pause = item["pause"]
            first, last = pause["call"]
            p = Pause(kernel=str(pause["kernel"]), first=first, last=last, unit=unit)
            unit.pauses.append(p)
            nodes.append(Node("pause", first=first, last=last, pause=p))
        elif "unit" in item:
            call = item["unit"]
            first, last = call["call"]
            c = UnitCall(unit=str(call["name"]), first=first, last=last)
            unit.unit_calls.append(c)
            nodes.append(Node("unit", first=first, last=last, call=c))
        elif "if" in item:
            nodes.append(Node("if", line=int(item["if"]),
                              children=_parse_body(item.get("then", []), unit),
                              orelse=_parse_body(item.get("else", []), unit),
                              elifs=[(int(e["line"]), _parse_body(e.get("then", []), unit)) for e in item.get("elif") or []]))
        elif "do" in item:
            nodes.append(Node("do", line=int(item["do"]), children=_parse_body(item.get("body", []), unit)))
        elif "select" in item:
            cases = {str(k): _parse_body(v, unit) for k, v in (item.get("cases") or {}).items()}
            nodes.append(Node("select", line=int(item["select"]), cases=cases,
                              refused=[str(x) for x in item.get("refused") or []]))
        else:
            raise SystemExit(f"{unit.key}: unknown body node {item}")
    return nodes


GETOPT_TYPES = {"character": "character(len=16)", "logical": "logical", "integer": "integer"}
#: option -> Fortran type of the module-private options phys_getopts reports
GETOPTS: dict[str, str] = {}


def _getopts(record) -> list[str]:
    """`getopts:` as a list (character options) or a mapping option -> character | logical | integer."""

    if not record:
        return []
    if isinstance(record, Mapping):
        for name, kind in record.items():
            if str(kind) not in GETOPT_TYPES:
                raise SystemExit(f"getopts: {name} has type {kind!r}; one of {sorted(GETOPT_TYPES)}")
            GETOPTS[str(name)] = GETOPT_TYPES[str(kind)]
        return [str(name) for name in record]
    for name in record:
        GETOPTS.setdefault(str(name), GETOPT_TYPES["character"])
    return [str(x) for x in record]


def load_spec(path: Path) -> Spec:
    payload = yaml.safe_load(path.read_text())
    if int(payload.get("schema_version", 0)) != 1:
        raise SystemExit(f"{path}: schema_version must be 1")
    prefix = str(payload["prefix"])
    units: dict[str, Unit] = {}
    for key, record in payload["units"].items():
        unit = Unit(
            key=str(key), prefix=prefix, source=str(record["source"]), routine=str(record["routine"]),
            declarations=tuple(record["declarations"]),
            module_header=tuple(record["module_header"]) if record.get("module_header") else None,
            body=[], dummies={k: str(v) for k, v in (record.get("dummies") or {}).items()},
            preamble=[str(x) for x in record.get("preamble") or []],
            carries={str(k): int(v) for k, v in (record.get("carries") or {}).items()},
            pbuf_indices=record.get("pbuf_indices"),
            uses=[str(x) for x in record.get("uses") or []],
            records={str(k): str(v) for k, v in (record.get("records") or {}).items()},
            getopts=_getopts(record.get("getopts")),
            locals={str(k): str(v) for k, v in (record.get("locals") or {}).items()},
            module_private=[(int(a), int(b)) for a, b in record.get("module_private") or []],
            helpers=[(int(a), int(b)) for a, b in record.get("helpers") or []],
            postamble=[str(x) for x in record.get("postamble") or []],
            automatic=[str(x).lower() for x in record.get("automatic") or []],
            carries_entry=str(record.get("carries_entry") or "pycam_macro_forcing_v1"),
            elided=[(int(a), int(b)) for a, b in record.get("elided") or []],
        )
        unit.body = _parse_body(record["body"], unit)
        units[str(key)] = unit
    if "glue" not in units:
        raise SystemExit(f"{path}: a spec needs a unit named glue (the action's tphysbc/tphysac block)")
    kernels = {str(name): Kernel(name=str(name), source=str(k["source"]), routine=str(k["routine"]))
               for name, k in payload["kernels"].items()}
    spec = Spec(path=path, prefix=prefix, stage=str(payload["stage"]),
                refuse=list(payload.get("refuse") or []), getopts=[str(x) for x in payload.get("getopts") or []],
                units=units, kernels=kernels, hosts=str(payload.get("hosts", "pycam_stage_hosts")),
                runner_uses=[str(x) for x in payload.get("runner_uses") or []])
    _resolve(spec)
    return spec


def _walk(nodes: list[Node]):
    for node in nodes:
        yield node
        yield from _walk(node.children)
        yield from _walk(node.orelse)
        for _, body in node.elifs:
            yield from _walk(body)
        for case in node.cases.values():
            yield from _walk(case)


def _resolve(spec: Spec) -> None:
    for unit in spec.units.values():
        unit.lines = read_lines(unit.source)
        unit.decls = parse_declarations(unit.lines, *unit.declarations)
        unit.dummy_names = signature(unit.lines, unit.declarations[0], unit.declarations[1], unit.routine)
        pieces = [n for n in _walk(unit.body) if n.kind == "piece"]
        for index, node in enumerate(pieces, start=1):
            node.name = f"{unit.key}_piece_{index}"
        unit.pieces = pieces
        spans = [(n.first, n.last) for n in _walk(unit.body) if n.kind in ("piece", "pause", "unit")]
        spans += [(n.line, n.line) for n in _walk(unit.body) if n.kind in ("if", "do", "select")]
        unit.body_range = (min(s[0] for s in spans), max(s[1] for s in spans))
        for pause in unit.pauses:
            (pause.statement,) = [t for _, _, t in statements(unit.lines, pause.first, pause.last)] or [""]
            if not pause.statement.lower().startswith("call "):
                raise SystemExit(f"{unit.key}: lines {pause.first}-{pause.last} are not one call statement: {pause.statement[:60]}")
            if pause.kernel not in spec.kernels:
                raise SystemExit(f"{unit.key}: pause names kernel {pause.kernel!r}, which the spec does not describe")
    # a kernel called at several sites pauses at each; the sites are numbered in source order
    sites: dict[str, list[Pause]] = {}
    for unit in spec.units.values():
        for pause in unit.pauses:
            sites.setdefault(pause.kernel, []).append(pause)
    for kernel, pauses in sites.items():
        for index, pause in enumerate(pauses, start=1):
            pause.site = index if len(pauses) > 1 else 0
            pause.pc_at = f"pc_at_{pause.tag}"
            pause.pc_after = f"pc_after_{pause.tag}"
    for unit in spec.units.values():
        for call in unit.unit_calls:
            (call.statement,) = [t for _, _, t in statements(unit.lines, call.first, call.last)] or [""]
            if call.unit not in spec.units:
                raise SystemExit(f"{unit.key}: unit call names {call.unit!r}, which the spec does not define")
    for kernel in spec.kernels.values():
        kernel.lines = read_lines(kernel.source)
        start = next((i + 1 for i, line in enumerate(kernel.lines)
                      if re.match(rf"^\s*(?:recursive\s+)?subroutine\s+{kernel.routine}\b", line, re.I)), None)
        if start is None:
            raise SystemExit(f"kernel {kernel.name}: no subroutine {kernel.routine} in {kernel.source}")
        end = next((i + 1 for i, line in enumerate(kernel.lines[start:], start=start)
                    if re.match(rf"^\s*end\s+subroutine\s+{kernel.routine}\b", line, re.I)), None)
        kernel.body_range = (start, end or len(kernel.lines))
        kernel.dummies = signature(kernel.lines, start, kernel.body_range[1], kernel.routine)
        kernel.decls = parse_declarations(kernel.lines, start, kernel.body_range[1])
        # a procedure dummy (an interface block, or a bare `optional ::`) has no data the frame serves
        for name in _procedure_dummies(kernel.lines, start, kernel.body_range[1]):
            if name in kernel.dummies and name not in kernel.decls:
                kernel.decls[name] = Decl(name, "procedure", "", "", "", None)


# ---------------------------------------------------------------------------
# coverage: every executable line of a unit's body is accounted for
# ---------------------------------------------------------------------------


def coverage_gaps(unit: Unit) -> list[int]:
    """Executable lines of the unit's body range that no piece, skeleton line, pause or unit call covers."""

    covered: set[int] = set()
    for node in _walk(unit.body):
        if node.kind in ("piece", "pause", "unit"):
            covered.update(range(node.first, node.last + 1))
        elif node.kind in ("if", "do", "select"):
            covered.add(node.line)
            covered.update(_skeleton_closers(unit, node))
            if node.kind == "select":
                covered.update(_refused_case_lines(unit, node))
    for first, last in unit.elided:
        covered.update(range(first, last + 1))
    gaps = []
    for number in range(unit.body_range[0], unit.body_range[1] + 1):
        if number in covered:
            continue
        text = strip_comment(unit.lines[number - 1]).strip()
        if not text or is_preprocessor(unit.lines[number - 1]):
            continue
        gaps.append(number)
    return gaps


def _skeleton_closers(unit: Unit, node: Node) -> set[int]:
    """The `else`, `end if`, `end do`, `case`, `end select` lines that belong to a skeleton node."""

    closers: set[int] = set()
    lines = unit.lines
    depth = 0
    number = node.line
    opener = strip_comment(lines[number - 1]).strip().lower()
    if node.kind == "if":
        pattern_open, pattern_close = r"^if\s*\(.*\)\s*then$", r"^(end\s*if|endif)\b"
        mid = r"^else\b"
    elif node.kind == "do":
        pattern_open, pattern_close = r"^do\b", r"^(end\s*do|enddo)\b"
        mid = None
    else:
        pattern_open, pattern_close = r"^select\s*case\b", r"^end\s*select\b"
        mid = r"^case\b"
    # a `do` may be a labelled `do 10 ...`; a `do while` counts as well
    for number in range(node.line + 1, unit.body_range[1] + 1):
        text = strip_comment(lines[number - 1]).strip().lower()
        if not text:
            continue
        if re.match(pattern_open, text) and _opens_block(text, node.kind):
            depth += 1
        elif re.match(pattern_close, text):
            if depth == 0:
                closers.add(number)
                break
            depth -= 1
        elif mid and depth == 0 and re.match(mid, text):
            closers.add(number)
    return closers


def _refused_case_lines(unit: Unit, node: Node) -> set[int]:
    """The lines of the select's cases the runner refuses: covered, since create refuses the configuration."""

    closers = sorted(_skeleton_closers(unit, node))
    covered: set[int] = set()
    lines = unit.lines
    for start, end in zip(closers, closers[1:]):
        text = strip_comment(lines[start - 1]).strip().lower()
        match = re.match(r"^case\s*\((.*)\)", text)
        if not match:
            continue
        values = [v.strip() for v in _split_top(match.group(1))]
        kept = {v.strip().lower() for v in node.cases}
        if any(v.lower() in kept for v in values):
            continue
        covered.update(range(start, end))
    return covered


def _opens_block(text: str, kind: str) -> bool:
    if kind == "if":
        return bool(re.match(r"^if\s*\(.*\)\s*then$", text))
    if kind == "do":
        return bool(re.match(r"^do\b", text))
    return bool(re.match(r"^select\s*case\b", text))


# ---------------------------------------------------------------------------
# frames: the paused call's arguments, in the callee's order
# ---------------------------------------------------------------------------


@dataclass
class Slot:
    """One argument of a paused call as the frame serves it.

    ``by_address`` marks a scalar served where it lives rather than from a
    copy: an intent(out) or inout scalar whose actual is a variable, so a model
    can answer it.  ``helper`` marks storage whose address is taken through a
    TARGET dummy (a module array with neither attribute, say).
    """

    dummy: str
    actual: str
    intent: str
    kind: str              # array | scalar | pointer | derived-component
    rank: int
    dtype: int
    expression: str        # what the frame takes the address of
    shape: list[str]       # Fortran extent expressions, per dimension
    guard: str | None      # "associated(x)" / "allocated(x)" or None
    component: str | None = None
    by_address: bool = False
    helper: bool = False



def _dummy_dims(decl: Decl, scalar_actuals: Mapping[str, str]) -> list[str]:
    dims = []
    for extent in _split_top(decl.dims) if decl.dims else []:
        if ":" in extent:
            lower, _, upper = extent.partition(":")
            extent = f"({upper})-({lower})+1" if upper.strip() and lower.strip() else ""
        for dummy, actual in scalar_actuals.items():
            extent = re.sub(rf"\b{dummy}\b", f"({actual})", extent)
        dims.append(extent.strip())
    return dims


def frame_slots(pause: Pause, kernel: Kernel) -> list[Slot]:
    """One slot per argument of the paused call, less character arguments.

    Positional actuals pair with the callee's dummies in order; keyword
    actuals by name.  Scalars are served by value from a frame copy; arrays
    by the address of their first element and the callee's declared extents
    (with scalar dummies replaced by the actuals bound to them), so an
    element or a contiguous section passed by sequence association is served
    with the shape the callee sees; pointers unassociated in this
    configuration are served as zeros of that shape.  A derived-type dummy
    is served component by component, each component the callee's body
    names, as `dummy.component` slots; the frame does not carry the type.
    """

    inside = pause.statement.split("(", 1)[1].rsplit(")", 1)[0]
    actuals = _split_top(inside)
    by_dummy: dict[str, str] = {}
    for index, actual in enumerate(actuals):
        match = re.match(r"^(\w+)\s*=\s*(.+)$", actual)
        if match and match.group(1).lower() in kernel.dummies:
            by_dummy[match.group(1).lower()] = match.group(2).strip()
        else:
            by_dummy[kernel.dummies[index]] = actual.strip()
    scalar_actuals = {d: by_dummy[d] for d in kernel.dummies
                      if d in by_dummy and kernel.decls.get(d) is not None and kernel.decls[d].rank == 0
                      and not kernel.decls[d].is_derived}
    slots: list[Slot] = []
    unit_decls = pause.unit.decls
    for dummy in kernel.dummies:
        decl = kernel.decls.get(dummy)
        if decl is None:
            raise SystemExit(f"kernel {kernel.name}: dummy {dummy!r} has no declaration")
        if decl.base_type.startswith("character") or decl.kind == "procedure":
            continue
        intent = decl.intent or "inout"
        if dummy not in by_dummy:
            # an optional argument this site omits: the slot stays, empty, so every
            # site of the kernel serves the same frame
            if decl.is_derived or decl.base_type.startswith("character"):
                continue
            slots.append(Slot(dummy=dummy, actual="", intent=intent, kind="absent", rank=decl.rank,
                              dtype=DTYPE_CODE.get(decl.base_type, 1), expression="", shape=[], guard=None))
            continue
        actual = by_dummy[dummy]
        if decl.is_derived:
            if decl.kind.lower() not in DERIVED_TYPES:
                # a type the frame does not know (private components, say): the
                # original passes it, the frame serves nothing for it
                slots.append(Slot(dummy=dummy, actual=actual, intent=intent, kind="opaque", rank=0, dtype=1,
                                  expression="", shape=[], guard=None))
                continue
            components = _components_used(kernel, dummy)
            for component, (crank, cbase, written) in components.items():
                cintent = "in" if intent == "in" else ("out" if written and not _read(kernel, dummy, component) else ("inout" if written else "in"))
                expression = f"{actual}%{component}"
                slots.append(Slot(dummy=f"{dummy}.{component}", actual=expression, intent=cintent,
                                  kind="derived-component", rank=crank, dtype=DTYPE_CODE.get(cbase, 1),
                                  expression=expression, shape=[f"size({expression},{i + 1})" for i in range(crank)],
                                  guard=None, component=component))
            continue
        dtype = DTYPE_CODE.get(decl.base_type, 1)
        if decl.rank == 0:
            designator = bool(re.match(r"^[\w%]+(\s*\([^()]*\))?$", actual.strip())) and not re.fullmatch(r"[\d.]+(_\w+)?", actual.strip())
            base_name = actual.strip().split("%")[0].split("(")[0].lower()
            by_address = designator and intent != "in" and decl.base_type != "logical"
            slots.append(Slot(dummy=dummy, actual=actual, intent=intent, kind="scalar", rank=0, dtype=dtype,
                              expression=actual.strip(), shape=[], guard=None, by_address=by_address,
                              helper=by_address and base_name not in unit_decls and base_name not in pause.unit.dummy_names))
            continue
        shape = _dummy_dims(decl, scalar_actuals) if decl.dims and ":" not in decl.dims.replace("(:,", "(") else []
        base, section = _actual_base(actual)
        # the callee's declared extents, where the unit can evaluate them; an extent
        # naming something only the callee imports (wtrc_ntype(iwtice), say) is the
        # actual's own extent on that axis
        resolvable = _resolvable_names(pause.unit)
        shape = [extent if identifiers(extent) <= resolvable else f"size({base},{i + 1})"
                 for i, extent in enumerate(shape)]
        if not shape:
            shape = [f"size({base},{i + 1})" for i in range(decl.rank)]
        guard = None
        base_name = base.split("%")[0].lower()
        base_decl = unit_decls.get(base_name)
        if base_decl is not None and "%" not in base:
            if base_decl.is_pointer:
                guard = f"associated({base})"
            elif base_decl.is_allocatable:
                guard = f"allocated({base})"
        # a module variable of the driver's own module (a `use`d array with neither
        # target nor pointer attribute) is addressed through a TARGET dummy
        helper = base_decl is None and base_name not in pause.unit.dummy_names and base_name not in pause.unit.locals \
            and base_name not in pause.unit.carries and base_name not in pause.unit.records
        slots.append(Slot(dummy=dummy, actual=actual, intent=intent,
                          kind="pointer" if guard and "associated" in guard else "array",
                          rank=decl.rank, dtype=dtype,
                          expression=_first_element(actual, base, section, decl.rank, base_decl),
                          shape=shape, guard=guard, helper=helper))
    return slots


#: Extents a frame slot carries: the ABI's `shapes(FRAME_MAX_RANK, count)`, the same number
#: freecam.pi_cam.segment_runner reads (a water-tracer ratio is rank 4).
FRAME_MAX_RANK = 5

#: Names every unit can evaluate in an extent: the grid's parameters (`use ppgrid` is unqualified
#: in these drivers) and the constituent count.
GRID_NAMES = {"pcols", "pver", "pverp", "pcnst", "psubcols", "begchunk", "endchunk"}


def _resolvable_names(unit: Unit) -> set[str]:
    """Identifiers a unit module can evaluate in a frame extent: its declarations, dummies,
    resolved indices, options, and every name its use statements import by name."""

    names = set(unit.decls) | set(unit.dummy_names) | set(unit.getopts) | GRID_NAMES
    names |= {name for name, _ in _unit_pbuf_indices(unit)}
    ranges = ([unit.module_header] if unit.module_header else []) + [unit.declarations]
    for statement in use_statements(unit.lines, ranges) + unit.uses:
        only = re.search(r"only\s*:\s*(.*)$", strip_comment(statement), re.I)
        if only:
            for item in _split_top(only.group(1)):
                names.add(item.split("=>")[0].strip().lower())
    return names


def _actual_base(actual: str) -> tuple[str, str | None]:
    """`ptend%q(1,1,1)` -> ('ptend%q', '1,1,1'); `state%q(:,:,ixcldliq)` -> ('state%q', ':,:,ixcldliq')."""

    match = re.match(r"^([\w%]+)\s*\((.*)\)\s*$", actual)
    if match:
        return match.group(1), match.group(2)
    return actual, None


def _lower_bounds(decl: Decl | None, rank: int, base: str = "") -> list[str]:
    """The lower bound of every axis: the declaration's `lo` in `lo:hi`, 1 for a plain extent, and
    `lbound(base, axis)` for a deferred-shape axis (an allocatable or pointer, whose bounds are set
    at run time: gw_tend's tau is allocated from -ngwv)."""

    bounds = []
    dims = _split_top(decl.dims) if decl is not None and decl.dims else []
    for axis in range(rank):
        extent = dims[axis].strip() if axis < len(dims) else ""
        if extent == ":" and base:
            bounds.append(f"lbound({base},{axis + 1})")
            continue
        lower = extent.partition(":")[0].strip() if ":" in extent else ""
        bounds.append(lower if lower else "1")
    return bounds


def _first_element(actual: str, base: str, section: str | None, rank: int, decl: Decl | None = None) -> str:
    """The Fortran expression whose address is the argument's first element.

    A whole array passes its first element, at the declared lower bounds
    (`aer_tau(pcols,0:pver,nbndsw)` starts at `aer_tau(1,0,1)`); a section's
    omitted subscripts take the lower bound of their axis.
    """

    if not rank:
        return base
    bounds = _lower_bounds(decl, rank, base)
    if section is None:
        return f"{base}({','.join(bounds)})"
    indices = []
    for i, subscript in enumerate(_split_top(section)):
        subscript = subscript.strip()
        if ":" in subscript:
            lower = subscript.partition(":")[0].strip()
            subscript = lower if lower else (bounds[i] if i < len(bounds) else "1")
        indices.append(subscript)
    return f"{base}({','.join(indices)})"


def _components_used(kernel: Kernel, dummy: str) -> dict[str, tuple[int, str, bool]]:
    """component -> (rank, base type, written) for every `dummy%component` the callee's body names.

    Rank and type come from the derived type's definition when it is one the
    generator knows (physics_state, physics_ptend); otherwise the component
    is assumed real(r8) of the rank its first use indexes.
    """

    body = "\n".join(kernel.lines[kernel.body_range[0] - 1:kernel.body_range[1]])
    found: dict[str, tuple[int, str, bool]] = {}
    for line in body.splitlines():
        code = strip_comment(line)
        for match in re.finditer(rf"\b{dummy}%(\w+)\s*(\([^)]*\))?", code, re.I):
            component = match.group(1).lower()
            known = DERIVED_TYPES.get(kernel.decls[dummy].kind.lower())
            typed = (known or {}).get(component)
            if typed is None:
                if known is not None:
                    continue                          # a type-bound procedure, not a data component
                indexed = match.group(2) or ""
                rank = indexed.count(",") + 1 if indexed else 0
                base = "real"
            else:
                rank, base = typed
            written = bool(re.match(rf"^\s*{dummy}%{component}\b[^=]*=(?!=)", code, re.I))
            previous = found.get(component)
            found[component] = (rank, base, (previous[2] if previous else False) or written)
    return found


def _read(kernel: Kernel, dummy: str, component: str) -> bool:
    body = "\n".join(kernel.lines[kernel.body_range[0] - 1:kernel.body_range[1]])
    for line in body.splitlines():
        code = strip_comment(line)
        if re.search(rf"\b{dummy}%{component}\b", code, re.I) and not re.match(rf"^\s*{dummy}%{component}\b[^=]*=(?!=)", code, re.I):
            return True
        if re.match(rf"^\s*{dummy}%{component}\b[^=]*=(?!=)(.*)$", code, re.I):
            rhs = re.match(rf"^\s*{dummy}%{component}\b[^=]*=(?!=)(.*)$", code, re.I).group(1)
            if re.search(rf"\b{dummy}%{component}\b", rhs, re.I):
                return True
    return False


#: Components of CAM's derived types, by name: (rank, base type).  Enough for
#: the frames to serve them with the right shape and dtype.
DERIVED_TYPES: dict[str, dict[str, tuple[int, str]]] = {
    "type(physics_state)": {
        "lchnk": (0, "integer"), "ncol": (0, "integer"), "psetcols": (0, "integer"), "ngrdcol": (0, "integer"),
        "lat": (1, "real"), "lon": (1, "real"), "ps": (1, "real"), "phis": (1, "real"),
        "t": (2, "real"), "u": (2, "real"), "v": (2, "real"), "s": (2, "real"), "omega": (2, "real"),
        "pmid": (2, "real"), "pdel": (2, "real"), "rpdel": (2, "real"), "lnpmid": (2, "real"), "exner": (2, "real"),
        "zm": (2, "real"), "q": (3, "real"), "pint": (2, "real"), "pintdry": (2, "real"), "lnpint": (2, "real"),
        "zi": (2, "real"), "pmiddry": (2, "real"), "pdeldry": (2, "real"), "rpdeldry": (2, "real"),
        "lnpmiddry": (2, "real"), "lnpintdry": (2, "real"), "te_ini": (1, "real"), "te_cur": (1, "real"),
        "tw_ini": (1, "real"), "tw_cur": (1, "real"), "psdry": (1, "real"), "count": (0, "integer"),
    },
    "type(rrtmg_state_t)": {
        name: (2, "real") for name in (
            "h2ovmr", "o3vmr", "co2vmr", "ch4vmr", "o2vmr", "n2ovmr", "cfc11vmr", "cfc12vmr", "cfc22vmr",
            "ccl4vmr", "pmidmb", "pintmb", "tlay", "tlev")
    },
    "type(gwband)": {
        "ngwv": (0, "integer"), "dc": (0, "real"), "cref": (1, "real"), "fcrit2": (0, "real"),
        "kwv": (0, "real"), "effkwv": (0, "real"),
    },
    "type(coords1d)": {
        "n": (0, "integer"), "d": (0, "integer"), "ifc": (2, "real"), "mid": (2, "real"), "del": (2, "real"),
        "dst": (2, "real"), "rdel": (2, "real"), "rdst": (2, "real"),
    },
    "type(physics_ptend)": {
        "s": (2, "real"), "u": (2, "real"), "v": (2, "real"), "q": (3, "real"),
        "hflux_srf": (1, "real"), "hflux_top": (1, "real"), "taux_srf": (1, "real"), "taux_top": (1, "real"),
        "tauy_srf": (1, "real"), "tauy_top": (1, "real"), "cflx_srf": (2, "real"), "cflx_top": (2, "real"),
        "psetcols": (0, "integer"), "ls": (0, "logical"), "lu": (0, "logical"), "lv": (0, "logical"),
        "lq": (1, "logical"), "top_level": (0, "integer"), "bot_level": (0, "integer"), "name": (0, "character"),
    },
}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _hoisted_state(unit: Unit) -> list[str]:
    """The unit's locals as module state: every declared name the body or the preamble uses."""

    used: set[str] = set()
    for number in range(unit.body_range[0], unit.body_range[1] + 1):
        used |= identifiers(unit.lines[number - 1])
    for text in unit.preamble + unit.postamble:
        used |= identifiers(text)
    for call in unit.unit_calls:
        used |= identifiers(call.statement)
    rows: list[str] = []
    for name in sorted(used):
        decl = unit.decls.get(name)
        if decl is None or name in unit.dummy_names or name in unit.carries or name in unit.records \
                or name in unit.getopts or name in unit.locals or name in KEYWORDS:
            continue
        attrs = f", {decl.attributes}" if decl.attributes else ""
        save = "" if decl.is_parameter else ", save"
        # frames take the address of a local or of its components; a pointer or
        # parameter is left as declared, everything else gets the target attribute
        target = "" if (decl.is_parameter or decl.is_pointer or "target" in decl.attributes) else ", target"
        init = decl.init
        if decl.is_pointer and not init:
            init = "=> null()"
        shape = f"({decl.dims})" if decl.dims else ""
        if _is_automatic(unit, name):
            # an automatic array of the routine (an extent set at run time): module
            # state cannot size it, so it is allocated at bind, again when the shape changes
            shape = "(" + ",".join(":" for _ in range(decl.rank)) + ")"
            rows.append(f"  {decl.kind}{attrs}, allocatable{save}{target}, public :: {name}{shape}")
            continue
        rows.append(f"  {decl.kind}{attrs}{save}{target}, public :: {name}{shape}{(' ' + init) if init else ''}")
    return rows


def _is_automatic(unit: Unit, name: str) -> bool:
    """A local listed as automatic, or whose extents name a dummy's component or a dummy (state%ncol, ncol)."""

    decl = unit.decls.get(name)
    if decl is None or not decl.dims or decl.is_pointer or decl.is_allocatable or decl.is_parameter or decl.is_derived:
        return False
    if name in unit.automatic:
        return True
    if "%" in decl.dims:
        return True
    return bool(identifiers(decl.dims) & set(unit.dummy_names))


def _automatic_names(unit: Unit) -> list[str]:
    used: set[str] = set()
    for number in range(unit.body_range[0], unit.body_range[1] + 1):
        used |= identifiers(unit.lines[number - 1])
    return sorted(n for n in used if n in unit.decls and n not in unit.dummy_names and _is_automatic(unit, n))


def _automatic_allocations(unit: Unit, indent: str = "    ") -> list[str]:
    rows = []
    for name in _automatic_names(unit):
        decl = unit.decls[name]
        extents = ", ".join(f"({e.partition(':')[2]})-({e.partition(':')[0]})+1" if ":" in e else e for e in _split_top(decl.dims))
        rows.append(f"{indent}if (allocated({name})) then")
        rows.append(f"{indent}  if (any(shape({name}) /= (/ {extents} /))) deallocate({name})")
        rows.append(f"{indent}end if")
        rows.append(f"{indent}if (.not. allocated({name})) allocate({name}({decl.dims}))")
    return rows


def _dummy_state(unit: Unit) -> list[str]:
    """The unit's dummies as module variables the caller binds: pointers for arrays and derived types."""

    rows: list[str] = []
    for name in unit.dummy_names:
        decl = unit.decls.get(name)
        if decl is None:
            raise SystemExit(f"{unit.key}: dummy {name!r} of {unit.routine} has no declaration in {unit.declarations}")
        if decl.is_derived or decl.rank > 0 or decl.is_pointer:
            dims = ",".join(":" for _ in range(decl.rank)) if decl.rank else ""
            shape = f"({dims})" if dims else ""
            rows.append(f"  {decl.kind}, pointer, save, public :: {name}{shape} => null()")
        else:
            rows.append(f"  {decl.kind}, save, public :: {name}")
    return rows


def _record_state(unit: Unit) -> list[str]:
    rows = []
    for name in sorted(unit.records):
        decl = unit.decls.get(name)
        if decl is None:
            raise SystemExit(f"{unit.key}: record {name!r} is not a local of {unit.routine}")
        rows.append(f"  {decl.kind}, pointer, save, public :: {name} => null()")
    return rows


def _local_state(unit: Unit) -> list[str]:
    """Locals the glue points at storage of its own choosing (`locals:` in the spec)."""

    rows = []
    for name in sorted(unit.locals):
        decl = unit.decls.get(name)
        if decl is None:
            raise SystemExit(f"{unit.key}: local {name!r} is not a local of {unit.routine}")
        dims = ",".join(":" for _ in range(decl.rank))
        shape = f"({dims})" if dims else ""
        rows.append(f"  {decl.kind}, pointer, save, public :: {name}{shape} => null()")
    return rows


def _verbatim(unit: Unit, first: int, last: int, indent: str = "  ") -> list[str]:
    return [(indent + unit.lines[n - 1].rstrip()) if unit.lines[n - 1].strip() else "" for n in range(first, last + 1)]


def _getopts_state(unit: Unit) -> list[str]:
    rows = []
    for name in unit.getopts:
        kind = GETOPTS.get(name, GETOPT_TYPES["character"])
        init = " = ' '" if kind.startswith("character") else (" = .false." if kind == "logical" else " = 0")
        rows.append(f"  {kind}, save, public :: {name}{init}")
    return rows


def _carry_state(unit: Unit) -> list[str]:
    rows = []
    for name in sorted(unit.carries):
        decl = unit.decls.get(name)
        if decl is None:
            raise SystemExit(f"{unit.key}: carry {name!r} is not a local of {unit.routine}")
        dims = ",".join(":" for _ in range(decl.rank))
        rows.append(f"  {decl.kind}, pointer, save, public :: {name}({dims}) => null()")
    return rows


FLOW_CODES = {"cycle": 1, "exit": 2, "return": 3}
_BLOCK_OPEN = re.compile(r"^(?:\w+\s*:\s*)?(?:if\s*\(.*\)\s*then|do\b|select\s*case\b|select\s*type\b|where\s*\(.*\)\s*$|forall\s*\(.*\)\s*$|associate\s*\()", re.I)
_BLOCK_CLOSE = re.compile(r"^(?:end\s*(?:if|do|select|where|forall|associate)|endif|enddo|endwhere|endforall)\b", re.I)
_FLOW = re.compile(r"^(?:if\s*\((?P<cond>.*)\)\s*)?(?P<verb>return|cycle|exit)(?:\s+\w+)?$", re.I)


def _piece_flows(unit: Unit, first: int, last: int) -> dict[int, tuple[int, int, str, str | None]]:
    """start line -> (last line, flow code, verb, condition) for every return / cycle / exit at
    the piece's own block level: they leave the enclosing routine or the runner's loop, which the
    piece cannot do itself, so the piece reports them through `flow` and the runner acts."""

    found: dict[int, tuple[int, int, str, str | None]] = {}
    loops = 0
    for start, end, text in statements(unit.lines, first, last):
        low = text.strip().lower()
        if re.match(r"^(?:end\s*do|enddo)\b", low):
            loops = max(loops - 1, 0)
            continue
        if re.match(r"^(?:\w+\s*:\s*)?do\b", low) and not re.match(r"^do\s*\d", low):
            loops += 1
            continue
        match = _FLOW.match(low)
        if not match:
            continue
        verb = match.group("verb").lower()
        # a return leaves the routine from any depth; a cycle or exit outside every loop
        # of the piece belongs to the runner's loop
        if verb == "return" or loops == 0:
            found[start] = (end, FLOW_CODES[verb], verb, match.group("cond"))
    return found


def _piece_text(unit: Unit, node: Node) -> str:
    body = []
    flows = _piece_flows(unit, node.first, node.last)
    node.flows = {code for _, code, _, _ in flows.values()}
    skip_to = 0
    for number in range(node.first, node.last + 1):
        if number <= skip_to:
            continue
        line = unit.lines[number - 1].rstrip()
        if number in flows:
            end, code, verb, condition = flows[number]
            indent = " " * (len(line) - len(line.lstrip()))
            note = f"! {verb} at the routine's level: the runner {'leaves the routine' if verb == 'return' else 'moves its loop'}"
            if condition is None:
                body += [f"    {indent}flow = {code}   {note}", f"    {indent}return"]
            else:
                body += [f"    {indent}if ({condition}) then   {note}", f"    {indent}  flow = {code}",
                         f"    {indent}  return", f"    {indent}end if"]
            skip_to = end
            continue
        body.append(("    " + line) if line.strip() else "")
    return "\n".join([f"  subroutine {node.name}()",
                      f"    ! {Path(unit.source).name}:{node.first}-{node.last}, verbatim"
                      + (" but for the flow statements the runner carries out" if flows else ""), ""] + body
                     + ["", f"  end subroutine {node.name}"])


def _bind_routine(unit: Unit, caller: Unit) -> str:
    """`<unit>_bind(<actuals>)`: what the caller's call statement hands the unit's dummies."""

    args = ", ".join(f"a_{d}" for d in unit.dummy_names)
    lines = [f"  subroutine {unit.key}_bind({args})",
             f"    ! the dummies of {unit.routine}, as the caller's statement passes them"]
    for name in unit.dummy_names:
        decl = unit.decls[name]
        if decl.is_derived or decl.rank > 0 or decl.is_pointer:
            dims = ",".join(":" for _ in range(decl.rank))
            shape = f"({dims})" if dims else ""
            lines.append(f"    {decl.kind}, target, intent(inout) :: a_{name}{shape}")
        else:
            lines.append(f"    {decl.kind}, intent(in) :: a_{name}")
    for name in unit.dummy_names:
        decl = unit.decls[name]
        if decl.is_derived or decl.rank > 0 or decl.is_pointer:
            lines.append(f"    {name} => a_{name}")
        else:
            lines.append(f"    {name} = a_{name}")
    lines += _automatic_allocations(unit)
    lines.append(f"  end subroutine {unit.key}_bind")
    return "\n".join(lines)


def _frame_routine(spec: Spec, pause: Pause, slots: list[Slot]) -> str:
    name = f"{pause.tag}_frame"
    lines = [f"  subroutine {name}(ptrs, ndims, shapes, dtypes, intents, ncol_out)",
             f"    ! the paused `{pause.statement[:70]}...` in the callee's argument order",
             "    type(c_ptr), intent(inout) :: ptrs(:)",
             "    integer(c_int), intent(inout) :: ndims(:), dtypes(:), intents(:)",
             "    integer(c_int64_t), intent(inout) :: shapes(:,:)",
             "    integer(c_int), intent(out) :: ncol_out"]
    scalars = [s for s in slots if s.rank == 0 and not s.by_address and s.kind not in ("absent", "opaque")]
    for index, slot in enumerate(scalars, start=1):
        kind = "real(c_double)" if slot.dtype == 1 else "integer(c_int32_t)"
        lines.append(f"    {kind}, save, target :: sc_{index}")
    if any(s.helper for s in slots):
        lines.append("    type(c_ptr) :: address")
    lines.append(f"    ncol_out = int({_ncol_expression(pause.unit)}, c_int)")
    for index, slot in enumerate(slots, start=1):
        if slot.kind in ("absent", "opaque"):
            lines.append(f"    call empty_slot({index}, {slot.rank}, {slot.dtype}, {INTENT_CODE[slot.intent]}, ptrs, ndims, shapes, dtypes, intents)")
            continue
        if slot.rank == 0 and not slot.by_address:
            si = scalars.index(slot) + 1
            if slot.dtype == 1:
                lines.append(f"    sc_{si} = real({slot.expression}, c_double)")
            else:
                lines.append(f"    sc_{si} = int(merge(1, 0, {slot.expression}), c_int32_t)" if _is_logical(slot, spec, pause)
                             else f"    sc_{si} = int({slot.expression}, c_int32_t)")
            lines.append(f"    call put_slot({index}, c_loc(sc_{si}), 0, zero_shape, {slot.dtype}, {INTENT_CODE[slot.intent]}, "
                         f"ptrs, ndims, shapes, dtypes, intents)")
            continue
        shape = "zero_shape" if slot.rank == 0 else "(/ " + ", ".join(f"int({e}, c_int64_t)" for e in slot.shape) + " /)"
        if slot.helper:
            base, section = _actual_base(slot.actual)
            colons = [s for s in _split_top(section)] if section else []
            sectioned = [s for s in colons if ":" in s]
            if sectioned:
                # a section (mu(:,:,lchnk), cpairv(:,:,state%lchnk)): the whole section goes to an
                # assumed-shape TARGET dummy of its rank, which is legal for a pointer array too
                taker = f"slot_address_{'r8' if slot.dtype == 1 else 'i4'}_{len(sectioned)}"
                lines.append(f"    call {taker}({slot.actual}, address)")
            else:
                taker = "slot_address_r8" if slot.dtype == 1 else "slot_address_i4"
                lines.append(f"    call {taker}({slot.expression}, address)")
            location = "address"
        else:
            location = f"c_loc({slot.expression})"
        put = (f"call put_slot({index}, {location}, {slot.rank}, {shape}, {slot.dtype}, "
               f"{INTENT_CODE[slot.intent]}, ptrs, ndims, shapes, dtypes, intents)")
        if slot.guard:
            lines.append(f"    if ({slot.guard}) then")
            lines.append(f"      {put}")
            lines.append(f"    else")
            lines.append(f"      call empty_slot({index}, {slot.rank}, {slot.dtype}, {INTENT_CODE[slot.intent]}, ptrs, ndims, shapes, dtypes, intents)")
            lines.append(f"    end if")
        else:
            lines.append(f"    {put}")
    lines.append(f"  end subroutine {name}")
    return "\n".join(lines)


def _ncol_expression(unit: Unit) -> str:
    """The chunk's column count as the unit knows it: its `ncol`, else its physics state's."""

    if "ncol" in unit.decls or "ncol" in unit.dummy_names:
        return "ncol"
    for name in unit.dummy_names:
        decl = unit.decls.get(name)
        if decl is not None and decl.kind.lower() == "type(physics_state)":
            return f"{name}%ncol"
    raise SystemExit(f"{unit.key}: {unit.routine} has neither an ncol local nor a physics_state dummy; "
                     f"a frame cannot report the live columns")


def _is_logical(slot: Slot, spec: Spec, pause: Pause) -> bool:
    kernel = spec.kernels[pause.kernel]
    decl = kernel.decls.get(slot.dummy.split(".")[0])
    return decl is not None and decl.base_type == "logical"


def _original_routine(pause: Pause) -> str:
    body = []
    for number in range(pause.first, pause.last + 1):
        line = pause.unit.lines[number - 1].rstrip()
        body.append(("    " + line) if line.strip() else "")
    return "\n".join([f"  subroutine {pause.tag}_original()",
                      f"    ! the original call, {Path(pause.unit.source).name}:{pause.first}-{pause.last}, verbatim"]
                     + body + [f"  end subroutine {pause.tag}_original"])


def _section_helpers(ranks: set[tuple[int, int]]) -> str:
    """slot_address_<kind>_<rank>: an assumed-shape TARGET dummy per (kind, rank) a frame needs."""

    out = []
    for dtype, rank in sorted(ranks):
        kind = "real(r8)" if dtype == 1 else "integer(c_int32_t)"
        name = f"slot_address_{'r8' if dtype == 1 else 'i4'}_{rank}"
        dims = ",".join(":" for _ in range(rank))
        ones = ",".join("1" for _ in range(rank))
        out.append(f"""  subroutine {name}(field, address)
    ! a section's address through an assumed-shape TARGET dummy; a contiguous
    ! section is passed without a copy
    {kind}, intent(inout), target :: field({dims})
    type(c_ptr), intent(out) :: address
    address = c_loc(field({ones}))
  end subroutine {name}""")
    return "\n\n".join(out)


def _section_helper_ranks(spec: Spec, unit: Unit) -> set[tuple[int, int]]:
    ranks: set[tuple[int, int]] = set()
    for pause in unit.pauses:
        for slot in frame_slots(pause, spec.kernels[pause.kernel]):
            if not slot.helper:
                continue
            _, section = _actual_base(slot.actual)
            sectioned = [s for s in _split_top(section)] if section else []
            sectioned = [s for s in sectioned if ":" in s]
            if sectioned:
                ranks.add((slot.dtype, len(sectioned)))
    return ranks


ADDRESS_HELPERS = """  subroutine slot_address_r8(field, address)
    ! a TARGET dummy so c_loc is legal whatever the actual argument's attributes
    ! (a module array of the driver's own module, say); a contiguous actual is
    ! passed without a copy, and an element stands for the rest of its array
    real(r8), intent(inout), target :: field(*)
    type(c_ptr), intent(out) :: address
    address = c_loc(field(1))
  end subroutine slot_address_r8

  subroutine slot_address_i4(field, address)
    integer(c_int32_t), intent(inout), target :: field(*)
    type(c_ptr), intent(out) :: address
    address = c_loc(field(1))
  end subroutine slot_address_i4"""

SLOT_HELPERS = """  subroutine put_slot(index, address, rank, shape, dtype, intent, ptrs, ndims, shapes, dtypes, intents)
    integer, intent(in) :: index, rank, dtype, intent
    type(c_ptr), intent(in) :: address
    integer(c_int64_t), intent(in) :: shape(:)
    type(c_ptr), intent(inout) :: ptrs(:)
    integer(c_int), intent(inout) :: ndims(:), dtypes(:), intents(:)
    integer(c_int64_t), intent(inout) :: shapes(:,:)
    integer :: axis
    ptrs(index) = address
    ndims(index) = int(rank, c_int)
    shapes(:, index) = 0_c_int64_t
    do axis = 1, rank
      shapes(axis, index) = shape(axis)
    end do
    dtypes(index) = int(dtype, c_int)
    intents(index) = int(intent, c_int)
  end subroutine put_slot

  subroutine empty_slot(index, rank, dtype, intent, ptrs, ndims, shapes, dtypes, intents)
    ! an argument with no storage in this call: a null address and zero extents
    integer, intent(in) :: index, rank, dtype, intent
    type(c_ptr), intent(inout) :: ptrs(:)
    integer(c_int), intent(inout) :: ndims(:), dtypes(:), intents(:)
    integer(c_int64_t), intent(inout) :: shapes(:,:)
    ptrs(index) = c_null_ptr
    ndims(index) = int(rank, c_int)
    shapes(:, index) = 0_c_int64_t
    dtypes(index) = int(dtype, c_int)
    intents(index) = int(intent, c_int)
  end subroutine empty_slot"""


def _only_r8(use: str) -> bool:
    """A shr_kind_mod use the unit module already covers with its own `r8 => shr_kind_r8`."""

    if not re.match(r"^\s*use\s+shr_kind_mod\b", use, re.I):
        return False
    only = re.search(r"only\s*:\s*(.*)$", strip_comment(use), re.I)
    if not only:
        return False
    names = {item.split("=>")[0].strip().lower() for item in _split_top(only.group(1))}
    return names <= {"r8"}


def _without_r8(use: str) -> str:
    """A shr_kind_mod use with its `r8 => shr_kind_r8` item dropped: the unit module imports r8 itself,
    and ifort refuses the same local name from two use statements."""

    if not re.match(r"^\s*use\s+shr_kind_mod\b", use, re.I):
        return use
    head, _, only = strip_comment(use).partition(":")
    items = [item.strip() for item in _split_top(only) if item.split("=>")[0].strip().lower() != "r8"]
    return f"{head.strip()}: {', '.join(items)}"


def render_unit(spec: Spec, unit: Unit) -> str:
    """One unit's module: hoisted locals, dummies, pieces, frames, originals, binder."""

    uses = use_statements(unit.lines, ([unit.module_header] if unit.module_header else []) + [unit.declarations])
    uses = [_without_r8(u) for u in uses if not re.match(r"^\s*use\s+iso_c_binding\b", u, re.I) and not _only_r8(u)]
    header = [
        f"! {unit.routine} ({Path(unit.source).name}:{unit.body_range[0]}-{unit.body_range[1]}) hoisted for the",
        f"! {spec.prefix} segment runner: its locals as module state, its body in pieces the",
        "! runner calls in the source's order, its dummies bound by the caller.",
        "!",
        "! GENERATED by tools/generate_pi_cam_pausable_runners.py from",
        f"! native/pi_cam/pausable/{spec.path.name}.  Do not edit by hand; edit the spec.",
        f"module {unit.module}",
        "  use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t, c_double, c_ptr, c_loc, c_null_ptr, c_f_pointer",
        "  use shr_kind_mod, only: r8 => shr_kind_r8",
    ]
    if unit.key == "glue":
        header.append(f"  use {spec.hosts}, only: host_state, host_tend, host_pbuf2d, host_cam_in, host_cam_out")
        header.append("  use physics_buffer, only: pbuf_get_chunk")
        header.append("  use time_manager, only: get_nstep, get_step_size")
    for use in uses:
        header.append("  " + use)
    for use in unit.uses:
        header.append("  " + use)
    for call in unit.unit_calls:
        header.append(f"  use pycam_{spec.prefix}_{call.unit}, only: {call.unit}_bind")
    header += ["  implicit none", "  private", "  public :: " + ", ".join(
        [n.name for n in unit.pieces] + [f"{p.tag}_frame" for p in unit.pauses] + [f"{p.tag}_original" for p in unit.pauses]
        + ([f"{unit.key}_bind"] if unit.key != "glue" else ["glue_enter"] + (["glue_leave"] if unit.postamble else []))
        + [f"{unit.key}_bind_{c.unit}" for c in unit.unit_calls]
        + ([f"{unit.key}_resolve_indices"] if unit.pbuf_indices else []) + ([f"{unit.key}_configure"] if unit.getopts else [])),
        "", "  integer(c_int64_t), parameter :: zero_shape(1) = (/ 0_c_int64_t /)", "",
        "  ! the routine's dummies, bound by the caller"]
    header += _dummy_state(unit)
    if unit.key == "glue" and _copied_dummies(unit):
        header += ["", "  ! read-only dummies served from a copy: their storage has no target attribute"] + _copy_state(unit)
    if unit.carries:
        header += ["", "  ! tphysbc locals that are pycesm carries: views of physpkg's storage for this chunk"] + _carry_state(unit)
    if unit.records:
        header += ["", "  ! locals that are pycesm carries of derived type: pointers to physpkg's records"] + _record_state(unit)
    if unit.getopts:
        header += ["", "  ! the module's private options, read through phys_getopts at configure"] + _getopts_state(unit)
    if unit.locals:
        header += ["", "  ! locals pointed at storage the stage's Python side reads back"] + _local_state(unit)
    header += ["", "  ! a piece's return, cycle or exit at the routine's level, for the runner to carry out",
               "  integer, save, public :: flow = 0"]
    header += ["", "  ! the routine's locals, held for the chunk in flight"] + _hoisted_state(unit)
    for first, last in unit.module_private:
        header += ["", f"  ! {Path(unit.source).name}:{first}-{last}: the module's own declaration, verbatim"]
        header += _verbatim(unit, first, last)
    if unit.pbuf_indices:
        indices = _unit_pbuf_indices(unit)
        header += ["", "  ! the module's private physics-buffer indices, resolved by name at configure"]
        header += [f"  integer, save, public :: {name} = -1" for name, _ in indices]
    body = ["", "contains", ""]
    if unit.key == "glue":
        body.append(_glue_enter(spec, unit))
        body.append("")
        if unit.postamble:
            body.append("\n".join(["  subroutine glue_leave()", "    ! the block's closing statements, after the last piece"]
                                  + ["    " + text for text in unit.postamble] + ["  end subroutine glue_leave"]))
            body.append("")
    for call in unit.unit_calls:
        body.append(_unit_call_binder(spec, unit, call))
        body.append("")
    if unit.key != "glue":
        body.append(_bind_routine(unit, spec.units["glue"]))
        body.append("")
    if unit.pbuf_indices:
        body.append(_resolve_indices(unit))
        body.append("")
    if unit.getopts:
        body.append("\n".join([f"  subroutine {unit.key}_configure()",
                               "    ! the module's private options, as phys_getopts reports them",
                               "    use phys_control, only: phys_getopts"]
                              + [f"    call phys_getopts({name}_out={name})" for name in unit.getopts]
                              + [f"  end subroutine {unit.key}_configure"]))
        body.append("")
    for node in unit.pieces:
        body.append(_piece_text(unit, node))
        body.append("")
    for pause in unit.pauses:
        slots = frame_slots(pause, spec.kernels[pause.kernel])
        body.append(_frame_routine(spec, pause, slots))
        body.append("")
        body.append(_original_routine(pause))
        body.append("")
    for first, last in unit.helpers:
        body.append("\n".join([f"  ! {Path(unit.source).name}:{first}-{last}: a private procedure of the module, verbatim"]
                              + _verbatim(unit, first, last)))
        body.append("")
    if unit.key == "glue" and _viewed_dummies(unit):
        body.append(VIEW_HELPER)
        body.append("")
    if any(s.helper for p in unit.pauses for s in frame_slots(p, spec.kernels[p.kernel])):
        body.append(ADDRESS_HELPERS)
        body.append("")
        ranks = _section_helper_ranks(spec, unit)
        if ranks:
            body.append(_section_helpers(ranks))
            body.append("")
    body.append(SLOT_HELPERS)
    body.append("")
    body.append(f"end module {unit.module}")
    return "\n".join(header + body) + "\n"


def _unit_pbuf_indices(unit: Unit) -> list[tuple[str, str]]:
    if unit.pbuf_indices == "auto":
        return buffer_indices(unit.lines, 1, len(unit.lines))
    if isinstance(unit.pbuf_indices, Mapping):
        return sorted((str(k).lower(), str(v)) for k, v in unit.pbuf_indices.items())
    return []


def _resolve_indices(unit: Unit) -> str:
    lines = [f"  subroutine {unit.key}_resolve_indices()",
             "    ! the module's register/init indices, by the same field names (-1 where unregistered)",
             "    use physics_buffer, only: pbuf_get_index",
             "    integer :: ierr"]
    for name, fld in _unit_pbuf_indices(unit):
        lines.append(f"    {name} = pbuf_get_index('{fld}', ierr)")
    lines.append(f"  end subroutine {unit.key}_resolve_indices")
    return "\n".join(lines)


def _copied_dummies(unit: Unit) -> dict[str, str]:
    """Dummies the glue serves from a copy: `copy: <expression>` in the spec (read-only inputs
    whose storage has no target attribute, such as comsrf's module arrays)."""

    return {name: target.split(":", 1)[1].strip() for name, target in unit.dummies.items()
            if target.strip().lower().startswith("copy:")}


def _viewed_dummies(unit: Unit) -> dict[str, str]:
    """Dummies the glue serves as views of storage with no target attribute: `view: <expression>`
    (comsrf's module arrays, say).  The address is taken through a target dummy, the idiom of the
    handles' view routines; a contiguous actual is passed without a copy."""

    return {name: target.split(":", 1)[1].strip() for name, target in unit.dummies.items()
            if target.strip().lower().startswith("view:")}


def _view_shape(decl: Decl) -> str:
    dims = []
    for extent in _split_top(decl.dims) if decl.dims else []:
        if ":" in extent:
            lower, _, upper = extent.partition(":")
            extent = f"({upper})-({lower})+1"
        dims.append(extent.strip())
    return "(/ " + ", ".join(dims) + " /)"


VIEW_HELPER = """  subroutine glue_view(field, address)
    ! a TARGET dummy so c_loc is legal whatever the actual argument's attributes;
    ! a contiguous actual is passed without a copy
    real(r8), intent(inout), target :: field(*)
    type(c_ptr), intent(out) :: address
    address = c_loc(field(1))
  end subroutine glue_view"""


def _copy_state(unit: Unit) -> list[str]:
    rows = []
    for name in sorted(_copied_dummies(unit)):
        decl = unit.decls[name]
        rows.append(f"  {decl.kind}, save, target :: copy_{name}({decl.dims})")
    return rows


def _glue_enter(spec: Spec, unit: Unit) -> str:
    """`glue_enter(lchnk)`: the tphysbc dummies for this chunk, the carries, the entry preamble."""

    head = ["  subroutine glue_enter(lchnk_in)",
            "    ! tphysbc's dummies for this chunk, its carries as views, and the entry",
            "    ! statements every stage relies on (lchnk, ncol, nstep, ...)"]
    uses = ["    use cam_abortutils, only: endrun"] if (unit.carries or unit.records) else []
    interfaces = []
    if unit.carries:
        entry = unit.carries_entry
        interfaces.append(f"    interface\n      integer(c_int) function {entry}(lchnk, code, ptr, ndims, extents) &\n"
                          f"           bind(C, name='{entry}')\n        import :: c_int, c_int64_t, c_ptr\n"
                          "        integer(c_int), value, intent(in) :: lchnk, code\n        type(c_ptr), intent(out) :: ptr\n"
                          "        integer(c_int), intent(out) :: ndims\n        integer(c_int64_t), intent(out) :: extents(4)\n"
                          f"      end function {entry}\n    end interface")
    for name, entry in sorted(unit.records.items()):
        interfaces.append(f"    interface\n      integer(c_int) function {entry}(lchnk, ptr) bind(C, name='{entry}')\n"
                          f"        import :: c_int, c_ptr\n        integer(c_int), value, intent(in) :: lchnk\n"
                          f"        type(c_ptr), intent(out) :: ptr\n      end function {entry}\n    end interface")
    decls = ["    integer, intent(in) :: lchnk_in",
             "    type(c_ptr) :: address", "    integer(c_int) :: ndims, status", "    integer(c_int64_t) :: extents(4)"]
    body = []
    copied = _copied_dummies(unit)
    viewed = _viewed_dummies(unit)
    for name in unit.dummy_names:
        target = unit.dummies.get(name)
        if target is None:
            continue
        decl = unit.decls[name]
        if name in copied:
            body.append(f"    copy_{name} = {copied[name].replace('lchnk', 'lchnk_in')}")
            body.append(f"    {name} => copy_{name}")
        elif name in viewed:
            body.append(f"    call glue_view({viewed[name].replace('lchnk', 'lchnk_in')}, address)")
            body.append(f"    call c_f_pointer(address, {name}, {_view_shape(decl)})")
        elif decl.is_derived or decl.rank > 0 or decl.is_pointer:
            body.append(f"    {name} => {target.replace('lchnk', 'lchnk_in')}")
        else:
            body.append(f"    {name} = {target.replace('lchnk', 'lchnk_in')}")
    for name, code in sorted(unit.carries.items(), key=lambda kv: kv[1]):
        decl = unit.decls[name]
        body.append(f"    status = {unit.carries_entry}(int(lchnk_in, c_int), {code}_c_int, address, ndims, extents)")
        body.append(f"    if (status /= 0_c_int .or. ndims /= {decl.rank}_c_int) call endrun('{spec.runner_module}: carry {name} refused')")
        shape = ", ".join(f"int(extents({i + 1}))" for i in range(decl.rank))
        body.append(f"    call c_f_pointer(address, {name}, (/ {shape} /))")
    for name, entry in sorted(unit.records.items()):
        body.append(f"    status = {entry}(int(lchnk_in, c_int), address)")
        body.append(f"    if (status /= 0_c_int) call endrun('{spec.runner_module}: record {name} refused')")
        body.append(f"    call c_f_pointer(address, {name})")
    for name, target in sorted(unit.locals.items()):
        body.append(f"    {name} => {target.replace('lchnk', 'lchnk_in')}")
    body += _automatic_allocations(unit)
    for text in unit.preamble:
        body.append("    " + text)
    return "\n".join(head + uses + interfaces + decls + body + ["  end subroutine glue_enter"])


def _unit_call_binder(spec: Spec, unit: Unit, call: UnitCall) -> str:
    inside = call.statement.split("(", 1)[1].rsplit(")", 1)[0]
    return "\n".join([f"  subroutine {unit.key}_bind_{call.unit}()",
                      f"    ! {Path(unit.source).name}:{call.first}-{call.last}: the call's actuals bound to the unit's dummies",
                      f"    call {call.unit}_bind({inside})",
                      f"  end subroutine {unit.key}_bind_{call.unit}"])


# ---------------------------------------------------------------------------
# the runner: state machine and ABI
# ---------------------------------------------------------------------------


@dataclass
class State:
    name: str
    code: str          # the Fortran executed on entering this state (may set pc / return)


def _flatten(spec: Spec, unit: Unit, nodes: list[Node], states: list[State], after: str, counters: dict,
             targets: dict | None = None) -> str:
    """Return the pc that starts `nodes`; states appended; flow continues to `after`.

    `targets` names where a piece's flow statements go: ``cycle`` and ``exit`` to the
    enclosing skeleton loop's next / after states, ``return`` to the routine's end.
    """

    if not nodes:
        return after
    targets = targets or {"cycle": None, "exit": None, "return": "pc_chunk_end"}
    first_pc = None
    # build backwards so each state knows its successor
    next_pc = after
    for node in reversed(nodes):
        pc = _node_pc(spec, unit, node, states, next_pc, counters, targets)
        next_pc = pc
        first_pc = pc
    return first_pc


def _new_pc(counters: dict, base: str) -> str:
    counters[base] = counters.get(base, 0) + 1
    return f"pc_{base}_{counters[base]}"


def _node_pc(spec: Spec, unit: Unit, node: Node, states: list[State], after: str, counters: dict,
             targets: dict) -> str:
    if node.kind == "piece":
        pc = _new_pc(counters, node.name)
        _piece_text(unit, node)          # records the flow codes the piece can set
        if not node.flows:
            states.append(State(pc, f"call {node.name}()\n        pc = {after}"))
            return pc
        branches = []
        for code, verb in ((1, "cycle"), (2, "exit"), (3, "return")):
            if code not in node.flows:
                continue
            target = targets.get(verb)
            if target is None:
                raise SystemExit(f"{unit.key}: piece {node.first}-{node.last} has a `{verb}` outside its loops "
                                 f"but no skeleton loop encloses it")
            branches.append(f"        case ({code})\n          {unit.key}_flow = 0\n          pc = {target}")
        text = "\n".join(branches)
        states.append(State(pc, f"call {node.name}()\n        select case ({unit.key}_flow)\n{text}\n"
                                f"        case default\n          pc = {after}\n        end select"))
        return pc
    if node.kind == "unit":
        target = spec.units[node.call.unit]
        pc = _new_pc(counters, f"enter_{target.key}")
        inner_after = _new_pc(counters, f"leave_{target.key}")
        first = _flatten(spec, target, target.body, states, inner_after, counters,
                         {"cycle": None, "exit": None, "return": inner_after})
        states.append(State(inner_after, f"pc = {after}"))
        states.append(State(pc, f"call {unit.key}_bind_{target.key}()\n        pc = {first}"))
        return pc
    if node.kind == "pause":
        pause = node.pause
        at, resumed = pause.pc_at, pause.pc_after
        kernel_id = list(spec.kernels).index(pause.kernel) + 1
        pc = _new_pc(counters, f"before_{pause.kernel}")
        states.append(State(pc, f"if (replace({kernel_id})) then\n          token = token + 1_c_int\n          pc = {at}\n"
                                f"          event = ev_needs_kernel\n          return\n        end if\n"
                                f"        call {pause.tag}_original()\n        pc = {resumed}"))
        states.append(State(at, f"last_error = '{spec.prefix} is paused; only resume continues it'\n        event = ev_error\n        return"))
        states.append(State(resumed, f"call_index = call_index + 1_c_int\n        pc = {after}"))
        return pc
    line = strip_comment(unit.lines[node.line - 1]).strip()
    if node.kind == "if":
        condition = re.match(r"^if\s*\((.*)\)\s*then$", line, re.I)
        if not condition:
            raise SystemExit(f"{unit.key}: line {node.line} is not `if (...) then`: {line}")
        then_pc = _flatten(spec, unit, node.children, states, after, counters, targets)
        else_pc = _flatten(spec, unit, node.orelse, states, after, counters, targets)
        chain = [f"if ({condition.group(1)}) then\n          pc = {then_pc}"]
        for line, body in node.elifs:
            text = strip_comment(unit.lines[line - 1]).strip()
            elif_condition = re.match(r"^else\s*if\s*\((.*)\)\s*then$", text, re.I)
            if not elif_condition:
                raise SystemExit(f"{unit.key}: line {line} is not `else if (...) then`: {text}")
            elif_pc = _flatten(spec, unit, body, states, after, counters, targets)
            chain.append(f"        else if ({elif_condition.group(1)}) then\n          pc = {elif_pc}")
        pc = _new_pc(counters, "if")
        states.append(State(pc, "\n".join(chain) + f"\n        else\n          pc = {else_pc}\n        end if"))
        return pc
    if node.kind == "do":
        header = re.match(r"^do\s+(\w+)\s*=\s*(.+?)\s*,\s*(.+?)(?:\s*,\s*(.+))?$", line, re.I)
        if not header:
            raise SystemExit(f"{unit.key}: line {node.line} is not a counted `do`: {line}")
        var, start, stop, step = header.groups()
        step = step or "1"
        check = _new_pc(counters, f"do_{var}")
        body_pc = _flatten(spec, unit, node.children, states, f"{check}_next", counters,
                           {**targets, "cycle": f"{check}_next", "exit": after})
        states.append(State(f"{check}_next", f"{var} = {var} + ({step})\n        pc = {check}"))
        test = f"{var} > {stop}" if not step.strip().startswith("-") else f"{var} < {stop}"
        states.append(State(check, f"if ({test}) then\n          pc = {after}\n        else\n          pc = {body_pc}\n        end if"))
        pc = _new_pc(counters, f"do_{var}_init")
        states.append(State(pc, f"{var} = {start}\n        pc = {check}"))
        return pc
    if node.kind == "select":
        selector = re.match(r"^select\s*case\s*\((.*)\)\s*$", line, re.I)
        if not selector:
            raise SystemExit(f"{unit.key}: line {node.line} is not `select case (...)`: {line}")
        pc = _new_pc(counters, "select")
        branches = []
        for value, body in node.cases.items():
            case_pc = _flatten(spec, unit, body, states, after, counters, targets)
            branches.append(f"        case ({value})\n          pc = {case_pc}")
        refused = "\n".join(branches)
        default = (f"        case default\n          last_error = '{spec.prefix}: {selector.group(1)} takes a case this runner "
                   f"was not written for'\n          event = ev_error\n          return")
        states.append(State(pc, f"select case ({selector.group(1)})\n{refused}\n{default}\n        end select"))
        return pc
    raise SystemExit(f"{unit.key}: unknown node kind {node.kind}")


def render_runner(spec: Spec) -> str:
    glue = spec.units["glue"]
    states: list[State] = []
    counters: dict = {}
    first_pc = _flatten(spec, glue, glue.body, states, "pc_chunk_end", counters)
    kernels = list(spec.kernels)
    kernel_constants = "\n".join(f"  integer(c_int), parameter :: kernel_{k} = {i}_c_int" for i, k in enumerate(kernels, start=1))
    pause_states = [s for s in states]
    pc_names = ["pc_idle", "pc_chunk_begin", "pc_chunk_end"] + [s.name for s in pause_states]
    pc_params = "\n".join(f"  integer, parameter :: {name} = {i}" for i, name in enumerate(pc_names))
    all_pauses = [p for u in spec.units.values() for p in u.pauses]
    frame_slots_max = max(len(frame_slots(p, spec.kernels[p.kernel])) for p in all_pauses) if all_pauses else 1
    unit_uses = []
    for unit in spec.units.values():
        names = [n.name for n in unit.pieces] + [f"{p.tag}_frame" for p in unit.pauses] + [f"{p.tag}_original" for p in unit.pauses]
        names += [f"{unit.key}_bind_{c.unit}" for c in unit.unit_calls]
        if unit.key == "glue":
            names.append("glue_enter")
            if unit.postamble:
                names.append("glue_leave")
        if any(_piece_flows(unit, n.first, n.last) for n in unit.pieces):
            names.append(f"{unit.key}_flow => flow")
        if unit.pbuf_indices:
            names.append(f"{unit.key}_resolve_indices")
        if unit.getopts:
            names.append(f"{unit.key}_configure")
        # the skeleton's conditions and loop variables live in the unit, as do
        # the indices and options the refusals test
        names += sorted(_skeleton_names(unit) | set(unit.getopts) | _refusal_names(spec, unit))
        unit_uses.append(f"  use {unit.module}, only: " + ", &\n       ".join(", ".join(names[i:i + 6]) for i in range(0, len(names), 6)))
    unit_uses += ["  " + use for use in spec.runner_uses]
    getopts = "".join(f"    call {u.key}_configure()\n" for u in spec.units.values() if u.getopts)
    getopt_decls = ""
    refusals = "".join(f"    if ({r['when']}) then\n      last_error = '{spec.prefix}: {r['message']}'; return\n    end if\n" for r in spec.refuse)
    resolves = "".join(f"    call {u.key}_resolve_indices()\n" for u in spec.units.values() if u.pbuf_indices)
    frame_cases = "\n".join(f"    case ({p.pc_at})\n      call {p.tag}_frame(ptrs, ndims, shapes, dtypes, intents, ncol_out)\n"
                            f"      kernel = kernel_{p.kernel}" for p in all_pauses)
    resume_cases = "\n".join(f"    case ({p.pc_at})\n      if (kernel /= kernel_{p.kernel}) then\n"
                             f"        last_error = '{spec.prefix} is paused on {p.kernel}, not on the kernel resumed'; status = 3_c_int; return\n"
                             f"      end if\n      pc = {p.pc_after}" for p in all_pauses)
    original_cases = "\n".join(f"    case ({p.pc_at})\n      if (kernel /= kernel_{p.kernel}) then\n"
                               f"        last_error = '{spec.prefix} is paused on {p.kernel}, not on the kernel asked for'; status = 3_c_int; return\n"
                               f"      end if\n      call {p.tag}_original()" for p in all_pauses)
    paused_pcs = ", ".join(p.pc_at for p in all_pauses) or "-1"
    advance_cases = "\n".join(f"      case ({s.name})\n        {s.code}" for s in states)
    ep = spec.entry_prefix
    return f'''! The segment runner for {spec.stage}: the original Fortran, pausable at
! {", ".join(kernels)}.
!
! GENERATED by tools/generate_pi_cam_pausable_runners.py from
! native/pi_cam/pausable/{spec.path.name}.  Do not edit by hand; edit the spec.
!
! The runner is a state machine over the units' pieces: start() runs from the
! top of the action for every chunk of this rank and returns either DONE or,
! when a replaced kernel's call is reached, NEEDS_PYTHON_KERNEL with the
! program counter parked before the call; frame() describes the call's
! arguments where they live; resume() continues past the call; original()
! runs the very call on the paused frame.  Python makes every call; Fortran
! never calls Python.
module {spec.runner_module}
  use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t, c_double, c_ptr, c_char, c_null_ptr, c_loc
  use ppgrid, only: begchunk, endchunk
  use {spec.hosts}, only: stage_hosts_ok
{chr(10).join(unit_uses)}
  implicit none
  private

  integer(c_int), parameter :: ev_done = 0_c_int, ev_needs_kernel = 1_c_int, ev_error = 2_c_int
{pc_params}
{kernel_constants}
  integer, parameter :: nkernels = {len(kernels)}
  integer, parameter, public :: frame_slots = {frame_slots_max}
  integer, parameter :: context_id = 1

  logical, save :: created = .false.
  integer, save :: pc = pc_idle
  integer, save :: lchnk = 0
  integer(c_int), save :: token = 0_c_int, call_index = 0_c_int
  logical, save :: replace(nkernels) = .false.
  character(len=256), save :: last_error = ' '
{getopt_decls}
contains

  ! ------------------------------------------------------------------ !
  ! The ABI Python drives
  ! ------------------------------------------------------------------ !

  integer(c_int) function {ep}_create_v1(context) bind(C, name='{ep}_create_v1') result(status)
    integer(c_int), intent(out) :: context
    context = 0_c_int
    status = 1_c_int
    if (created) then
      last_error = '{spec.prefix} context already exists'; return
    end if
    if (.not. stage_hosts_ok(int(begchunk, c_int))) then
      last_error = '{spec.prefix}: the stage hosts are not bound (pycam_stagehost_bind_v1 first)'; return
    end if
{getopts}{resolves}    status = 2_c_int
{refusals}    created = .true.
    pc = pc_idle
    context = int(context_id, c_int)
    status = 0_c_int
  end function {ep}_create_v1

  integer(c_int) function {ep}_start_v1(context, count, mask, event) bind(C, name='{ep}_start_v1') result(status)
    integer(c_int), value, intent(in) :: context, count
    integer(c_int), intent(in) :: mask(count)
    integer(c_int), intent(out) :: event
    integer :: k
    event = ev_error
    status = 1_c_int
    if (.not. created .or. context /= context_id) then
      last_error = 'no {spec.prefix} context'; return
    end if
    if (pc /= pc_idle) then
      last_error = '{spec.prefix} is not idle; resume or reset it first'; status = 2_c_int; return
    end if
    if (count < nkernels) then
      last_error = 'replacement mask is too short'; status = 3_c_int; return
    end if
    do k = 1, nkernels
      replace(k) = mask(k) /= 0_c_int
    end do
    call_index = 0_c_int
    lchnk = begchunk
    pc = pc_chunk_begin
    call advance(event)
    status = 0_c_int
  end function {ep}_start_v1

  integer(c_int) function {ep}_resume_v1(context, kernel, token_in, event) bind(C, name='{ep}_resume_v1') result(status)
    integer(c_int), value, intent(in) :: context, kernel, token_in
    integer(c_int), intent(out) :: event
    event = ev_error
    status = 1_c_int
    if (.not. created .or. context /= context_id) then
      last_error = 'no {spec.prefix} context'; return
    end if
    if (token_in /= token) then
      last_error = 'stale resume: the frame token does not match the pause'; status = 4_c_int; return
    end if
    select case (pc)
{resume_cases}
    case default
      last_error = '{spec.prefix} is not paused'; status = 2_c_int; return
    end select
    token = token + 1_c_int
    call advance(event)
    status = 0_c_int
  end function {ep}_resume_v1

  integer(c_int) function {ep}_frame_v1(context, kernel, index_out, lchnk_out, ncol_out, &
       substep_out, token_out, count, ptrs, ndims, shapes, dtypes, intents) bind(C, name='{ep}_frame_v1') result(status)
    integer(c_int), value, intent(in) :: context, count
    integer(c_int), intent(out) :: kernel, index_out, lchnk_out, ncol_out, substep_out, token_out
    type(c_ptr), intent(out) :: ptrs(count)
    integer(c_int), intent(out) :: ndims(count), dtypes(count), intents(count)
    integer(c_int64_t), intent(out) :: shapes({FRAME_MAX_RANK}, count)
    kernel = 0_c_int; index_out = 0_c_int; lchnk_out = 0_c_int; ncol_out = 0_c_int
    substep_out = 0_c_int; token_out = 0_c_int
    status = 1_c_int
    if (.not. created .or. context /= context_id) then
      last_error = 'no {spec.prefix} context'; return
    end if
    if (count < frame_slots) then
      last_error = 'frame table is too short'; status = 3_c_int; return
    end if
    select case (pc)
{frame_cases}
    case default
      last_error = '{spec.prefix} is not paused; there is no frame'; status = 2_c_int; return
    end select
    index_out = call_index
    lchnk_out = int(lchnk, c_int)
    substep_out = 1_c_int
    token_out = token
    status = 0_c_int
  end function {ep}_frame_v1

  integer(c_int) function {ep}_original_v1(context, kernel) bind(C, name='{ep}_original_v1') result(status)
    ! Run the original call on the paused frame: the validation gate's
    ! replacement, exercising the pause, the frame and the resume.
    integer(c_int), value, intent(in) :: context, kernel
    status = 1_c_int
    if (.not. created .or. context /= context_id) then
      last_error = 'no {spec.prefix} context'; return
    end if
    select case (pc)
{original_cases}
    case default
      last_error = '{spec.prefix} is not paused; there is nothing to run'; status = 2_c_int; return
    end select
    status = 0_c_int
  end function {ep}_original_v1

  integer(c_int) function {ep}_error_v1(context, buffer, length) bind(C, name='{ep}_error_v1') result(status)
    integer(c_int), value, intent(in) :: context, length
    character(kind=c_char), intent(out) :: buffer(length)
    integer :: n, i
    n = min(length - 1, len_trim(last_error))
    do i = 1, n
      buffer(i) = last_error(i:i)
    end do
    buffer(n + 1) = c_char_'\\0'
    status = 0_c_int
    if (context /= context_id) status = 1_c_int
  end function {ep}_error_v1

  integer(c_int) function {ep}_reset_v1(context) bind(C, name='{ep}_reset_v1') result(status)
    integer(c_int), value, intent(in) :: context
    status = 1_c_int
    if (.not. created .or. context /= context_id) return
    pc = pc_idle
    status = 0_c_int
  end function {ep}_reset_v1

  integer(c_int) function {ep}_destroy_v1(context) bind(C, name='{ep}_destroy_v1') result(status)
    integer(c_int), value, intent(in) :: context
    status = 1_c_int
    if (.not. created .or. context /= context_id) return
    created = .false.
    pc = pc_idle
    status = 0_c_int
  end function {ep}_destroy_v1

  ! ------------------------------------------------------------------ !
  ! The state machine
  ! ------------------------------------------------------------------ !

  subroutine advance(event)
    integer(c_int), intent(out) :: event
    do
      select case (pc)
      case (pc_chunk_begin)
        if (lchnk > endchunk) then
          pc = pc_idle
          event = ev_done
          return
        end if
        call glue_enter(lchnk)
        pc = {first_pc}
      case (pc_chunk_end)
{"        call glue_leave()" + chr(10) if glue.postamble else ""}        lchnk = lchnk + 1
        pc = pc_chunk_begin
{advance_cases}
      case default
        last_error = '{spec.prefix} program counter is corrupt'
        event = ev_error
        return
      end select
    end do
  end subroutine advance

end module {spec.runner_module}
'''


def _refusal_names(spec: Spec, unit: Unit) -> set[str]:
    """Names of the unit the spec's refusals test: its options and its resolved indices."""

    names: set[str] = set()
    for refusal in spec.refuse:
        names |= identifiers(str(refusal["when"]))
    names -= KEYWORDS
    indices = {name for name, _ in _unit_pbuf_indices(unit)}
    return {n for n in names if n in unit.getopts or n in indices or n in unit.decls}


def _skeleton_names(unit: Unit) -> set[str]:
    """Names the runner's skeleton statements reference, which must be public in the unit."""

    names: set[str] = set()
    for node in _walk(unit.body):
        if node.kind in ("if", "do", "select"):
            names |= identifiers(unit.lines[node.line - 1])
            for line, _ in node.elifs:
                names |= identifiers(unit.lines[line - 1])
    names -= KEYWORDS
    known = {n for n in names if n in unit.decls or n in unit.dummy_names or n in unit.carries
             or n in unit.records or n in unit.getopts}
    return known


# ---------------------------------------------------------------------------
# frame descriptors for the Python side
# ---------------------------------------------------------------------------


def frame_descriptors(spec: Spec) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for unit in spec.units.values():
        for pause in unit.pauses:
            out[pause.kernel] = [{
                "name": s.dummy, "actual": s.actual, "rank": s.rank, "dtype": "float64" if s.dtype == 1 else "int32",
                "intent": s.intent, "kind": s.kind,
            } for s in frame_slots(pause, spec.kernels[pause.kernel])]
    return out


def spec_digest(spec: Spec) -> dict[str, str]:
    """The pinned ranges every unit and kernel transcribes, hashed."""

    out = {}
    for unit in spec.units.values():
        out[f"{unit.key}:{Path(unit.source).name}:{unit.body_range[0]}-{unit.body_range[1]}"] = \
            range_digest(unit.lines, *unit.body_range)
        out[f"{unit.key}:{Path(unit.source).name}:declarations {unit.declarations[0]}-{unit.declarations[1]}"] = \
            range_digest(unit.lines, *unit.declarations)
        for first, last in unit.module_private + unit.helpers + unit.elided:
            out[f"{unit.key}:{Path(unit.source).name}:verbatim {first}-{last}"] = range_digest(unit.lines, first, last)
    for kernel in spec.kernels.values():
        out[f"{kernel.name}:{Path(kernel.source).name}:{kernel.body_range[0]}-{kernel.body_range[1]}"] = \
            range_digest(kernel.lines, *kernel.body_range)
    return out


def render_all(spec: Spec) -> dict[Path, str]:
    """path -> text for every module the spec produces."""

    files = {}
    for unit in spec.units.values():
        files[SUPPORT / f"{unit.module}.F90"] = render_unit(spec, unit)
    files[SUPPORT / f"{spec.runner_module}.F90"] = render_runner(spec)
    return files


__all__ = ["FRAMES", "REPO", "SPECS", "SUPPORT", "Spec", "Unit", "Kernel", "coverage_gaps", "frame_descriptors",
           "frame_slots", "load_spec", "parse_declarations", "range_digest", "render_all", "render_runner",
           "render_unit", "signature", "spec_digest", "statements"]
