"""The kernel-API closure record: what the PI-atm physics step can reach.

Phase A of opening every numerical kernel to Python: a machine-readable list
of every procedure the configured physics step reaches from its two drivers,
each call site with its preprocessor condition and namelist guards, each
procedure classified by reviewable rules, and every unresolved reference
named rather than dropped.  The record is generated from the pinned source
with the configuration's real macros and build search path; ``--check``
proves the committed record still matches its inputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from .call_tree import CallSite, CallTree, GuardPart, ProcedureScope, expand_templates, select_sources
from .errors import PICAMConfigurationError
from .plan import PICAMStepPlan

CLOSURE_SCHEMA_VERSION = 1
DEFAULT_RULES = Path("native/pi_cam/kernel_api_closure_rules.yaml")
DEFAULT_RECORD = Path("validation/pi_cam_kernel_api_closure.json")
SOURCE_ROOT = Path("external/iCESM1.3.1_fzhu")
GENERATED_DIR = Path("build/pi_cam_kernel_api_closure/generated")
GENERATOR_SOURCES = ("call_tree.py", "kernel_api_closure.py")
PATCHED_PHYSPKG = Path("build/iCESM1.3.1_PI_cam_only/components/cam/src/physics/cam/physpkg.F90")
# freeCAM's patched physpkg dispatches tphysbc stages 1..11 as plan ids 421..431
# and tphysac stages 1..10 as plan ids 402..411 (atm_comp_mct: action_id-420 / -401).
STAGE_BASES = {"tphysbc": 420, "tphysac": 401}
_STAGE_LABEL = re.compile(r"^\s*(\d+)00\s+continue\b", re.IGNORECASE)
_STAGE_END = re.compile(r"^\s*if\s*\(\s*stage\s*==\s*(\d+)\s*\)", re.IGNORECASE)
CATEGORIES = (
    "numeric_kernel",
    "process_control",
    "internal_service",
    "lifecycle",
    "shared_numeric",
    "out_of_scope",
)


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CategoryRule:
    id: str
    category: str
    name_regex: str | None = None
    module_regex: str | None = None
    source_regex: str | None = None
    qualified_regex: str | None = None
    kind: str | None = None
    note: str | None = None

    def matches(self, scope: ProcedureScope) -> bool:
        checks = (
            (self.name_regex, scope.name),
            (self.module_regex, scope.module or ""),
            (self.source_regex, scope.source),
            (self.qualified_regex, scope.qualified),
        )
        for pattern, value in checks:
            if pattern is not None and re.search(pattern, value, re.IGNORECASE) is None:
                return False
        if self.kind is not None and self.kind != scope.kind:
            return False
        return True


@dataclass(frozen=True, slots=True)
class ClosureRules:
    source_directories: tuple[str, ...]
    templates: tuple[str, ...]
    genf90: str
    macros: tuple[str, ...]
    step_roots: tuple[str, ...]
    init_roots: tuple[str, ...]
    selectors: tuple[str, ...]
    build_options: tuple[str, ...]
    option_functions: Mapping[str, str]
    constants: Mapping[str, object]
    library_names: tuple[str, ...]
    action_procedures: Mapping[str, tuple[str, ...]]
    categories: tuple[CategoryRule, ...]
    default_category: str
    sha256: str
    text: str


_RULE_KEYS = {
    "schema_version",
    "source_directories",
    "templates",
    "genf90",
    "macros",
    "step_roots",
    "init_roots",
    "selectors",
    "build_options",
    "option_functions",
    "constants",
    "library_names",
    "action_procedures",
    "categories",
    "default_category",
}
_CATEGORY_KEYS = {"id", "category", "match", "note"}
_MATCH_KEYS = {"name_regex", "module_regex", "source_regex", "qualified_regex", "kind"}


def load_closure_rules(path: str | Path) -> ClosureRules:
    text = Path(path).read_text()
    payload = yaml.safe_load(text) or {}
    unknown = set(payload) - _RULE_KEYS
    if unknown:
        raise PICAMConfigurationError(f"closure rules have unknown keys {sorted(unknown)}")
    if int(payload.get("schema_version", 0)) != CLOSURE_SCHEMA_VERSION:
        raise PICAMConfigurationError("closure rules schema_version mismatch")
    categories: list[CategoryRule] = []
    for raw in payload.get("categories", ()):
        unknown = set(raw) - _CATEGORY_KEYS
        if unknown:
            raise PICAMConfigurationError(f"category rule {raw.get('id')!r} has unknown keys {sorted(unknown)}")
        match = dict(raw.get("match") or {})
        unknown = set(match) - _MATCH_KEYS
        if unknown:
            raise PICAMConfigurationError(f"category rule {raw.get('id')!r} has unknown match keys {sorted(unknown)}")
        if raw.get("category") not in CATEGORIES:
            raise PICAMConfigurationError(f"category rule {raw.get('id')!r} names unknown category {raw.get('category')!r}")
        for pattern in match.values():
            if pattern is not None and not isinstance(pattern, str):
                raise PICAMConfigurationError(f"category rule {raw.get('id')!r} patterns must be strings")
        categories.append(CategoryRule(id=str(raw["id"]), category=str(raw["category"]), note=raw.get("note"), **match))
    default = str(payload.get("default_category", "numeric_kernel"))
    if default not in CATEGORIES:
        raise PICAMConfigurationError(f"default_category {default!r} is not a known category")
    return ClosureRules(
        source_directories=tuple(str(item) for item in payload.get("source_directories", ())),
        templates=tuple(str(item) for item in payload.get("templates", ())),
        genf90=str(payload.get("genf90", "cime/src/externals/genf90/genf90.pl")),
        macros=tuple(str(item) for item in payload.get("macros", ())),
        step_roots=tuple(str(item) for item in payload.get("step_roots", ())),
        init_roots=tuple(str(item) for item in payload.get("init_roots", ())),
        selectors=tuple(str(item) for item in payload.get("selectors", ())),
        build_options=tuple(str(item) for item in payload.get("build_options", ())),
        option_functions=dict(payload.get("option_functions") or {}),
        constants=dict(payload.get("constants") or {}),
        library_names=tuple(str(item) for item in payload.get("library_names", ())),
        action_procedures={
            str(action): tuple(str(item) for item in names or ())
            for action, names in (payload.get("action_procedures") or {}).items()
        },
        categories=tuple(categories),
        default_category=default,
        sha256=sha256(text.encode()).hexdigest(),
        text=text,
    )


# --------------------------------------------------------------------------
# configuration values
# --------------------------------------------------------------------------

_NAMELIST_LINE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*,?\s*$")
_CONFIG_ENTRY = re.compile(r'<entry id="([^"]+)" value="([^"]*)"')


def namelist_values(text: str, names: Iterable[str]) -> dict[str, object]:
    """Scalar values of ``names`` from a Fortran namelist file."""

    wanted = {name.lower() for name in names}
    values: dict[str, object] = {}
    for line in text.splitlines():
        match = _NAMELIST_LINE.match(line)
        if match is None:
            continue
        name = match.group(1).lower()
        if name in wanted and name not in values:
            values[name] = fortran_literal(match.group(2))
    return values


def config_cache_values(text: str, names: Iterable[str]) -> dict[str, str]:
    wanted = {name.lower() for name in names}
    return {key: value for key, value in _CONFIG_ENTRY.findall(text) if key.lower() in wanted}


def fortran_literal(text: str) -> object:
    """A Fortran scalar literal as a Python value; anything else stays text."""

    value = text.strip()
    if value.startswith(("'", '"')) and value.endswith(value[0]) and len(value) >= 2:
        return value[1:-1].replace(value[0] * 2, value[0])
    lower = value.lower()
    if lower in (".true.", ".t.", "t", ".true"):
        return True
    if lower in (".false.", ".f.", "f", ".false"):
        return False
    number = re.sub(r"_\w+$", "", lower).replace("d", "e")
    try:
        if re.fullmatch(r"[+-]?\d+", number):
            return int(number)
        return float(number)
    except ValueError:
        return value


# --------------------------------------------------------------------------
# guard evaluation
# --------------------------------------------------------------------------

_TOKEN = re.compile(
    r"""\s*(?:
    (?P<string>'(?:[^']|'')*'|"(?:[^"]|"")*")|
    (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?(?:_\w+)?)|
    (?P<logop>\.(?:and|or|not|eqv|neqv|true|false|eq|ne|lt|le|gt|ge)\.)|
    (?P<op>==|/=|<=|>=|//|<|>|\(|\)|,|\+|-|\*|/|%)|
    (?P<name>[A-Za-z_]\w*)
    )""",
    re.VERBOSE | re.IGNORECASE,
)


class ConditionEvaluator:
    """Tri-state evaluation of Fortran conditions against known values.

    ``values`` are namelist selectors, build options and reviewed constants;
    ``functions`` answer calls such as ``cam_physpkg_is('cam5')``.  Anything
    the evaluator does not know evaluates to ``None`` and the guard stays
    undecided -- it never guesses.
    """

    def __init__(
        self,
        values: Mapping[str, object],
        functions: Mapping[str, Callable[[list[object]], object | None]] | None = None,
    ) -> None:
        self.values = {str(key).lower(): value for key, value in values.items()}
        self.functions = {str(key).lower(): fn for key, fn in (functions or {}).items()}
        self.functions.setdefault("trim", lambda args: args[0].rstrip() if isinstance(args[0], str) else None)
        self.functions.setdefault("adjustl", lambda args: args[0].lstrip() if isinstance(args[0], str) else None)
        self.functions.setdefault("len_trim", lambda args: len(args[0].rstrip()) if isinstance(args[0], str) else None)

    # -- public --------------------------------------------------------------

    def value(self, text: str) -> object | None:
        tokens = self._tokens(text)
        if tokens is None:
            return None
        parser = _Parser(tokens, self)
        try:
            result = parser.expression()
        except _Unparsable:
            return None
        if not parser.done():
            return None
        return result

    def truth(self, text: str) -> bool | None:
        result = self.value(text)
        return result if isinstance(result, bool) else None

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _tokens(text: str) -> list[tuple[str, str]] | None:
        tokens: list[tuple[str, str]] = []
        position = 0
        stripped = text.strip()
        while position < len(stripped):
            match = _TOKEN.match(stripped, position)
            if match is None or match.end() == position:
                return None
            position = match.end()
            kind = match.lastgroup
            if kind is None:
                continue
            tokens.append((kind, match.group(kind)))
        return tokens


class _Unparsable(Exception):
    pass


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]], evaluator: ConditionEvaluator) -> None:
        self.tokens = tokens
        self.index = 0
        self.evaluator = evaluator

    def done(self) -> bool:
        return self.index >= len(self.tokens)

    def _peek(self) -> tuple[str, str] | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _take(self, kind: str | None = None, text: str | None = None) -> tuple[str, str]:
        token = self._peek()
        if token is None:
            raise _Unparsable
        if kind is not None and token[0] != kind:
            raise _Unparsable
        if text is not None and token[1].lower() != text:
            raise _Unparsable
        self.index += 1
        return token

    def _accept(self, kind: str, *texts: str) -> str | None:
        token = self._peek()
        if token is not None and token[0] == kind and token[1].lower() in texts:
            self.index += 1
            return token[1].lower()
        return None

    # grammar, lowest precedence first
    def expression(self) -> object | None:
        left = self._or()
        while (op := self._accept("logop", ".eqv.", ".neqv.")) is not None:
            right = self._or()
            if isinstance(left, bool) and isinstance(right, bool):
                left = (left == right) if op == ".eqv." else (left != right)
            else:
                left = None
        return left

    def _or(self) -> object | None:
        values = [self._and()]
        while self._accept("logop", ".or.") is not None:
            values.append(self._and())
        if len(values) == 1:
            return values[0]
        if any(item is True for item in values):
            return True
        if all(item is False for item in values):
            return False
        return None

    def _and(self) -> object | None:
        values = [self._not()]
        while self._accept("logop", ".and.") is not None:
            values.append(self._not())
        if len(values) == 1:
            return values[0]
        if any(item is False for item in values):
            return False
        if all(item is True for item in values):
            return True
        return None

    def _not(self) -> object | None:
        if self._accept("logop", ".not.") is not None:
            value = self._not()
            return (not value) if isinstance(value, bool) else None
        return self._comparison()

    def _comparison(self) -> object | None:
        left = self._additive()
        token = self._peek()
        operators = {
            "==": "eq", ".eq.": "eq", "/=": "ne", ".ne.": "ne",
            "<": "lt", ".lt.": "lt", "<=": "le", ".le.": "le",
            ">": "gt", ".gt.": "gt", ">=": "ge", ".ge.": "ge",
        }
        if token is not None and token[0] in ("op", "logop") and token[1].lower() in operators:
            self.index += 1
            right = self._additive()
            return _compare(operators[token[1].lower()], left, right)
        return left

    def _additive(self) -> object | None:
        left = self._term()
        while (op := self._accept("op", "+", "-", "//")) is not None:
            right = self._term()
            left = _arith(op, left, right)
        return left

    def _term(self) -> object | None:
        left = self._unary()
        while (op := self._accept("op", "*", "/")) is not None:
            right = self._unary()
            left = _arith(op, left, right)
        return left

    def _unary(self) -> object | None:
        if (op := self._accept("op", "+", "-")) is not None:
            value = self._unary()
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return -value if op == "-" else value
            return None
        return self._primary()

    def _primary(self) -> object | None:
        token = self._peek()
        if token is None:
            raise _Unparsable
        kind, text = token
        if kind == "op" and text == "(":
            self.index += 1
            value = self.expression()
            self._take("op", ")")
            return value
        if kind == "string":
            self.index += 1
            return fortran_literal(text)
        if kind == "number":
            self.index += 1
            return fortran_literal(text)
        if kind == "logop" and text.lower() in (".true.", ".false."):
            self.index += 1
            return text.lower() == ".true."
        if kind == "name":
            self.index += 1
            name = text.lower()
            while self._accept("op", "%") is not None:
                component = self._take("name")[1].lower()
                name = f"{name}%{component}"
            if self._accept("op", "(") is not None:
                arguments: list[object | None] = []
                if self._accept("op", ")") is None:
                    arguments.append(self.expression())
                    while self._accept("op", ",") is not None:
                        arguments.append(self.expression())
                    self._take("op", ")")
                function = self.evaluator.functions.get(name)
                if function is None:
                    return None
                if any(argument is None for argument in arguments):
                    return None
                try:
                    return function(arguments)
                except Exception:
                    return None
            return self.evaluator.values.get(name)
        raise _Unparsable


def _compare(operator: str, left: object | None, right: object | None) -> bool | None:
    if left is None or right is None:
        return None
    if isinstance(left, str) and isinstance(right, str):
        left, right = left.rstrip(), right.rstrip()
    elif isinstance(left, bool) or isinstance(right, bool):
        return None
    elif isinstance(left, str) or isinstance(right, str):
        return None
    try:
        return {
            "eq": left == right, "ne": left != right, "lt": left < right,
            "le": left <= right, "gt": left > right, "ge": left >= right,
        }[operator]
    except TypeError:
        return None


def _arith(operator: str, left: object | None, right: object | None) -> object | None:
    if left is None or right is None:
        return None
    if operator == "//":
        return left + right if isinstance(left, str) and isinstance(right, str) else None
    if isinstance(left, bool) or isinstance(right, bool) or isinstance(left, str) or isinstance(right, str):
        return None
    try:
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        return left / right
    except (TypeError, ZeroDivisionError):
        return None


def guard_state(guards: Sequence[GuardPart], evaluator: ConditionEvaluator) -> str | None:
    """``enabled``, ``disabled`` or ``undecided`` for a call site's guards; ``None`` when unguarded."""

    if not guards:
        return None
    outcomes: list[bool | None] = []
    for part in guards:
        if part.kind == "if":
            result = evaluator.truth(part.text)
        else:
            selector = evaluator.value(part.text)
            if selector is None:
                result = None
            else:
                matches: list[bool | None] = []
                for value in part.values:
                    if ":" in value and not value.startswith(("'", '"')):
                        matches.append(None)      # a range: not decided here
                        continue
                    matches.append(_compare("eq", selector, fortran_literal(value)))
                if any(item is True for item in matches):
                    result = True
                elif all(item is False for item in matches):
                    result = False
                else:
                    result = None
        if result is not None and part.negate:
            result = not result
        outcomes.append(result)
    if any(item is False for item in outcomes):
        return "disabled"
    if all(item is True for item in outcomes):
        return "enabled"
    return "undecided"


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

@dataclass(slots=True)
class ClosureInputs:
    project_root: Path
    rules_path: Path
    rules: ClosureRules
    reference_run: Path | None = None
    reference_case: Path | None = None
    native_manifest: Path | None = None
    previous_record: Mapping[str, Any] | None = None
    patched_physpkg: Path | None = None

    @property
    def source_root(self) -> Path:
        return self.project_root / SOURCE_ROOT


def source_revision(source_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def gather_configuration(inputs: ClosureInputs) -> dict[str, Any]:
    """Selector and build-option values, from the reference case when present."""

    rules = inputs.rules
    previous = (inputs.previous_record or {}).get("configuration", {})
    selectors: dict[str, object] | None = None
    options: dict[str, str] | None = None
    sources: dict[str, str] = {}
    atm_in = inputs.reference_run / "atm_in" if inputs.reference_run is not None else None
    if atm_in is not None and atm_in.is_file():
        selectors = namelist_values(atm_in.read_text(errors="replace"), rules.selectors)
        sources["selectors"] = "reference run atm_in"
    cache = (
        inputs.reference_case / "Buildconf" / "camconf" / "config_cache.xml"
        if inputs.reference_case is not None
        else None
    )
    if cache is not None and cache.is_file():
        options = config_cache_values(cache.read_text(errors="replace"), rules.build_options)
        sources["build_options"] = "reference case config_cache.xml"
    if selectors is None:
        if "selectors" not in previous:
            raise PICAMConfigurationError(
                "the reference run's atm_in is not available and no previous record supplies the selectors"
            )
        selectors = dict(previous["selectors"])
        sources["selectors"] = "previous record"
    if options is None:
        if "build_options" not in previous:
            raise PICAMConfigurationError(
                "the reference case's config_cache.xml is not available and no previous record supplies the build options"
            )
        options = dict(previous["build_options"])
        sources["build_options"] = "previous record"
    missing = [name for name in rules.selectors if name not in selectors]
    filepath_check = _filepath_check(inputs)
    macro_check = _macro_check(inputs)
    return {
        "selectors": {name: selectors[name] for name in sorted(selectors)},
        "missing_selectors": missing,
        "build_options": {name: options[name] for name in sorted(options)},
        "constants": dict(rules.constants),
        "sources": sources,
        "filepath_verified": filepath_check,
        "macros_verified": macro_check,
    }


def _filepath_check(inputs: ClosureInputs) -> dict[str, Any] | None:
    """Compare the rules' source directories with the reference case's Filepath."""

    if inputs.reference_case is None:
        return None
    filepath = inputs.reference_case / "Buildconf" / "camconf" / "Filepath"
    if not filepath.is_file():
        return None
    listed: list[str] = []
    for line in filepath.read_text().splitlines():
        line = line.strip()
        marker = "components/"
        if marker in line and "unit_drivers" not in line:
            index = line.index(marker)
            listed.append(re.sub(r"/+", "/", line[index:]))
    declared = [item for item in inputs.rules.source_directories if item.startswith("components/")]
    return {"matches": listed == declared, "case_filepath": listed}


def _macro_check(inputs: ClosureInputs) -> dict[str, Any] | None:
    """Every macro the rules declare must appear in the image's compile commands."""

    if inputs.native_manifest is None or not inputs.native_manifest.is_file():
        return None
    try:
        manifest = json.loads(inputs.native_manifest.read_text())
    except (OSError, ValueError):
        return None
    commands = manifest.get("compile_commands")
    if not commands:
        return None
    sample = next(iter(commands.values())) if isinstance(commands, dict) else commands[0]
    text = sample if isinstance(sample, str) else json.dumps(sample)
    defined = set(re.findall(r"-D(\w+(?:=[^\s\",]*)?)", text))
    declared = {macro[2:] if macro.startswith("-D") else macro for macro in inputs.rules.macros}
    declared = {item[:-1] if item.endswith("=") else item for item in declared}
    defined_clean = {item[:-1] if item.endswith("=") else item for item in defined}
    return {
        "matches": declared <= defined_clean,
        "missing_from_image": sorted(declared - defined_clean),
        "image_only": sorted(defined_clean - declared),
    }


# --------------------------------------------------------------------------
# building the record
# --------------------------------------------------------------------------

def scan_tree(inputs: ClosureInputs, *, workers: int = 8) -> tuple[CallTree, dict[str, Any]]:
    """Select the configured sources, expand templates, and scan them."""

    rules = inputs.rules
    root = inputs.source_root
    generated_dir = inputs.project_root / GENERATED_DIR
    produced = expand_templates(
        [root / template for template in rules.templates],
        genf90=root / rules.genf90,
        output_dir=generated_dir,
    )
    stub_dir = generated_dir / "include"
    stub_dir.mkdir(parents=True, exist_ok=True)
    for header in ("mpif.h", "netcdf.inc"):
        (stub_dir / header).write_text("! stub for call-tree analysis; the build uses the MPI library's own\n")
    directories = [root / item for item in rules.source_directories]
    selected = select_sources(directories, generated=list(produced.values()))
    portable_names = {
        str(path): f"{Path(template).relative_to(root).as_posix()} (genf90)"
        for template, path in produced.items()
    }
    include_dirs = [*directories, root / "cime/src/share/include", stub_dir]
    files = sorted(selected.values(), key=lambda path: path.as_posix())
    tree = CallTree.scan(
        files,
        source_root=root,
        macros=rules.macros,
        include_dirs=include_dirs,
        workers=workers,
        portable_names=portable_names,
    )
    scan_record = {
        "files": len(files),
        "generated_from_templates": sorted(Path(t).relative_to(root).as_posix() for t in produced),
        "directories": list(rules.source_directories),
    }
    return tree, scan_record


def build_closure(inputs: ClosureInputs, *, workers: int = 8, tree: CallTree | None = None) -> dict[str, Any]:
    rules = inputs.rules
    configuration = gather_configuration(inputs)
    if tree is None:
        tree, scan_record = scan_tree(inputs, workers=workers)
    else:
        scan_record = {"files": len(tree.scans), "generated_from_templates": [], "directories": list(rules.source_directories)}
    evaluator = _evaluator(rules, configuration, parameters=_literal_parameters(tree))
    library_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in rules.library_names]

    def decide(site: CallSite) -> bool | None:
        return guard_state(site.guards, evaluator) != "disabled"

    step_roots = [_root(tree, name) for name in rules.step_roots]
    init_roots = [_root(tree, name) for name in rules.init_roots]
    missing_roots = [name for name, found in zip([*rules.step_roots, *rules.init_roots], [*step_roots, *init_roots]) if found is None]
    step_roots = [item for item in step_roots if item is not None]
    init_roots = [item for item in init_roots if item is not None]
    configured = tree.reach(step_roots, decide=decide)
    static = tree.reach(step_roots)
    init_only = {name: d for name, d in tree.reach(init_roots, decide=decide).items() if name not in static}

    # plan actions: attribution roots from the operation name, freeCAM's
    # patched physpkg stage blocks, and reviewed overrides in the rules
    plan = PICAMStepPlan.default()
    stage_roots, stage_note = _stage_roots(inputs, tree, static)
    action_records: list[dict[str, Any]] = []
    parent_actions: dict[str, set[str]] = defaultdict(set)
    action_roots: set[str] = set()
    for action in plan.actions:
        operation = action.operation
        candidate = operation[5:] if operation.startswith("leaf_") else operation
        procedures_for_action: list[str] = []
        basis: list[str] = []
        named = _roots(tree, candidate, within=static)
        if named:
            procedures_for_action.extend(named)
            basis.append("operation name")
        if action.native_id in stage_roots:
            for name in stage_roots[action.native_id]:
                if name not in procedures_for_action:
                    procedures_for_action.append(name)
            basis.append(f"patched physpkg stage {stage_roots[action.native_id].stage}")
        for name in rules.action_procedures.get(action.qualified_name, ()):
            if name in tree.procedures and name not in procedures_for_action:
                procedures_for_action.append(name)
                basis.append("rules")
        reached: dict[str, int] = tree.reach(procedures_for_action, decide=decide) if procedures_for_action else {}
        for name in reached:
            parent_actions[name].add(action.qualified_name)
        action_roots.update(procedures_for_action)
        action_records.append(
            {
                "id": action.qualified_name,
                "native_id": action.native_id,
                "operation": operation,
                "enabled": bool(action.enabled),
                "kind": action.kind,
                "procedures": procedures_for_action,
                "basis": sorted(set(basis)),
                "reachable": len(reached),
            }
        )

    evidence = _runtime_evidence(inputs.project_root)
    kernel_evidence = evidence.get("kernels", {})
    action_evidence = evidence.get("actions", {})
    procedures: list[dict[str, Any]] = []
    sites_by_kind: Counter[str] = Counter()
    unresolved: list[dict[str, Any]] = []
    library_calls: Counter[str] = Counter()
    undecided: Counter[str] = Counter()
    dynamic_guards: Counter[str] = Counter()
    disabled_sites = 0
    categories: Counter[str] = Counter()
    expression_functions = 0
    callers: dict[str, set[str]] = defaultdict(set)
    for qualified in static:
        for callee in tree.callees(qualified):
            callers[callee].add(qualified)
    service_like = {"internal_service", "out_of_scope"}
    for qualified in sorted(static):
        scope = tree.procedures[qualified]
        category, rule_id = _classify(scope, rules)
        in_configuration = qualified in configured
        entry_states = [evaluator.truth(text) for text in scope.entry_guards]
        inert = any(state is True for state in entry_states)
        if not in_configuration:
            categories["config_disabled"] += 1
        elif inert:
            categories["inert_in_configuration"] += 1
        else:
            categories[category] += 1
        site_records: list[dict[str, Any]] = []
        kinds: Counter[str] = Counter()
        max_loop = 0
        for site in tree.sites.get(qualified, ()):
            if not site.active:
                kinds["cpp_removed"] += 1
                continue
            state = guard_state(site.guards, evaluator)
            kind = site.kind
            if kind == "unresolved" and any(pattern.search(site.name) for pattern in library_patterns):
                kind = "library"
            kinds[kind] += 1
            sites_by_kind[kind] += 1
            max_loop = max(max_loop, site.loop_depth)
            if state == "disabled":
                disabled_sites += 1
            elif state == "undecided":
                targets = [site.target, *site.candidates]
                target_categories = {
                    _classify(tree.procedures[name], rules)[0] for name in targets if name in tree.procedures
                }
                if target_categories and not target_categories <= service_like:
                    for part in site.guards:
                        if part.kind == "if" and evaluator.truth(part.text) is None:
                            (undecided if _looks_static(part.text) else dynamic_guards)[part.text] += 1
            if kind == "function" and site.in_expression:
                expression_functions += 1
            if kind == "library":
                library_calls[site.module or site.name] += 1
            if kind == "unresolved":
                unresolved.append({"caller": qualified, "name": site.name, "line": site.line, "statement": site.statement})
            record = site.as_dict()
            record["kind"] = kind
            record["guard_state"] = state
            for redundant in ("caller", "statement_kind", "active"):
                record.pop(redundant, None)
            site_records.append(record)
        procedures.append(
            {
                "qualified": qualified,
                "name": scope.name,
                "module": scope.module,
                "kind": scope.kind,
                "host": scope.host,
                "source": scope.source,
                "line_start": scope.line_start,
                "line_end": scope.line_end,
                "public": scope.public,
                "dummies": list(scope.dummies),
                "result": scope.result,
                "category": category,
                "rule": rule_id,
                "action_root": qualified in action_roots,
                "runtime_evidence": _procedure_evidence(scope, parent_actions.get(qualified, ()), kernel_evidence, action_evidence),
                "in_configuration": in_configuration,
                "entry_guards": [
                    {"condition": text, "returns": state}
                    for text, state in zip(scope.entry_guards, entry_states)
                ],
                "inert_in_configuration": inert,
                "distance": configured.get(qualified, static.get(qualified)),
                "parent_actions": sorted(parent_actions.get(qualified, ())),
                "callers": sorted(callers.get(qualified, ())),
                "callees": sorted(tree.callees(qualified, decide=decide)),
                "site_kinds": dict(sorted(kinds.items())),
                "max_loop_depth": max_loop,
                # services keep their site counts; their call sites are not the closure's concern
                "sites": site_records if category not in service_like else [],
            }
        )
    initialization = [
        {
            "qualified": name,
            "category": _classify(tree.procedures[name], rules)[0],
            "source": tree.procedures[name].source,
        }
        for name in sorted(init_only)
    ]
    selected_sources = {scan.source for scan in tree.scans}
    failures = tree.failures()
    summary = {
        "procedures_static": len(static),
        "procedures_configured": len(configured),
        "procedures_config_disabled": len(static) - len(configured),
        "by_category": dict(sorted(categories.items())),
        "sites_by_kind": dict(sorted(sites_by_kind.items())),
        "sites_disabled_by_guards": disabled_sites,
        "function_sites_in_expressions": expression_functions,
        "unresolved_references": len(unresolved),
        "undecided_guard_conditions": len(undecided),
        "dynamic_guard_conditions": len(dynamic_guards),
        "initialization_only_procedures": len(initialization),
        "parse_failures": len(failures),
        "duplicate_modules": len(tree.duplicate_modules),
        "actions_with_procedures": sum(1 for item in action_records if item["procedures"]),
        "actions": len(action_records),
        "stage_attribution": stage_note,
    }
    record = {
        "schema_version": CLOSURE_SCHEMA_VERSION,
        "generator": "tools/build_pi_cam_kernel_api_closure.py",
        "what": (
            "Every procedure the configured PI-atm physics step reaches from its drivers, "
            "with each call site's preprocessor condition, namelist guards and context, "
            "classified by native/pi_cam/kernel_api_closure_rules.yaml.  Phase A of the "
            "kernel-API closure: an inventory, not a claim that anything is callable yet."
        ),
        "baseline": {
            "source_root": SOURCE_ROOT.as_posix(),
            "source_revision": source_revision(inputs.source_root),
            "rules": inputs.rules_path.as_posix() if not inputs.rules_path.is_absolute() else _portable(inputs.rules_path, inputs.project_root),
            "rules_sha256": rules.sha256,
            "generator_sha256": generator_sha256(),
            "native_manifest_sha256": _file_sha256(inputs.native_manifest),
            "patched_physpkg_sha256": _file_sha256(inputs.patched_physpkg),
            "plan_actions": len(plan.actions),
        },
        "configuration": configuration,
        "runtime_evidence": {
            "source": evidence.get("source"),
            "note": (
                "trace counts from the recorded 50-step and month runs, by plan action, and the "
                "in-model gates of the kernels already opened; static reachability is never "
                "narrowed by a count of zero"
            ),
            "actions": action_evidence,
        },
        "scan": scan_record,
        "roots": {"step": step_roots, "initialization": init_roots, "missing": missing_roots},
        "actions": action_records,
        "summary": summary,
        "procedures": procedures,
        "initialization": initialization,
        "unresolved": {
            "references": unresolved,
            "undecided_guards": [{"condition": text, "sites": count} for text, count in undecided.most_common()],
            "dynamic_guards": len(dynamic_guards),
            "library_calls": [{"module_or_name": name, "sites": count} for name, count in library_calls.most_common()],
            "parse_failures": [item for item in failures if item["source"] in selected_sources],
            "duplicate_modules": {name: sources for name, sources in sorted(tree.duplicate_modules.items())},
            "missing_roots": missing_roots,
        },
    }
    record["inputs_hash"] = inputs_hash(inputs, configuration)
    return record


_STATIC_GUARD = re.compile(r"""^[\w\s.=/<>'"+\-*]+$""")


def _looks_static(text: str) -> bool:
    """A condition over plain names and literals, the shape a configuration flag takes."""

    return _STATIC_GUARD.match(text) is not None


def _literal_parameters(tree: CallTree) -> dict[str, object]:
    """Module ``parameter`` literals whose name has one value across the tree."""

    seen: dict[str, set[str]] = defaultdict(set)
    for table in tree.modules.values():
        for name, text in table.parameters.items():
            seen[name].add(text)
    return {name: fortran_literal(next(iter(texts))) for name, texts in seen.items() if len(texts) == 1}


LEDGER = Path("validation/physics_kernel_decoupling.json")


def _runtime_evidence(project_root: Path) -> dict[str, Any]:
    """What the decoupling ledger already proved about actions and kernels at run time."""

    path = project_root / LEDGER
    if not path.is_file():
        return {"source": None, "actions": {}, "kernels": {}}
    try:
        ledger = json.loads(path.read_text())
    except ValueError:
        return {"source": None, "actions": {}, "kernels": {}}
    actions = {
        str(item["id"]): {"activity": item.get("activity"), "execution": item.get("execution", {})}
        for item in ledger.get("actions", ())
        if item.get("id")
    }
    kernels = {
        str(item["routine"]).lower(): {
            "kernel": item.get("kernel"),
            "owner_class": item.get("owner_class"),
            "in_model_gates": list(item.get("in_model_gates", ())),
            "status": item.get("status"),
        }
        for item in ledger.get("kernels", ())
        if item.get("routine")
    }
    return {"source": LEDGER.as_posix(), "actions": actions, "kernels": kernels}


def _procedure_evidence(
    scope: ProcedureScope,
    actions: Iterable[str],
    kernel_evidence: Mapping[str, Any],
    action_evidence: Mapping[str, Any],
) -> dict[str, Any] | None:
    record: dict[str, Any] = {}
    kernel = kernel_evidence.get(scope.name)
    if kernel is not None:
        record["gated_kernel"] = kernel
    executed = [name for name in actions if action_evidence.get(name, {}).get("execution")]
    if executed:
        record["executed_through_actions"] = sorted(executed)
    return record or None


def _evaluator(
    rules: ClosureRules, configuration: Mapping[str, Any], *, parameters: Mapping[str, object] | None = None
) -> ConditionEvaluator:
    values: dict[str, object] = {}
    values.update(parameters or {})
    values.update(configuration.get("constants", {}))
    values.update(configuration.get("selectors", {}))
    options = configuration.get("build_options", {})
    functions: dict[str, Callable[[list[object]], object | None]] = {}
    for function, option in rules.option_functions.items():
        expected = options.get(option)

        def answer(arguments: list[object], expected: object = expected) -> object | None:
            if expected is None or not arguments or not isinstance(arguments[0], str):
                return None
            return arguments[0].strip().lower() == str(expected).strip().lower()

        functions[function] = answer
    return ConditionEvaluator(values, functions)


class _StageRoots(list):
    """Procedures a patched-physpkg stage block calls, remembering the stage."""

    def __init__(self, stage: int, names: Iterable[str]) -> None:
        super().__init__(names)
        self.stage = stage


def _roots(tree: CallTree, name: str, *, within: Mapping[str, int] | None) -> list[str]:
    """Procedures a plan operation names: one procedure, or a generic's specifics."""

    single = _root(tree, name, within=within)
    if single is not None:
        return [single]
    found: list[str] = []
    for table in tree.modules.values():
        specifics = table.generics.get(name.lower())
        if not specifics:
            continue
        for specific in specifics:
            qualified = table.procedures.get(specific)
            if qualified is not None and (within is None or qualified in within):
                found.append(qualified)
    return sorted(set(found))


def _stage_roots(inputs: ClosureInputs, tree: CallTree, static: Mapping[str, int]) -> tuple[dict[int, _StageRoots], str]:
    """Plan id -> procedures called inside that stage block of freeCAM's patched physpkg."""

    from .call_tree import scan_file

    path = inputs.patched_physpkg
    if path is None or not path.is_file():
        return {}, "patched physpkg not available; attribution by operation name and rules only"
    scan = scan_file(
        path,
        source_root=inputs.source_root,
        macros=inputs.rules.macros,
        include_dirs=[inputs.source_root / "cime/src/share/include", path.parent],
        portable="components/cam/src/physics/cam/physpkg.F90 (patched)",
    )
    if scan.failure is not None:
        return {}, f"patched physpkg did not parse: {scan.failure['message']}"
    lines = path.read_text(errors="replace").splitlines()
    result: dict[int, _StageRoots] = {}
    for scope in scan.procedures:
        base = STAGE_BASES.get(scope.name)
        if base is None or scope.line_start is None or scope.line_end is None:
            continue
        labels: dict[int, int] = {1: scope.line_start}
        for number in range(scope.line_start, scope.line_end + 1):
            label = _STAGE_LABEL.match(lines[number - 1])
            if label is not None:
                labels.setdefault(int(label.group(1)), number)
        pinned = tree.procedures.get(f"physpkg::{scope.name}")
        if pinned is None:
            continue
        # a stage block runs from its label to the next label: the ``if
        # (stage == N)`` tests inside it are its own control flow
        ordered = sorted(labels.items(), key=lambda item: item[1])
        for index, (stage, start) in enumerate(ordered):
            stop = ordered[index + 1][1] - 1 if index + 1 < len(ordered) else scope.line_end
            names: list[str] = []
            for reference in scope.references:
                if reference.line is None or not (start <= reference.line <= stop):
                    continue
                if reference.hint not in ("call", "ref"):
                    continue
                entity = tree.lookup(pinned, reference.name)
                targets = [entity.qualified] if entity.qualified else list(entity.specifics)
                for target in targets:
                    if target not in static or target in names:
                        continue
                    if _classify(tree.procedures[target], inputs.rules)[0] in ("internal_service", "out_of_scope"):
                        continue
                    names.append(target)
            result[base + stage] = _StageRoots(stage, names)
    return result, f"stage blocks of {PATCHED_PHYSPKG.as_posix()} mapped to plan ids"


def _root(tree: CallTree, name: str, *, within: Mapping[str, int] | None = None) -> str | None:
    """The qualified procedure ``name`` denotes; a bare name must be unique."""

    if "::" in name:
        return name if name in tree.procedures else None
    matches = sorted(
        qualified
        for qualified, scope in tree.procedures.items()
        if scope.name == name.lower() and scope.host is None and (within is None or qualified in within)
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # a module of the same name is the conventional home (dadadj::dadadj)
        same_module = [item for item in matches if item.split("::")[0] == name.lower()]
        if len(same_module) == 1:
            return same_module[0]
    return None


def _classify(scope: ProcedureScope, rules: ClosureRules) -> tuple[str, str | None]:
    for rule in rules.categories:
        if rule.matches(scope):
            return rule.category, rule.id
    return rules.default_category, None


def generator_sha256() -> str:
    digest = sha256()
    here = Path(__file__).resolve().parent
    for name in GENERATOR_SOURCES:
        digest.update((here / name).read_bytes())
    return digest.hexdigest()


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return sha256(path.read_bytes()).hexdigest()


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def inputs_hash(inputs: ClosureInputs, configuration: Mapping[str, Any]) -> str:
    plan = PICAMStepPlan.default()
    payload = {
        "source_revision": source_revision(inputs.source_root),
        "rules_sha256": inputs.rules.sha256,
        "generator_sha256": generator_sha256(),
        "selectors": configuration.get("selectors", {}),
        "build_options": configuration.get("build_options", {}),
        "constants": configuration.get("constants", {}),
        "plan": [(action.qualified_name, action.operation, bool(action.enabled)) for action in plan.actions],
    }
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def check_closure(record: Mapping[str, Any], inputs: ClosureInputs) -> list[str]:
    """Why the committed record no longer matches its inputs; empty when current."""

    problems: list[str] = []
    if int(record.get("schema_version", 0)) != CLOSURE_SCHEMA_VERSION:
        problems.append("schema_version differs from the generator's")
    inputs.previous_record = record
    configuration = gather_configuration(inputs)
    expected = inputs_hash(inputs, configuration)
    if record.get("inputs_hash") != expected:
        baseline = record.get("baseline", {})
        if baseline.get("source_revision") != source_revision(inputs.source_root):
            problems.append("the pinned source revision changed")
        if baseline.get("rules_sha256") != inputs.rules.sha256:
            problems.append("the closure rules changed")
        if baseline.get("generator_sha256") != generator_sha256():
            problems.append("the generator changed")
        stored = record.get("configuration", {})
        if stored.get("selectors") != configuration.get("selectors"):
            problems.append("the namelist selectors changed")
        if stored.get("build_options") != configuration.get("build_options"):
            problems.append("the build options changed")
        if not problems:
            problems.append("the inputs hash differs (plan actions or constants changed)")
    return problems


def write_record(record: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=1, sort_keys=False, default=str) + "\n")
    return target
