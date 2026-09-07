"""The PI-atm physics call tree from the pinned Fortran, one call site at a time.

The catalog in :mod:`freecam.pi_cam.source_catalog` recovers procedures and
their ``call`` statements; this module goes further for the kernel-API
closure: every source file is preprocessed with the configuration's real
macros, every reference that could be a function call is resolved through
Fortran scoping (locals, host association, ``use`` with ``only`` and renames,
generic interfaces, internal procedures, external units), and every call site
carries the preprocessor condition, the enclosing constructs, the namelist
guards it sits under, and whether it is a plain ``call`` or a function inside
an expression.  Nothing here executes or rewrites the numerical code; the
output is evidence for the closure record and input for the later phases.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from .errors import PICAMConfigurationError

FORTRAN_SUFFIXES = frozenset({".f90", ".F90", ".f", ".F"})
STATEMENT_TEXT_LIMIT = 240
_MARKER = re.compile(r'^# (\d+) "([^"]*)"')
_DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")
_FUNCTION_RESULT = re.compile(r"result\s*\(\s*([A-Za-z_]\w*)\s*\)", re.IGNORECASE)


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PreprocessedSource:
    """One file after ``cpp`` with the configuration's macros.

    ``text`` is what the parser reads (markers removed); ``origin[i]`` is the
    original line of ``text`` line ``i+1`` (``None`` for text pulled in from
    an include file); ``conditions`` maps original lines inside conditional
    blocks to the conjunction of directives guarding them, whether or not the
    block survived; ``active_lines`` are the original lines that survived.
    """

    path: str
    text: str
    origin: tuple[int | None, ...]
    active_lines: frozenset[int]
    conditions: dict[int, str]
    sha256: str


def cpp_conditions(original: str) -> dict[int, str]:
    """Map each original line under a preprocessor conditional to its guard."""

    conditions: dict[int, str] = {}
    stack: list[list[str]] = []   # per open #if: the branch conditions seen so far
    for number, line in enumerate(original.splitlines(), start=1):
        match = _DIRECTIVE.match(line)
        if match is None:
            if stack:
                conditions[number] = " && ".join(_branch_condition(level) for level in stack)
            continue
        keyword, rest = match.group(1), match.group(2).strip()
        if keyword == "if":
            stack.append([rest])
        elif keyword == "ifdef":
            stack.append([f"defined({rest})"])
        elif keyword == "ifndef":
            stack.append([f"!defined({rest})"])
        elif keyword == "elif" and stack:
            stack[-1].append(rest)
        elif keyword == "else" and stack:
            stack[-1].append("")
        elif keyword == "endif" and stack:
            stack.pop()
    return conditions


def _branch_condition(branches: list[str]) -> str:
    *previous, current = branches
    negated = [f"!({item})" for item in previous]
    if current:
        return " && ".join([*negated, f"({current})"]) if negated else f"({current})"
    return " && ".join(negated) if negated else "else"


def preprocess_source(
    path: str | Path,
    *,
    macros: Sequence[str],
    include_dirs: Sequence[str | Path] = (),
    cpp: str = "cpp",
    root: str | Path | None = None,
) -> PreprocessedSource:
    """Run the C preprocessor the way the model build does and keep the line map.

    With ``root`` the preprocessor runs from that directory on the relative
    path, so ``__FILE__`` expansions (``errMsg(__FILE__, __LINE__)``) name
    the file portably instead of by its absolute location.
    """

    source = Path(path)
    original = source.read_text(errors="replace")
    command = [cpp, "-traditional-cpp", "-nostdinc", "-undef"]
    for macro in macros:
        command.append(macro if macro.startswith("-D") else f"-D{macro}")
    for directory in include_dirs:
        command.extend(["-I", str(Path(directory).resolve())])
    cwd = None
    argument = str(source)
    if root is not None:
        try:
            argument = str(source.resolve().relative_to(Path(root).resolve()))
            cwd = str(Path(root).resolve())
        except ValueError:
            pass
    command.append(argument)
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace", cwd=cwd)
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        raise PICAMConfigurationError(
            f"cpp failed on {source}: {detail[-1] if detail else completed.returncode}"
        )
    original_lines = original.splitlines()
    lines: list[str] = []
    origin: list[int | None] = []
    active: set[int] = set()
    current_file = str(source)
    next_line = 1
    in_main = True
    for raw in completed.stdout.splitlines():
        marker = _MARKER.match(raw)
        if marker is not None:
            next_line = int(marker.group(1))
            current_file = marker.group(2)
            in_main = _same_file(current_file, source, cwd)
            continue
        lines.append(raw)
        if in_main:
            origin.append(next_line)
            # cpp keeps short removed blocks as blank lines rather than
            # markers: a blank output line for a non-blank, non-directive
            # original line means the preprocessor dropped it
            source_line = original_lines[next_line - 1] if 0 < next_line <= len(original_lines) else ""
            dropped = raw.strip() == "" and source_line.strip() != ""
            if not dropped and not source_line.lstrip().startswith("#"):
                active.add(next_line)
        else:
            origin.append(None)
        next_line += 1
    return PreprocessedSource(
        path=str(source),
        text="\n".join(lines) + "\n",
        origin=tuple(origin),
        active_lines=frozenset(active),
        conditions=cpp_conditions(original),
        sha256=sha256(source.read_bytes()).hexdigest(),
    )


def _same_file(marker_path: str, source: Path, cwd: str | None = None) -> bool:
    if marker_path in ("<built-in>", "<command-line>", "<stdin>"):
        return False
    candidate = Path(marker_path)
    if not candidate.is_absolute() and cwd is not None:
        candidate = Path(cwd) / candidate
    try:
        return candidate.resolve() == source.resolve()
    except OSError:
        return marker_path == str(source)


# --------------------------------------------------------------------------
# scoping records (picklable: they cross the process pool)
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class UseSpec:
    module: str
    only: tuple[tuple[str, str], ...] | None    # (local, remote); None = wildcard
    renames: tuple[tuple[str, str], ...] = ()   # renames on a wildcard use


@dataclass(slots=True)
class ModuleTable:
    name: str
    source: str
    procedures: dict[str, str] = field(default_factory=dict)      # name -> qualified
    generics: dict[str, tuple[str, ...]] = field(default_factory=dict)  # name -> specific names
    variables: set[str] = field(default_factory=set)
    types: set[str] = field(default_factory=set)
    externals: set[str] = field(default_factory=set)               # explicit interfaces / external attr
    uses: list[UseSpec] = field(default_factory=list)
    type_bindings: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)  # type -> binding -> procedures
    variable_types: dict[str, str] = field(default_factory=dict)   # module variable -> derived type
    parameters: dict[str, str] = field(default_factory=dict)       # literal ``parameter`` values as written
    default_private: bool = False
    public_names: set[str] = field(default_factory=set)
    private_names: set[str] = field(default_factory=set)

    def is_public(self, name: str) -> bool:
        if name in self.public_names:
            return True
        if name in self.private_names:
            return False
        return not self.default_private


@dataclass(frozen=True, slots=True)
class GuardPart:
    """One enclosing branch: an ``if`` condition or a ``select case`` value."""

    kind: str                     # "if" | "case"
    text: str                     # the condition, or the case selector expression
    values: tuple[str, ...] = ()  # case values; empty with kind "case" means "case default"
    negate: bool = False


@dataclass(frozen=True, slots=True)
class RawReference:
    name: str
    hint: str                     # "call" | "ref"
    line: int | None
    cpp_condition: str | None
    active: bool
    loop_depth: int
    constructs: tuple[str, ...]
    guards: tuple[GuardPart, ...]
    in_expression: bool
    as_argument: bool
    statement_kind: str
    statement: str


@dataclass(slots=True)
class ProcedureScope:
    name: str
    qualified: str
    kind: str                     # subroutine | function
    module: str | None
    host: str | None              # qualified host procedure for internal procedures
    source: str
    line_start: int | None
    line_end: int | None
    dummies: tuple[str, ...]
    result: str | None
    locals: set[str]
    local_externals: set[str]
    proc_decls: set[str]
    internal: dict[str, str]      # name -> qualified
    local_types: set[str]
    local_generics: dict[str, tuple[str, ...]]
    uses: list[UseSpec]
    references: list[RawReference]
    public: bool
    cpp_condition: str | None = None
    var_types: dict[str, str] = field(default_factory=dict)        # local/dummy -> derived type
    entry_guards: tuple[str, ...] = ()                             # leading ``if (cond) return`` conditions


@dataclass(slots=True)
class FileScan:
    source: str
    sha256: str
    parser: str
    modules: list[ModuleTable] = field(default_factory=list)
    externals: dict[str, str] = field(default_factory=dict)   # program-unit name -> qualified
    procedures: list[ProcedureScope] = field(default_factory=list)
    failure: dict[str, str] | None = None
    inactive_lines: int = 0


# --------------------------------------------------------------------------
# one file: parse and collect scopes and references
# --------------------------------------------------------------------------

def scan_file(
    path: str | Path,
    *,
    source_root: str | Path,
    macros: Sequence[str],
    include_dirs: Sequence[str | Path],
    portable: str | None = None,
) -> FileScan:
    """Preprocess and parse one file, collecting scopes and raw references.

    ``portable`` names the file in the record; a file generated from a
    template is recorded under its template's path.
    """

    source = Path(path)
    portable = portable or _portable(source, Path(source_root))
    try:
        pre = preprocess_source(source, macros=macros, include_dirs=include_dirs, root=source_root)
    except PICAMConfigurationError as exc:
        return FileScan(
            source=portable,
            sha256=sha256(source.read_bytes()).hexdigest(),
            parser="none",
            failure={"error_type": "PreprocessError", "message": str(exc)},
        )
    original_lines = len(source.read_text(errors="replace").splitlines())
    inactive = sum(
        1
        for number in range(1, original_lines + 1)
        if number not in pre.active_lines and number in pre.conditions
    )
    try:
        from fparser.common.readfortran import FortranStringReader
        from fparser.two.parser import ParserFactory

        reader = FortranStringReader(_parser_text(pre.text), ignore_comments=True)
        tree = ParserFactory().create(std="f2008")(reader)
    except Exception as exc:  # fparser raises many concrete types
        message = str(exc).strip().splitlines()
        return FileScan(
            source=portable,
            sha256=pre.sha256,
            parser="none",
            failure={
                "error_type": type(exc).__name__,
                "message": message[-1] if message else repr(exc),
            },
            inactive_lines=inactive,
        )
    scan = FileScan(source=portable, sha256=pre.sha256, parser="fparser-cpp", inactive_lines=inactive)
    _Collector(pre, portable, scan).collect(tree)
    return scan


def _parser_text(text: str) -> str:
    """The preprocessed text as the parser reads it: tabs widened, comment lines blanked.

    Line numbers are preserved, so statement spans still map back through
    the preprocessor's line table.
    """

    lines: list[str] = []
    for line in text.splitlines():
        line = line.replace("\t", " ")
        if line.lstrip().startswith("!"):
            line = ""
        lines.append(line)
    return "\n".join(lines) + "\n"


def _scan_worker(arguments: tuple[str, str, tuple[str, ...], tuple[str, ...], str | None]) -> FileScan:
    path, root, macros, includes, portable = arguments
    return scan_file(path, source_root=root, macros=macros, include_dirs=includes, portable=portable)


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


class _Collector:
    """Walk one parsed file; fparser classes are imported lazily per process."""

    def __init__(self, pre: PreprocessedSource, portable: str, scan: FileScan) -> None:
        from fparser.two import Fortran2003 as F

        self.F = F
        self.pre = pre
        self.portable = portable
        self.scan = scan
        self.text_lines = pre.text.splitlines()
        self.stem = Path(pre.path).stem.lower()

    # -- entry -----------------------------------------------------------

    def collect(self, tree: Any) -> None:
        F = self.F
        for unit in getattr(tree, "children", ()):
            if isinstance(unit, F.Module):
                self._module(unit)
            elif isinstance(unit, (F.Subroutine_Subprogram, F.Function_Subprogram)):
                statement = unit.children[0]
                name = str(statement.items[1]).lower()
                qualified = f"{self.stem}::{name}"
                self.scan.externals[name] = qualified
                self._procedure(unit, module=None, host=None, qualified=qualified, public=True)

    # -- modules ---------------------------------------------------------

    def _module(self, node: Any) -> None:
        F = self.F
        name = str(node.children[0].items[1]).lower()
        table = ModuleTable(name=name, source=self.portable)
        spec = _child(node, F.Specification_Part)
        if spec is not None:
            self._specification(spec, table=table, scope=None)
        part = _child(node, F.Module_Subprogram_Part)
        if part is not None:
            for child in part.children:
                if isinstance(child, (F.Subroutine_Subprogram, F.Function_Subprogram)):
                    procedure_name = str(child.children[0].items[1]).lower()
                    table.procedures[procedure_name] = f"{name}::{procedure_name}"
        self.scan.modules.append(table)
        if part is not None:
            for child in part.children:
                if isinstance(child, (F.Subroutine_Subprogram, F.Function_Subprogram)):
                    procedure_name = str(child.children[0].items[1]).lower()
                    self._procedure(
                        child,
                        module=name,
                        host=None,
                        qualified=f"{name}::{procedure_name}",
                        public=table.is_public(procedure_name),
                    )

    def _specification(self, spec: Any, *, table: ModuleTable | None, scope: ProcedureScope | None) -> None:
        """Fill a module table or a procedure scope from a specification part."""

        F = self.F
        for child in spec.children:
            if isinstance(child, F.Use_Stmt):
                use = self._use(child)
                (table.uses if table is not None else scope.uses).append(use)
            elif isinstance(child, F.Access_Stmt) and table is not None:
                keyword, names = child.items
                if names is None:
                    table.default_private = str(keyword).upper() == "PRIVATE"
                else:
                    bucket = table.public_names if str(keyword).upper() == "PUBLIC" else table.private_names
                    bucket.update(_names(names))
            elif isinstance(child, F.Type_Declaration_Stmt):
                attributes = "" if child.items[1] is None else str(child.items[1]).upper()
                names = [str(entity.items[0]).lower() for entity in _walk(child, F.Entity_Decl)]
                external = "EXTERNAL" in attributes
                type_spec = child.items[0]
                if table is not None and "PARAMETER" in attributes:
                    for entity in _walk(child, F.Entity_Decl):
                        initial = entity.items[3]
                        if initial is not None and isinstance(initial, F.Initialization):
                            value = initial.items[1]
                            if type(value).__name__ in ("Logical_Literal_Constant", "Int_Literal_Constant", "Char_Literal_Constant", "Signed_Int_Literal_Constant"):
                                table.parameters[str(entity.items[0]).lower()] = str(value)
                if isinstance(type_spec, F.Declaration_Type_Spec) and str(type_spec.items[0]).upper() in ("TYPE", "CLASS"):
                    type_name = str(type_spec.items[1]).lower()
                    target_types = table.variable_types if table is not None else scope.var_types
                    for entity_name in names:
                        target_types[entity_name] = type_name
                if table is not None:
                    if external:
                        table.externals.update(names)
                    else:
                        table.variables.update(names)
                    if "PUBLIC" in attributes:
                        table.public_names.update(names)
                    if "PRIVATE" in attributes:
                        table.private_names.update(names)
                else:
                    (scope.local_externals if external else scope.locals).update(names)
            elif isinstance(child, F.External_Stmt):
                names = _names(child.items[1])
                (table.externals if table is not None else scope.local_externals).update(names)
            elif isinstance(child, F.Procedure_Declaration_Stmt):
                names = {str(item).split("=>")[0].strip().lower() for item in _items(child.items[2])}
                names = {re.sub(r"\(.*", "", item).strip() for item in names}
                if table is not None:
                    table.variables.update(names)
                else:
                    scope.proc_decls.update(names)
            elif isinstance(child, F.Derived_Type_Def):
                type_name = str(child.children[0].items[1]).lower()
                (table.types if table is not None else scope.local_types).add(type_name)
                bindings: dict[str, tuple[str, ...]] = {}
                for binding in _walk(child, F.Specific_Binding):
                    binding_name = str(binding.items[3]).lower()
                    procedure_name = str(binding.items[4]).lower() if binding.items[4] is not None else binding_name
                    bindings[binding_name] = (procedure_name,)
                for binding in _walk(child, F.Generic_Binding):
                    binding_name = str(binding.items[1]).lower()
                    specifics = tuple(str(item).lower() for item in _items(binding.items[2]))
                    bindings[binding_name] = tuple(
                        name for specific in specifics for name in bindings.get(specific, (specific,))
                    )
                if bindings and table is not None:
                    table.type_bindings[type_name] = bindings
            elif isinstance(child, F.Interface_Block):
                self._interface(child, table=table, scope=scope)
            elif isinstance(child, (F.Dimension_Stmt, F.Parameter_Stmt, F.Save_Stmt, F.Common_Stmt, F.Pointer_Stmt, F.Target_Stmt, F.Allocatable_Stmt)):
                names = {
                    str(name).lower() for name in _walk(child, F.Name)
                }
                if table is not None:
                    table.variables.update(names)
                else:
                    scope.locals.update(names)
            elif isinstance(child, F.Implicit_Part):
                for statement in child.children:
                    if isinstance(statement, F.Parameter_Stmt):
                        names = {str(name).lower() for name in _walk(statement, F.Name)}
                        (table.variables if table is not None else scope.locals).update(names)

    def _use(self, node: Any) -> UseSpec:
        F = self.F
        module = str(node.items[2]).lower()
        only_keyword = "ONLY" in str(node.items[3]).upper()
        items = node.items[4]
        pairs: list[tuple[str, str]] = []
        renames: list[tuple[str, str]] = []
        for item in _items(items):
            if isinstance(item, F.Rename):
                local, remote = [part.strip().lower() for part in str(item).split("=>")]
                (pairs if only_keyword else renames).append((local, remote))
            elif isinstance(item, F.Name):
                name = str(item).lower()
                pairs.append((name, name))
            # operator/assignment generic specs carry no callable name
        if only_keyword:
            return UseSpec(module=module, only=tuple(pairs))
        return UseSpec(module=module, only=None, renames=tuple(renames))

    def _interface(self, node: Any, *, table: ModuleTable | None, scope: ProcedureScope | None) -> None:
        F = self.F
        header = node.children[0]
        header_text = str(header).upper()
        if header_text.startswith("ABSTRACT"):
            return
        generic = header.items[0]
        generic_name = None
        if generic is not None and isinstance(generic, F.Name):
            generic_name = str(generic).lower()
        specifics: list[str] = []
        for child in node.children[1:-1]:
            if isinstance(child, F.Procedure_Stmt):
                specifics.extend(str(item).lower() for item in _items(child.items[0]))
            elif isinstance(child, (F.Subroutine_Body, F.Function_Body)):
                name = str(child.children[0].items[1]).lower()
                if generic_name is None:
                    (table.externals if table is not None else scope.local_externals).add(name)
                else:
                    specifics.append(name)
                    (table.externals if table is not None else scope.local_externals).add(name)
        if generic_name is not None:
            target = table.generics if table is not None else scope.local_generics
            target[generic_name] = tuple(dict.fromkeys([*target.get(generic_name, ()), *specifics]))

    # -- procedures ------------------------------------------------------

    def _procedure(self, node: Any, *, module: str | None, host: str | None, qualified: str, public: bool) -> None:
        F = self.F
        statement = node.children[0]
        name = str(statement.items[1]).lower()
        kind = "function" if isinstance(statement, F.Function_Stmt) else "subroutine"
        dummies = tuple(str(item).lower() for item in _items(statement.items[2]))
        result = None
        if kind == "function":
            suffix = statement.items[3]
            match = _FUNCTION_RESULT.search(str(suffix)) if suffix is not None else None
            result = match.group(1).lower() if match else name
        first_line, last_line = self._span(node)
        scope = ProcedureScope(
            name=name,
            qualified=qualified,
            kind=kind,
            module=module,
            host=host,
            source=self.portable,
            line_start=first_line,
            line_end=last_line,
            dummies=dummies,
            result=result,
            locals=set(dummies) | ({result} if result else set()),
            local_externals=set(),
            proc_decls=set(),
            internal={},
            local_types=set(),
            local_generics={},
            uses=[],
            references=[],
            public=public,
            cpp_condition=self.pre.conditions.get(first_line) if first_line else None,
        )
        spec = _child(node, F.Specification_Part)
        if spec is not None:
            self._specification(spec, table=None, scope=scope)
        internal_part = _child(node, F.Internal_Subprogram_Part)
        internal_nodes = []
        if internal_part is not None:
            for child in internal_part.children:
                if isinstance(child, (F.Subroutine_Subprogram, F.Function_Subprogram)):
                    internal_name = str(child.children[0].items[1]).lower()
                    scope.internal[internal_name] = f"{qualified}.{internal_name}"
                    internal_nodes.append((child, internal_name))
        execution = _child(node, F.Execution_Part)
        if execution is not None:
            scope.entry_guards = self._entry_guards(execution)
            self._visit(execution, scope, _Context())
        self.scan.procedures.append(scope)
        for child, internal_name in internal_nodes:
            self._procedure(
                child,
                module=module,
                host=qualified,
                qualified=f"{qualified}.{internal_name}",
                public=False,
            )

    def _entry_guards(self, execution: Any) -> tuple[str, ...]:
        """Conditions of the ``if (...) return`` statements that open the body."""

        F = self.F
        guards: list[str] = []
        for child in getattr(execution, "children", ()):
            if isinstance(child, F.If_Stmt) and isinstance(child.items[1], F.Return_Stmt):
                guards.append(_compact(str(child.items[0])))
                continue
            if isinstance(child, F.If_Construct):
                body = [item for item in child.children if not isinstance(item, (F.If_Then_Stmt, F.End_If_Stmt))]
                if len(body) == 1 and isinstance(body[0], F.Return_Stmt):
                    guards.append(_compact(str(child.children[0].items[0])))
                    continue
            break
        return tuple(guards)

    # -- execution part ----------------------------------------------------

    def _visit(self, node: Any, scope: ProcedureScope, context: "_Context") -> None:
        F = self.F
        if isinstance(node, F.If_Construct):
            self._visit_if(node, scope, context)
            return
        if isinstance(node, F.Case_Construct):
            self._visit_case(node, scope, context)
            return
        if isinstance(node, F.If_Stmt):
            condition, action = node.items
            self._references(node, scope, context, only=condition)
            guard = GuardPart(kind="if", text=_compact(str(condition)))
            # the action shares the if statement's line; it is not a statement node of its own
            self._references(node, scope, context.push("if", guard=guard), only=action)
            return
        class_name = type(node).__name__
        if "Do_Construct" in class_name or class_name in ("Do_Stmt", "Label_Do_Stmt", "Nonlabel_Do_Stmt"):
            inner = context.push("do", loop=True)
            for child in getattr(node, "children", ()):
                if isinstance(child, (F.Nonlabel_Do_Stmt, F.Label_Do_Stmt)):
                    self._references(child, scope, context)
                elif isinstance(child, F.End_Do_Stmt):
                    continue
                else:
                    self._visit(child, scope, inner)
            return
        if isinstance(node, (F.Where_Construct, F.Forall_Construct, F.Associate_Construct, F.Select_Type_Construct)):
            label = class_name.replace("_Construct", "").lower()
            inner = context.push(label)
            if isinstance(node, F.Associate_Construct):
                # associate names are local aliases for the construct's extent
                for association in _walk(node.children[0], F.Association):
                    scope.locals.add(str(association.items[0]).lower())
            for child in getattr(node, "children", ()):
                self._visit(child, scope, inner)
            return
        if getattr(node, "item", None) is not None and not isinstance(node, (F.Execution_Part,)):
            self._references(node, scope, context)
            return
        for child in getattr(node, "children", ()):
            if isinstance(child, str):
                continue
            self._visit(child, scope, context)

    def _visit_if(self, node: Any, scope: ProcedureScope, context: "_Context") -> None:
        F = self.F
        previous: list[str] = []
        branch: _Context | None = None
        for child in node.children:
            if isinstance(child, (F.If_Then_Stmt, F.Else_If_Stmt)):
                self._references(child, scope, context)
                condition = _compact(str(child.items[0]))
                guards = tuple(GuardPart(kind="if", text=text, negate=True) for text in previous)
                guards += (GuardPart(kind="if", text=condition),)
                previous.append(condition)
                branch = context.push("if", guards=guards)
            elif isinstance(child, F.Else_Stmt):
                guards = tuple(GuardPart(kind="if", text=text, negate=True) for text in previous)
                branch = context.push("if", guards=guards)
            elif isinstance(child, F.End_If_Stmt):
                continue
            else:
                self._visit(child, scope, branch or context.push("if"))

    def _visit_case(self, node: Any, scope: ProcedureScope, context: "_Context") -> None:
        F = self.F
        selector = ""
        branch: _Context | None = None
        seen: list[tuple[str, ...]] = []
        for child in node.children:
            if isinstance(child, F.Select_Case_Stmt):
                self._references(child, scope, context)
                selector = _compact(str(child.items[0]))
            elif isinstance(child, F.Case_Stmt):
                values = _case_values(str(child.items[0]))
                if values is None:   # case default: everything already seen is excluded
                    guards = tuple(
                        GuardPart(kind="case", text=selector, values=previous, negate=True) for previous in seen
                    )
                else:
                    guards = (GuardPart(kind="case", text=selector, values=values),)
                    seen.append(values)
                branch = context.push("select", guards=guards)
            elif isinstance(child, F.End_Select_Stmt):
                continue
            else:
                self._visit(child, scope, branch or context.push("select"))

    def _references(self, statement: Any, scope: ProcedureScope, context: "_Context", *, only: Any = None) -> None:
        """Record the calls and function-like references inside one statement."""

        F = self.F
        line = self._line(statement)
        active = line is None or line in self.pre.active_lines
        text = _compact(self._statement_text(statement))
        kind = type(statement).__name__
        base = dict(
            line=line,
            cpp_condition=self.pre.conditions.get(line) if line is not None else None,
            active=active,
            loop_depth=context.loop_depth,
            constructs=context.constructs,
            guards=context.guards,
            statement_kind=kind,
            statement=text,
        )
        root = statement if only is None else only
        designator = None
        if isinstance(root, F.Call_Stmt):
            designator = root.items[0]
            if isinstance(designator, F.Name):
                scope.references.append(
                    RawReference(name=str(designator).lower(), hint="call", in_expression=False, as_argument=False, **base)
                )
            else:
                # type-bound procedure call: ``object%binding``
                scope.references.append(
                    RawReference(name=_compact(str(designator)).replace(" ", "").lower(), hint="bound_call", in_expression=False, as_argument=False, **base)
                )
        for reference in _walk(root, (F.Part_Ref, F.Structure_Constructor, F.Function_Reference)):
            if reference is designator:
                continue
            if _is_component(reference, F):
                parent = reference.parent
                if isinstance(parent, F.Data_Ref) and len(parent.items) == 2 and isinstance(parent.items[0], F.Name):
                    # possibly a type-bound function: resolved against the type's bindings
                    scope.references.append(
                        RawReference(
                            name=f"{str(parent.items[0]).lower()}%{str(reference.items[0]).lower()}",
                            hint="bound_ref",
                            in_expression=True,
                            as_argument=_inside_call_arguments(reference, F),
                            **base,
                        )
                    )
                continue
            name_node = reference.items[0]
            if not isinstance(name_node, F.Name):
                continue
            scope.references.append(
                RawReference(
                    name=str(name_node).lower(),
                    hint="ref",
                    in_expression=True,
                    as_argument=_inside_call_arguments(reference, F),
                    **base,
                )
            )

    # -- helpers -----------------------------------------------------------

    def _line(self, node: Any) -> int | None:
        item = getattr(node, "item", None)
        if item is None or not getattr(item, "span", None):
            return None
        start = int(item.span[0])
        if 1 <= start <= len(self.pre.origin):
            return self.pre.origin[start - 1]
        return None

    def _span(self, node: Any) -> tuple[int | None, int | None]:
        first = self._line(node.children[0])
        last = self._line(node.children[-1])
        return first, last

    def _statement_text(self, node: Any) -> str:
        item = getattr(node, "item", None)
        if item is None or not getattr(item, "span", None):
            return str(node)
        start, end = int(item.span[0]), int(item.span[1])
        lines = self.text_lines[start - 1 : end]
        return " ".join(line.strip() for line in lines)


@dataclass(frozen=True, slots=True)
class _Context:
    constructs: tuple[str, ...] = ()
    loop_depth: int = 0
    guards: tuple[GuardPart, ...] = ()

    def push(self, label: str, *, loop: bool = False, guard: GuardPart | None = None, guards: tuple[GuardPart, ...] = ()) -> "_Context":
        extra = guards if guards else ((guard,) if guard is not None else ())
        return _Context(
            constructs=(*self.constructs, label),
            loop_depth=self.loop_depth + (1 if loop else 0),
            guards=(*self.guards, *extra),
        )


def _child(node: Any, cls: type) -> Any | None:
    for child in getattr(node, "children", ()):
        if isinstance(child, cls):
            return child
    return None


def _walk(node: Any, classes: type | tuple[type, ...]) -> list[Any]:
    from fparser.two.utils import walk

    return walk(node, classes)


def _items(node: Any) -> tuple[Any, ...]:
    if node is None:
        return ()
    items = getattr(node, "items", None)
    if items is None:
        return (node,)
    return tuple(items)


def _names(node: Any) -> set[str]:
    return {str(item).lower() for item in _items(node)}


def _compact(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) > STATEMENT_TEXT_LIMIT:
        return compact[: STATEMENT_TEXT_LIMIT - 3] + "..."
    return compact


def _case_values(text: str) -> tuple[str, ...] | None:
    body = text.strip()
    if body.upper() == "DEFAULT":
        return None
    body = body.strip("()")
    return tuple(part.strip() for part in body.split(",") if part.strip())


def _is_component(reference: Any, F: Any) -> bool:
    """A ``Part_Ref`` that names a derived-type component is not a call."""

    parent = getattr(reference, "parent", None)
    if isinstance(parent, (F.Data_Ref, F.Proc_Component_Ref)):
        return parent.items[0] is not reference
    return False


def _inside_call_arguments(reference: Any, F: Any) -> bool:
    node = getattr(reference, "parent", None)
    while node is not None and getattr(node, "item", None) is None:
        if isinstance(node, F.Actual_Arg_Spec_List):
            return True
        node = getattr(node, "parent", None)
    return False


# --------------------------------------------------------------------------
# whole-tree resolution
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Entity:
    kind: str                                 # procedure | generic | variable | type | external | dummy_procedure | procedure_pointer | library | intrinsic | unresolved
    qualified: str | None = None
    specifics: tuple[str, ...] = ()
    module: str | None = None


@dataclass(frozen=True, slots=True)
class CallSite:
    caller: str
    name: str
    kind: str          # call | function | generic | external | dummy_procedure | procedure_pointer | library | bound_call | unresolved
    target: str | None
    candidates: tuple[str, ...]
    module: str | None
    line: int | None
    active: bool
    cpp_condition: str | None
    guards: tuple[GuardPart, ...]
    loop_depth: int
    constructs: tuple[str, ...]
    in_expression: bool
    as_argument: bool
    statement_kind: str
    statement: str

    def as_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["guards"] = [asdict(part) for part in self.guards]
        return record


class CallTree:
    """Scopes from every scanned file, resolved into call sites."""

    def __init__(self, scans: Iterable[FileScan]) -> None:
        self.scans = tuple(scans)
        self.modules: dict[str, ModuleTable] = {}
        self.externals: dict[str, str] = {}
        self.procedures: dict[str, ProcedureScope] = {}
        self.duplicate_modules: dict[str, list[str]] = {}
        for scan in self.scans:
            for table in scan.modules:
                if table.name in self.modules:
                    self.duplicate_modules.setdefault(table.name, [self.modules[table.name].source]).append(table.source)
                    continue
                self.modules[table.name] = table
            for name, qualified in scan.externals.items():
                self.externals.setdefault(name, qualified)
            for scope in scan.procedures:
                self.procedures.setdefault(scope.qualified, scope)
        self._exports: dict[str, dict[str, Entity]] = {}
        self._exporting: set[str] = set()
        self.intrinsics = _intrinsic_names()
        self.sites: dict[str, tuple[CallSite, ...]] = {}

    @classmethod
    def scan(
        cls,
        files: Sequence[str | Path],
        *,
        source_root: str | Path,
        macros: Sequence[str],
        include_dirs: Sequence[str | Path] = (),
        workers: int = 1,
        portable_names: Mapping[str, str] | None = None,
    ) -> "CallTree":
        names = dict(portable_names or {})
        arguments = [
            (
                str(path),
                str(source_root),
                tuple(macros),
                tuple(str(item) for item in include_dirs),
                names.get(str(path)),
            )
            for path in files
        ]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                scans = list(pool.map(_scan_worker, arguments, chunksize=4))
        else:
            scans = [_scan_worker(item) for item in arguments]
        tree = cls(scans)
        tree.resolve()
        return tree

    # -- exports -------------------------------------------------------------

    def exports(self, module_name: str) -> dict[str, Entity]:
        """Every entity ``use module_name`` makes visible, by its local name."""

        if module_name in self._exports:
            return self._exports[module_name]
        table = self.modules.get(module_name)
        if table is None or module_name in self._exporting:
            return {}
        self._exporting.add(module_name)
        visible: dict[str, Entity] = {}
        for use in table.uses:
            for local, entity in self._imported(use).items():
                visible.setdefault(local, entity)
        for name in table.variables:
            visible[name] = Entity("variable", module=module_name)
        for name in table.types:
            visible[name] = Entity("type", module=module_name)
        for name in table.externals:
            visible[name] = Entity("external", qualified=self.externals.get(name), module=module_name)
        for name, qualified in table.procedures.items():
            visible[name] = Entity("procedure", qualified=qualified, module=module_name)
        for name, specifics in table.generics.items():
            visible[name] = Entity(
                "generic",
                specifics=tuple(self._specific(table, item) for item in specifics),
                module=module_name,
            )
        exported = {name: entity for name, entity in visible.items() if table.is_public(name)}
        self._exporting.discard(module_name)
        self._exports[module_name] = exported
        return exported

    def _specific(self, table: ModuleTable, name: str) -> str:
        if name in table.procedures:
            return table.procedures[name]
        for use in table.uses:
            entity = self._imported(use).get(name)
            if entity is not None and entity.qualified:
                return entity.qualified
        if name in table.externals:
            return self.externals.get(name, f"external::{name}")
        return f"?::{name}"

    def _imported(self, use: UseSpec) -> dict[str, Entity]:
        table = self.exports(use.module)
        known = use.module in self.modules
        result: dict[str, Entity] = {}
        if use.only is not None:
            for local, remote in use.only:
                entity = table.get(remote)
                if entity is None:
                    entity = Entity("library" if not known else "unresolved", module=use.module, qualified=remote)
                result[local] = entity
            return result
        renamed = {remote: local for local, remote in use.renames}
        for remote, entity in table.items():
            result[renamed.get(remote, remote)] = entity
        if not known:
            result["*"] = Entity("library", module=use.module)
        return result

    # -- resolution ----------------------------------------------------------

    def lookup(self, scope: ProcedureScope, name: str) -> Entity:
        """Resolve ``name`` as the procedure's statements would see it."""

        if name in scope.dummies and (name in scope.local_externals or name in scope.proc_decls):
            # a procedure argument, whether declared through an interface block or a procedure statement
            return Entity("dummy_procedure")
        if name in scope.local_externals:
            return Entity("external", qualified=self.externals.get(name))
        if name in scope.proc_decls:
            return Entity("procedure_pointer")
        if name in scope.locals:
            return Entity("variable")
        if name in scope.internal:
            return Entity("procedure", qualified=scope.internal[name])
        if name in scope.local_types:
            return Entity("type")
        if name in scope.local_generics:
            return Entity("generic", specifics=tuple(self._resolve_specific(scope, item) for item in scope.local_generics[name]))
        found = self._from_uses(scope.uses, name)
        if found is not None:
            return found
        if scope.host is not None and scope.host in self.procedures:
            host = self.procedures[scope.host]
            if name == host.name:
                return Entity("procedure", qualified=host.qualified)
            return self.lookup(host, name)
        if scope.module is not None:
            table = self.modules.get(scope.module)
            if table is not None:
                if name in table.procedures:
                    return Entity("procedure", qualified=table.procedures[name], module=scope.module)
                if name in table.generics:
                    return Entity("generic", specifics=tuple(self._specific(table, item) for item in table.generics[name]), module=scope.module)
                if name in table.variables:
                    return Entity("variable", module=scope.module)
                if name in table.types:
                    return Entity("type", module=scope.module)
                if name in table.externals:
                    return Entity("external", qualified=self.externals.get(name), module=scope.module)
                found = self._from_uses(table.uses, name)
                if found is not None:
                    return found
        if name in self.externals:
            return Entity("external", qualified=self.externals[name])
        if name in self.intrinsics:
            return Entity("intrinsic")
        return Entity("unresolved")

    def _resolve_specific(self, scope: ProcedureScope, name: str) -> str:
        entity = self.lookup(scope, name)
        return entity.qualified or f"?::{name}"

    def _from_uses(self, uses: Sequence[UseSpec], name: str) -> Entity | None:
        wildcard_library: Entity | None = None
        for use in uses:
            imported = self._imported(use)
            if name in imported:
                return imported[name]
            if "*" in imported and wildcard_library is None:
                wildcard_library = Entity("library", module=use.module)
        # a wildcard import from a module outside the tree may supply the name
        return wildcard_library

    def lookup_bound(self, scope: ProcedureScope, designator: str) -> Entity | None:
        """``object%binding`` through the object's declared type; ``None`` when it is a component."""

        base, _, binding = designator.partition("%")
        base = re.sub(r"\(.*", "", base)
        binding = re.sub(r"\(.*", "", binding)
        if not binding or "%" in binding:
            return Entity("unresolved") if "%" in binding else None
        type_name = self._variable_type(scope, base)
        if type_name is None:
            return None
        for table in self.modules.values():
            bindings = table.type_bindings.get(type_name)
            if bindings is None or binding not in bindings:
                continue
            targets = tuple(table.procedures.get(name, self._specific(table, name)) for name in bindings[binding])
            if len(targets) == 1:
                return Entity("procedure", qualified=targets[0], module=table.name)
            return Entity("generic", specifics=targets, module=table.name)
        return None

    def _variable_type(self, scope: ProcedureScope, name: str) -> str | None:
        if name in scope.var_types:
            return scope.var_types[name]
        if scope.host is not None and scope.host in self.procedures:
            found = self._variable_type(self.procedures[scope.host], name)
            if found is not None:
                return found
        table = self.modules.get(scope.module or "")
        if table is not None and name in table.variable_types:
            return table.variable_types[name]
        for use_list in (scope.uses, table.uses if table is not None else ()):
            for use in use_list:
                origin = self.modules.get(use.module)
                if origin is None:
                    continue
                remote = name
                if use.only is not None:
                    matches = [remote_name for local, remote_name in use.only if local == name]
                    if not matches:
                        continue
                    remote = matches[0]
                if remote in origin.variable_types:
                    return origin.variable_types[remote]
        return None

    def resolve(self) -> None:
        for qualified, scope in self.procedures.items():
            sites: list[CallSite] = []
            for reference in scope.references:
                if reference.hint in ("bound_call", "bound_ref"):
                    entity = self.lookup_bound(scope, reference.name)
                    if entity is None:
                        if reference.hint == "bound_call":
                            sites.append(self._site(scope, reference, "bound_call", Entity("unresolved")))
                        continue
                    kind = "call" if reference.hint == "bound_call" else "function"
                    if entity.kind == "generic":
                        kind = "generic"
                    elif entity.kind == "unresolved":
                        kind = "bound_call"
                    sites.append(self._site(scope, reference, kind, entity))
                    continue
                entity = self.lookup(scope, reference.name)
                if entity.kind in ("variable", "type", "intrinsic"):
                    continue
                if entity.kind == "procedure":
                    kind = "call" if reference.hint == "call" else "function"
                elif entity.kind == "external":
                    kind = "external"
                elif entity.kind == "library":
                    kind = "library"
                else:
                    kind = entity.kind
                sites.append(self._site(scope, reference, kind, entity))
            self.sites[qualified] = tuple(sites)

    @staticmethod
    def _site(scope: ProcedureScope, reference: RawReference, kind: str, entity: Entity) -> CallSite:
        return CallSite(
            caller=scope.qualified,
            name=reference.name,
            kind=kind,
            target=entity.qualified,
            candidates=entity.specifics,
            module=entity.module,
            line=reference.line,
            active=reference.active,
            cpp_condition=reference.cpp_condition,
            guards=reference.guards,
            loop_depth=reference.loop_depth,
            constructs=reference.constructs,
            in_expression=reference.in_expression,
            as_argument=reference.as_argument,
            statement_kind=reference.statement_kind,
            statement=reference.statement,
        )

    # -- graph ------------------------------------------------------------------

    def callees(self, qualified: str, *, decide: Any = None) -> set[str]:
        """Procedures ``qualified`` can call in this configuration.

        ``decide(site)`` returns ``False`` for a site the configuration rules
        out; sites the preprocessor removed never count.
        """

        result: set[str] = set()
        for site in self.sites.get(qualified, ()):
            if not site.active:
                continue
            if decide is not None and decide(site) is False:
                continue
            if site.target is not None and site.target in self.procedures:
                result.add(site.target)
            for candidate in site.candidates:
                if candidate in self.procedures:
                    result.add(candidate)
        return result

    def reach(self, roots: Iterable[str], *, decide: Any = None) -> dict[str, int]:
        """Breadth-first closure from ``roots``: qualified name -> distance."""

        distance: dict[str, int] = {}
        queue: list[str] = []
        for root in roots:
            if root in self.procedures and root not in distance:
                distance[root] = 0
                queue.append(root)
        index = 0
        while index < len(queue):
            current = queue[index]
            index += 1
            for callee in sorted(self.callees(current, decide=decide)):
                if callee not in distance:
                    distance[callee] = distance[current] + 1
                    queue.append(callee)
        return distance

    def failures(self) -> list[dict[str, Any]]:
        return [
            {"source": scan.source, **scan.failure}
            for scan in self.scans
            if scan.failure is not None
        ]


def _intrinsic_names() -> frozenset[str]:
    try:
        from fparser.two import Fortran2003 as F
    except Exception:  # pragma: no cover - fparser is a dependency
        return frozenset()
    names: set[str] = set()
    for attribute in ("generic_function_names", "specific_function_names", "subroutine_names", "function_names"):
        values = getattr(F.Intrinsic_Name, attribute, None)
        if values:
            names.update(str(item).lower() for item in values)
    names.update(EXTRA_INTRINSICS)
    return frozenset(names)


# Fortran 2008 and compiler-extension intrinsics fparser's table does not list.
EXTRA_INTRINSICS = frozenset(
    {
        "erf", "erfc", "erfc_scaled", "gamma", "log_gamma", "hypot", "norm2",
        "bessel_j0", "bessel_j1", "bessel_jn", "bessel_y0", "bessel_y1", "bessel_yn",
        "acosh", "asinh", "atanh", "is_iostat_end", "is_iostat_eor", "move_alloc",
        "command_argument_count", "get_command_argument", "get_environment_variable",
        "c_loc", "c_funloc", "c_associated", "c_f_pointer", "c_f_procpointer", "c_sizeof",
        "int8", "iargc", "getarg", "flush", "abort", "sleep", "system", "etime", "dtime",
        "ieee_is_nan", "ieee_is_finite", "ieee_value", "ieee_support_datatype",
    }
)


def select_sources(directories: Sequence[str | Path], *, generated: Sequence[str | Path] = ()) -> dict[str, Path]:
    """The files the build compiles: one per basename, the first directory wins.

    ``directories`` is the build's ordered search path (each directory
    itself, not its subtree), as CESM's ``Filepath`` lists it; ``generated``
    files (templates expanded by genf90) take the precedence of the
    directory their template lives in, so they are passed first.
    """

    chosen: dict[str, Path] = {}
    for path in generated:
        chosen.setdefault(Path(path).name.lower(), Path(path))
    for directory in directories:
        base = Path(directory)
        if not base.is_dir():
            continue
        for path in sorted(base.iterdir()):
            if path.is_file() and path.suffix in FORTRAN_SUFFIXES:
                chosen.setdefault(path.name.lower(), path)
    return chosen


def expand_templates(
    templates: Sequence[str | Path], *, genf90: str | Path, output_dir: str | Path
) -> dict[str, Path]:
    """Run genf90 on ``.F90.in`` templates the way the build does; template -> file."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    produced: dict[str, Path] = {}
    for template in templates:
        source = Path(template)
        target = out / source.name[: -len(".in")]
        completed = subprocess.run(
            ["perl", str(Path(genf90).resolve()), str(source.resolve())],
            capture_output=True,
            text=True,
            errors="replace",
            cwd=str(out),
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()
            raise PICAMConfigurationError(
                f"genf90 failed on {source}: {detail[-1] if detail else completed.returncode}"
            )
        target.write_text(completed.stdout)
        produced[str(source)] = target
    return produced


def fortran_files(roots: Sequence[str | Path], *, exclude: Sequence[str] = ()) -> list[Path]:
    """Every Fortran source under ``roots`` in a stable order."""

    files: list[Path] = []
    for root in roots:
        base = Path(root)
        for path in sorted(base.rglob("*")):
            if path.suffix in FORTRAN_SUFFIXES and path.is_file():
                portable = path.as_posix()
                if any(re.search(pattern, portable) for pattern in exclude):
                    continue
                files.append(path)
    return files
