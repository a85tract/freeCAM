"""Properties of the Python source itself that no behavioural test sees directly."""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "src" / "freecam"

#: decorators under which the same name is legitimately defined more than once in a class body
_REDEFINING = {"setter", "getter", "deleter", "overload", "register"}


def _duplicate_methods(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    duplicates: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        first: dict[str, int] = {}
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = {d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", None) for d in item.decorator_list}
            if decorators & _REDEFINING:
                continue
            if item.name in first:
                duplicates.append(f"{path.relative_to(REPO)}:{item.lineno} {node.name}.{item.name} "
                                  f"is already defined at line {first[item.name]}")
            first.setdefault(item.name, item.lineno)
    return duplicates


def test_no_class_defines_a_method_twice() -> None:
    """The later definition silently replaces the earlier: Radiation.prepare_segmented was defined twice
    and two fifty-step runs (7418304, 7418305) ran with the process plugin never bound."""

    duplicates = [line for path in sorted(SOURCE.rglob("*.py")) for line in _duplicate_methods(path)]
    assert not duplicates, "\n".join(duplicates)
