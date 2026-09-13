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


def test_every_record_names_the_image_by_a_repo_relative_manifest_path() -> None:
    """A record's ``native_manifest`` is the image's manifest under ``build/``: no site directory, no user."""

    offenders = []
    for path in sorted((REPO / "validation").glob("*.json")):
        for line in path.read_text().splitlines():
            if '"native_manifest": "/' in line:
                offenders.append(f"{path.relative_to(REPO)}: {line.strip()[:120]}")
    assert not offenders, "\n".join(offenders)


def _tracked_text_files() -> list[Path]:
    import subprocess

    names = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split("\n")
    return [REPO / name for name in names if name and (REPO / name).is_file()]


def test_tracked_files_name_no_site_directory_user_or_allocation() -> None:
    """No committed file names a user or an allocation, and no record names a site directory.

    Records spell their paths through the site's variables (``${FREECAM_SCRATCH}``,
    ``${FREECAM_CASES}``, ``${WORK}``) or relative to this checkout, as the health records
    always did.  The login name and the account are site facts (``site.env``): they are
    forbidden by value where a site is configured, and never appear in the source itself.
    """

    import getpass
    import re

    site = REPO / "site.env"
    tokens: list[str] = []
    if site.exists():
        tokens.append(getpass.getuser())
        for line in site.read_text().splitlines():
            if line.startswith("FREECAM_ACCOUNT=") and line.split("=", 1)[1].strip().strip('"'):
                tokens.append(line.split("=", 1)[1].strip().strip('"'))
    allocation = re.compile(r"(?<![A-Za-z0-9])UCUB\d{4}(?![A-Za-z0-9])")
    # a per-user root followed by a login name; "${USER}", "$USER", "<owner>" and "example_user" do not match
    user_root = re.compile(r"/glade/(?:derecho/scratch|work|u/home)/([A-Za-z][A-Za-z0-9]*)(?![A-Za-z0-9_])")
    offenders: list[str] = []
    for path in _tracked_text_files():
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        where = path.relative_to(REPO)
        for match in user_root.finditer(text):
            offenders.append(f"{where}: a site directory of a user ({match.group(1)})")
        if allocation.search(text):
            offenders.append(f"{where}: an allocation name")
        for token in tokens:
            if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])", text):
                offenders.append(f"{where}: names the site's user or account")
    assert not offenders, "\n".join(sorted(set(offenders))[:40])
