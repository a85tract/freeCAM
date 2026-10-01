#!/usr/bin/env python3
"""Prepare the CESM source a PI-atm case is built from: the pinned iCESM submodule and its patches.

    tools/prepare_cesm_source.py --output DIR [--variant cases|state]
    tools/prepare_cesm_source.py --output DIR --compare-with CHECKOUT --as-of 2026-08-05T21:20 [--record JSON]

The recipe is native/pi_cam/cesm_source/source.yaml.  The tree is a copy of the
submodule at its pinned revisions (checked as tools/prepare_pi_cam_source.py
checks them), the recipe's patches applied component by component, and a
``.cesm-source.json`` naming the revisions and the sha256 of every patch.

``--compare-with`` reads every source file of a hand-made checkout as it was at
``--as-of`` -- a file modified since is taken from the checkout's own git --
and requires the prepared tree to hold the same bytes, apart from the recipe's
known differences.  That is the proof the recipe rebuilds what a case was built from.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import yaml

REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "native/pi_cam/cesm_source/source.yaml"
sys.path.insert(0, str(REPO / "tools"))

from prepare_pi_cam_source import PINNED_REVISIONS, ensure_pinned_externals  # noqa: E402

#: what a copy and a comparison leave out: VCS data, interpreter caches, built tools
IGNORED = (".git", "__pycache__")
IGNORED_SUFFIXES = (".pyc",)
#: compiled Python 2 namelist builders and the CLM interpinic tool's own build, left in the checkouts
IGNORED_PATHS = ("tools/clm4_0/interpinic", "buildnmlc")
COMPONENTS = ("components/cam", "components/clm", "components/cice", "components/rtm", "components/pop", "cime")


def _sha256(data: bytes) -> str:
    return sha256(data).hexdigest()


def load_recipe(path: Path = RECIPE) -> dict[str, Any]:
    recipe = yaml.safe_load(path.read_text())
    if int(recipe.get("schema_version", 0)) != 1:
        raise SystemExit(f"{path}: schema_version 1 is required")
    return recipe


def _patches(recipe: Mapping[str, Any], variant: str) -> list[Mapping[str, Any]]:
    variants = recipe.get("variants") or {}
    if variant not in variants:
        raise SystemExit(f"unknown variant {variant!r}; the recipe has {sorted(variants)}")
    return [*(recipe.get("patches") or ()), *(variants[variant].get("patches") or ())]


def _shown(path: Path) -> str:
    """``path`` as the repository names it; a file elsewhere by its name alone."""

    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return path.name


def _ignored(relative: str) -> bool:
    parts = relative.split("/")
    return (any(part in IGNORED for part in parts) or relative.endswith(IGNORED_SUFFIXES)
            or any(item in relative for item in IGNORED_PATHS))


def apply_patch(patch: Path, target: Path) -> None:
    """Apply ``patch`` to the directory ``target`` and prove it is there.

    Inside some other git work tree (``build/`` of this repository, say) ``git apply``
    reads the paths as relative to that tree's top and skips them without an error;
    the ceiling keeps it to ``target``.  The reverse check fails unless every hunk landed.
    """

    environment = {**os.environ, "GIT_CEILING_DIRECTORIES": str(target.resolve().parent)}
    for arguments, failure in ((["--whitespace=nowarn"], "does not apply to"),
                               (["--reverse", "--check"], "is not in")):
        result = subprocess.run(["git", "apply", *arguments, str(patch.resolve())], cwd=target,
                                env=environment, capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(f"{patch.name} {failure} {target.name}: {result.stderr.strip()}")


def prepare(output: Path, *, variant: str = "cases", recipe_path: Path = RECIPE, force: bool = False) -> Path:
    """Copy the pinned submodule to ``output`` and apply the recipe's patches for ``variant``."""

    recipe = load_recipe(recipe_path)
    source = (REPO / recipe["base"]).resolve()
    revisions = ensure_pinned_externals(source)
    if output.exists():
        if not force:
            raise SystemExit(f"{output} exists; pass --force to replace it")
        shutil.rmtree(output)
    shutil.copytree(source, output, symlinks=True,
                    ignore=lambda directory, names: [n for n in names if n in IGNORED or n.endswith(IGNORED_SUFFIXES)])
    applied = []
    for patch in _patches(recipe, variant):
        path = recipe_path.parent / patch["file"]
        apply_patch(path, output / patch["component"])
        applied.append({"component": patch["component"], "file": patch["file"], "sha256": _sha256(path.read_bytes())})
    record = {"schema_version": 1, "recipe": _shown(recipe_path), "variant": variant,
              "base": recipe["base"], "revisions": revisions, "patches": applied}
    (output / ".cesm-source.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return output


def _component_of(relative: str) -> str:
    for component in COMPONENTS:
        if relative == component or relative.startswith(component + "/"):
            return component
    return "."


def known_differences(recipe: Mapping[str, Any], variant: str) -> dict[str, str]:
    """The files a prepared ``variant`` may differ in from its hand-made checkout, and why."""

    _patches(recipe, variant)                       # refuses an unknown variant
    items = [*(recipe.get("known_differences") or ()),
             *(recipe["variants"][variant].get("known_differences") or ())]
    return {item["path"]: item["why"] for item in items}


def compare(prepared: Path, checkout: Path, as_of: datetime, *, variant: str = "cases",
            recipe_path: Path = RECIPE) -> dict[str, Any]:
    """Every source file of ``checkout`` as it was at ``as_of`` against ``prepared``."""

    known = known_differences(load_recipe(recipe_path), variant)
    cutoff = as_of.timestamp()
    counts = {"same": 0, "taken_from_git": 0, "known_differences": 0}
    later: list[str] = []
    problems: list[str] = []
    seen: set[str] = set()
    for path in sorted(checkout.rglob("*")):
        relative = str(path.relative_to(checkout))
        if _ignored(relative) or not (path.is_file() or path.is_symlink()):
            continue
        seen.add(relative)
        other = prepared / relative
        if relative in known:
            counts["known_differences"] += 1
            continue
        if path.is_symlink():
            # a link is the name of its target; the target is compared as a file of its own
            if other.is_symlink() and os.readlink(other) == os.readlink(path):
                counts["same"] += 1
            else:
                problems.append(f"link differs: {relative}")
            continue
        data = path.read_bytes()
        if path.stat().st_mtime > cutoff:
            # changed after the build: the build read what the component's git holds
            component = _component_of(relative)
            inner = relative[len(component) + 1:] if component != "." else relative
            shown = subprocess.run(["git", "-C", str(checkout / component), "show", f"HEAD:{inner}"], capture_output=True)
            if shown.returncode:
                problems.append(f"changed after {as_of.isoformat()} and not in git: {relative}")
                continue
            data = shown.stdout
            counts["taken_from_git"] += 1
            later.append(relative)
        if not other.is_file():
            problems.append(f"missing from the prepared tree: {relative}")
        elif _sha256(other.read_bytes()) != _sha256(data):
            problems.append(f"differs: {relative}")
        else:
            counts["same"] += 1
    for path in prepared.rglob("*"):
        relative = str(path.relative_to(prepared))
        if (path.is_file() or path.is_symlink()) and relative not in seen and not _ignored(relative) \
                and relative != ".cesm-source.json":
            problems.append(f"only in the prepared tree: {relative}")
    return {"identical": not problems, **counts, "taken_from_git_files": later, "problems": problems,
            "known": known, "as_of": as_of.isoformat(), "compared_with": checkout.name}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="cases", choices=("cases", "state"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compare-with", type=Path, help="a hand-made checkout the prepared tree must reproduce")
    parser.add_argument("--as-of", help="when that checkout's case was built (ISO time): later edits are read from its git")
    parser.add_argument("--record", type=Path, help="write the comparison here (JSON)")
    arguments = parser.parse_args(argv)
    prepare(arguments.output, variant=arguments.variant, force=arguments.force)
    print(f"prepared {arguments.output} ({arguments.variant}); pinned {len(PINNED_REVISIONS)} revisions")
    if arguments.compare_with is None:
        return 0
    if not arguments.as_of:
        raise SystemExit("--compare-with needs --as-of")
    result = compare(arguments.output, arguments.compare_with, datetime.fromisoformat(arguments.as_of),
                     variant=arguments.variant)
    summary = {key: result[key] for key in ("identical", "same", "taken_from_git", "known_differences")}
    print(json.dumps(summary), *result["problems"][:40], sep="\n")
    if arguments.record is not None:
        record = {"schema_version": 1, "variant": arguments.variant,
                  "source": json.loads((arguments.output / ".cesm-source.json").read_text()),
                  **{key: result[key] for key in ("identical", "same", "taken_from_git", "taken_from_git_files",
                                                  "known_differences", "problems", "known", "as_of",
                                                  "compared_with")}}
        arguments.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
