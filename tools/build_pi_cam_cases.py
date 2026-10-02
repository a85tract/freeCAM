#!/usr/bin/env python3
"""Make the PI-atm CESM cases from the repository's recipe, under one build root.

    tools/build_pi_cam_cases.py --root DIR source              # each variant's CESM source, and the control source
    tools/build_pi_cam_cases.py --root DIR create [--case NAME ...] [--pcols N]
    tools/build_pi_cam_cases.py --root DIR compare --with CASES [--record JSON]
    tools/build_pi_cam_cases.py --root DIR build [--case NAME ...]   # case.build: inside a CPU job
    tools/build_pi_cam_cases.py --root DIR run --case oracle [--no-batch]   # case.submit: the 50-step run

The recipe is native/pi_cam/cesm_source/cases.yaml.  The root holds

    source/<variant>/     tools/prepare_cesm_source.py's tree for each variant (provider: the coupler library's)
    source/control/       tools/prepare_pi_cam_source.py's tree, which the state SourceMods are made from
    cases/<case name>/    the case directories
    output/<output>/      each case's CIME_OUTPUT_ROOT: its bld/ and run/

``compare`` sets every configured value of a case beside the hand-made case of
the same name -- env_*.xml, user_nl_*, Macros.make, the machine environment --
with each case's own roots, mapping directory, account and user written as
names, and reports what differs.  Nothing here is submitted to a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import string
import subprocess
import sys
from typing import Any, Iterable, Iterator, Mapping
import xml.etree.ElementTree as ElementTree

import yaml

REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "native/pi_cam/cesm_source/cases.yaml"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from freecam import site  # noqa: E402

#: the configuration files a case's compare reads besides env_*.xml
CONFIGURATION_FILES = ("Macros.make", "env_mach_specific.xml")
#: env entries that say where a case is, which every case has its own of
LOCATIONS = ("CASEROOT", "SRCROOT", "CIMEROOT", "CIME_OUTPUT_ROOT", "ATM_DOMAIN_PATH")
#: the generated SourceMod the state case's own build leaves out
STATE_BUILD_EXCLUDES = ("physpkg.F90",)
#: CAM's columns a chunk when its configure is not told otherwise
DEFAULT_PCOLS = 16


def load_recipe(path: Path = RECIPE) -> dict[str, Any]:
    recipe = yaml.safe_load(path.read_text())
    if int(recipe.get("schema_version", 0)) != 1:
        raise SystemExit(f"{path}: schema_version 1 is required")
    for key, case in recipe["cases"].items():
        if case.get("clone") and case["clone"] not in recipe["cases"]:
            raise SystemExit(f"case {key} clones {case['clone']}, which the recipe does not make")
    return recipe


def fill(template: str, values: Mapping[str, str]) -> str:
    """``template`` with its ``${NAME}`` filled; a name with no value is refused."""

    try:
        return string.Template(template).substitute(values)
    except KeyError as missing:
        raise SystemExit(f"the recipe names ${{{missing.args[0]}}}, which has no value") from None


def xmlchange_commands(recipe: Mapping[str, Any], values: Mapping[str, str]) -> list[list[str]]:
    """Every ``./xmlchange`` the recipe makes, in order, with its values filled."""

    return [["./xmlchange", *shlex.split(fill(change, values))] for change in recipe["xmlchange"]]


def mappings() -> str:
    """The domain and mapping directory: FREECAM_PI_ATM_MAPPINGS, else the reference case's."""

    value = site.setting("FREECAM_PI_ATM_MAPPINGS")
    if value:
        return value
    reference = site.resolved()["reference case"]
    if isinstance(reference, Path) and (reference / "env_run.xml").is_file():
        return env_entries(reference)["ATM_DOMAIN_PATH"]
    raise SystemExit("set FREECAM_PI_ATM_MAPPINGS in site.env: the directory of the ne16np4/gx1v6/r05 "
                     "domain and mapping files the PI-atm cases name")


class Layout:
    """Where a build root keeps each part."""

    def __init__(self, root: Path, recipe: Mapping[str, Any]):
        self.root = root.resolve()
        self.recipe = recipe

    def source(self, variant: str) -> Path:
        return self.root / "source" / variant

    @property
    def control(self) -> Path:
        return self.root / "source/control"

    def case(self, key: str) -> Path:
        return self.root / "cases" / self.recipe["cases"][key]["name"]

    def output(self, key: str) -> Path:
        return self.root / "output" / self.recipe["cases"][key]["output"]

    def values(self, key: str) -> dict[str, str]:
        case = self.recipe["cases"][key]
        account = site.setting("FREECAM_ACCOUNT")
        if not account:
            raise SystemExit("set FREECAM_ACCOUNT in site.env: the cases charge their runs to it")
        return {"MAPPINGS": mappings(), "OUTPUT_ROOT": str(self.output(key)), "CASE": case["name"],
                "ACCOUNT": account, "QUEUE": case.get("queue", "develop")}


def _run(command: list[str], cwd: Path, **options: Any) -> None:
    print("+", " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True, **options)


def prepare_sources(layout: Layout, *, force: bool = False) -> None:
    """Every variant's CESM source (the cases', and the coupler library's), and the control source."""

    import prepare_cesm_source

    for variant in prepare_cesm_source.load_recipe()["variants"]:
        prepare_cesm_source.prepare(layout.source(variant), variant=variant, force=force)
    if layout.control.exists() and not force:
        raise SystemExit(f"{layout.control} exists; pass --force to replace it")
    shutil.rmtree(layout.control, ignore_errors=True)
    _run([sys.executable, str(REPO / "tools/prepare_pi_cam_source.py"), "--output", str(layout.control)], REPO)


def create(layout: Layout, key: str, *, pcols: int | None = None) -> Path:
    """Make one case: create_newcase (or create_clone), its settings, case.setup, its namelists.

    ``pcols`` other than CAM's default (16) is appended to CAM_CONFIG_OPTS before case.setup;
    a clone takes it from the case it clones.
    """

    case = layout.recipe["cases"][key]
    directory = layout.case(key)
    if directory.exists():
        raise SystemExit(f"{directory} exists; a case is made once")
    directory.parent.mkdir(parents=True, exist_ok=True)
    scripts = layout.source(case["variant"]) / "cime/scripts"
    values = layout.values(key)
    if case.get("clone"):
        _run([str(scripts / "create_clone"), "--case", str(directory), "--clone", str(layout.case(case["clone"])),
              "--cime-output-root", str(layout.output(key)), "--project", values["ACCOUNT"],
              # a clone keeps its original's machine directory unless told: its own source's
              "--mach-dir", str(layout.source(case["variant"]) / "cime/config/cesm/machines")], layout.root)
        _run(["./case.setup"], directory)
    else:
        _run([str(scripts / "create_newcase"), "--case", str(directory), *layout.recipe["create_newcase"],
              "--project", values["ACCOUNT"]], layout.root)
        for command in xmlchange_commands(layout.recipe, values):
            _run(command, directory)
        if pcols is not None and int(pcols) != DEFAULT_PCOLS:
            _run(["./xmlchange", f"CAM_CONFIG_OPTS=-pcols {int(pcols)}", "--append"], directory)
        _run(["./case.setup"], directory)
        for component, lines in layout.recipe["user_nl"].items():
            with (directory / f"user_nl_{component}").open("a") as handle:
                handle.write(lines)
    set_model_version(directory, layout.source(case["variant"]))
    if case.get("python_state"):
        write_python_state(layout, directory)
    return directory


def set_model_version(directory: Path, source: Path) -> None:
    """MODEL_VERSION as CIME would have named the source by ``git describe``, which a prepared tree has no git for.

    case.setup has locked env_case.xml by then; it is locked again with the value, as CIME's own lock_file does.
    """

    version = json.loads((source / ".cesm-source.json").read_text())["model_version"]
    _run(["./xmlchange", f"MODEL_VERSION={version}"], directory)
    locked = directory / "LockedFiles/env_case.xml"
    if locked.is_file():
        shutil.copy2(directory / "env_case.xml", locked)


def write_python_state(layout: Layout, directory: Path) -> None:
    """The state case's SourceMods, without the ones its own build leaves to the image."""

    output = directory / "SourceMods/src.cam"
    _run([sys.executable, str(REPO / "tools/generate_pi_cam_python_state_source.py"),
          "--source-root", str(layout.control), "--output-dir", str(output)], REPO)
    for name in STATE_BUILD_EXCLUDES:
        (output / name).unlink(missing_ok=True)


def build(layout: Layout, key: str) -> None:
    _run(["./case.build"], layout.case(key))


def submit(layout: Layout, key: str, queue: str | None = None, *, no_batch: bool = False) -> None:
    """case.submit; ``queue`` replaces the recipe's, which only says where the run waits.

    ``no_batch`` runs the case in this shell -- inside a job that already holds its nodes.
    """

    if queue:
        _run(["./xmlchange", f"JOB_QUEUE={queue}", "--force"], layout.case(key))
    _run(["./case.submit", *(["--no-batch"] if no_batch else [])], layout.case(key))


def _env_items(directory: Path) -> Iterator[tuple[str, str, str]]:
    """Every ``<entry id= value=>`` of a case's env_*.xml, as (file, id, value)."""

    for path in sorted(directory.glob("env_*.xml")):
        if path.name == "env_mach_specific.xml":
            continue
        for element in ElementTree.parse(path).iter("entry"):
            if "value" in element.attrib:
                yield path.name, element.attrib["id"], element.attrib["value"]


def env_entries(directory: Path) -> dict[str, str]:
    """A case's env values by id."""

    return {name: value for _, name, value in _env_items(directory)}


def _spellings(directory: Path) -> list[tuple[str, str]]:
    """What a case says about where it is and whose it is, longest first, and the name each is read as."""

    entries = env_entries(directory)
    names: dict[str, str] = {}
    for name in LOCATIONS:
        value = entries.get(name)
        if value:
            for spelling in {value, os.path.normpath(value)}:
                names[spelling] = f"<{name}>"
    for name in ("PROJECT", "CHARGE_ACCOUNT"):
        if entries.get(name):
            names[entries[name]] = "<ACCOUNT>"
    for name in ("USER", "REALUSER", "CCSMUSER"):
        if entries.get(name):
            names[entries[name]] = "<USER>"
    return sorted(names.items(), key=lambda item: -len(item[0]))


def configuration(directory: Path) -> dict[str, str]:
    """A case's configured values with its own locations and owner written as names."""

    spellings = _spellings(directory)

    def spelled(text: str) -> str:
        for value, name in spellings:
            text = text.replace(value, name)
        return text

    values = {f"{file}:{name}": spelled(value) for file, name, value in _env_items(directory)}
    for path in sorted(directory.glob("user_nl_*")):
        values[path.name] = path.read_text()
    for name in CONFIGURATION_FILES:
        if (directory / name).is_file():
            values[name] = spelled((directory / name).read_text())
    return values


def compare_case(made: Path, hand_made: Path) -> dict[str, Any]:
    ours, theirs = configuration(made), configuration(hand_made)
    differ = sorted(key for key in ours.keys() & theirs.keys() if ours[key] != theirs[key])
    return {"case": made.name, "same": len(ours.keys() & theirs.keys()) - len(differ), "differ": differ,
            "only_made": sorted(ours.keys() - theirs.keys()), "only_hand_made": sorted(theirs.keys() - ours.keys()),
            "values": {key: {"made": ours[key], "hand_made": theirs[key]} for key in differ if "\n" not in ours[key]}}


def _selected(recipe: Mapping[str, Any], wanted: Iterable[str] | None) -> list[str]:
    keys = list(recipe["cases"])
    chosen = list(wanted or keys)
    unknown = [key for key in chosen if key not in keys]
    if unknown:
        raise SystemExit(f"the recipe makes no case {unknown}; it makes {keys}")
    return [key for key in keys if key in chosen]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("stage", choices=("source", "create", "compare", "build", "run"))
    parser.add_argument("--case", action="append", help="a case of the recipe (repeatable; default every case)")
    parser.add_argument("--with", dest="hand_made", type=Path, help="compare: the directory of hand-made cases")
    parser.add_argument("--record", type=Path, help="compare: write the result here (JSON)")
    parser.add_argument("--force", action="store_true", help="source: replace prepared trees")
    parser.add_argument("--queue", help="run: the PBS queue (develop takes at most two CPU nodes; 512 ranks need main)")
    parser.add_argument("--no-batch", action="store_true", help="run: in this shell, inside a job holding the nodes")
    parser.add_argument("--pcols", type=int, help="create: CAM's columns a chunk (default 16, CAM's own)")
    arguments = parser.parse_args(argv)
    recipe = load_recipe()
    layout = Layout(arguments.root, recipe)
    keys = _selected(recipe, arguments.case)
    if arguments.stage == "source":
        prepare_sources(layout, force=arguments.force)
    elif arguments.stage == "create":
        for key in keys:
            create(layout, key, pcols=arguments.pcols)
    elif arguments.stage == "build":
        for key in keys:
            build(layout, key)
    elif arguments.stage == "run":
        for key in keys:
            submit(layout, key, arguments.queue, no_batch=arguments.no_batch)
    else:
        if arguments.hand_made is None:
            raise SystemExit("compare needs --with")
        results = [compare_case(layout.case(key), arguments.hand_made / recipe["cases"][key]["name"]) for key in keys]
        for result in results:
            print(json.dumps({key: result[key] for key in ("case", "same", "differ", "only_made", "only_hand_made")}))
        if arguments.record is not None:
            arguments.record.write_text(json.dumps({"schema_version": 1, "cases": results}, indent=2) + "\n")
        return 0 if all(not (r["differ"] or r["only_made"] or r["only_hand_made"]) for r in results) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
