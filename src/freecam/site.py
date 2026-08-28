"""Where this installation sits, and which allocation it may charge.

Nothing in this repository names a user or a project.  A site is described
once, in ``site.env`` at the repository root, in a form both readers
understand: plain ``KEY=value`` lines that bash can ``source`` and this
module can parse.  ``validation/jobs/common.sh`` reads the same file, so a
notebook and a PBS job cannot disagree about where the model lives.

Resolution order for every setting, most specific first:

1. an explicit argument, where the caller passes one;
2. the process environment;
3. ``site.env`` at the repository root;
4. a derived default, where one exists that names nobody.

A setting with no value is reported as unset by name.  It is never quietly
replaced by whatever the login shell happened to export -- charging someone
else's allocation is the kind of mistake that should be loud.

``site.env`` is deliberately not committed; ``site.env.example`` is.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

__all__ = [
    "Check",
    "Setting",
    "SETTINGS",
    "load",
    "main",
    "missing",
    "origin",
    "path",
    "preflight",
    "report",
    "repository_root",
    "resolved",
    "setting",
    "site_file",
]

SITE_FILE_NAME = "site.env"

#: A value that means "explicitly nothing", as PBS writes for an unset
#: allocation.  Treated as absent rather than as the literal string.
_EMPTY = {None, "", "N/A"}


@dataclass(frozen=True, slots=True)
class Setting:
    """One site fact, its meaning, and how it is obtained when unset."""

    name: str
    summary: str
    produced_by: str | None = None
    required: bool = False


SETTINGS: tuple[Setting, ...] = (
    Setting(
        "FREECAM_ACCOUNT",
        "PBS allocation charged for every job this checkout submits",
        "your NCAR project code, e.g. from `sacctmgr` or the CESM case's "
        "env_batch.xml CHARGE_ACCOUNT",
        required=True,
    ),
    Setting(
        "FREECAM_SCRATCH",
        "root for run directories and generated data",
        "defaults to $SCRATCH, then /glade/derecho/scratch/$USER",
    ),
    Setting(
        "FREECAM_CASES",
        "directory holding the configured CESM cases",
        "defaults to <repository>/../CESM_cases",
    ),
    Setting(
        "FREECAM_REFERENCE_CASE",
        "the configured CESM case supplying the machine environment",
        "CESM create_newcase/case.setup; overrides FREECAM_CASES for the "
        "reference",
    ),
    Setting(
        "FREECAM_REFERENCE_RUN",
        "run directory of the oracle whose atm_in and initial state are copied",
        "one completed run of the reference case",
    ),
    Setting(
        "FREECAM_QUEUE",
        "PBS queue for interactive sessions",
        "defaults to develop",
    ),
    Setting(
        "FREECAM_CESM_PROVIDER_LIBRARY",
        "the online CESM surface and coupler components, which the default "
        "case runs live",
        "validation/jobs/pi_cam_online_coupler_build.pbs",
    ),
    Setting(
        "FREECAM_CESM_PROVIDER_SEED",
        "the CESM run the online provider seeds its components from",
        "one completed run of the original coupled model",
    ),
    Setting(
        "FREECAM_NATIVE_MANIFEST",
        "an existing native image to run against instead of building one",
        "validation/jobs/pi_cam_promoted_statepool_build.pbs; the image is "
        "compiled in place and records absolute paths, so it is pointed at, "
        "never copied",
    ),
    Setting(
        "FREECAM_SURROGATE",
        "directory holding a trained kernel surrogate and its anchors",
        "validation/jobs/pi_cam_mmacro_dataset_build.pbs; the most recent "
        "such run under FREECAM_SCRATCH is used when this is unset",
    ),
    Setting(
        "FREECAM_CAPTURE",
        "directory of published kernel-argument capture bundles",
        "validation/jobs/pi_cam_function_capture_training.pbs, or an "
        "existing readable one",
    ),
)

_SETTINGS_BY_NAME = {entry.name: entry for entry in SETTINGS}

# ``KEY=value`` and ``export KEY=value``.  Anything else in the file is an
# error rather than a line silently skipped: bash would act on it.
_ASSIGNMENT = re.compile(
    r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$"
)
_REFERENCE = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)"
)

_cache: dict[Path, tuple[float, dict[str, str]]] = {}


def repository_root(start: str | Path | None = None) -> Path:
    """The checkout containing ``start``, identified by ``pyproject.toml``."""

    here = Path(start or __file__).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(
        f"no freeCAM checkout above {here}: no pyproject.toml found"
    )


def site_file(repo: str | Path | None = None) -> Path:
    """Path of the site description, whether or not it exists."""

    return Path(repo or repository_root()).resolve() / SITE_FILE_NAME


def _expand(value: str, known: Mapping[str, str], source: Path, line: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(3)
        default = match.group(2)
        for space in (known, os.environ):
            resolved = space.get(name)
            if resolved not in _EMPTY:
                return str(resolved)
        if default is not None:
            return default
        raise ValueError(
            f"{source}: {line.strip()!r} refers to ${name}, which is unset; "
            f"write ${{{name}:-<default>}} or set it before reading the file"
        )

    return _REFERENCE.sub(replace, value)


def load(repo: str | Path | None = None) -> Mapping[str, str]:
    """Parse ``site.env``; an absent file is an empty site, not an error."""

    source = site_file(repo)
    try:
        stamp = source.stat().st_mtime
    except FileNotFoundError:
        return {}
    cached = _cache.get(source)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    values: dict[str, str] = {}
    for line in source.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ASSIGNMENT.match(line)
        if match is None:
            raise ValueError(
                f"{source}: {stripped!r} is not KEY=value; the file is read "
                "by both bash and Python, so it holds assignments only"
            )
        name, raw = match.groups()
        raw = raw.split(" #", 1)[0].rstrip() if not raw.startswith(("'", '"')) else raw
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
            literal = raw[0] == "'"
            raw = raw[1:-1]
        else:
            literal = False
        values[name] = raw if literal else _expand(raw, values, source, line)
    _cache[source] = (stamp, values)
    return values


def setting(
    name: str,
    default: str | None = None,
    *,
    repo: str | Path | None = None,
) -> str | None:
    """Resolve one setting: environment, then ``site.env``, then ``default``."""

    from_environment = os.environ.get(name)
    if from_environment not in _EMPTY:
        return from_environment
    from_file = load(repo).get(name)
    if from_file not in _EMPTY:
        return from_file
    return default


def origin(name: str, *, repo: str | Path | None = None) -> str:
    """Where :func:`setting` would take ``name`` from, for a report."""

    if os.environ.get(name) not in _EMPTY:
        return "environment"
    if load(repo).get(name) not in _EMPTY:
        return str(site_file(repo))
    return "unset"


def path(
    name: str,
    default: str | Path | None = None,
    *,
    repo: str | Path | None = None,
) -> Path | None:
    """:func:`setting` as a resolved path."""

    value = setting(name, None, repo=repo)
    if value is None:
        value = None if default is None else str(default)
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def missing(
    names: Iterable[str] | None = None, *, repo: str | Path | None = None
) -> tuple[Setting, ...]:
    """Declared settings with no value, so a preflight can name them."""

    wanted = (
        tuple(_SETTINGS_BY_NAME[name] for name in names)
        if names is not None
        else tuple(entry for entry in SETTINGS if entry.required)
    )
    return tuple(
        entry for entry in wanted if setting(entry.name, repo=repo) is None
    )


def report(
    *, repo: str | Path | None = None
) -> tuple[tuple[str, str | None, str, str], ...]:
    """``(name, value, origin, summary)`` for every declared setting."""

    return tuple(
        (
            entry.name,
            setting(entry.name, repo=repo),
            origin(entry.name, repo=repo),
            entry.summary,
        )
        for entry in SETTINGS
    )


# --- what a run needs that a clone cannot bring with it ----------------------


@dataclass(frozen=True, slots=True)
class Check:
    """One prerequisite, whether it is satisfied, and what produces it."""

    name: str
    ok: bool
    detail: str
    produced_by: str

    def __str__(self) -> str:
        mark = "ok " if self.ok else "MISSING"
        return f"{mark:<8}{self.name:<18}{self.detail}"


DEFAULT_CONFIG = "configs/pi_cam_icesm131.yaml"


def resolved(
    *, repo: str | Path | None = None, config: str | Path | None = None
) -> dict[str, Path | str | None]:
    """Every site path this checkout would use, after the resolution chain.

    The same answers ``freecam.Driver`` reaches, computed without importing
    the driver, so a preflight can run before anything is launched.
    """

    import yaml

    root = Path(repo or repository_root()).resolve()
    config_path = Path(config or root / DEFAULT_CONFIG)
    declared = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
    case_name = declared.get("case_name", "")
    scratch = path(
        "FREECAM_SCRATCH",
        os.environ.get("SCRATCH")
        or f"/glade/derecho/scratch/{os.environ.get('USER', 'unknown')}",
        repo=root,
    )
    cases = path("FREECAM_CASES", root.parent / "CESM_cases", repo=root)
    assert scratch is not None and cases is not None
    return {
        "repository": root,
        "site file": site_file(root),
        "account": setting("FREECAM_ACCOUNT", repo=root),
        "python": root / ".venv" / "bin" / "python",
        "scratch": scratch,
        "cases": cases,
        "reference case": path(
            "FREECAM_REFERENCE_CASE", cases / case_name, repo=root
        ),
        "reference run": path(
            "FREECAM_REFERENCE_RUN",
            scratch / "pyCAM" / "PI-cam" / case_name / "run",
            repo=root,
        ),
        "provider library": path(
            "FREECAM_CESM_PROVIDER_LIBRARY",
            root / "build/cesm/pi_atm/production-components/libpycesm_external_atm.so",
            repo=root,
        ),
        "provider seed": path(
            "FREECAM_CESM_PROVIDER_SEED",
            scratch / "pyCESM" / "PI-atm" / "oracle-1month" / "run",
            repo=root,
        ),
        "native manifest": path(
            "FREECAM_NATIVE_MANIFEST",
            root
            / declared.get(
                "native_manifest", "build/pi_cam_promoted/native_cam_manifest.json"
            ),
            repo=root,
        ),
        "queue": setting("FREECAM_QUEUE", "develop", repo=root),
    }


def preflight(
    *, repo: str | Path | None = None, config: str | Path | None = None
) -> tuple[Check, ...]:
    """Check what a 512-rank run needs, and name what produces anything absent."""

    from shutil import which

    where = resolved(repo=repo, config=config)
    account = where["account"]
    python = where["python"]
    reference_case = where["reference case"]
    reference_run = where["reference run"]
    manifest = where["native manifest"]
    library = where["provider library"]
    seed = where["provider seed"]
    assert isinstance(library, Path) and isinstance(seed, Path)
    assert isinstance(python, Path)
    assert isinstance(reference_case, Path) and isinstance(reference_run, Path)
    assert isinstance(manifest, Path)
    return (
        Check(
            "allocation",
            account is not None,
            str(account) if account else "no FREECAM_ACCOUNT",
            "set FREECAM_ACCOUNT in site.env; `qsub` lists the projects you "
            "may charge",
        ),
        Check(
            "environment",
            python.is_file(),
            str(python),
            "uv sync --extra notebook --extra test",
        ),
        Check(
            "scheduler",
            which("qsub") is not None,
            "qsub on PATH" if which("qsub") else "no qsub: not an NCAR machine",
            "run on Derecho; there is no local mode for the 512-rank case",
        ),
        Check(
            "native image",
            manifest.is_file(),
            str(manifest),
            "validation/jobs/pi_cam_promoted_statepool_build.pbs, after both "
            "CESM cases are built -- see the README's \"Building the native "
            "image\".  It is compiled in place and records absolute paths, so "
            "FREECAM_NATIVE_MANIFEST points at an existing one rather than "
            "copying it",
        ),
        Check(
            "provider library",
            library.is_file(),
            str(library),
            "validation/jobs/pi_cam_online_coupler_build.pbs; the default "
            "case runs the original CESM surface components and coupler "
            "live, or point FREECAM_CESM_PROVIDER_LIBRARY at an existing "
            "build",
        ),
        Check(
            "provider seed",
            (seed / "drv_in").is_file(),
            str(seed),
            "one completed run of the original coupled model, which the "
            "online provider seeds from; or point FREECAM_CESM_PROVIDER_SEED "
            "at one",
        ),
        Check(
            "reference case",
            (reference_case / ".env_mach_specific.sh").is_file(),
            str(reference_case),
            "CESM create_newcase and case.setup; or point "
            "FREECAM_REFERENCE_CASE at an existing one",
        ),
        Check(
            "reference run",
            (reference_run / "atm_in").is_file(),
            str(reference_run),
            "one completed run of the reference case; or point "
            "FREECAM_REFERENCE_RUN at an existing one",
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Report what this checkout resolves to, and what it still needs."""

    root = repository_root()
    file = site_file(root)
    print(f"repository  {root}")
    print(f"site file   {file}{'' if file.is_file() else '   (absent)'}")
    print()
    sources = {
        "account": "FREECAM_ACCOUNT",
        "scratch": "FREECAM_SCRATCH",
        "cases": "FREECAM_CASES",
        "reference case": "FREECAM_REFERENCE_CASE",
        "reference run": "FREECAM_REFERENCE_RUN",
        "queue": "FREECAM_QUEUE",
    }
    where = resolved(repo=root)
    for name, value in where.items():
        if name in {"repository", "site file"}:
            continue
        key = sources.get(name)
        source = origin(key, repo=root) if key else ""
        if source == "unset":
            source = "derived"
        elif source.endswith(SITE_FILE_NAME):
            source = SITE_FILE_NAME
        print(f"  {name:<17}{value if value is not None else '-'}")
        if source:
            print(f"  {'':<17}from {source}")
    print()
    checks = preflight(repo=root)
    for check in checks:
        print(f"  {check}")
    absent = [check for check in checks if not check.ok]
    if absent:
        print()
        print("what is missing, and what produces it:")
        for check in absent:
            print(f"  {check.name}: {check.produced_by}")
    return 1 if absent else 0


if __name__ == "__main__":  # pragma: no cover - a command, exercised by hand
    raise SystemExit(main())
