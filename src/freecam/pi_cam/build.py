"""Builds of the PI-atm model for compile-time options: ``fc.build(pcols=32)``.

A compile-time option -- CAM's columns a chunk (``pcols``), the device FTorch
is linked for -- changes machine code, so it changes everything made from it:
the CESM cases, their oracle, the native image, the coupler library, and the
evidence that they agree.  The installed model is the build of the default
options and is returned as it is; any other options get a build root of their
own under ``$FREECAM_SCRATCH/freeCAM/builds/`` and a chain of CPU jobs:

====================  ==================================================  =========================
stage                 job                                                 done when
====================  ==================================================  =========================
``cases``             the CESM sources and cases from the recipe          every case is set up
``cases-build``       case.build of the three cases                       every case is built
``runs``              the oracle's and the coupled model's 50 steps       both ran
``image``             the native image from the oracle's own objects      its manifest exists
``coupler``           the online coupler library linked to that image     the library exists
``online-gate``       the exact online 50 steps against the oracle        bit-for-bit recorded
====================  ==================================================  =========================

Each job waits on the one before (``afterok``).  Calling again resumes: a stage
whose product exists is skipped, a queued or running one is waited on, a failed
one is submitted again.  ``build.json`` in the root records the options, the
fingerprint and every stage's job.  A build is *validated* once its online gate
is bit-for-bit with its own oracle; ``Driver(build=b)`` runs only a validated
build unless told ``exploratory=True``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import yaml

#: the devices FTorch can be linked for
DEVICES = ("cpu", "cuda")
#: FTorch's installation for each device, under the repository's build/
FTORCH_ROOTS = {"cpu": "build/ftorch", "cuda": "build/ftorch-cuda"}
#: what decides the machine code besides the options: hashed into the fingerprint
PROVENANCE = ("native/pi_cam/cesm_source", "native/pi_cam/control_patches", "native/pi_cam/support",
              "native/pi_cam/hooks.yaml")
#: the exact online gate's configuration, derived for the build
GATE_CONFIG = "configs/pi_cam_icesm131_exact_online_50step.yaml"
#: the path entries of a case configuration, made absolute in a derived copy
CONFIG_PATHS = ("source_root", "native_manifest", "initial_conditions", "namelist")


class BuildError(RuntimeError):
    """A build that cannot be made, or a stage that failed."""


@dataclass(frozen=True)
class BuildOptions:
    """The compile-time options; the defaults are the installed, admitted model's."""

    #: CAM's columns a chunk (CAM's configure -pcols)
    pcols: int = 16
    #: what FTorch is linked for: the image's TorchScript models run there
    device: str = "cpu"

    def __post_init__(self) -> None:
        if isinstance(self.pcols, bool) or int(self.pcols) < 1:
            raise BuildError(f"pcols must be a positive integer, not {self.pcols!r}")
        object.__setattr__(self, "pcols", int(self.pcols))
        if self.device not in DEVICES:
            raise BuildError(f"device must be one of {DEVICES}, not {self.device!r}")

    @property
    def is_default(self) -> bool:
        return self == BuildOptions()

    def changed(self) -> dict[str, Any]:
        """The options that differ from the defaults."""

        default = BuildOptions()
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != getattr(default, f.name)}

    def fingerprint(self, repo: Path) -> str:
        """The options and everything the machine code is made from: the recipe, patches and support sources."""

        digest = sha256(json.dumps(asdict(self), sort_keys=True).encode())
        for entry in PROVENANCE:
            path = repo / entry
            for item in sorted(path.rglob("*")) if path.is_dir() else [path]:
                if item.is_file():
                    digest.update(str(item.relative_to(repo)).encode())
                    digest.update(item.read_bytes())
        submodule = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD:external/iCESM1.3.1_fzhu"],
                                   capture_output=True, text=True)
        digest.update(submodule.stdout.strip().encode())
        return digest.hexdigest()[:12]

    @property
    def label(self) -> str:
        return "-".join(f"{name}{value}" for name, value in self.changed().items()) or "default"


@dataclass(frozen=True)
class Stage:
    """One job of a build: what it runs, with what, and how its product is recognised."""

    name: str
    job: str
    what: str
    environment: Mapping[str, str]
    done: Callable[[], bool] = field(compare=False)
    failed: Callable[[], bool] = field(default=lambda: False, compare=False)


def _last(case_status: Path, action: str) -> str | None:
    """The last outcome CIME recorded for ``action`` (starting, success, error), or None."""

    if not case_status.is_file():
        return None
    outcomes = re.findall(rf"{re.escape(action)} (starting|success|error)", case_status.read_text())
    return outcomes[-1] if outcomes else None


def _derive_yaml(source: Path, destination: Path, values: Mapping[str, Any], base: Path) -> Path:
    """A copy of a case configuration with ``values`` set and its paths made absolute against ``base``."""

    document = yaml.safe_load(source.read_text())
    for name in CONFIG_PATHS:
        value = document.get(name)
        if value is not None and not Path(value).is_absolute():
            document[name] = str((base / value).resolve())
    document.update(values)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(f"# derived from {source.name} by freecam.pi_cam.build\n"
                           + yaml.safe_dump(document, sort_keys=False))
    return destination


def submit_job(repo: Path, job: str, environment: Mapping[str, str], after: str | None) -> str:
    """Submit ``job`` through validation/jobs/submit.sh (the site's allocation) and return its id."""

    command = [str(repo / "validation/jobs/submit.sh"), job]
    if after:
        command += ["-W", f"depend=afterok:{after}"]
    if environment:
        if any("," in str(value) for value in environment.values()):
            raise BuildError("a job variable holds a comma, which qsub -v cannot pass")
        command += ["-v", ",".join(f"{key}={value}" for key, value in environment.items())]
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True)
    if result.returncode:
        raise BuildError(f"cannot submit {job}: {(result.stderr or result.stdout).strip()}")
    return result.stdout.strip().splitlines()[-1]


def inspect_job(job_id: str) -> tuple[str, int | None]:
    """A job's state (queued, running, finished, unknown) and, once finished, its exit status."""

    result = subprocess.run(["qstat", "-x", "-f", job_id], capture_output=True, text=True)
    if result.returncode:
        return "unknown", None
    state = re.search(r"job_state = (\w)", result.stdout)
    status = re.search(r"Exit_status = (-?\d+)", result.stdout)
    code = {"Q": "queued", "H": "queued", "W": "queued", "R": "running", "E": "running", "F": "finished"}
    return code.get(state.group(1) if state else "", "unknown"), (int(status.group(1)) if status else None)


class Build:
    """The model built for a set of compile-time options: its products, its jobs, and whether it is validated."""

    def __init__(self, options: BuildOptions | None = None, *, repo: str | Path | None = None,
                 scratch: str | Path | None = None,
                 submitter: Callable[[Path, str, Mapping[str, str], str | None], str] = submit_job,
                 inspector: Callable[[str], tuple[str, int | None]] = inspect_job) -> None:
        from .. import site

        self.options = options or BuildOptions()
        self.repo = Path(repo or Path(__file__).resolve().parents[3]).resolve()
        self.installed = self.options.is_default
        self._submit, self._inspect = submitter, inspector
        if self.installed:
            self.fingerprint = None
            self.root = None
            return
        self.fingerprint = self.options.fingerprint(self.repo)
        scratch_root = Path(scratch or site.setting("FREECAM_SCRATCH", repo=self.repo)
                            or os.environ.get("SCRATCH") or f"/glade/derecho/scratch/{os.environ.get('USER', 'unknown')}")
        self.root = (scratch_root / "freeCAM" / "builds" / f"{self.options.label}-{self.fingerprint}").resolve()
        recipe = yaml.safe_load((self.repo / "native/pi_cam/cesm_source/cases.yaml").read_text())
        self._cases = {key: (self.root / "cases" / case["name"], self.root / "output" / case["output"] / case["name"])
                       for key, case in recipe["cases"].items()}

    # ------------------------------------------------------------------ products
    def case(self, key: str) -> Path:
        return self._cases[key][0]

    def run_dir(self, key: str) -> Path:
        return self._cases[key][1] / "run"

    @property
    def native_manifest(self) -> Path | None:
        return None if self.installed else self.root / "image/native_cam_manifest.json"

    @property
    def online_library(self) -> Path | None:
        return None if self.installed else self.root / "provider/production-components/libpycesm_external_atm.so"

    @property
    def gate_record(self) -> Path | None:
        return None if self.installed else self.root / "validation/pi_cam_exact_cesm_online_build_50step_bfb.json"

    def _gate(self) -> bool | None:
        record = self.gate_record
        if record is None or not record.is_file():
            return None
        return bool(json.loads(record.read_text()).get("bfb"))

    @property
    def validated(self) -> bool:
        """The installed build carries the repository's evidence; another, its own online gate."""

        return True if self.installed else self._gate() is True

    def config_for(self, base: str | Path) -> Path:
        """A case configuration for this build: its pcols and its image, every path absolute."""

        source = Path(base).resolve()
        if self.installed:
            return source
        return _derive_yaml(source, self.root / "configs" / source.name,
                            {"pcols": self.options.pcols, "native_manifest": str(self.native_manifest)},
                            source.parent.parent if source.parent.name == "configs" else source.parent)

    def driver_inputs(self) -> dict[str, Path]:
        """What ``Driver(build=...)`` runs on; the online seed contributes configuration only and is the site's."""

        if self.installed:
            return {}
        return {"reference_case": self.case("oracle"), "reference_run": self.run_dir("oracle"),
                "online_library": self.online_library}

    # ------------------------------------------------------------------ stages
    def stages(self) -> list[Stage]:
        if self.installed:
            return []
        root, repo = self.root, self.repo
        oracle_bld = self._cases["oracle"][1] / "bld"
        cases = ("oracle", "state", "pycesm")
        gate = self.gate_record
        return [
            Stage("cases", "validation/jobs/pi_cam_cases_prepare.pbs",
                  "the CESM sources and the oracle, python-state and pyCESM cases from the recipe",
                  {"FREECAM_BUILD_ROOT": str(root), "FREECAM_BUILD_PCOLS": str(self.options.pcols)},
                  lambda: all(_last(self.case(c) / "CaseStatus", "case.setup") == "success" for c in cases)),
            Stage("cases-build", "validation/jobs/pi_cam_cases_build.pbs", "case.build of the three cases",
                  {"FREECAM_BUILD_ROOT": str(root)},
                  lambda: all(_last(self.case(c) / "CaseStatus", "case.build") == "success" for c in cases)),
            Stage("runs", "validation/jobs/pi_cam_cases_run.pbs",
                  "the oracle's 50 steps and the original coupled model's",
                  {"FREECAM_BUILD_ROOT": str(root)},
                  lambda: all(_last(self.case(c) / "CaseStatus", "case.run") == "success" for c in ("oracle", "pycesm"))),
            Stage("image", "validation/jobs/pi_cam_promoted_statepool_build.pbs",
                  "the native image from the oracle's own objects",
                  {"FREECAM_STATE_CASE": str(self.case("state")), "FREECAM_NUMERICAL_BUILD": str(oracle_bld),
                   "FREECAM_IMAGE_ROOT": str(root / "image"), "FREECAM_PREPARED_SOURCE": str(root / "source/control"),
                   "FREECAM_DIRECT_KERNELS": str(root / "image/direct_kernels_promoted.yaml"),
                   "FREECAM_FTORCH_ROOT": str(repo / FTORCH_ROOTS[self.options.device])},
                  lambda: self.native_manifest.is_file()),
            Stage("coupler", "validation/jobs/pi_cam_online_coupler_build.pbs",
                  "the online coupler library linked to the image",
                  {"FREECAM_PYCESM_CASE": str(self.case("pycesm")), "FREECAM_ICESM_SOURCE": str(root / "source/provider"),
                   "FREECAM_IMAGE_ROOT": str(root / "image"), "FREECAM_PROVIDER_ROOT": str(root / "provider"),
                   "FREECAM_PROVIDER_RECORD": str(root / "provider/external_atm_build.json")},
                  lambda: self.online_library.is_file() and (root / "provider/external_atm_build.json").is_file()),
            Stage("online-gate", "validation/jobs/pi_cam_exact_cesm_online_50step.pbs",
                  "the exact online 50 steps, bit for bit against the build's own oracle",
                  {"FREECAM_REFERENCE_CASE": str(self.case("oracle")), "FREECAM_REFERENCE_RUN": str(self.run_dir("oracle")),
                   "FREECAM_CESM_REFERENCE_RUN": str(self.run_dir("pycesm")),
                   "FREECAM_CESM_PROVIDER_LIBRARY": str(self.online_library),
                   "PYCAM_NATIVE_MANIFEST": str(self.native_manifest),
                   "PYCAM_CONFIG": str(root / "configs" / Path(GATE_CONFIG).name),
                   # the captured exchange the default provider checks itself against is the installed build's
                   "FREECAM_PROVIDER_ORACLE": "", "FREECAM_RECORD_DIR": str(gate.parent), "PYCAM_RUN_TAG": "build"},
                  lambda: self._gate() is True, failed=lambda: self._gate() is False),
        ]

    # ------------------------------------------------------------------ record
    @property
    def record_path(self) -> Path | None:
        return None if self.installed else self.root / "build.json"

    def _record(self) -> dict[str, Any]:
        if self.record_path is not None and self.record_path.is_file():
            return json.loads(self.record_path.read_text())
        return {"schema_version": 1, "options": asdict(self.options), "fingerprint": self.fingerprint,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stages": {}}

    def _save(self, record: Mapping[str, Any]) -> None:
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self.record_path.write_text(json.dumps(record, indent=2) + "\n")

    def _state(self, stage: Stage, entry: Mapping[str, Any] | None) -> str:
        if stage.done():
            return "done"
        if stage.failed():
            return "failed"
        if not entry or not entry.get("job"):
            return "pending"
        state, _ = self._inspect(entry["job"])
        if state == "finished":
            return "failed"                         # it ended without its product
        return "submitted" if state == "unknown" else state

    # ------------------------------------------------------------------ actions
    def submit(self) -> "Build":
        """Submit every stage that is neither done nor waiting, each after the one before."""

        if self.installed:
            return self
        record = self._record()
        self.config_for(self.repo / GATE_CONFIG)        # the configuration the online gate runs
        after: str | None = None
        resubmitting = False
        for stage in self.stages():
            entry = record["stages"].get(stage.name)
            state = self._state(stage, entry)
            if state == "done":
                after = None
                continue
            # a job that waited on one submitted again waited on a failed job: PBS has dropped it
            if state in ("queued", "running", "submitted") and not resubmitting:
                after = entry["job"]
                continue
            resubmitting = True
            job = self._submit(self.repo, stage.job, stage.environment, after)
            record["stages"][stage.name] = {"job": job, "after": after,
                                            "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            after = job
        self._save(record)
        return self

    def status(self) -> dict[str, Any]:
        """Each stage's state: pending, queued, running, submitted, done or failed."""

        if self.installed:
            return {"build": "installed", "options": asdict(self.options), "validated": True, "stages": {}}
        record = self._record()
        stages = {}
        for stage in self.stages():
            entry = record["stages"].get(stage.name)
            stages[stage.name] = {"state": self._state(stage, entry), "job": (entry or {}).get("job"), "what": stage.what}
        return {"build": self.root.name, "root": str(self.root), "options": asdict(self.options),
                "validated": self.validated, "stages": stages}

    def wait(self, *, poll_seconds: float = 60.0, timeout: float | None = None) -> dict[str, Any]:
        """Return once every stage is done; a failed stage raises :class:`BuildError`."""

        start = time.monotonic()
        while True:
            status = self.status()
            states = {name: entry["state"] for name, entry in status["stages"].items()}
            failed = [name for name, state in states.items() if state == "failed"]
            if failed:
                raise BuildError(f"stage {failed[0]} of {status.get('build')} failed (job "
                                 f"{status['stages'][failed[0]]['job']}); its log is in logs/")
            if all(state == "done" for state in states.values()):
                return status
            if timeout is not None and time.monotonic() - start > timeout:
                return status
            time.sleep(poll_seconds)

    def __repr__(self) -> str:
        if self.installed:
            return "Build(installed)"
        return f"Build({self.options.label}, {self.root})"


def build(*, pcols: int = 16, device: str = "cpu", submit: bool = True, repo: str | Path | None = None,
          scratch: str | Path | None = None, **hooks: Any) -> Build:
    """The model for these compile-time options.

    The defaults are the installed model, returned without building anything.  Other options
    get their own build root; ``submit`` (the default) sends its stages to PBS, CPU nodes only,
    and returns at once -- follow it with ``status()`` and ``wait()``.  Calling it again
    resumes rather than starts over.
    """

    made = Build(BuildOptions(pcols=pcols, device=device), repo=repo, scratch=scratch, **hooks)
    if submit and not made.installed:
        made.submit()
    return made


def main(argv: Sequence[str] | None = None) -> int:
    """``freecam build``: make, follow or describe the build of a set of compile-time options."""

    import argparse

    parser = argparse.ArgumentParser(
        prog="freecam build",
        description="Build the PI-atm model for compile-time options (CPU jobs); the defaults are the installed model.")
    parser.add_argument("--pcols", type=int, default=16, help="CAM's columns a chunk (default 16)")
    parser.add_argument("--device", choices=DEVICES, default="cpu", help="what FTorch is linked for (default cpu)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="list the stages and what each runs; submit nothing")
    action.add_argument("--status", action="store_true", help="each stage's state; submit nothing")
    action.add_argument("--wait", action="store_true", help="submit, then wait until every stage is done")
    arguments = parser.parse_args(argv)
    made = build(pcols=arguments.pcols, device=arguments.device, submit=not (arguments.plan or arguments.status))
    if arguments.plan:
        print(json.dumps({"build": repr(made), "stages": [{"name": stage.name, "job": stage.job, "what": stage.what,
                                                           "environment": dict(stage.environment)}
                                                          for stage in made.stages()]}, indent=2))
        return 0
    status = made.wait() if arguments.wait else made.status()
    print(json.dumps(status, indent=2))
    return 0


__all__ = ["Build", "BuildError", "BuildOptions", "Stage", "build", "main"]
