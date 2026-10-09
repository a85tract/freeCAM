"""Write the run configuration for ``recast run refactor-todo`` on this site.

RecastEngine's refactor recipe walks freeCAM's carve-out (recast/carve.json)
through its gates: the static arithmetic check, the pinned original run as the
oracle, and the 512-rank 50-step run held to it bit for bit.  What those gates
run here is site knowledge -- the account, the scratch filesystem, the oracle's
run directory -- so this tool resolves it the way the jobs do (through
validation/jobs/common.sh and site.env) and writes a configuration that is
never committed:

    python tools/recast_run_config.py --output $SCRATCH/recast-freecam.json
    recast run refactor-todo . --config $SCRATCH/recast-freecam.json

The full-model gate is two jobs.  The first builds an image from the
candidate's tree into the run's own workspace (pi_cam_promoted_statepool_build.pbs
re-prepares build/iCESM1.3.1_PI_cam_only from the patches and support sources
first), so the image the second job runs is the candidate's and no image in
build/ is replaced; the second is the exact online 50-step gate, pointed at the
oracle's run directory and that image, writing its record into the workspace.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SETTINGS = ("FREECAM_SCRATCH", "FREECAM_ACCOUNT", "FREECAM_REFERENCE_RUN")


def site(repo: Path) -> dict[str, str]:
    """The settings as the jobs see them: common.sh, which reads site.env."""
    script = "source validation/jobs/common.sh >/dev/null 2>&1; " + "; ".join(
        f'printf "%s\\n" "${{{name}:-}}"' for name in SETTINGS
    )
    shown = subprocess.run(["bash", "-c", script], cwd=repo, capture_output=True, text=True,
                           check=True).stdout.splitlines()
    values = dict(zip(SETTINGS, shown))
    missing = [name for name in SETTINGS if not values.get(name)]
    if missing:
        raise SystemExit(f"site.env does not resolve {', '.join(missing)}; see site.env.example")
    return values


def build(repo: Path, workspace: Path, manifest: str) -> dict:
    values = site(repo)
    sys.path.insert(0, str(repo / "tools"))
    import prepare_pi_cam_source as prepare

    reference = Path(values["FREECAM_REFERENCE_RUN"])
    return {
        "reference_commit": prepare.PINNED_REVISIONS["."],
        "executor": "pbs",
        "workspace": str(workspace),
        "stages": {
            "pbs": {"account": values["FREECAM_ACCOUNT"], "poll_seconds": 60},
            "carve": {"manifest": manifest},
            "pinned-run": {
                "output": str(reference),
                "expect": ["*.cam.h0.*.nc", "atm_in", "drv_in"],
                "key_files": [str(reference.parent / "bld/cesm.exe"), str(reference / "atm_in"),
                              str(reference / "drv_in")],
                "steps": 50,
            },
            "fullmodel.bitwise": {
                "job": [
                    {
                        "argv": ["validation/jobs/pi_cam_promoted_statepool_build.pbs"],
                        "env": {
                            "FREECAM_IMAGE_ROOT": "{workspace}/image",
                            "FREECAM_DIRECT_KERNELS": "{workspace}/direct_kernels_promoted.yaml",
                        },
                        "timeout_s": 4 * 3600,
                    },
                    {
                        "argv": ["validation/jobs/pi_cam_exact_cesm_online_50step.pbs"],
                        "env": {
                            "FREECAM_REFERENCE_RUN": "{reference}",
                            "FREECAM_RECORD_DIR": "{workspace}",
                            "PYCAM_RUN_TAG": "recast",
                            "PYCAM_NATIVE_MANIFEST": "{workspace}/image/native_cam_manifest.json",
                        },
                        "timeout_s": 6 * 3600,
                    },
                ],
                "result": "{workspace}/pi_cam_exact_cesm_online_recast_50step_bfb.json",
                "result_key": "bfb",
                "metrics_keys": ["reference_files", "candidate_files", "compared_files",
                                 "missing_in_candidate", "extra_in_candidate", "first_difference"],
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=REPO,
                        help="the checkout whose site.env, build products and jobs the run uses")
    parser.add_argument("--workspace", type=Path,
                        help="where the engine works (default: a new directory under the scratch area)")
    parser.add_argument("--manifest", default="recast/carve.json",
                        help="the carve manifest, relative to the repo or absolute")
    arguments = parser.parse_args(argv)
    repo = arguments.repo.resolve()
    workspace = arguments.workspace or (
        Path(site(repo)["FREECAM_SCRATCH"]) / "freeCAM" / "recast"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    config = build(repo, workspace, arguments.manifest)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(config, indent=2) + "\n")
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
