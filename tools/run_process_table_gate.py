#!/usr/bin/env python
"""Drive a 512-rank run through ``fc.Driver`` with a radiation process model set on ``driver.processes``.

    python tools/run_process_table_gate.py --root <dir> --config configs/pi_cam_icesm131_replay.yaml \\
        --native-manifest build/<image>/native_cam_manifest.json --boundary <replay dir> \\
        --block-replay <capture dir> --reference <oracle run> [--cli-run <run dir>] --record <json>

The notebook's way in, on the model: ``driver.processes["radiation"].process = load_block_model("replay:DIR")``
before the first step, nothing else.  The run is compared byte for byte with the oracle (the state must be
bit-for-bit, the history restart differs in the branch's diagnostics) and, when given, with the same replay
made through the command line, which must be identical in every file: one path, two ways in.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--block-replay", type=Path, required=True, help="a radiation process capture directory to replay through the block slot")
    parser.add_argument("--reference", type=Path, required=True, help="the oracle run directory")
    parser.add_argument("--cli-run", type=Path, default=None, help="the same replay made through the command line, for the two-ways-in comparison")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--pbs-job-id", default=None)
    arguments = parser.parse_args(argv)

    import freecam as fc
    from freecam.physics.radiation_process import load_block_model
    from freecam.pi_cam.validation import compare_pi_cam_directories
    from freecam.site import site_relative

    root = arguments.root
    root.mkdir(parents=True, exist_ok=True)
    # the case's configuration with this image: the notebook user points at an image the same way
    payload = yaml.safe_load(arguments.config.read_text())
    payload["native_manifest"] = str(arguments.native_manifest.resolve())
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False))

    run_dir = root / "run"
    driver = fc.Driver(case="PI-atm", nsteps=arguments.steps, config=config, run_dir=run_dir,
                       boundary=arguments.boundary, trace_limit=None)
    started = time.time()
    radiation = driver.processes["radiation"]
    radiation.process = load_block_model(f"replay:{arguments.block_replay}", rank=0)
    result = driver.run()
    finished = time.time()
    status = dict(driver.status)
    processes = status.get("processes", {})
    driver.close()

    bfb = compare_pi_cam_directories(arguments.reference, run_dir).to_payload()
    same_path = compare_pi_cam_directories(arguments.cli_run, run_dir).to_payload() if arguments.cli_run else None
    record = {
        "schema_version": 1,
        "what": "the radiation process replaced from driver.processes, the notebook's way in, on 512 ranks",
        "pbs_job_id": arguments.pbs_job_id,
        "steps": arguments.steps,
        "how": "fc.Driver; driver.processes['radiation'].process = load_block_model('replay:...'); driver.run()",
        "native_manifest": site_relative(arguments.native_manifest),
        "block_replay": site_relative(arguments.block_replay),
        "processes": processes,
        "seconds": {"run": finished - started},
        "against_oracle": bfb,
        "against_cli_replay": same_path,
        "cli_run": None if arguments.cli_run is None else site_relative(arguments.cli_run),
        "run_dir": site_relative(run_dir),
    }
    arguments.record.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("against_oracle", "against_cli_replay")}, indent=1))
    print("against oracle: bfb", bfb.get("bfb"), "| against the CLI replay: bfb", None if same_path is None else same_path.get("bfb"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
