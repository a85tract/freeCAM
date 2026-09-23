"""The online month's legs in one record: loops, the model's cost and where it ran, ratios."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

GPTL = """  "CPL:INIT"                                -       1    -      10.000000    10.000000    10.000000         0.000000
    "CPL:RUN_LOOP"                          -    1488    -     400.000000     5.829652     0.241992         0.000136
    "CPL:FINAL"                             -       1    -       1.000000     0.002280     0.002280         0.000000
"""


def _freecam(root: Path, leg: str, seconds: float, *, device: str | None = None, mps: bool = False) -> None:
    directory = root / f"{leg}-freecam"
    directory.mkdir(parents=True)
    hooks = {}
    models = None
    if device is not None:
        # 512 ranks, 2976 calls a rank; the first call a rank costs 2 s, every later one 4 ms
        modeled = 512 * 2976
        hooks = {"compute_uwshcu_inv": {"calls": modeled, "modeled": modeled, "ranks_called": 512,
                                        "first_call_seconds": 512 * 2.0, "first_call_seconds_max": 2.5,
                                        "call_seconds": 512 * 2.0 + (modeled - 512) * 0.004,
                                        "call_seconds_max": 15.0, "warm_seconds": 512 * 1.0}}
        models = {"compute_uwshcu_inv": {"file": "m.pt", "binding": "torchscript", "device": device}}
    (directory / "summary.json").write_text(json.dumps({
        "run_status": "passed", "steps": 1488, "boundary_mode": "online",
        "boundary_provider": "CESMOnlineBoundaryProvider",
        "timing": {"advance_seconds": seconds, "initialize_seconds": 20.0, "finalize_seconds": 0.5,
                   "total_seconds": seconds + 20.5},
        "python_stages": ["shallow_convection"], "kernel_models": models, "hooks": hooks,
        "stage_execution": {"shallow_convection_python": {"execution_mode": "native-model" if device else "native-whole"}},
        "memory": {"samples": [{"maximum_rank_rss_bytes": 1, "total_rss_bytes": 2}]}}))
    if device is None:
        (directory / "bfb.json").write_text(json.dumps({"bfb": True, "compared_files": 18}))
    (directory / "health.json").write_text(json.dumps({"counts": {"big_error": 3}}))
    if mps:
        (directory / "mps.txt").write_text("".join(
            f"node{n}: GPU {k} servers [1234 ] client disconnects 32 log faults 0\n" for n in range(4) for k in range(4)))


def test_every_leg_is_recorded_with_its_position_cost_and_ratios(tmp_path: Path) -> None:
    root = tmp_path / "root"
    timing = root / "A-original-fortran" / "run" / "timing"
    timing.mkdir(parents=True)
    (timing / "cesm_timing.000").write_text(GPTL)
    _freecam(root, "M", 420.0, device="cpu")
    _freecam(root, "G", 480.0, device="cuda", mps=True)
    output = tmp_path / "record.json"
    subprocess.run([sys.executable, str(REPO / "tools" / "report_pi_cam_online_legs.py"), "--root", str(root),
                    "--legs", "AMG", "--output", str(output), "--pbs-job-id", "1"], check=True, capture_output=True)
    record = json.loads(output.read_text())
    assert record["boundary"] == "online" and record["order"] == "AMG"
    legs = record["legs"]
    assert [legs[x]["position"] for x in "AMG"] == [1, 2, 3]
    assert legs["A"]["coupling_loop_seconds"] == 400.0 and legs["A"]["lifecycle_seconds"] == 411.0
    assert legs["G"]["kernel_models"]["compute_uwshcu_inv"]["device"] == "cuda"
    cost = legs["M"]["model"]
    assert abs(cost["ms_per_call_after_first"] - 4.0) < 1e-9 and cost["first_call_seconds_per_rank"] == 2.0
    assert cost["original"] == 0                                 # every call the model's
    assert legs["G"]["mps"] == {"gpus": 16, "servers_running": 16, "client_disconnects": 512, "log_faults": 0}
    assert legs["M"]["mps"] is None and legs["M"]["health"] == {"big_error": 3}
    assert record["ratios"] == {"M/A": 1.05, "G/A": 1.2, "G/M": 480.0 / 420.0}


def test_a_leg_that_did_not_finish_is_named_and_left_out_of_the_ratios(tmp_path: Path) -> None:
    import report_pi_cam_online_legs as rl

    root = tmp_path / "root"
    _freecam(root, "C", 400.0)
    legs = {"C": rl.freecam_leg(root / "C-freecam", "compute_uwshcu_inv"),
            "M": rl.freecam_leg(root / "M-freecam", "compute_uwshcu_inv"),
            "A": rl.original_leg(root / "A-original-fortran" / "run", None)}
    assert legs["C"]["completed"] is True and legs["C"]["bfb"] is True and legs["C"]["model"]["modeled"] == 0
    assert legs["M"] == {"completed": False} and legs["A"] == {"completed": False}
    assert rl.ratios(legs) == {}
