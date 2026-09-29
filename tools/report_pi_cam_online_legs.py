#!/usr/bin/env python3
"""One online month's legs, run back to back in one allocation, as one record.

The legs are the ones ``validation/jobs/pi_cam_online_model_1month.pbs`` runs:

    A  the original Fortran model, every component live (GPTL ``CPL:RUN_LOOP``);
    C  freeCAM online, the stage classes installed, nothing replaced;
    M  C with the kernel answered by a TorchScript model inside the image, on the host;
    G  C with the same model on the node's GPUs through MPS: one server of the job's own a
       GPU, or the site's server of a node (``--gpu-mps``).

Every leg is timed over the same coupling loop (freeCAM's ``advance_seconds``
against A's ``CPL:RUN_LOOP``); the record keeps each leg's position in the
order, the model's cost a call from the image's hook counters, what proves
the model ran where it says (the slot's binding and device, the calls it
answered, the MPS servers' client counts), the leg's health and its drift
from the oracle month, and the ratios between the legs.

    tools/report_pi_cam_online_legs.py --root <job root> --legs ACM \\
        --kernel compute_uwshcu_inv --hardware "..." --output <record>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_pi_cam_month import parse_gptl  # noqa: E402
from report_pi_cam_pair import gptl_seconds, sha256  # noqa: E402

LEGS = {
    "A": "the original Fortran model (the pyCESM case executable), every component live",
    "C": "freeCAM online (live CLM, CICE, DOCN, RTM and coupler), the stage classes installed, nothing replaced",
    "M": "freeCAM online, the kernel answered by the TorchScript model inside the image on the host CPU",
    "G": "freeCAM online, the kernel answered by the TorchScript model inside the image on the node's GPUs "
         "through MPS",
    "N": "M with every chunk of a rank answered by one forward a step (--batch-chunks): the kernel's inputs "
         "gathered for every chunk before the stage runs, each call taking its chunk's rows",
    "H": "G with every chunk of a rank answered by one forward a step (--batch-chunks)",
}
_MPS_LINE = re.compile(r"GPU (?P<gpu>\d+) servers \[(?P<servers>[^\]]*)\] client disconnects (?P<clients>\d+) "
                       r"log faults (?P<faults>\d+)")
_SITE_LINE = re.compile(r"site MPS exclusive GPUs (?P<exclusive>\d+) of (?P<gpus>\d+)")
_REFUSED_LINE = re.compile(r"refused ranks (?P<refused>\d+)")


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def model_cost(hooks: dict[str, Any] | None, kernel: str) -> dict[str, Any]:
    """The model's calls and seconds from the hook counters (sums over the ranks, slowest rank beside)."""

    row = (hooks or {}).get(kernel) or {}
    modeled, ranks = int(row.get("modeled", 0)), int(row.get("ranks_called", 0))
    cost: dict[str, Any] = {"calls": row.get("calls"), "modeled": modeled, "original": row.get("calls", 0) - modeled
                            if row.get("calls") is not None else None, "ranks": ranks}
    total, first = row.get("call_seconds"), row.get("first_call_seconds")
    if modeled and total is not None and first is not None and modeled > ranks:
        cost["ms_per_call_after_first"] = 1e3 * (total - first) / (modeled - ranks)
    if ranks and first is not None:
        cost["first_call_seconds_per_rank"] = first / ranks
        cost["first_call_seconds_slowest_rank"] = row.get("first_call_seconds_max")
    if ranks and row.get("warm_seconds") is not None:
        cost["warm_up_seconds_per_rank"] = row["warm_seconds"] / ranks
    if ranks and total is not None:
        cost["model_seconds_per_rank"] = total / ranks
        cost["model_seconds_slowest_rank"] = row.get("call_seconds_max")
    batch = row.get("batch")
    if batch and batch.get("forwards"):
        # the hook batch: a call only takes its chunk's rows (call_seconds); the model's work is
        # the forwards, one a step, timed apart -- the model's cost a rank is both together
        forwards, brank = int(batch["forwards"]), int(batch.get("ranks") or ranks or 1)
        cost.pop("ms_per_call_after_first", None)
        cost.pop("first_call_seconds_per_rank", None)
        cost.pop("first_call_seconds_slowest_rank", None)
        # the slowest rank's take time alone says nothing of its model: the forwards are its cost
        cost.pop("model_seconds_slowest_rank", None)
        cost["batch"] = {
            "forwards": forwards,
            "chunks_per_forward": batch["chunks"] / forwards,
            "columns_per_forward": batch["rows"] / forwards,
            "ms_per_forward": 1e3 * batch["forward_seconds"] / forwards,
            "forward_seconds_per_rank": batch["forward_seconds"] / brank,
            "forward_seconds_slowest_rank": batch.get("forward_seconds_max"),
            "take_seconds_per_rank": None if total is None or not ranks else total / ranks,
            "untaken_chunks": batch.get("untaken", 0) + batch.get("pending", 0),
        }
        if total is not None and ranks:
            cost["model_seconds_per_rank"] = (total + batch["forward_seconds"]) / ranks
    return cost


def mps_evidence(path: Path) -> dict[str, Any] | None:
    """What the MPS servers said at the end: one line a GPU a node from gpu_mps_per_gpu.sh stop."""

    if not path.is_file():
        return None
    lines = path.read_text().splitlines()
    nodes = [m.groupdict() for m in map(_SITE_LINE.search, lines) if m]
    if nodes:
        # the site's server of each node: its GPUs in Exclusive_Process mode refuse any rank that
        # missed MPS, so all of them in that mode and no rank refused mean every rank used it
        refused = [int(m.group("refused")) for m in map(_REFUSED_LINE.search, lines) if m]
        return {"server": "site", "nodes": len(nodes),
                "gpus": sum(int(n["gpus"]) for n in nodes),
                "exclusive_process_gpus": sum(int(n["exclusive"]) for n in nodes),
                "refused_ranks": refused[-1] if refused else None}
    gpus = [m.groupdict() for m in map(_MPS_LINE.search, lines) if m]
    return {"server": "own, one a GPU", "gpus": len(gpus),
            "servers_running": sum(1 for g in gpus if g["servers"].strip()),
            "client_disconnects": sum(int(g["clients"]) for g in gpus),
            "log_faults": sum(int(g["faults"]) for g in gpus)}


def original_leg(run: Path, executable: Path | None) -> dict[str, Any]:
    timing = run / "timing" / "cesm_timing.000"
    if not timing.is_file():
        return {"completed": False}
    gptl = parse_gptl(timing)
    loop = gptl_seconds(gptl, "CPL:RUN_LOOP")
    init, final = gptl_seconds(gptl, "CPL:INIT"), gptl_seconds(gptl, "CPL:FINAL")
    return {
        "completed": loop is not None,
        "coupling_loop_seconds": loop,
        "init_seconds": init,
        "final_seconds": final,
        "lifecycle_seconds": None if None in (loop, init, final) else init + loop + final,
        "executable_sha256": None if executable is None or not executable.is_file() else sha256(executable),
        "timing_source": "GPTL CPL:RUN_LOOP in timing/cesm_timing.000",
    }


def freecam_leg(directory: Path, kernel: str) -> dict[str, Any]:
    summary = _load(directory / "summary.json")
    if summary is None:
        return {"completed": False}
    timing = summary["timing"]
    bfb = _load(directory / "bfb.json") or {}
    health = _load(directory / "health.json")
    drift = _load(directory / "drift.json")
    samples = (summary.get("memory") or {}).get("samples") or []
    stages = summary.get("stage_execution") or {}
    return {
        "completed": summary.get("run_status") == "passed",
        "steps": summary.get("steps"),
        "coupling_loop_seconds": float(timing["advance_seconds"]),
        "init_seconds": float(timing.get("initialize_seconds", 0.0)),
        "final_seconds": float(timing.get("finalize_seconds", 0.0)),
        "lifecycle_seconds": float(timing.get("total_seconds", 0.0)),
        "sypd": timing.get("advance_sypd"),
        "boundary_mode": summary.get("boundary_mode"),
        "boundary_provider": summary.get("boundary_provider"),
        "native_manifest": summary.get("native_manifest"),
        "native_library_sha256": summary.get("native_library_sha256"),
        "python_stages": summary.get("python_stages"),
        "cloud_macro_micro_python": summary.get("cloud_macro_micro_python"),
        "stage_execution_modes": {name: row.get("execution_mode") for name, row in stages.items()
                                  if isinstance(row, dict)},
        "kernel_models": summary.get("kernel_models"),
        "model": model_cost(summary.get("hooks"), kernel),
        "mps": mps_evidence(directory / "mps.txt"),
        "bfb": bfb.get("bfb"),
        "bfb_files": bfb.get("compared_files"),
        "peak_rank_rss_bytes": samples[-1].get("maximum_rank_rss_bytes") if samples else None,
        "total_rss_bytes": samples[-1].get("total_rss_bytes") if samples else None,
        "health": None if health is None else health.get("counts"),
        "drift": None if drift is None else {key: drift.get(key) for key in (
            "fields_compared", "fields_identical", "fields_with_non_finite_values", "relative_rms")},
    }


def ratios(legs: dict[str, dict[str, Any]]) -> dict[str, float]:
    loops = {name: leg.get("coupling_loop_seconds") for name, leg in legs.items() if leg.get("completed")}
    pairs = [(x, "A") for x in "CMGNH"] + [("M", "C"), ("G", "C"), ("G", "M"), ("N", "M"), ("H", "G"), ("H", "N")]
    return {f"{x}/{y}": loops[x] / loops[y] for x, y in pairs if loops.get(x) and loops.get(y)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="the job's root: <leg>-... directories")
    parser.add_argument("--legs", required=True, help="the legs in the order they ran, e.g. ACM")
    parser.add_argument("--kernel", default="compute_uwshcu_inv")
    parser.add_argument("--a-executable", type=Path)
    parser.add_argument("--hardware", default="", help="the nodes and layout the legs shared")
    parser.add_argument("--gpu-mps", choices=("own", "site"), default="own",
                        help="whose MPS server G's ranks reached: the job's own, one a GPU, or the site's")
    parser.add_argument("--root-label", default="", help="the root as the record names it (no site directory)")
    parser.add_argument("--pbs-job-id")
    parser.add_argument("--git-commit")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    unknown = sorted(set(arguments.legs) - set(LEGS))
    if unknown or len(set(arguments.legs)) != len(arguments.legs):
        raise SystemExit(f"--legs takes each of {''.join(LEGS)} at most once, got {arguments.legs!r}")
    legs: dict[str, dict[str, Any]] = {}
    for position, name in enumerate(arguments.legs, start=1):
        if name == "A":
            leg = original_leg(arguments.root / "A-original-fortran" / "run", arguments.a_executable)
        else:
            leg = freecam_leg(arguments.root / f"{name}-freecam", arguments.kernel)
        legs[name] = {"what": LEGS[name], "position": position, **leg}
    record = {
        "schema_version": 1,
        "what": "One allocation's online PI-atm month (1488 steps, 512 ranks), legs back to back: the original "
                "Fortran, freeCAM with nothing replaced, and freeCAM with a model in the kernel's slot. Every "
                "component is live in every leg (online coupling, not the offline replay). Times are the "
                "coupling loop; the model legs are not bit-for-bit by construction, their drift and health "
                "are recorded instead.",
        "boundary": "online",
        "pbs_job_id": arguments.pbs_job_id,
        "git_commit": arguments.git_commit,
        "hardware": arguments.hardware,
        "gpu_mps": arguments.gpu_mps if set("GH") & set(arguments.legs) else None,
        "root": arguments.root_label or None,
        "order": arguments.legs,
        "kernel": arguments.kernel,
        "legs": legs,
        "ratios": ratios(legs),
    }
    arguments.output.write_text(json.dumps(record, indent=2) + "\n")
    line = " | ".join(f"{name} {leg.get('coupling_loop_seconds') or float('nan'):.1f} s"
                      for name, leg in legs.items())
    print(f"online month {arguments.legs} job {arguments.pbs_job_id}: {line}")
    print("ratios:", json.dumps({k: round(v, 4) for k, v in record["ratios"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
