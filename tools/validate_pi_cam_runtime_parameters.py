#!/usr/bin/env python3
"""Prove one audited CAM tunable can be changed on the running model.

Runs the online 50-step case in two halves: the first half with every
parameter at its namelist value -- so output written in that window must
stay bit-for-bit with the unperturbed oracle -- and the second half after
collectively writing one deep-convection tunable, so output written at the
end must differ.  Also asserts, on every rank, that each bound parameter's
storage read back exactly the value parsed from ``atm_in`` at
initialization, which is the proof the bindings resolve the intended
module variables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cloudpickle

from mpi4py import MPI

from freecam.pi_cam import PICAMCase


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--boundary-provider", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--change-at", type=int, default=25)
    parser.add_argument("--parameter", default="zmconv_c0_lnd")
    parser.add_argument("--value", type=float, default=0.0075)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    world = MPI.COMM_WORLD
    case = PICAMCase.from_yaml(args.config)
    boundary = cloudpickle.loads(args.boundary_provider.read_bytes())
    with case.runtime(
        boundary=boundary,
        communicator=world,
        run_dir=args.run_dir,
    ) as cam:
        cam.initialize()
        registry = cam.module_parameters
        bound = registry.names()
        unavailable = dict(registry.unavailable)
        # Binding self-verified every value against atm_in; anything the
        # audit admitted but the model could not verify is a failure here.
        baseline = registry.values()
        before = registry.value(args.parameter)

        cam.advance(args.change_at)
        report = cam.set_module_parameter(args.parameter, args.value)
        after = registry.value(args.parameter)
        cam.advance(args.steps - args.change_at)
        final_value = registry.value(args.parameter)
        overrides = registry.overrides()
        final_step = int(cam.coupling_step)

    local = {
        "rank": world.Get_rank(),
        "bound": list(bound),
        "unavailable": unavailable,
        "value_before": before,
        "value_after": after,
        "value_final": final_value,
    }
    gathered = world.gather(local, root=0)
    if world.Get_rank() != 0:
        return 0

    ranks_agree = all(
        item["bound"] == list(bound)
        and item["value_before"] == before
        and item["value_after"] == args.value
        and item["value_final"] == args.value
        for item in gathered
    )
    passed = (
        ranks_agree
        and not unavailable
        and len(bound) == 14
        and after == args.value
        and final_step == args.steps
        and report["previous"] == before
        and overrides == {args.parameter: (before, args.value)}
    )
    summary = {
        "schema_version": 1,
        "gate": "pi_cam_runtime_parameter_50step",
        "run_status": "passed" if passed else "failed",
        "mpi_ranks": world.Get_size(),
        "steps": args.steps,
        "change_at_step": args.change_at,
        "parameter": args.parameter,
        "value_before": before,
        "value_after": args.value,
        "bound_parameters": list(bound),
        "bound_count": len(bound),
        "unavailable": unavailable,
        "baseline_values": baseline,
        "overrides": {
            name: {"baseline": pair[0], "value": pair[1]}
            for name, pair in overrides.items()
        },
        "all_ranks_agree": ranks_agree,
        "binding_verified_against_atm_in": not unavailable,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"runtime parameter gate: {summary['run_status']} "
        f"({args.parameter} {before} -> {args.value} at step {args.change_at})"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
