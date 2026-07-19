#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycam_sima import FULL_CAM_PHASES, NotebookSession


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--env-script", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    args = parser.parse_args()

    with NotebookSession(
        args.config,
        run_dir=args.run_dir,
        env_script=args.env_script,
        log_path=args.log_path,
    ) as model:
        if model.phase_names != FULL_CAM_PHASES:
            raise RuntimeError(f"unexpected phase contract: {model.phase_names}")
        if model.next_phase != "cam_run2":
            raise RuntimeError(f"unexpected initial phase: {model.phase_status}")

        samples: dict[str, float] = {}
        # Explicitly finish CAM-SIMA's nstep=0 initial-send cycle.
        for phase in FULL_CAM_PHASES:
            status = model.run_phase(phase)
            field = model.get_field("air_temperature", rank=0)
            if not np.isfinite(field).all():
                raise RuntimeError(f"non-finite temperature after {phase}")
            samples[f"initial:{phase}"] = float(field[0, 0])
            if status["last_phase"] != phase:
                raise RuntimeError(f"phase status mismatch: {status}")

        if model.current_step != 0 or model.phase_status["native_nstep"] != 1:
            raise RuntimeError(f"initial-send accounting failed: {model.phase_status}")

        # Execute requested step 1 with a Notebook-visible pause after each phase.
        for phase in FULL_CAM_PHASES:
            model.run_phase(phase)
            field = model.get_field("air_temperature", rank=0)
            samples[f"step1:{phase}"] = float(field[0, 0])

        if model.current_step != 1 or model.phase_status["native_nstep"] != 2:
            raise RuntimeError(f"requested-step accounting failed: {model.phase_status}")
        if not model.phase_status["sequence_safe"] or not model.phase_status["cycle_complete"]:
            raise RuntimeError(f"phase state machine did not return to boundary: {model.phase_status}")

        print(
            "PYCAM_SIMA_PHASE_SESSION_OK "
            f"suite={model.config.physics_suite} step={model.current_step} "
            f"phases={len(FULL_CAM_PHASES)} samples={len(samples)} "
            f"T={samples['step1:cam_run1']:.17g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
