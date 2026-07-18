#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycam_sima import NotebookSession


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
        if len(model.field_names) != 21:
            raise RuntimeError(f"expected 21 fields, got {len(model.field_names)}")
        initial = model.get_field("air_temperature", rank=0)
        model.set_field("air_temperature", initial, rank=0)
        roundtrip = model.get_field("air_temperature", rank=0)
        if not np.array_equal(initial, roundtrip):
            raise RuntimeError("NotebookSession set/get roundtrip changed the field")
        if model.step() != 1 or model.step() != 2:
            raise RuntimeError(f"unexpected Python step counter {model.current_step}")
        final = model.get_field("air_temperature", rank=0)
        statistics = model.get_field_stats("air_temperature", rank="all")
        if len(statistics) != 24:
            raise RuntimeError(f"expected 24 rank statistics, got {len(statistics)}")
        print(
            "PYCAM_SIMA_NOTEBOOK_SESSION_OK "
            f"steps={model.current_step} fields={len(model.field_names)} "
            f"shape={final.shape} min={final.min():.17g} max={final.max():.17g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
