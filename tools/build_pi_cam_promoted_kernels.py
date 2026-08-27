#!/usr/bin/env python3
"""Regenerate direct-kernel YAML with every safe StatePool-promoted process."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import yaml

from freecam.pi_cam.kernel_codegen import load_direct_kernels
from freecam.pi_cam.process_codegen import (
    direct_kernel_payload,
    generated_promoted_kernels,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        action="append",
        default=None,
        help="descriptor of reviewed kernels (repeatable; default: direct_kernels.yaml "
             "direct_kernels_macrophysics.yaml, direct_kernels_radiation.yaml, direct_kernels_microphysics.yaml and direct_kernels_mm.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("native/pi_cam/direct_kernels_promoted.yaml"),
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path("native/pi_cam/process_promotion_rules.yaml"),
    )
    args = parser.parse_args()
    bases = args.base or [
        Path("native/pi_cam/direct_kernels.yaml"),
        Path("native/pi_cam/direct_kernels_macrophysics.yaml"),
        Path("native/pi_cam/direct_kernels_radiation.yaml"),
        Path("native/pi_cam/direct_kernels_microphysics.yaml"),
        Path("native/pi_cam/direct_kernels_mm.yaml"),
    ]
    base = tuple(
        kernel
        for path in bases
        for kernel in load_direct_kernels(path)
        if not kernel.name.startswith("__generated_")
    )
    reviewed_routines = {kernel.routine.casefold() for kernel in base}
    rules = yaml.safe_load(args.rules.read_text())
    if not isinstance(rules, dict) or int(rules.get("schema_version", 0)) != 1:
        raise RuntimeError(f"invalid process promotion rules: {args.rules}")
    excluded = {
        str(name): str(reason)
        for name, reason in dict(
            rules.get("exclude_from_native_image", {})
        ).items()
    }
    generated = tuple(
        kernel
        for kernel in generated_promoted_kernels()
        if kernel.name not in excluded
        and kernel.routine.casefold() not in reviewed_routines
    )
    stable_ids: dict[str, int] = {}
    if args.output.exists():
        stable_ids = {
            kernel.name: kernel.action_id
            for kernel in load_direct_kernels(args.output)
            if kernel.action_id > 0
        }
    next_action_id = max(
        (kernel.action_id for kernel in (*base, *generated)), default=999
    ) + 1
    stable_generated = []
    for kernel in generated:
        action_id = stable_ids.get(kernel.name)
        if action_id is None:
            action_id = next_action_id
            next_action_id += 1
        stable_generated.append(replace(kernel, action_id=action_id))
    generated = tuple(stable_generated)
    duplicates = {kernel.name for kernel in base} & {kernel.name for kernel in generated}
    if duplicates:
        raise RuntimeError(
            "reviewed and generated direct kernels overlap: "
            + ", ".join(sorted(duplicates))
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# Generated from the flat PI-CAM catalog. Do not edit by hand.\n"
        + yaml.safe_dump(
            direct_kernel_payload((*base, *generated)),
            sort_keys=False,
        )
    )
    print(f"wrote {len(base) + len(generated)} kernels to {args.output}")
    for name, reason in excluded.items():
        print(f"skipped {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
