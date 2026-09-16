#!/usr/bin/env python3
"""Write docs/contracts.md from the contracts the code reads: every function contract
(native/pi_cam/functions/*.yaml), the hook each serves (native/pi_cam/hooks.yaml), and the
block contracts of the Python-driver form (the two cloud blocks, the radiation branch).

    tools/export_contracts_doc.py          # write docs/contracts.md
    tools/export_contracts_doc.py --check  # exit 1 if the committed page is stale

The page is generated: edit the contracts, not the page.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

TARGET = REPO / "docs/contracts.md"
FUNCTIONS = REPO / "native/pi_cam/functions"


def _shape(items) -> str:
    return "scalar" if not items else ",".join(str(x) for x in items)


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def function_sections() -> list[str]:
    from freecam.physics.spec import load_function_spec
    from freecam.pi_cam.hooks import load_hooks

    hooks = {hook.contract: hook for hook in load_hooks().hooks}
    lines: list[str] = []
    for path in sorted(FUNCTIONS.glob("*.yaml")):
        spec = load_function_spec(str(path))
        rel = path.relative_to(REPO).as_posix()
        lines.append(f"### `{spec.routine}` ({spec.module or 'no module'})\n")
        lines.append(f"Contract `{rel}`; source `{spec.source}`; binding `{spec.binding}`, layout `{spec.layout}`.  ")
        dims = ", ".join(f"{k}={v}" for k, v in sorted(spec.dimensions.items()))
        axes = ", ".join(f"{k} is the `{v}` axis" for k, v in sorted(spec.public_axes.items()))
        lines.append(f"Extents: {dims or 'none'}.  Public axes: {axes or 'none'}.\n")
        hook = hooks.get(rel)
        if hook is not None:
            callers = ", ".join(f"`{c.routine}` ({c.object})" for c in hook.callers)
            lines.append(f"Hook `{hook.kernel}`: `{hook.redirect}` on {callers or 'no caller object'}; "
                         f"binding `{hook.binding}`; callee `{hook.callee_symbol}`.  ")
            if hook.takes_model:
                lines.append(f"Model block: {len(hook.model_inputs)} inputs "
                             f"({', '.join(hook.model_inputs)}), {len(hook.model_outputs)} outputs "
                             f"({', '.join(hook.model_outputs)}).\n")
            else:
                lines.append("No model block: the hook counts and pauses only.\n")
        else:
            lines.append("No hook: reached at a runner pause or as a standalone function.\n")
        lines.append("| argument | role | intent | type | native shape | public shape | units | notes |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for a in spec.arguments:
            notes = []
            if a.role == "structural":
                notes.append(f"value {a.value}")
            if a.pointer:
                notes.append("pointer")
            if a.optional:
                notes.append("optional")
            if a.carrier:
                notes.append(f"{a.carrier} carrier")
            if a.lower_bounds and any(b != 1 for b in a.lower_bounds):
                notes.append(f"lower bounds {','.join(str(b) for b in a.lower_bounds)}")
            if a.description:
                notes.append(_cell(a.description))
            lines.append(f"| `{a.name}` | {a.role} | {a.intent} | {a.fortran_type}/{a.dtype} | {_shape(a.native_shape)} | "
                         f"{_shape(a.public_shape) if a.public_shape is not None else '-'} | {_cell(a.units or '')} | "
                         f"{'; '.join(notes)} |")
        if spec.parameters:
            lines.append(f"\nParameters read from the module: {', '.join(f'`{k}`' for k in sorted(spec.parameters))}.")
        if spec.module_state:
            lines.append(f"\nModule state the routine depends on: {', '.join(f'`{e.symbol}`' for e in spec.module_state)}.")
        lines.append("")
    return lines


def block_sections() -> list[str]:
    from freecam.physics import cloud_block, radiation_process

    lines: list[str] = []
    for block in (cloud_block.MACRO_BLOCK, cloud_block.MICRO_BLOCK):
        lines.append(f"### Cloud block `{block.name}`\n")
        driver = {"macro": "`macrop_driver_tend`", "micro": "`microp_aero_run`, `micro_mg_cam_tend` and the tendency sum"}[block.name]
        lines.append(f"The compute block of {driver}; its tendency object is `{block.ptend_name}`.  "
                     f"{len(block.inputs)} inputs, {len(block.outputs)} outputs, {len(block.buffers)} buffer fields.")
        if block.cloud_borne:
            lines.append("Also writes the cloud-borne aerosol fields, registered per constituent at run time.")
        if block.tracer_precipitation:
            lines.append("Also writes the water tracers' surface precipitation fields.")
        buffer_names = set(block.buffer_names) | {f.name for f in block.input_buffers}
        plain_in = [n for n in block.inputs if n not in buffer_names]
        plain_out = [n for n in block.outputs if n not in buffer_names]
        lines.append(f"\nInputs besides the buffer: {', '.join(f'`{n}`' for n in plain_in)}.\n")
        lines.append(f"Outputs besides the buffer: {', '.join(f'`{n}`' for n in plain_out)}.\n")
        lines.append("| buffer field | index symbol | older time sample | rank | dtype | read only |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for f in block.buffers + tuple(x for x in block.input_buffers if x.name not in block.buffer_names):
            read_only = "yes" if f.name in {x.name for x in block.input_buffers} else ""
            lines.append(f"| `{f.name}` | `{f.symbol}` | {'yes' if f.time_sliced else ''} | {f.rank} | {f.dtype} | {read_only} |")
        lines.append("")
    lines.append("### Radiation branch\n")
    rank = dict(radiation_process.TABLE_INPUTS) | dict(radiation_process.TABLE_OUTPUTS)
    lines.append("The computing branch of `radiation_tend` (both solvers and the diagnostics around them), "
                 "run on the radiative steps; captured, replayed or answered by a block model.\n")
    lines.append("| plane | fields (rank) |")
    lines.append("| --- | --- |")
    for label, names in (("scalar inputs", radiation_process.SCALAR_INPUTS), ("array inputs", radiation_process.ARRAY_INPUTS),
                         ("rrtmg state inputs", radiation_process.RSTATE_INPUTS), ("outputs", radiation_process.OUTPUTS)):
        lines.append(f"| {label} | " + ", ".join(f"`{n}`" + (f" ({rank[n]})" if n in rank else "") for n in names) + " |")
    lines.append(f"\nThe block model's inputs (`BLOCK_INPUTS`): {', '.join(f'`{n}`' for n in radiation_process.BLOCK_INPUTS)}.\n")
    return lines


def render() -> str:
    head = [
        "# Contracts\n",
        "Generated by `tools/export_contracts_doc.py` from the contracts the code reads; do not edit.  ",
        "A **function contract** describes one Fortran kernel as its dummies see it: each argument's role, ",
        "intent, type, the shape the kernel declares (native, with the chunk's `pcols` columns) and the ",
        "shape a replacement sees (public, one column at a time in the `column` layout).  A **hook** puts ",
        "a wrapper of that signature at the kernel's call sites so a model or plugin can answer it inside ",
        "the image.  A **block contract** describes a whole compute block of a driver by the names of what ",
        "it reads and writes, for the Python-driver form (capture, replay, block model).\n",
        "## Function contracts\n",
    ]
    return "\n".join(head + function_sections() + ["## Block contracts\n"] + block_sections()).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 if docs/contracts.md is not what would be written")
    args = parser.parse_args()
    text = render()
    if args.check:
        current = TARGET.read_text() if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET.relative_to(REPO)} is stale; run tools/export_contracts_doc.py")
            return 1
        print(f"{TARGET.relative_to(REPO)} is current")
        return 0
    TARGET.write_text(text)
    print(f"wrote {TARGET.relative_to(REPO)} ({len(text) // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
