"""Assemble the training dataset of compute_uwshcu_inv from captures of the running model.

The UW shallow cumulus kernel takes the whole constituent array (57 tracers, the
water isotopes among them, each isotope a fixed ratio of the water it mirrors) and
returns their tendencies with its own fractionation.  Drawing those inputs from
distributions, as ``generate_mmacro_pcond_dataset.py`` does for its kernel, would
put the isotopes off their water; the kernel's own calls inside the running model
are the states it will be asked about.  So this script does not sample: it reads
the frames captured at the kernel's hook -- every argument of every call, live
columns only -- and writes them, one sample per column, as one NetCDF file in the
layout the physics-function datasets use (``input__<name>``, ``output__<name>``
over ``sample`` and the contract's axes ``lev``, ``ilev``, ``cnst``; ``sample_id``,
``status``; the provenance of every capture as attributes), plus where each sample
came from (``sample_step``, ``sample_rank``, ``sample_column``, ``sample_call``)
and the constituent names on the ``cnst`` axis.

A capture is a run of the model with the kernel's frames recorded::

    PYCAM_CAPTURE_KERNELS=compute_uwshcu_inv PYCAM_CAPTURE_EVERY=25 \\
        validation/jobs/submit.sh validation/jobs/pi_cam_pausable_1month.pbs

(with the Python stage classes installed as the job does by default; ``every``
records one call in N of each rank's, 1 for all of them).  The run stays
bit-for-bit -- the capture only reads -- and leaves ``<run>/frame-capture/`` with
``compute_uwshcu_inv.rank-NNNN.npz`` per rank and ``capture.json``.  Then::

    uv run python examples/generate_compute_uwshcu_inv_dataset.py \\
        --capture <run>/frame-capture --output compute_uwshcu_inv_training.nc

    uv run python examples/generate_compute_uwshcu_inv_dataset.py \\
        --capture <run-a>/frame-capture --capture <run-b>/frame-capture \\
        --ranks 0:512:4 --no-tracers --output compute_uwshcu_inv_small.nc

The dataset holds the kernel's model block: 20 inputs (``dt`` per column, the
column's pressure, height, wind, water, temperature, static energy, tracers,
turbulence, cloud fractions, boundary-layer height and cumulus scale height) and
30 outputs (the fluxes, tendencies, precipitation, cumulus properties, cloud top
and base levels, the tracer tendencies and the water-tracer precipitation and
detrainment).  ``cush`` is both: ``input__cush`` before the call, ``output__cush``
after.  Level axes run top down as CAM stores them (the ``_inv`` names); a month
capture of every 25th call is about 0.9 million columns and 45 GB uncompressed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from freecam.physics.spec import load_function_spec  # noqa: E402
from freecam.pi_cam.hooks import load_hooks  # noqa: E402

KERNEL = "compute_uwshcu_inv"
CONTRACT = REPO / "native/pi_cam/functions/uwshcu.yaml"

#: the water-tracer arrays: the 57-constituent tracer state and what the kernel returns for it
TRACER_ARGUMENTS = ("tr0_inv", "trten_inv", "wtqc_inv", "wtprec", "wtsnow")

#: the advected constituents of the PI-atm configuration in the kernel's order (CAM's
#: constituent list as its log prints it): bulk water, the pseudo-prognostic precipitation,
#: four water-isotope species in seven phases each, then the chemistry and MAM3 aerosols
CONSTITUENTS = (
    "Q", "CLDLIQ", "CLDICE", "NUMLIQ", "NUMICE", "QRAINC", "QSNOWC", "QRAINS", "QSNOWS",
    "H2OV", "H2OL", "H2OI", "H2OR", "H2OS", "H2Or", "H2Os",
    "H216OV", "H216OL", "H216OI", "H216OR", "H216OS", "H216Or", "H216Os",
    "HDOV", "HDOL", "HDOI", "HDOR", "HDOS", "HDOr", "HDOs",
    "H218OV", "H218OL", "H218OI", "H218OR", "H218OS", "H218Or", "H218Os",
    "H2O2", "H2SO4", "SO2", "DMS", "SOAG", "so4_a1", "pom_a1", "soa_a1", "bc_a1", "dst_a1", "ncl_a1", "num_a1",
    "so4_a2", "soa_a2", "ncl_a2", "num_a2", "dst_a3", "ncl_a3", "so4_a3", "num_a3",
)

AXIS_NAMES = {"pver": "lev", "pverp": "ilev", "pcnst": "cnst"}

NOTES = """\
One sample is one live column of one call of compute_uwshcu_inv inside the running
model, every argument taken from that call: the states are the model's own and the
isotope tracers sit on their water at the model's ratios.  Nothing is sampled or
perturbed; dt is the run's timestep.  The dead columns a chunk may hold beyond its
live count are not written.  A model trained on these must be used inside the model
at the same slot, or on columns from the same configuration.
"""


def parse_ranks(text: str | None) -> slice:
    if not text:
        return slice(None)
    parts = [int(p) if p else None for p in text.split(":")]
    if len(parts) > 3:
        raise SystemExit(f"--ranks takes START:STOP:STEP, got {text!r}")
    return slice(*parts)


def rank_files(capture: Path) -> list[tuple[int, Path]]:
    files = []
    for path in sorted(capture.glob(f"{KERNEL}.rank-*.npz")):
        match = re.search(r"rank-(\d+)\.npz$", path.name)
        if match:
            files.append((int(match.group(1)), path))
    if not files:
        raise SystemExit(f"{capture}: no {KERNEL}.rank-NNNN.npz files")
    return files


def capture_provenance(capture: Path) -> dict:
    record = {}
    meta = capture / "capture.json"
    if meta.is_file():
        record = json.loads(meta.read_text())
    keep = ("run_tag", "pbs_job_id", "native_library_sha256", "every", "calls_total_by_kernel", "bfb_record")
    return {"directory": capture.name, **{k: record[k] for k in keep if k in record}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", type=Path, action="append", required=True, metavar="DIR",
                        help="a run's frame-capture directory; repeatable, the samples are concatenated")
    parser.add_argument("--output", type=Path, default=Path("compute_uwshcu_inv_training.nc"), help="the NetCDF file to write")
    parser.add_argument("--ranks", default=None, metavar="START:STOP:STEP", help="which rank files to take, as a Python slice (default all)")
    parser.add_argument("--every-call", type=int, default=1, metavar="N", help="keep one captured call in N of each rank (default 1)")
    parser.add_argument("--max-samples", type=int, default=None, help="stop after this many columns")
    parser.add_argument("--no-tracers", action="store_true",
                        help=f"leave out the tracer arrays {', '.join(TRACER_ARGUMENTS)} (the file shrinks about fourfold)")
    parser.add_argument("--arguments", default=None, metavar="NAME,...",
                        help="only these arguments of the model block (default: all of them)")
    parser.add_argument("--compress", action="store_true", help="zlib level 1 on every array (smaller, slower to write)")
    parser.add_argument("--list", action="store_true", help="print what each capture holds and exit")
    arguments = parser.parse_args(argv)

    from netCDF4 import Dataset as NetCDF

    spec = load_function_spec(str(CONTRACT))
    hook = load_hooks().hook(KERNEL)
    by_name = {item.name: item for item in spec.arguments}
    inputs, outputs = list(hook.model_inputs), list(hook.model_outputs)
    if arguments.no_tracers:
        inputs = [n for n in inputs if n not in TRACER_ARGUMENTS]
        outputs = [n for n in outputs if n not in TRACER_ARGUMENTS]
    if arguments.arguments:
        wanted = {n.strip() for n in arguments.arguments.split(",") if n.strip()}
        unknown = sorted(wanted - set(inputs) - set(outputs))
        if unknown:
            raise SystemExit(f"--arguments: {unknown} are not in the model block ({', '.join(inputs + outputs)})")
        inputs = [n for n in inputs if n in wanted]
        outputs = [n for n in outputs if n in wanted]
    provenance = [capture_provenance(c) for c in arguments.capture]
    if arguments.list:
        for capture, record in zip(arguments.capture, provenance):
            files = rank_files(capture)
            print(f"{capture}: {len(files)} rank files, {json.dumps(record)}")
        return 0

    def public_dims(name: str) -> list[tuple[str, int]]:
        item = by_name[name]
        return [(AXIS_NAMES.get(str(axis), f"dim_{axis}"), int(spec.dimensions[str(axis)]) if str(axis) in spec.dimensions else int(axis))
                for axis in item.native_shape[1:]]

    probe = [("in", n) for n in inputs if by_name[n].rank >= 1] + [("out", n) for n in outputs if by_name[n].rank >= 1]
    if not probe:
        raise SystemExit("at least one array argument is needed to know a call's live column count")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    written = 0
    with NetCDF(str(arguments.output), "w") as handle:
        handle.createDimension("sample", None)
        declared: set[str] = set()
        variables = {}

        def declare(prefix: str, names: list[str]) -> None:
            for name in names:
                dims = ["sample"]
                for axis, extent in public_dims(name):
                    if axis not in declared:
                        handle.createDimension(axis, extent)
                        declared.add(axis)
                    dims.append(axis)
                chunk = (1024,) + tuple(extent for _, extent in public_dims(name))
                variable = handle.createVariable(f"{prefix}__{name}", "f8", tuple(dims), zlib=arguments.compress,
                                                 complevel=1 if arguments.compress else 0, chunksizes=chunk)
                variable.units = by_name[name].units or ""
                variable.long_name = by_name[name].description or name
                variables[(prefix, name)] = variable

        declare("input", inputs)
        declare("output", outputs)
        if "cnst" in declared:
            names = handle.createVariable("constituent", str, ("cnst",))
            for index, name in enumerate(CONSTITUENTS):
                names[index] = name
        for name, kind in (("sample_id", "i8"), ("sample_step", "i8"), ("sample_call", "i8"), ("sample_token", "i8"),
                           ("sample_rank", "i4"), ("sample_column", "i4"), ("sample_capture", "i4")):
            handle.createVariable(name, kind, ("sample",), chunksizes=(4096,))
        handle.createVariable("status", str, ("sample",))
        handle.createVariable("message", str, ("sample",))

        for capture_index, capture in enumerate(arguments.capture):
            files = rank_files(capture)[parse_ranks(arguments.ranks)]
            for rank, path in files:
                z = np.load(path, allow_pickle=True)
                meta = json.loads(str(z["meta"])) if "meta" in z.files else []
                calls = sorted({int(k.split("/")[1]) for k in z.files if k.startswith("in/")})
                for call in calls[:: max(1, arguments.every_call)]:
                    record = meta[call] if call < len(meta) else {}
                    if "ncol" in record:
                        ncol = int(record["ncol"])
                    else:                                   # the live count from the first array asked for
                        side, name = probe[0]
                        ncol = int(np.asarray(z[f"{side}/{call}/{name}"]).shape[0])
                    if ncol <= 0:
                        continue
                    if arguments.max_samples is not None:
                        ncol = min(ncol, arguments.max_samples - written)
                        if ncol <= 0:
                            break
                    lo, hi = written, written + ncol
                    for prefix, names, side in (("input", inputs, "in"), ("output", outputs, "out")):
                        for name in names:
                            values = np.asarray(z[f"{side}/{call}/{name}"], dtype=np.float64)
                            if values.ndim == 0:
                                values = np.full(ncol, float(values))
                            variables[(prefix, name)][lo:hi, ...] = values[:ncol]
                    handle.variables["sample_id"][lo:hi] = np.arange(lo, hi)
                    handle.variables["sample_step"][lo:hi] = int(record.get("step", -1))
                    handle.variables["sample_call"][lo:hi] = call
                    handle.variables["sample_token"][lo:hi] = int(record.get("token", -1))
                    handle.variables["sample_rank"][lo:hi] = rank
                    handle.variables["sample_column"][lo:hi] = np.arange(ncol)
                    handle.variables["sample_capture"][lo:hi] = capture_index
                    for index in range(lo, hi):
                        handle.variables["status"][index] = "ok"
                        handle.variables["message"][index] = ""
                    written = hi
                if arguments.max_samples is not None and written >= arguments.max_samples:
                    break
                elapsed = time.monotonic() - started
                print(f"  {capture.name} rank {rank:4d}: {written:9d} columns  {elapsed:7.1f} s", flush=True)
            if arguments.max_samples is not None and written >= arguments.max_samples:
                break

        handle.function = spec.qualified_name
        handle.kernel = KERNEL
        handle.generator = "examples/generate_compute_uwshcu_inv_dataset.py"
        handle.captures = json.dumps(provenance)
        handle.inputs = ",".join(inputs)
        handle.outputs = ",".join(outputs)
        handle.tracers = "left out" if arguments.no_tracers else "included"
        handle.every_call = int(arguments.every_call)
        handle.ranks = arguments.ranks or "all"
        handle.constituents = ",".join(CONSTITUENTS)
        handle.notes = NOTES
    size = arguments.output.stat().st_size
    print(f"{arguments.output}: {written} samples, {len(inputs)} inputs, {len(outputs)} outputs, {size / 1e9:.2f} GB, {time.monotonic() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
