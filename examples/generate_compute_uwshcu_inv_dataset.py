"""Generate one compute_uwshcu_inv training dataset: real columns, perturbed, answered by the Fortran.

The UW shallow cumulus kernel (``uwshcu::compute_uwshcu_inv``, the most expensive kernel
of the physics) runs here as a Python function in a standalone image linked from the
pinned iCESM object code, so every answer is the model's own arithmetic.  Each sample
draws one real column the model computed, perturbs it, draws the tunable parameter,
and calls the Fortran.  Every one of the kernel's 20 inputs is drawn for every sample
and every one of its 30 outputs is written: the 57-constituent tracer array among
them, with the water isotopes on their water at the column's own ratios.

Two ways to place the state, as for ``generate_mmacro_pcond_dataset.py``.  Without
``--anchor-bundle`` the samples are perturbed around the shipped example column, a
demonstration space.  With ``--anchor-bundle`` (``tools/extract_pi_cam_anchor_columns.py
--frame-capture``) the anchor is drawn per sample from the columns the model actually
gave the kernel, and the perturbation adds a stated budget of states the kernel would
meet through a surrogate's own error.

Run it with::

    uv run python examples/generate_compute_uwshcu_inv_dataset.py \\
        --samples 2000 --seed 42 --output compute_uwshcu_inv_example.nc

    uv run python examples/generate_compute_uwshcu_inv_dataset.py \\
        --samples 200000 --anchor-bundle anchors.npz \\
        --output compute_uwshcu_inv_capture_anchored.nc
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import freecam as fc

FUNCTION = "uwshcu"
KERNEL = "compute_uwshcu_inv"

#: CAM's constants (physconst): dry static energy is cp*T + g*z + phis
CP = 1004.64
GRAVITY = 9.80616

#: where the bulk water sits in the tracer array (0-based constituents Q, CLDLIQ, CLDICE)
WATER = {"qv0_inv": 0, "ql0_inv": 1, "qi0_inv": 2}
#: the water-isotope species (bulk H2O, H216O, HDO, H218O) as vapour, liquid, ice
ISOTOPE_SETS = ((9, 10, 11), (16, 17, 18), (23, 24, 25), (30, 31, 32))
#: the cloud droplet and crystal numbers ride with their condensate
NUMBERS = {3: ("ql0_inv", 1.0 / (4.0 / 3.0 * np.pi * 1.0e-5 ** 3 * 1000.0)),     # 10 um droplets
           4: ("qi0_inv", 1.0 / (4.0 / 3.0 * np.pi * 2.5e-5 ** 3 * 500.0))}      # 25 um crystals

#: the hydrostatic set comes from the anchor as one piece: interface and mid-level
#: pressure and height, the layer's pressure thickness wet and dry
HYDROSTATIC = ("ps0_inv", "zs0_inv", "p0_inv", "z0_inv", "dp0_inv", "dpdry0_inv")

#: perturbation of the state a surrogate's own error would move: a relative term keeps
#: an exact zero exactly zero, an absolute term lets a clear level take on condensate
#: at the gated rate, temperature and wind take an absolute nudge everywhere
RELATIVE = {"qv0_inv": 0.02, "ql0_inv": 0.05, "qi0_inv": 0.05, "tke_inv": 0.05, "cldfrct_inv": 0.05, "pblh": 0.05}
ABSOLUTE = {"t0_inv": 0.5, "u0_inv": 0.5, "v0_inv": 0.5, "ql0_inv": 1.0e-6, "qi0_inv": 1.0e-7}
GATE = {"ql0_inv": 0.03, "qi0_inv": 0.03}
CLIP = {"t0_inv": (150.0, 330.0), "qv0_inv": (0.0, 0.04), "ql0_inv": (0.0, 0.01), "qi0_inv": (0.0, 0.01),
        "tke_inv": (0.0, 100.0), "cldfrct_inv": (0.0, 1.0), "concldfrct_inv": (0.0, 1.0), "pblh": (10.0, 6000.0),
        "u0_inv": (-150.0, 150.0), "v0_inv": (-150.0, 150.0)}
#: the perturbation of the two arguments the rules below rebuild themselves
CUSH_RELATIVE = 0.05
CONCLD_RELATIVE = 0.05

SAMPLING_NOTES = """\
Every one of the kernel's 20 inputs is drawn for every sample and its one tunable
parameter with them; every one of its 30 outputs is written, the tracer arrays
included.

The state is one real column per sample -- the model's own, taken whole from a
capture of the kernel's calls (or, without an anchor bundle, the shipped example
column) -- perturbed at a stated budget: temperature and wind by half a unit
everywhere, water vapour by 2 % of itself, condensate by 5 % of itself plus a
seed of cloud at 3 % of clear levels, turbulence, cloud fraction and boundary
layer height by 5 % of themselves.  Relative terms leave the model's exact
zeros zero.

Four arguments are not perturbed on their own but rebuilt so the column stays
one the kernel recognises:

  s0_inv        the dry static energy follows the perturbed temperature,
                s0 = s0_anchor + cp (t0 - t0_anchor), with cp = 1004.64 J/kg/K;
                the anchor's own s0 - cp t0 - g z0 is the surface geopotential,
                constant over the column to 1e-10 in every captured column.
  tr0_inv       the tracer array's Q, CLDLIQ and CLDICE are the drawn water; each
                water-isotope species (bulk H2O, H216O, HDO, H218O; constituents
                10-12, 17-19, 24-26, 31-33) keeps the anchor's ratio to its water
                at every level (the vapour ratio where the anchor holds no
                condensate); the droplet and crystal numbers scale with their
                condensate (a seeded cloud gets 10 um droplets or 25 um crystals);
                every other constituent is the anchor's.
  concldfrct_inv perturbed by 5 % then held at or below the drawn cloud fraction.
  cush          the cumulus scale height is -1 where the anchor had no cumulus
                (56 % of columns) and stays so; elsewhere perturbed by 5 %.

The hydrostatic set (ps0, zs0, p0, z0, dp0, dpdry0) is the anchor's, unperturbed,
so pressure and height never contradict each other.  dt is drawn over 900-3600 s,
the timesteps CAM runs at; the capture ran at one.  uwshcu_rpen, the penetrative
entrainment efficiency, is drawn over its reviewed range [1, 20].
"""


def _ratio(numerator: np.ndarray, denominator: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denominator > 0.0, numerator / np.where(denominator > 0.0, denominator, 1.0), fallback)
    return ratio


def rebuild_tracers(anchor: dict, drawn: dict) -> np.ndarray:
    """The tracer array of the perturbed column: water from the draw, isotopes at the anchor's ratios."""

    tracers = np.array(anchor["tr0_inv"], dtype=np.float64)
    for name, index in WATER.items():
        tracers[:, index] = drawn[name]
    vapour_ratio = {}
    for vapour, liquid, ice in ISOTOPE_SETS:
        ones = np.ones_like(anchor["qv0_inv"])
        r_v = _ratio(anchor["tr0_inv"][:, vapour], anchor["qv0_inv"], ones)
        r_l = _ratio(anchor["tr0_inv"][:, liquid], anchor["ql0_inv"], r_v)
        r_i = _ratio(anchor["tr0_inv"][:, ice], anchor["qi0_inv"], r_v)
        tracers[:, vapour] = r_v * drawn["qv0_inv"]
        tracers[:, liquid] = r_l * drawn["ql0_inv"]
        tracers[:, ice] = r_i * drawn["qi0_inv"]
        vapour_ratio[vapour] = r_v
    for index, (water, per_kg) in NUMBERS.items():
        seeded = np.full_like(anchor[water], per_kg) * drawn[water]
        tracers[:, index] = _ratio(anchor["tr0_inv"][:, index], anchor[water], np.zeros_like(seeded)) * drawn[water]
        tracers[:, index] = np.where(anchor[water] > 0.0, tracers[:, index], seeded)
    return tracers


def static_energy(rng, anchor: dict, drawn: dict) -> np.ndarray:
    return anchor["s0_inv"] + CP * (drawn["t0_inv"] - anchor["t0_inv"])


def tracers(rng, anchor: dict, drawn: dict) -> np.ndarray:
    return rebuild_tracers(anchor, drawn)


def concld(rng, anchor: dict, drawn: dict) -> np.ndarray:
    value = anchor["concldfrct_inv"] * (1.0 + rng.standard_normal(anchor["concldfrct_inv"].shape) * CONCLD_RELATIVE)
    return np.minimum(np.clip(value, 0.0, 1.0), drawn["cldfrct_inv"])


def cush(rng, anchor: dict, drawn: dict) -> np.ndarray:
    value = np.asarray(anchor["cush"], dtype=np.float64)
    if value <= 0.0:
        return value
    return np.maximum(value * (1.0 + rng.standard_normal() * CUSH_RELATIVE), 1.0)


def build_parameters():
    return {"uwshcu_rpen": fc.physics.Uniform(1.0, 20.0)}


def build_capture_space(scheme, anchors, column, *, gate_scale: float,
                        part: int = 0, parts: int = 1, limit: int | None = None):
    """Distributions anchored on a capture: real columns, drawn whole, with the rules above.

    ``limit`` takes a seeded random ``limit`` of the anchors; ``part``/``parts`` split
    them between processes so each holds its share only.
    """

    names = [item.name for item in scheme.spec.user_arguments if item.name != "dt"]
    held = int(np.asarray(anchors[names[0]]).shape[0])
    take = slice(None) if limit is None or limit >= held else \
        np.sort(np.random.default_rng(limit).choice(held, limit, replace=False))
    columns = {name: np.array(anchors[name][take][part::parts], copy=True) for name in names}
    gate = {}
    for name, rate in GATE.items():
        clouds = (columns[name] > 1.0e-12).any(axis=0)          # levels the model ever clouds
        gate[name] = clouds.astype(np.float64) * rate * gate_scale
    # the order matters: the rules read what was drawn before them
    produces = tuple(n for n in names if n not in ("s0_inv", "tr0_inv", "concldfrct_inv", "cush")) + \
        ("concldfrct_inv", "cush", "s0_inv", "tr0_inv")
    captured = fc.physics.CapturedColumns(
        columns=columns, produces=produces,
        relative_scale={k: v for k, v in RELATIVE.items() if k in names},
        absolute_scale={k: v for k, v in ABSOLUTE.items() if k in names},
        absolute_probability=gate,
        clip={k: v for k, v in CLIP.items() if k in names},
        derived={"s0_inv": static_energy, "tr0_inv": tracers, "concldfrct_inv": concld, "cush": cush},
    )
    inputs = {produces[0]: captured, "dt": fc.physics.Uniform(900.0, 3600.0)}
    return scheme.sampling_space(base=column, inputs=inputs, parameters=build_parameters())


def build_space(scheme, column):
    """The demonstration space: the shipped example column perturbed the same way."""

    anchor = {name: np.asarray(column[name], dtype=np.float64) for name in column}
    inputs = {"dt": fc.physics.Uniform(900.0, 3600.0)}
    for name in ("t0_inv", "u0_inv", "v0_inv", "qv0_inv", "ql0_inv", "qi0_inv", "tke_inv", "cldfrct_inv", "pblh"):
        inputs[name] = fc.physics.Anchored(anchor[name], relative_scale=RELATIVE.get(name, 0.0),
                                           absolute_scale=ABSOLUTE.get(name, 0.0), clip=CLIP.get(name))
    inputs["s0_inv"] = fc.physics.Derived(lambda rng, t0_inv: static_energy(rng, anchor, {"t0_inv": t0_inv}), depends=("t0_inv",))
    inputs["tr0_inv"] = fc.physics.Derived(lambda rng, **drawn: rebuild_tracers(anchor, drawn),
                                           depends=("qv0_inv", "ql0_inv", "qi0_inv"))
    inputs["concldfrct_inv"] = fc.physics.Derived(lambda rng, cldfrct_inv: concld(rng, anchor, {"cldfrct_inv": cldfrct_inv}),
                                                  depends=("cldfrct_inv",))
    inputs["cush"] = fc.physics.Derived(lambda rng: cush(rng, anchor, {}), depends=())
    for name in HYDROSTATIC:
        inputs[name] = fc.physics.Constant(anchor[name])
    return scheme.sampling_space(base=column, inputs=inputs, parameters=build_parameters())


def _cover(space, spec) -> None:
    """Fail closed if any input or parameter is left un-drawn."""

    drawn = set(space.distributions) | set(space.produced)
    missing = [item.name for item in spec.user_arguments if item.name not in drawn]
    missing += [name for name in spec.parameters if name not in drawn]
    if missing:
        raise SystemExit("not every knob is drawn; missing: " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=2000, help="number of samples (default 2000)")
    parser.add_argument("--seed", type=int, default=42, help="seed for the sampling generator (default 42)")
    parser.add_argument("--output", type=Path, default=Path("compute_uwshcu_inv_training.nc"), help="the single NetCDF file to write")
    parser.add_argument("--example", default="captured-anchor", help="the example column to perturb around")
    parser.add_argument("--anchor-bundle", type=Path, default=None,
                        help="anchor on a capture's real columns (tools/extract_pi_cam_anchor_columns.py --frame-capture)")
    parser.add_argument("--gate-scale", type=float, default=1.0,
                        help="multiplies the rate at which a clear level is seeded with condensate (default 1.0)")
    parser.add_argument("--anchor-columns", type=int, default=None, help="use only a random this many anchors (default: all)")
    parser.add_argument("--anchor-part", type=int, default=0, help="which share of the anchors this process takes (default 0)")
    parser.add_argument("--anchor-parts", type=int, default=1, help="how many shares the anchors are split into, one per process (default 1)")
    arguments = parser.parse_args()

    scheme = fc.physics.load_function(FUNCTION, max_restarts=max(100, arguments.samples))
    anchors = None
    try:
        column = scheme.example_input(arguments.example)
        if arguments.anchor_bundle is not None:
            anchors = np.load(arguments.anchor_bundle, allow_pickle=True)
            space = build_capture_space(scheme, anchors, column, gate_scale=arguments.gate_scale,
                                        part=arguments.anchor_part, parts=arguments.anchor_parts,
                                        limit=arguments.anchor_columns)
        else:
            space = build_space(scheme, column)
        _cover(space, scheme.spec)
        print(space.describe(), flush=True)

        started = time.monotonic()
        step = max(1, arguments.samples // 20)

        def progress(done: int, total: int, status: str) -> None:
            if done % step and done != total:
                return
            elapsed = time.monotonic() - started
            print(f"  {done:6d}/{total}  {elapsed:7.1f} s  {done / max(elapsed, 1e-9):6.1f} samples/s  last={status}", flush=True)

        dataset = scheme.generate_dataset(arguments.samples, space, seed=arguments.seed, progress=progress)
    finally:
        scheme.close()

    dataset.attributes["generator"] = "examples/generate_compute_uwshcu_inv_dataset.py"
    dataset.attributes["kernel"] = KERNEL
    dataset.attributes["example_column"] = str(arguments.example)
    if anchors is not None:
        dataset.attributes["anchor_bundle"] = str(arguments.anchor_bundle)
        dataset.attributes["anchor_provenance"] = str(anchors["provenance"])
        dataset.attributes["gate_scale"] = float(arguments.gate_scale)
        dataset.attributes["anchor_share"] = f"{arguments.anchor_part}/{arguments.anchor_parts}"
    dataset.attributes["sampling_notes"] = SAMPLING_NOTES
    dataset.attributes["worker_restarts"] = int(getattr(scheme.host, "restarts", 0))

    path = dataset.save(arguments.output)
    print(dataset)
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
