"""Generate one mmacro_pcond training dataset with every knob perturbed.

The notebook's demonstration space moves five inputs and one parameter around
a captured column.  This script moves all of them: each of the 35 user inputs
and all 9 tunable parameters is drawn for every sample, from one seeded
generator, and the whole run lands in a single NetCDF file.

Ranges follow the reviewed specification
(``native/pi_cam/functions/mmacro_pcond.yaml``) wherever its ``range`` is a
physical tuning range -- that is, for all nine parameters.  Where the declared
range is only an admissibility bound rather than a climatology (a fraction
lies in [0, 1]; an advective tendency of 0.01 K/s is 864 K/day) the draw uses
the physical sub-range instead, and every such narrowing is listed in
``SAMPLING_NOTES`` and written into the file.

There are two ways to place the state.  Without ``--anchor-bundle`` the
states are perturbed around one captured column, which is a demonstration
space: one column's vertical structure is rank one, and the model's own cloud
water needs twenty-four principal components, so no retuning of the noise
reaches it.  With ``--anchor-bundle`` the anchor is drawn per sample from a
capture's real columns (see ``tools/extract_pi_cam_anchor_columns.py``), which
puts the state on the model's manifold and leaves the perturbation to do the
one job it is good at: a stated budget of off-manifold states, which a
surrogate run inside the model will visit through its own error.

Run it with::

    uv run python examples/generate_mmacro_pcond_dataset.py \
        --samples 10000 --seed 42 --output mmacro_pcond_training.nc

    uv run python examples/generate_mmacro_pcond_dataset.py \
        --samples 200000 --anchor-bundle anchors.npz \
        --output mmacro_pcond_capture_anchored.nc
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import freecam as fc

FUNCTION = "mmacro_pcond"

# Tendency arguments are all zero in the captured column (it was taken at
# nstep = 1), so they are drawn over a range of their own rather than
# perturbed around an anchor.
#
# Temperature is the only one of them with a height-uniform range: T is
# O(200-300 K) at every level, so one bound fits the whole column.  1e-3 K/s
# is 86 K/day, already above the largest convective heating rate in the
# model; the spec's declared bound, 1e-2, is a loose admissibility check.
TEMPERATURE_FORCING = 1.0e-3  # K/s, for A_T, C_T and D_T

# The other five forced quantities span four decades in the vertical -- water
# vapour is 2e-6 kg/kg at 3 hPa and 1e-2 kg/kg near the surface -- so no
# height-uniform bound works for them: a bound large enough to matter near
# the surface drives the top of the column strongly negative, which is what a
# first pass at this dataset did (a quarter of all level values came back
# with negative vapour).  Each is drawn instead as a fraction of that level's
# own drawn state per RELAXATION_TIME, so the forcing scales with the column
# it forces.  Three forcings act on each quantity (A, C, D), each at most
# state/RELAXATION_TIME, over a step of at most 3600 s: a level can lose at
# most half of what it holds, and positivity never depends on the routine's
# own repair.
RELAXATION_TIME = 21600.0  # s

RELAXED_FORCING = {
    "a_qv": "qv0", "c_qv": "qv0", "d_qv": "qv0",
    "a_ql": "ql0", "c_ql": "ql0", "d_ql": "ql0", "c_qlst": "ql0",
    "a_qi": "qi0", "c_qi": "qi0", "d_qi": "qi0",
    "a_nl": "nl0", "c_nl": "nl0", "d_nl": "nl0",
    "a_ni": "ni0", "c_ni": "ni0", "d_ni": "ni0",
}


def relaxation(state: str, scale: float):
    """U(-1, 1) x the level's own drawn ``state`` / RELAXATION_TIME."""

    def draw(rng, **drawn):
        value = np.abs(np.asarray(drawn[state], dtype=np.float64))
        return (2.0 * rng.uniform(0.0, 1.0, value.shape) - 1.0) * value * scale / RELAXATION_TIME

    draw.__name__ = f"fraction_of_{state}_per_{int(RELAXATION_TIME)}s"
    return fc.physics.Derived(draw, depends=(state,))


SAMPLING_NOTES = """\
Every user input and every tunable parameter is drawn per sample.

Parameters (9) use the reviewed range in the specification unchanged, except
cldfrc_premit, whose declared range [20000, 60000] overlaps cldfrc_premib's
[50000, 90000]; it is drawn over [20000, 50000] so the spec's constraint
premib > premit holds for every sample.  cldfrc_iceopt is categorical and is
drawn from its declared value set, not rounded from a continuous draw.

State inputs are perturbed around the captured column so the samples stay on
the model's own manifold: t0 with additive noise, qv0/ql0/qi0/nl0/ni0 with a
relative term plus an absolute term that lets a clear or ice-free layer take
on a little condensate.  qi0 and ni0 get an absolute term far above their
anchor on purpose: the captured column is nearly ice-free, and without ice
the four ice-cloud parameters would have nothing to act on.

p and dp come from one drawn surface pressure through CAM's own hybrid
coordinate (pint = hyai*p0 + hybi*ps), so the two profiles can never
contradict each other.  Surface pressure spans 700-1030 hPa.

snowh is drawn from a half-normal clipped at zero, which puts about half the
samples at exactly zero.  This is deliberate: the snow-free-land adjustment
only applies when nint(landfrac) == 1 and snowh <= 1e-6 m, so a draw that is
positive everywhere would leave cldfrc_rhminl_adj_land with no effect
anywhere in the dataset.

Narrowed from a definitional to a physical range: dt (declared [1, 7200] s,
drawn 900-3600 s, the timesteps CAM runs at); a_cud and a_cu0 (declared
[0, 1] because they are fractions, drawn from a half-normal that is exactly
zero about half the time and reaches ~0.36 otherwise -- cld = a_st_star +
a_cu0, so an always-positive cumulus fraction would both overwrite the
stratiform signal and leave the dataset without a clear level); the tendency
arguments, see
TEMPERATURE_FORCING and RELAXATION_TIME in the generator.

A_*, C_* and D_* are drawn independently even though the routine's own header
says A and C are exclusive in the model.  For vapour they enter as the sum
(qv_05 = qv_0 + (A_qv + C_qv)*dt), so the pair is largely redundant rather
than wrong, and the argument space itself admits both.

clrw_old and clri_old are drawn over the full [0, 1] but are inert in this
configuration: cldwat2m_macro reads them only under i_rhminl > 0 / i_rhmini
> 0, and the reviewed module state declares both switches 0 (they are
non-zero only under UNICON).  The same holds for the six workspace pointer
arguments, which are not user-visible at all.
"""


def build_space(scheme, column, *, forcing_scale: float, vary_do_cldice: bool):
    """Distributions for all 35 inputs and all 9 parameters."""

    inputs = {
        "dt": fc.physics.Uniform(900.0, 3600.0),
        # One draw, three consistent profiles: p and dp (this routine takes no
        # interface pressure) from pint = hyai*p0 + hybi*ps.
        "p": fc.physics.HybridPressure.from_column(
            column, surface_pressure=fc.physics.Uniform(70000.0, 103000.0)
        ),
        "t0": fc.physics.Anchored(column["t0"], absolute_scale=5.0, clip=(150.0, 330.0)),
        "qv0": fc.physics.Anchored(column["qv0"], relative_scale=0.35, absolute_scale=1.0e-7, clip=(0.0, 0.04)),
        "ql0": fc.physics.Anchored(column["ql0"], relative_scale=0.50, absolute_scale=2.0e-6, clip=(0.0, 0.01)),
        "qi0": fc.physics.Anchored(column["qi0"], relative_scale=0.50, absolute_scale=1.0e-6, clip=(0.0, 0.01)),
        "nl0": fc.physics.Anchored(column["nl0"], relative_scale=0.50, absolute_scale=1.0e6, clip=(0.0, 1.0e9)),
        "ni0": fc.physics.Anchored(column["ni0"], relative_scale=0.50, absolute_scale=1.0e3, clip=(0.0, 1.0e8)),
        # Half-normal clipped at zero: about half the draws are exactly zero,
        # which is what a cumulus fraction usually is, and cld = a_st_star +
        # a_cu0, so a draw that is positive everywhere would leave the dataset
        # without a single clear level.
        "a_cud": fc.physics.Normal(0.0, 0.12, clip=(0.0, 1.0)),
        "a_cu0": fc.physics.Normal(0.0, 0.12, clip=(0.0, 1.0)),
        "clrw_old": fc.physics.Uniform(0.0, 1.0),
        "clri_old": fc.physics.Uniform(0.0, 1.0),
        "landfrac": fc.physics.Uniform(0.0, 1.0),
        "snowh": fc.physics.Normal(0.0, 0.5, clip=(0.0, 10.0)),
    }
    for name in ("a_t", "c_t", "d_t"):
        width = TEMPERATURE_FORCING * forcing_scale
        inputs[name] = fc.physics.Uniform(-width, width)
    for name, state in RELAXED_FORCING.items():
        inputs[name] = relaxation(state, forcing_scale)
    if vary_do_cldice:
        inputs["do_cldice"] = fc.physics.Choice([0, 1])

    return scheme.sampling_space(base=column, inputs=inputs, parameters=build_parameters())


def build_parameters():
    """The nine tunable parameters, over the reviewed ranges.

    A capture holds one namelist, so this is the only source of a parameter
    axis; both sampling spaces use it unchanged.
    """

    return {
        "cldfrc_rhminl": fc.physics.Uniform(0.70, 0.99),
        "cldfrc_rhminl_adj_land": fc.physics.Uniform(0.0, 0.2),
        "cldfrc_rhminh": fc.physics.Uniform(0.60, 0.99),
        # Trimmed from [20000, 60000] to keep premib > premit for every sample.
        "cldfrc_premit": fc.physics.Uniform(20000.0, 50000.0),
        "cldfrc_premib": fc.physics.Uniform(50000.0, 90000.0),
        "cldfrc2m_rhmini": fc.physics.Uniform(0.60, 0.99),
        "cldfrc2m_rhmaxi": fc.physics.Uniform(1.00, 1.30),
        "cldfrc_icecrit": fc.physics.Uniform(0.80, 1.00),
        "cldfrc_iceopt": fc.physics.Choice([1, 2, 3, 4, 5]),
    }


# Perturbation around a captured column, by argument.  Scales are read off
# the capture itself (tools/extract_pi_cam_anchor_columns.py prints them):
# a relative term leaves an exact zero exactly zero, so the model's own
# fraction of clear levels survives -- 73% for ql0, 85% for the cumulus
# fractions -- and an absolute term is gated where seeding a clear level is
# the point rather than an accident.
CAPTURE_RELATIVE = {
    "qv0": 0.02, "ql0": 0.05, "qi0": 0.05, "nl0": 0.05, "ni0": 0.05,
    "a_cud": 0.05, "a_cu0": 0.05,
    **{f"{prefix}_{q}": 0.05
       for prefix in ("a", "c", "d")
       for q in ("t", "qv", "ql", "qi", "nl", "ni")},
    "c_qlst": 0.05,
}

#: Ungated unless listed in CAPTURE_GATE: t0 wants half a degree everywhere.
CAPTURE_ABSOLUTE = {"t0": 0.5, "ql0": 1.0e-6, "qi0": 1.0e-7}

#: The off-manifold budget: how often a level the model left clear takes on
#: condensate.  Small on purpose -- the first surrogate failed because its
#: training set had cloud in 51% of levels against the model's 16%.  The rate
#: is masked to the levels the capture ever clouds: the top of the column
#: holds no liquid water in any column of any rank at any step, and a budget
#: spent there buys nothing a surrogate will ever be asked about.
CAPTURE_GATE = {"ql0": 0.03, "qi0": 0.03}

#: A level counts as one the model clouds if any captured column holds this
#: much there.  Below it the capture's values are denormal residue, not cloud.
CAPTURE_CLOUD_FLOOR = 1.0e-12

CAPTURE_CLIP = {
    "t0": (150.0, 330.0), "qv0": (0.0, 0.04),
    "ql0": (0.0, 0.01), "qi0": (0.0, 0.01),
    "nl0": (0.0, 1.0e9), "ni0": (0.0, 1.0e8),
    "a_cud": (0.0, 1.0), "a_cu0": (0.0, 1.0),
}

#: Taken from the capture untouched.  p and dp are one hydrostatic pair and
#: perturbing them apart makes them contradict each other; landfrac and snowh
#: are surface properties the routine reads through nint() switches, where a
#: nudge is a category change rather than a perturbation.
CAPTURE_UNPERTURBED = ("p", "dp", "landfrac", "snowh", "clrw_old", "clri_old")

CAPTURE_NOTES = """\
State, forcing and surface arguments are drawn as whole captured columns:
one real column per sample, every argument taken from that same column, so a
sample is a state the model actually visited rather than independent draws
per variable.  This is what a single-column anchor cannot give -- the model's
cloud water spans 24 principal components over 30 levels and one anchor
supplies exactly one -- and it carries the correlations with it: cloud water
sits where the column reached saturation, and the zero fractions (73% for
ql0, 85% for the cumulus fractions, 64% ocean for landfrac) are the model's.

Perturbation is kept, at a stated budget, for the states a surrogate running
inside the model reaches through its own error.  Relative terms preserve
exact zeros; the absolute term for ql0/qi0 is gated so a clear level takes on
condensate at CAPTURE_GATE rather than everywhere.

dt is drawn over 900-3600 s rather than taken from the capture, which ran at
one timestep: the routine's dependence on it is real and a surrogate that
never saw it vary cannot represent it.  All nine parameters are drawn exactly
as in the single-anchor space -- a capture holds one namelist, so the
parameter axis can only come from the sampler.
"""


def build_capture_space(scheme, anchors, column, *, gate_scale: float,
                        part: int = 0, parts: int = 1, limit: int | None = None):
    """Distributions anchored on a capture: real columns, drawn whole.

    ``limit`` uses a random ``limit`` of the anchors rather than the first of
    them, so a published set can be asked for fewer without rewriting it.  The
    first of them is not a subset to take: extraction sorts the indices it drew,
    and the global column order runs record then rank, so the leading anchors
    are the lowest MPI ranks -- a fixed piece of the globe rather than a sample
    of it.  ``part``/``parts`` split the anchor set between processes.  Two hundred
    thousand columns are 1.4 GB once resident, and generating in parallel
    means that much per process -- five of them was enough to have one killed
    outright.  Taking every ``parts``-th column instead keeps the total
    constant however many processes run, and every anchor still reaches
    exactly one of them.
    """

    produced = tuple(name for name in anchors.files
                     if not name.startswith("meta_") and name != "provenance"
                     and name not in ("dt", "do_cldice"))
    # .copy(), not the strided view a slice would give: a view keeps the
    # whole decompressed member alive and saves nothing at all.
    held = int(np.asarray(anchors[produced[0]]).shape[0])
    if limit is None or limit >= held:
        take = slice(None)
    else:
        # Seeded on the size alone, so every process of one run takes the same
        # subset and their shares stay disjoint.
        take = np.sort(np.random.default_rng(limit).choice(held, limit, replace=False))
    columns = {name: np.array(anchors[name][take][part::parts], copy=True)
               for name in produced}
    gate = {}
    for name, rate in CAPTURE_GATE.items():
        if name not in columns:
            continue
        clouds = (columns[name] > CAPTURE_CLOUD_FLOOR).any(axis=0)
        gate[name] = clouds.astype(np.float64) * rate * gate_scale
    captured = fc.physics.CapturedColumns(
        columns=columns,
        produces=produced,
        relative_scale={k: v for k, v in CAPTURE_RELATIVE.items()
                        if k in produced and k not in CAPTURE_UNPERTURBED},
        absolute_scale={k: v for k, v in CAPTURE_ABSOLUTE.items()
                        if k in produced and k not in CAPTURE_UNPERTURBED},
        absolute_probability=gate,
        clip={k: v for k, v in CAPTURE_CLIP.items() if k in produced},
    )
    inputs = {produced[0]: captured, "dt": fc.physics.Uniform(900.0, 3600.0)}
    return scheme.sampling_space(base=column, inputs=inputs,
                                 parameters=build_parameters())


def _cover(space, spec, held: frozenset[str]) -> None:
    """Fail closed if a knob is left un-drawn other than one deliberately held."""

    drawn = set(space.distributions) | set(space.produced) | set(held)
    missing = [item.name for item in spec.user_arguments if item.name not in drawn]
    missing += [name for name in spec.parameters if name not in drawn]
    if missing:
        raise SystemExit("not every knob is perturbed; missing: " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=10000, help="number of samples (default 10000)")
    parser.add_argument("--seed", type=int, default=42, help="seed for the sampling generator (default 42)")
    parser.add_argument("--output", type=Path, default=Path("mmacro_pcond_training.nc"), help="the single NetCDF file to write")
    parser.add_argument("--forcing-scale", type=float, default=1.0, help="multiplies every tendency range (default 1.0)")
    parser.add_argument("--vary-do-cldice", action="store_true", help="also draw do_cldice; off by default because 0 is not the admitted CAM5 configuration")
    parser.add_argument("--example", default="captured-anchor", help="the example column to perturb around")
    parser.add_argument("--anchor-bundle", type=Path, default=None,
                        help="anchor on a capture's real columns (tools/extract_pi_cam_anchor_columns.py) "
                             "instead of perturbing one example column")
    parser.add_argument("--gate-scale", type=float, default=1.0,
                        help="multiplies the rate at which a clear level is seeded with condensate (default 1.0)")
    parser.add_argument("--anchor-columns", type=int, default=None,
                        help="use only the first this many anchors (default: all of them)")
    parser.add_argument("--anchor-part", type=int, default=0,
                        help="which share of the anchors this process takes (default 0)")
    parser.add_argument("--anchor-parts", type=int, default=1,
                        help="how many shares the anchors are split into, one per parallel "
                             "process; each holds 1/parts of them in memory (default 1)")
    arguments = parser.parse_args()

    scheme = fc.physics.load_function(FUNCTION, max_restarts=max(100, arguments.samples))
    try:
        column = scheme.example_input(arguments.example)
        if arguments.anchor_bundle is not None:
            anchors = np.load(arguments.anchor_bundle, allow_pickle=True)
            space = build_capture_space(scheme, anchors, column,
                                        gate_scale=arguments.gate_scale,
                                        part=arguments.anchor_part,
                                        parts=arguments.anchor_parts,
                                        limit=arguments.anchor_columns)
            held = frozenset({"do_cldice"})
        else:
            space = build_space(
                scheme, column,
                forcing_scale=arguments.forcing_scale,
                vary_do_cldice=arguments.vary_do_cldice,
            )
            held = frozenset() if arguments.vary_do_cldice else frozenset({"do_cldice"})
        _cover(space, scheme.spec, held)
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

    dataset.attributes["generator"] = "examples/generate_mmacro_pcond_dataset.py"
    dataset.attributes["forcing_scale"] = float(arguments.forcing_scale)
    dataset.attributes["example_column"] = str(arguments.example)
    if arguments.anchor_bundle is not None:
        dataset.attributes["anchor_bundle"] = str(arguments.anchor_bundle)
        dataset.attributes["anchor_provenance"] = str(anchors["provenance"])
        dataset.attributes["gate_scale"] = float(arguments.gate_scale)
        dataset.attributes["anchor_share"] = f"{arguments.anchor_part}/{arguments.anchor_parts}"
        dataset.attributes["capture_notes"] = CAPTURE_NOTES
    dataset.attributes["do_cldice"] = "drawn" if arguments.vary_do_cldice else "held at 1, the admitted CAM5 configuration"
    dataset.attributes["sampling_notes"] = SAMPLING_NOTES
    dataset.attributes["worker_restarts"] = int(getattr(scheme.host, "restarts", 0))

    path = dataset.save(arguments.output)
    print(dataset)
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
