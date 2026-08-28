#!/usr/bin/env python3
"""What a trained surrogate reproduces, argument by argument.

One number per argument is not enough for this kernel, and the obvious
number is misleading twice over.

The first trap is the state arguments.  ``t0`` and ``qv0`` are learned as a
change, and the change over one step is small against the state itself --
0.1 K against 250 K.  A coefficient of determination on the *value* is then
about 1.0 for a network that predicts no change at all, because the variance
it is scored against is the input's.  What matters is the coefficient on the
**change**, which this reports as ``r2_delta``.

The second trap is intermittency.  Most of what the routine answers is
exactly zero most of the time, and what is left spans decades, so a squared
error over all rows is set by a handful of them.  Whether the term fires is
reported on its own terms -- precision and recall against the truth -- and
the magnitude is scored where it fires, in decades, because an answer that
is right to within a factor of two is a different thing from one that is
wrong by twenty orders.

    tools/evaluate_pi_cam_kernel_surrogate.py --model <model>.pt \\
        --training <set dir> --output <report>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from freecam.physics.surrogate import load_surrogate  # noqa: E402

#: Below this many decades of separation a column is a continuous response
#: and "does it fire" is not a question about it.  Matches the trainer.
SEPARATION_DECADES = 5.0


def holdout_rows(kernel, meta, requested: int) -> np.ndarray:
    """Rows the model was not fitted on.

    A trainer that recorded its own holdout is believed.  Otherwise the split
    is by timestep, because the columns of one chunk at one step are not
    independent draws and a random split would flatter the fit.
    """

    recorded = getattr(kernel, "holdout_rows", None)
    if recorded is None and "meta_nstep" in meta:
        steps = np.asarray(meta["meta_nstep"])
        late = set(np.unique(steps)[-10:].tolist())
        recorded = np.flatnonzero(np.isin(steps, list(late)))
    if recorded is None:
        raise SystemExit("the model records no holdout and the set has no timestep to split on")
    recorded = np.asarray(recorded)
    return recorded[:requested] if requested and recorded.size > requested else recorded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=20000)
    arguments = parser.parse_args()

    import torch

    kernel = load_surrogate(arguments.model)
    payload = torch.load(arguments.model, map_location="cpu", weights_only=False)
    kernel.holdout_rows = payload.get("holdout_rows")
    thresholds = np.asarray(payload.get("thresholds", np.full(len(kernel.y_names), -np.inf)))
    separations = np.asarray(payload.get("separations", np.zeros(len(kernel.y_names))))

    X = np.load(arguments.training / "X.npy", mmap_mode="r")
    Y = np.load(arguments.training / "Y.npy", mmap_mode="r")
    meta = np.load(arguments.training / "meta.npz", allow_pickle=True)

    rows = holdout_rows(kernel, meta, arguments.rows)
    x = np.asarray(X[rows], dtype=np.float64)
    truth_all = np.asarray(Y[rows], dtype=np.float64)
    truth_names = [str(name) for name in meta["y_names"]]

    # A model that derives some of its answers has fewer output columns than
    # the training set has target columns, so the truth is looked up by name.
    # Indexing it with the model's own offsets would compare the right
    # prediction against the wrong argument, silently.
    where_in_truth = {name: index for index, name in enumerate(truth_names)}
    answer = kernel.batched_answer(x)
    prediction, truth, layout = [], [], {}
    at = 0
    for argument, values in answer.items():
        columns = [where_in_truth[name] for name in truth_names
                   if name == argument or name.startswith(argument + "[")]
        block = np.atleast_2d(values.T).T if values.ndim > 1 else values.reshape(-1, 1)
        if block.shape[1] != len(columns):
            raise SystemExit(f"{argument}: model gives {block.shape[1]} columns, "
                             f"the training set records {len(columns)}")
        prediction.append(block)
        truth.append(truth_all[:, columns])
        layout[argument] = slice(at, at + block.shape[1])
        at += block.shape[1]
    prediction = np.concatenate(prediction, axis=1)
    truth = np.concatenate(truth, axis=1)
    # Firing thresholds live in the model's own column order.  A derived
    # answer has none: nothing decided whether it fires.
    by_name = {name: index for index, name in enumerate(kernel.y_names)}
    def threshold_block(argument, width):
        columns = [by_name[n] for n in kernel.y_names
                   if n == argument or n.startswith(argument + "[")]
        if len(columns) != width:
            return np.full(width, -np.inf), np.zeros(width)
        return thresholds[columns], separations[columns]

    cut_blocks, sep_blocks = zip(*(threshold_block(a, w.stop - w.start)
                                   for a, w in layout.items()))
    thresholds, separations = np.concatenate(cut_blocks), np.concatenate(sep_blocks)
    derived = set(getattr(kernel, "derived", []) or
                  [i.target for i in kernel.identities])

    def coefficient(t: np.ndarray, p: np.ndarray):
        variance = t.var()
        return float(1 - ((p - t) ** 2).mean() / variance) if variance > 0 else None

    report = []
    for argument, where in layout.items():
        t, p = truth[:, where], prediction[:, where]
        entry = {"argument": argument, "r2": coefficient(t, p),
                 "derived": argument in derived,
                 "truth_rms": float(np.sqrt((t ** 2).mean()))}

        # What the model was actually fitted to.  For a state that is the
        # change, and every question about it -- does the term fire, how big is
        # it, is the sign right -- is a question about the change.  Asking them
        # of the value instead reports a magnitude error of zero for a column
        # whose change is not reproduced at all, because the value is the input
        # plus something four decades smaller.
        learned_t, learned_p = t, p
        if argument in kernel.delta_columns:
            base = x[:, kernel._x_slices[argument]]
            learned_t, learned_p = t - base, p - base
            entry["r2_delta"] = coefficient(learned_t, learned_p)

        # Firing is judged against the same per-column thresholds the model was
        # trained on, so the report and the model agree on what "zero" means.
        cut = thresholds[where]
        gated = np.isfinite(cut) & (separations[where] >= SEPARATION_DECADES)
        with np.errstate(divide="ignore"):
            truth_log = np.log10(np.abs(learned_t),
                                 out=np.full_like(learned_t, -np.inf),
                                 where=np.abs(learned_t) > 0)
        fires = np.isfinite(truth_log) & (truth_log >= cut[None, :])
        said = learned_p != 0.0
        entry["zero_fraction"] = float(1.0 - fires.mean())
        entry["gated_columns"] = int(gated.sum())
        if fires.any() and (~fires).any():
            hit = float((said & fires).sum())
            entry["gate_precision"] = hit / max(float(said.sum()), 1.0)
            entry["gate_recall"] = hit / max(float(fires.sum()), 1.0)
            entry["gate_accuracy"] = float((said == fires).mean())

        both = fires & said & (np.abs(learned_p) > 0)
        if both.any():
            # Error in decades, where the term actually fires: 0.3 is a factor
            # of two, 1.0 is an order of magnitude.
            decades = np.abs(np.log10(np.abs(learned_p[both]))
                             - np.log10(np.abs(learned_t[both])))
            entry["log_error_p50"] = float(np.median(decades))
            entry["log_error_p90"] = float(np.percentile(decades, 90))
            entry["sign_agreement"] = float(
                (np.sign(learned_p[both]) == np.sign(learned_t[both])).mean())
            entry["r2_firing"] = coefficient(learned_t[both], learned_p[both])
        report.append(entry)

    # Two separate questions, which one number would confuse.  The identity
    # is exact by construction, so its residual before any repair must be
    # zero; a non-zero one means the derivation is wrong.  The floor is a
    # repair *on top* of it, and how often it fires is a measurement of the
    # tendency heads -- a floor that catches a third of the levels is saying
    # the tendencies drive condensate negative a third of the time.  The
    # repair also invents the water it clips, so what it invents is reported
    # rather than left implicit.
    identity_residual, repair = {}, {}
    column = {name: x[:, where] if where.stop - where.start > 1 else x[:, where.start]
              for name, where in kernel._x_slices.items()}
    dt = column["dt"].reshape(-1, 1)
    for identity in kernel.identities:
        derived_value = identity(column, {n: prediction[:, layout[n]]
                                          for n in identity.derives_from}, dt)
        final = prediction[:, layout[identity.target]]
        floored = np.maximum(derived_value, 0.0) if identity.target in kernel.non_negative \
            else derived_value
        scale = max(float(np.percentile(np.abs(final), 99)), 1e-300)
        identity_residual[identity.target] = float(np.abs(floored - final).max() / scale)
        if identity.target in kernel.non_negative:
            below = derived_value < 0.0
            repair[identity.target] = {
                "levels_clipped": float(below.mean()),
                "invented_over_truth_rms": float(
                    np.abs(np.minimum(derived_value, 0.0)).sum()
                    / max(np.abs(truth[:, layout[identity.target]]).sum(), 1e-300)),
            }

    summary = {
        "identity_residual": identity_residual,
        "non_negative_repair": repair,
        "schema_version": 3,
        "model": str(arguments.model), "training": str(arguments.training),
        "kind": kernel.kind, "rows": int(rows.size),
        "provenance": kernel.provenance,
        "arguments": report,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=2, default=float) + "\n")

    if identity_residual:
        worst = max(identity_residual.values())
        print(f"identity residual after repair: worst {worst:.2e} relative"
              f"  --  {'exact' if worst < 1e-12 else 'NOT EXACT: the derivation is wrong'}")
    if repair:
        print("non-negativity floor (a repair, not an identity):")
        for name, item in sorted(repair.items(), key=lambda kv: -kv[1]["levels_clipped"]):
            print(f"  {name:6s} clipped {100 * item['levels_clipped']:5.2f}% of levels, "
                  f"inventing {100 * item['invented_over_truth_rms']:6.3f}% of the argument's mass")
    print(f"{'argument':14s} {'zero':>7s} {'r2':>8s} {'r2_delta':>9s} "
          f"{'gate P/R':>13s} {'decades p50':>11s} {'sign':>6s}")
    for item in report:
        def show(key, spec="8.3f"):
            value = item.get(key)
            return format(value, spec) if isinstance(value, float) else " " * int(spec.split(".")[0])
        gate = (f"{item['gate_precision']:.2f}/{item['gate_recall']:.2f}"
                if "gate_precision" in item else "     -")
        print(f"{item['argument']:14s} {item['zero_fraction']:7.3f} {show('r2')} "
              f"{show('r2_delta', '9.3f')} {gate:>13s} {show('log_error_p50', '11.2f')} "
              f"{show('sign_agreement', '6.2f')}")
    print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
