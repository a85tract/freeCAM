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
    truth = np.asarray(Y[rows], dtype=np.float64)
    prediction = kernel.predict_rows(x)

    def coefficient(t: np.ndarray, p: np.ndarray):
        variance = t.var()
        return float(1 - ((p - t) ** 2).mean() / variance) if variance > 0 else None

    report = []
    for argument, where in kernel._y_slices.items():
        t, p = truth[:, where], prediction[:, where]
        entry = {"argument": argument, "r2": coefficient(t, p),
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

    summary = {
        "schema_version": 2,
        "model": str(arguments.model), "training": str(arguments.training),
        "kind": kernel.kind, "rows": int(rows.size),
        "provenance": kernel.provenance,
        "arguments": report,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=2, default=float) + "\n")

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
