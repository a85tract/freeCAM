#!/usr/bin/env python3
"""Train a gated surrogate: decide whether a term fires, then how big it is.

The first surrogate for this kernel was one regression head under mean
squared error, and it failed in the model at step 3 with -6.1e+02 kg/kg of
cloud liquid.  The reason is in the data rather than in the fitting: most of
what this routine answers is exactly zero most of the time -- ``qiadj`` in
94.6% of levels, ``qladj`` in 90.5% -- and a single squared-error head cannot
answer "nothing happened here".  It answers a small non-zero number instead,
CAM's own bounds checks see negative condensate, and the run stops.

So each target column gets three heads:

* **significance** -- does this term fire at all;
* **sign** -- which way, given that it fires;
* **magnitude** -- how big, in decades above the firing threshold, given that
  it fires.

The threshold is not "different from zero".  The captured answers are
bimodal in log magnitude with a gap of ten to seventeen decades between the
physics and the round-off residue left by the routine's own arithmetic
(``qladj``: half its non-zero values are below 1e-26, and the next quarter
are above 1e-9).  Calling the residue a signal would train the classifier on
floating-point noise.  The split is found per column by Otsu's method on the
log-magnitude histogram and is only used where the two classes really are
far apart; otherwise the column is continuous and the gate stays open.

    tools/train_pi_cam_gated_surrogate.py \\
        --training <dir with X.npy, Y.npy, meta.npz> --output surrogate.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: Two log-magnitude classes count as separated -- and so as a gate worth
#: fitting -- only when their means are this many decades apart.  Below it the
#: column is a continuous response and Otsu's split would be an artefact.
SEPARATION_DECADES = 5.0

#: Never gate a column that fires almost always: the classifier would be a
#: constant and the masked magnitude head would lose the few zeros anyway.
ALWAYS_ON = 0.995

#: Clamp on the magnitude head's answer, in decades either side of what
#: training saw.  A surrogate asked about a state its training never held
#: must answer badly, not answer with an overflow.
DECADE_CLAMP = 2.0

#: Ceiling on the weight given to a column's firing samples.  ``qiadj`` fires
#: in 4% of rows and ``qilim`` in 6%, and an unweighted cross-entropy answers
#: "never" for both -- correct 95% of the time and useless.  The weight is the
#: ratio of quiet rows to firing ones, capped so a column that almost never
#: fires cannot dominate the gradient of every other column.
POSITIVE_WEIGHT_CAP = 20.0


def columns_of(names: list[str], argument: str) -> list[int]:
    return [index for index, name in enumerate(names)
            if name == argument or name.startswith(argument + "[")]


def otsu_threshold(magnitudes: np.ndarray) -> tuple[float, float]:
    """Split log10|y| into two classes; return (threshold, separation).

    Otsu's method: the split that minimises the variance within the two
    classes it makes.  On this data the low class is the routine's round-off
    residue and the high class is its physics.
    """

    if magnitudes.size < 64:
        return -np.inf, 0.0
    low, high = np.percentile(magnitudes, [0.1, 99.9])
    if not np.isfinite([low, high]).all() or high - low < 1e-6:
        return -np.inf, 0.0
    counts, edges = np.histogram(magnitudes, bins=128, range=(low, high))
    centres = 0.5 * (edges[:-1] + edges[1:])
    weight = np.cumsum(counts) / max(counts.sum(), 1)
    mean = np.cumsum(counts * centres) / max(counts.sum(), 1)
    total = mean[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (total * weight - mean) ** 2 / (weight * (1.0 - weight))
    between[~np.isfinite(between)] = -np.inf
    cut = int(np.argmax(between))
    threshold = float(centres[cut])
    below, above = magnitudes < threshold, magnitudes >= threshold
    if below.sum() < 16 or above.sum() < 16:
        return -np.inf, 0.0
    return threshold, float(magnitudes[above].mean() - magnitudes[below].mean())


class GatedSurrogate(nn.Module):
    """One trunk, three heads per target column."""

    def __init__(self, features: int, targets: int, hidden: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        size = features
        for _ in range(depth):
            layers += [nn.Linear(size, hidden), nn.SiLU()]
            size = hidden
        self.trunk = nn.Sequential(*layers)
        self.significance = nn.Linear(size, targets)
        self.sign = nn.Linear(size, targets)
        self.magnitude = nn.Linear(size, targets)

    def forward(self, x):
        h = self.trunk(x)
        return self.significance(h), self.sign(h), self.magnitude(h)


def prepare(training: Path) -> dict:
    """Targets, thresholds and the three label sets, from X and Y on disk."""

    X = np.load(training / "X.npy", mmap_mode="r")
    Y = np.load(training / "Y.npy", mmap_mode="r")
    meta = np.load(training / "meta.npz", allow_pickle=True)
    x_names = [str(name) for name in meta["x_names"]]
    y_names = [str(name) for name in meta["y_names"]]
    y_arguments = [str(name) for name in meta["y_arguments"]]

    target = np.asarray(Y, dtype=np.float64)

    # The six states the routine updates are learned as a change, not as a
    # value: their answer is the input plus a small correction, and a network
    # asked for the value spends its capacity reproducing the input.
    delta_columns: dict[str, list[int]] = {}
    delta_inputs: dict[str, list[int]] = {}
    for argument in y_arguments:
        here = columns_of(y_names, argument)
        there = columns_of(x_names, argument)
        if there and len(there) == len(here):
            target[:, here] -= np.asarray(X[:, there], dtype=np.float64)
            delta_columns[argument] = here
            delta_inputs[argument] = there

    # Half a million samples by 690 targets is 2.8 GB per float64 copy, and
    # the obvious sequence holds five of them at once.  Each temporary is
    # released as soon as the next is derived from it.
    with np.errstate(divide="ignore"):
        logs = np.full(target.shape, -np.inf)
        np.log10(np.abs(target), out=logs, where=np.abs(target) > 0)

    thresholds = np.full(target.shape[1], -np.inf)
    separations = np.zeros(target.shape[1])
    for column in range(target.shape[1]):
        finite = logs[:, column][np.isfinite(logs[:, column])]
        if finite.size == 0:
            continue
        cut, separation = otsu_threshold(finite)
        live = np.isfinite(logs[:, column])
        fires = float((live & (logs[:, column] >= cut)).mean()) if np.isfinite(cut) else float(live.mean())
        if separation >= SEPARATION_DECADES and fires < ALWAYS_ON:
            thresholds[column], separations[column] = cut, separation
        else:
            # Continuous, or effectively always on: the gate stays open and
            # every non-zero value is the magnitude head's business.
            thresholds[column] = -np.inf

    # An exact zero never fires, whatever the threshold.  log10(0) is -inf and
    # a column left ungated has a threshold of -inf, so the comparison alone
    # would call every zero a firing sample with a magnitude of 10**0.
    fires = np.isfinite(logs) & (logs >= thresholds[None, :])
    return {"X": X, "target": target, "logs": logs, "fires": fires,
            "thresholds": thresholds, "separations": separations,
            "x_names": x_names, "y_names": y_names,
            "x_arguments": [str(n) for n in meta["x_arguments"]],
            "y_arguments": y_arguments, "levels": int(meta["levels"]),
            "delta_columns": delta_columns, "delta_inputs": delta_inputs,
            "meta": meta}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--training", type=Path, required=True,
                        help="a directory holding X.npy, Y.npy and meta.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--holdout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    arguments = parser.parse_args()

    if arguments.threads:
        torch.set_num_threads(arguments.threads)
    torch.manual_seed(arguments.seed)
    rng = np.random.default_rng(arguments.seed)

    print(f"reading {arguments.training}", flush=True)
    data = prepare(arguments.training)

    # The parameter features carry the namelist.  A model run holds it at the
    # case's defaults, so the surrogate must know them: it is called with a
    # column, not with a namelist, and a missing parameter would otherwise
    # read as zero -- an rhminl of 0 rather than 0.87.
    dataset_provenance = json.loads(str(data["meta"]["provenance"])) if "provenance" in data["meta"] else {}
    parameter_defaults: dict[str, float] = {}
    function = dataset_provenance.get("function")
    if function:
        from freecam.physics.spec import load_function_spec

        for name, parameter in load_function_spec(function).parameters.items():
            parameter_defaults[name] = parameter.default
    print(f"  parameter defaults: {len(parameter_defaults)} from the {function!r} spec")
    X, target, logs, fires = data["X"], data["target"], data["logs"], data["fires"]
    samples, features = X.shape
    targets = target.shape[1]
    gated = int(np.isfinite(data["thresholds"]).sum())
    print(f"  {samples} samples, {features} features, {targets} targets")
    print(f"  gated columns: {gated} of {targets}"
          f" (separation >= {SEPARATION_DECADES} decades); the rest stay open")
    print(f"  firing rate over all targets: {100 * fires.mean():.2f}%")

    # Inputs: asinh over the argument's own p95, which is finite for a column
    # that is mostly zero and does not explode for one that spans decades.
    x = np.asarray(X, dtype=np.float32)
    x_scale = np.percentile(np.abs(x), 95, axis=0).astype(np.float64)
    x_scale[x_scale <= 0] = 1.0
    x_transformed = np.arcsinh(x / x_scale.astype(np.float32))
    del x

    # Magnitude is learned in decades above the firing threshold, centred and
    # scaled per column over the firing samples only.
    floor = np.where(np.isfinite(data["thresholds"]), data["thresholds"], 0.0)
    excess = np.where(fires, logs - floor[None, :], 0.0)
    excess[~np.isfinite(excess)] = 0.0
    del logs
    data["logs"] = None
    counts = fires.sum(axis=0)
    log_centre = np.where(counts > 0, excess.sum(axis=0) / np.maximum(counts, 1), 0.0)
    spread = np.where(fires, (excess - log_centre[None, :]) ** 2, 0.0).sum(axis=0)
    log_scale = np.sqrt(np.where(counts > 1, spread / np.maximum(counts - 1, 1), 1.0))
    log_scale[log_scale < 1e-3] = 1.0

    # Rarity, per column, as the cross-entropy's positive weight.
    firing_rate = fires.mean(axis=0)
    positive_weight = np.clip(
        np.where(firing_rate > 0, (1.0 - firing_rate) / np.maximum(firing_rate, 1e-6), 1.0),
        1.0, POSITIVE_WEIGHT_CAP)
    rare = int((positive_weight > 1.5).sum())
    print(f"  positive weight above 1.5 on {rare} columns, "
          f"largest {positive_weight.max():.1f}")

    u = np.where(fires, (excess - log_centre[None, :]) / log_scale[None, :], 0.0).astype(np.float32)
    positive = (target > 0).astype(np.float32)
    del target
    data["target"] = None

    # How many decades above its threshold each column was ever seen to reach.
    # Inference clamps to this band, widened by DECADE_CLAMP: a surrogate asked
    # about a state its training never held must answer badly, not overflow.
    excess_low = np.where(fires, excess, np.inf).min(axis=0)
    excess_high = np.where(fires, excess, -np.inf).max(axis=0)
    excess_low[~np.isfinite(excess_low)] = 0.0
    excess_high[~np.isfinite(excess_high)] = 0.0
    del excess
    firing = fires.astype(np.float32)

    order = rng.permutation(samples)
    cut = int(samples * (1.0 - arguments.holdout))
    train_rows, valid_rows = np.sort(order[:cut]), np.sort(order[cut:])
    print(f"  train {train_rows.size}, holdout {valid_rows.size}")

    def tensors(rows):
        return (torch.from_numpy(x_transformed[rows]),
                torch.from_numpy(firing[rows]),
                torch.from_numpy(positive[rows]),
                torch.from_numpy(u[rows]))

    xv, fv, pv, uv = tensors(valid_rows)
    model = GatedSurrogate(features, targets, arguments.hidden, arguments.depth)
    optimiser = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=arguments.epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    weights = torch.from_numpy(positive_weight.astype(np.float32))
    weighted_bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=weights)

    def losses(batch_x, batch_f, batch_p, batch_u):
        logit_f, logit_s, value_u = model(batch_x)
        significance = weighted_bce(logit_f, batch_f).mean()
        # Sign and magnitude are only defined where the term fires, so both
        # are masked: a column that answered zero teaches nothing about how
        # big or which way it would have been.
        mask = batch_f
        weight = mask.sum().clamp(min=1.0)
        sign = (bce(logit_s, batch_p) * mask).sum() / weight
        magnitude = (((value_u - batch_u) ** 2) * mask).sum() / weight
        return significance, sign, magnitude

    started = time.monotonic()
    for epoch in range(arguments.epochs):
        model.train()
        shuffled = torch.from_numpy(rng.permutation(train_rows.size))
        running = np.zeros(3)
        batches = 0
        for start in range(0, train_rows.size, arguments.batch):
            rows = train_rows[shuffled[start:start + arguments.batch].numpy()]
            parts = losses(*tensors(rows))
            loss = parts[0] + parts[1] + parts[2]
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            running += [float(part.detach()) for part in parts]
            batches += 1
        schedule.step()
        model.eval()
        with torch.no_grad():
            valid = losses(xv, fv, pv, uv)
            logit_f, logit_s, _ = model(xv)
            accuracy = float((((logit_f > 0).float() == fv).float()).mean())
        running /= max(batches, 1)
        print(f"  epoch {epoch + 1:3d}/{arguments.epochs}  "
              f"train sig {running[0]:.4f} sign {running[1]:.4f} mag {running[2]:.4f}  |  "
              f"valid sig {float(valid[0]):.4f} sign {float(valid[1]):.4f} mag {float(valid[2]):.4f}  "
              f"gate acc {100 * accuracy:.2f}%  {time.monotonic() - started:6.1f}s", flush=True)

    payload = {
        "kind": "gated",
        "state_dict": model.state_dict(),
        "features": features, "targets": targets,
        "hidden": arguments.hidden, "depth": arguments.depth,
        "x_names": data["x_names"], "y_names": data["y_names"],
        "x_arguments": data["x_arguments"], "y_arguments": data["y_arguments"],
        "x_scale": x_scale,
        "thresholds": data["thresholds"], "separations": data["separations"],
        "log_centre": log_centre, "log_scale": log_scale,
        "excess_low": excess_low, "excess_high": excess_high,
        "positive_weight": positive_weight,
        "decade_clamp": DECADE_CLAMP,
        "delta_columns": data["delta_columns"], "delta_inputs": data["delta_inputs"],
        "parameter_defaults": parameter_defaults,
        "levels": data["levels"],
        "holdout_rows": valid_rows,
        "provenance": {
            "training": str(arguments.training),
            "trainer": "tools/train_pi_cam_gated_surrogate.py",
            "epochs": arguments.epochs, "hidden": arguments.hidden,
            "depth": arguments.depth, "seed": arguments.seed,
            "samples": int(samples), "gated_columns": gated,
            "separation_decades": SEPARATION_DECADES,
            "dataset": dataset_provenance,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, arguments.output)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
