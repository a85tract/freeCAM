#!/usr/bin/env python3
"""Train a surrogate for one captured physics kernel.

The training set is what the model itself saw: one row per column the
kernel was called on, the arguments it read and the values it answered.
Three things about this target that decide the shape of the fit:

* the answers span twenty orders of magnitude, from 1e-12 tendencies to
  1e8 droplet counts, so every feature and every target is standardised
  and the loss is on the standardised scale -- otherwise the fit is
  entirely about the droplet counts;
* six of the answers are the state *after* the update, which is the state
  before plus a small tendency.  Those are learned as the difference, so
  the network is not asked to reproduce 235 K to five figures;
* neighbouring columns of one chunk are not independent draws, so the
  split is by timestep, not at random.  A random split would let the
  model see a column's neighbours during training and report a validation
  score that no held-out timestep will reproduce.

    tools/train_pi_cam_kernel_surrogate.py --training <set>.npz \\
        --output <model>.pt [--epochs 40] [--hidden 768]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

#: The answers that are the updated state, learned as a difference from the
#: input of the same name.
DELTA_OF = ("t0", "qv0", "ql0", "qi0", "nl0", "ni0")


def columns_of(names, argument: str) -> list[int]:
    return [index for index, name in enumerate(names)
            if name == argument or name.startswith(argument + "[")]


class Surrogate(nn.Module):
    """A plain fully-connected network: one column in, one column out."""

    def __init__(self, features: int, targets: int, hidden: int, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        size = features
        for _ in range(depth):
            layers += [nn.Linear(size, hidden), nn.SiLU()]
            size = hidden
        layers.append(nn.Linear(size, targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def transform(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """asinh, which is linear near zero and logarithmic far from it.

    These columns run from 1e-80 to 1e9 and most of them are zero most of
    the time.  Standardising such a column by its own spread makes a
    held-out sample explode; taking a logarithm cannot represent the zeros
    or the signs.  ``asinh(x / s)`` does both: it is ``x/s`` for small
    values and ``log(2x/s)`` for large ones, and it is odd, so the sign
    survives.
    """

    return np.arcsinh(values / scale)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True,
                        help="the directory build_pi_cam_kernel_training_set.py wrote")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-steps", type=int, default=10,
                        help="how many of the last timesteps to hold out")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    X = np.load(arguments.training / "X.npy", mmap_mode="r")
    Y = np.load(arguments.training / "Y.npy", mmap_mode="r")
    meta = np.load(arguments.training / "meta.npz", allow_pickle=True)
    x_names, y_names = list(meta["x_names"]), list(meta["y_names"])
    steps = meta["meta_nstep"]

    # a held-out timestep is a fair test; a held-out column is not, because
    # its neighbours in the same chunk were in training
    unique = np.unique(steps)
    held = unique[-arguments.validation_steps:]
    validation = np.isin(steps, held)
    train = ~validation
    print(f"{len(unique)} timesteps, holding out {held.min()}-{held.max()}")
    print(f"train {int(train.sum())} columns | validation {int(validation.sum())} columns")

    # the scale each column is measured against: a high percentile of its own
    # magnitude over the training rows, so the transform is set by the data
    # and not by an outlier
    sample = np.random.default_rng(arguments.seed).choice(
        np.flatnonzero(train), size=min(50_000, int(train.sum())), replace=False)
    sample.sort()
    x_scale = np.percentile(np.abs(np.asarray(X[sample], dtype=np.float64)), 95, axis=0)
    x_scale[x_scale <= 0] = 1.0

    delta_columns: dict[str, list[int]] = {}
    x_of: dict[str, list[int]] = {}
    for argument in DELTA_OF:
        y_columns = columns_of(y_names, argument)
        x_columns = columns_of(x_names, argument)
        if y_columns and len(y_columns) == len(x_columns):
            delta_columns[argument] = y_columns
            x_of[argument] = x_columns

    def targets_of(rows) -> np.ndarray:
        y = np.asarray(Y[rows], dtype=np.float64).copy()
        x = np.asarray(X[rows], dtype=np.float64)
        for argument, y_columns in delta_columns.items():
            y[:, y_columns] -= x[:, x_of[argument]]
        return y

    # The targets are scaled linearly, not through asinh.  Undoing an asinh
    # means a sinh, which turns a prediction error of 1 into a factor of e and
    # an error of 10 into e^10: the first fit of this surrogate had tendency
    # columns with R^2 of -1e15 for exactly that reason.  A linear scale can
    # only be off by what the network is off by.
    y_sample = targets_of(sample)
    y_scale = y_sample.std(axis=0)
    wide = np.percentile(np.abs(y_sample), 99.9, axis=0)
    y_scale = np.maximum(y_scale, 1e-3 * wide)
    y_scale[y_scale <= 0] = 1.0

    model = Surrogate(X.shape[1], Y.shape[1], arguments.hidden, arguments.depth)
    optimiser = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=arguments.epochs)
    loss_of = nn.MSELoss()

    train_rows = np.flatnonzero(train)
    validation_rows = np.flatnonzero(validation)
    hold_x = torch.from_numpy(transform(np.asarray(X[validation_rows], dtype=np.float64),
                                        x_scale).astype(np.float32))
    hold_y = torch.from_numpy((targets_of(validation_rows) / y_scale).astype(np.float32))

    rng = np.random.default_rng(arguments.seed)
    for epoch in range(arguments.epochs):
        model.train()
        order = rng.permutation(train_rows)
        total, seen = 0.0, 0
        for start in range(0, len(order), arguments.batch):
            rows = np.sort(order[start:start + arguments.batch])
            xb = torch.from_numpy(transform(np.asarray(X[rows], dtype=np.float64),
                                            x_scale).astype(np.float32))
            yb = torch.from_numpy((targets_of(rows) / y_scale).astype(np.float32))
            optimiser.zero_grad()
            loss = loss_of(model(xb), yb)
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(rows)
            seen += len(rows)
        schedule.step()
        model.eval()
        with torch.no_grad():
            held_out = float(loss_of(model(hold_x), hold_y))
        print(f"  epoch {epoch:3d}  train {total / seen:.5f}  validation {held_out:.5f}", flush=True)

    payload = {
        "state_dict": model.state_dict(),
        "features": int(X.shape[1]), "targets": int(Y.shape[1]),
        "hidden": arguments.hidden, "depth": arguments.depth,
        "x_names": x_names, "y_names": y_names,
        "x_arguments": list(meta["x_arguments"]), "y_arguments": list(meta["y_arguments"]),
        "x_scale": x_scale, "y_scale": y_scale,
        "delta_columns": delta_columns, "delta_inputs": x_of,
        "levels": int(meta["levels"]),
        "provenance": json.loads(str(meta["provenance"])),
        "training": {"epochs": arguments.epochs, "batch": arguments.batch,
                     "learning_rate": arguments.learning_rate, "seed": arguments.seed,
                     "validation_steps": held.tolist(), "validation_loss": held_out,
                     "transform": "inputs asinh(x / p95|x|); targets linear / robust std"},
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, arguments.output)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
