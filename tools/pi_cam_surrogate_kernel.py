#!/usr/bin/env python3
"""Put a trained surrogate in a stage's kernel slot.

The stage class has one definition of what computes its core -- the method
named after the routine -- so a surrogate replaces the kernel by replacing
that one thing, and the walk and any single-column caller both follow:

    from tools.pi_cam_surrogate_kernel import load_surrogate
    macro.kernels["mmacro_pcond"] = load_surrogate("mmacro_surrogate.pt")

or, if the caller would rather bind the method itself:

    macro.mmacro_pcond = MethodType(load_surrogate(path).as_method(), macro)

Both reach the same place.  What the surrogate is handed is one column
under the routine's own argument names, which is what the reviewed
standalone boundary is handed; what it must answer is every value the
routine answers, under the same names.

This module deliberately holds no science: it assembles the feature
vector in the order the training set recorded, runs the network, and
undoes the transform.  Whether the answer is any good is what the model
run measures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


class SurrogateKernel:
    """A trained network, callable the way the stage calls its kernel."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        import torch
        from torch import nn

        self.torch = torch
        self.x_names = list(payload["x_names"])
        self.y_names = list(payload["y_names"])
        self.x_arguments = list(payload["x_arguments"])
        self.y_arguments = list(payload["y_arguments"])
        self.x_scale = np.asarray(payload["x_scale"])
        self.y_scale = np.asarray(payload["y_scale"])
        self.delta_columns = dict(payload["delta_columns"])
        self.delta_inputs = dict(payload["delta_inputs"])
        self.levels = int(payload["levels"])
        self.provenance = payload.get("provenance", {})

        layers: list[Any] = []
        size = int(payload["features"])
        for _ in range(int(payload["depth"])):
            layers += [nn.Linear(size, int(payload["hidden"])), nn.SiLU()]
            size = int(payload["hidden"])
        layers.append(nn.Linear(size, int(payload["targets"])))
        self.net = nn.Sequential(*layers)
        self.net.load_state_dict(
            {key.removeprefix("net."): value for key, value in payload["state_dict"].items()})
        self.net.eval()

        # where each argument's numbers sit in the feature vector
        self._x_slices = self._slices(self.x_names, self.x_arguments)
        self._y_slices = self._slices(self.y_names, self.y_arguments)

    @staticmethod
    def _slices(names: list[str], arguments: list[str]) -> dict[str, slice]:
        out: dict[str, slice] = {}
        for argument in arguments:
            columns = [index for index, name in enumerate(names)
                       if name == argument or name.startswith(argument + "[")]
            out[argument] = slice(columns[0], columns[-1] + 1)
        return out

    def features(self, column: Mapping[str, Any]) -> np.ndarray:
        row = np.zeros(len(self.x_names), dtype=np.float64)
        for argument, where in self._x_slices.items():
            value = np.asarray(column[argument], dtype=np.float64).reshape(-1)
            row[where] = value if value.size == (where.stop - where.start) else value.item()
        return row

    def __call__(self, column: Mapping[str, Any]) -> dict[str, np.ndarray]:
        row = self.features(column)
        with self.torch.no_grad():
            answer = self.net(self.torch.from_numpy(
                np.arcsinh(row / self.x_scale).astype(np.float32)[None, :])).numpy()[0]
        # the targets were scaled linearly, so undoing it is a multiply: no
        # sinh, which would turn the network's error into an exponential one
        values = answer.astype(np.float64) * self.y_scale

        out: dict[str, np.ndarray] = {}
        for argument, where in self._y_slices.items():
            piece = values[where]
            if argument in self.delta_columns:
                # the network learned the change, so the answer is the state
                # it was given plus what the network says it becomes
                piece = piece + np.asarray(column[argument], dtype=np.float64).reshape(-1)
            out[argument] = piece if where.stop - where.start > 1 else piece.reshape(())
        return out

    def as_method(self):
        """The same thing shaped as a method, for ``MethodType`` binding."""

        kernel = self

        def mmacro_pcond(self, inputs, parameters=None):
            from freecam.physics.result import FunctionResult

            answer = kernel(inputs)
            updated = set(kernel.delta_columns)
            return FunctionResult(
                outputs={k: v for k, v in answer.items() if k not in updated},
                updated_inputs={k: v for k, v in answer.items() if k in updated})

        return mmacro_pcond


def load_surrogate(path: str | Path) -> SurrogateKernel:
    import torch

    return SurrogateKernel(torch.load(Path(path), map_location="cpu", weights_only=False))


__all__ = ["SurrogateKernel", "load_surrogate"]
