#!/usr/bin/env python3
"""Put a trained surrogate in a stage's kernel slot.

This lives in the package rather than in ``tools`` because it is runtime
code: every rank of a model run loads its own copy, and a notebook loads
one the same way.

The stage class has one definition of what computes its core -- the method
named after the routine -- so a surrogate replaces the kernel by replacing
that one thing, and the walk and any single-column caller both follow:

    from freecam.physics.surrogate import load_surrogate
    macro.kernels["mmacro_pcond"] = load_surrogate("mmacro_surrogate.pt")

or, if the caller would rather bind the method itself:

    macro.mmacro_pcond = MethodType(load_surrogate(path).as_method(), macro)

Both reach the same place.  What the surrogate is handed is one column
under the routine's own argument names, which is what the reviewed
standalone boundary is handed; what it must answer is every value the
routine answers, under the same names.

Two kinds of trained model load through the same class.  The first is one
regression head over linearly scaled targets.  The second is *gated*: three
heads per target -- does this term fire, which way, and how big in decades
above its firing threshold -- which is what it takes to answer "nothing
happened here" for a routine whose answers are exactly zero most of the
time.  A gated model also reads the nine tunable parameters as features, so
it is a function of the namelist rather than of one point in it; a column
arrives without a namelist, so the run's values are supplied by the caller
or taken from the case defaults recorded at training.

This module deliberately holds no science: it assembles the feature
vector in the order the training set recorded, runs the network, and
undoes the transform.  Whether the answer is any good is what the model
run measures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .identities import NON_NEGATIVE, identities_for


class SurrogateKernel:
    """A trained network, callable the way the stage calls its kernel."""

    #: A kernel in a stage's slot is called with the column alone -- that is
    #: the contract, and a plain ``lambda column: {...}`` must keep working.
    #: A gated surrogate reads the nine tunable parameters as features, so it
    #: wants the namelist too; this says so, and the stage passes it.  Without
    #: it the model would silently answer for the case defaults whatever the
    #: caller asked for, which is the quietest way to lose a parameter study.
    takes_parameters = True

    def __init__(self, payload: Mapping[str, Any]) -> None:
        import torch
        from torch import nn

        # One rank, one thread.  Five hundred ranks each spawning a thread per
        # core would thrash the node and buy nothing: a column is one small
        # matrix multiply.
        torch.set_num_threads(1)
        self.torch = torch
        self.x_names = list(payload["x_names"])
        self.y_names = list(payload["y_names"])
        self.x_arguments = list(payload["x_arguments"])
        self.y_arguments = list(payload["y_arguments"])
        self.x_scale = np.asarray(payload["x_scale"])
        self.delta_columns = dict(payload["delta_columns"])
        self.delta_inputs = dict(payload["delta_inputs"])
        self.levels = int(payload["levels"])
        self.provenance = payload.get("provenance", {})
        self.kind = str(payload.get("kind", "linear"))
        self.parameter_defaults = dict(payload.get("parameter_defaults", {}))
        # Answers the routine forms from its other answers.  The network never
        # saw them; they are computed from the identity they came from, so
        # state and tendency cannot drift apart the way a model run notices.
        self.function = payload.get("function")
        self.identities = identities_for(str(self.function or ""))
        self.non_negative = NON_NEGATIVE.get(str(self.function or ""), ())

        if self.kind == "gated":
            self._load_gated(payload, nn)
        else:
            self.y_scale = np.asarray(payload["y_scale"])
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
        self._x_slices = self._slices(self.x_names, self._plain_arguments())
        self._y_slices = self._slices(self.y_names, self.y_arguments)
        self._parameters = self._parameter_layout()

    def _plain_arguments(self) -> list[str]:
        """The arguments that are columns of X in their own right.

        A parameter is a feature too, but under a ``parameter:`` name and
        sometimes as one indicator per admitted value, so it is laid out
        separately rather than sliced like a profile.
        """

        return [name for name in self.x_arguments if name not in self.parameter_defaults]

    def _parameter_layout(self) -> dict[str, Any]:
        """Where each parameter sits: one column, or one per admitted value."""

        layout: dict[str, Any] = {}
        for index, name in enumerate(self.x_names):
            if not name.startswith("parameter:"):
                continue
            body = name[len("parameter:"):]
            if "==" in body:
                parameter, value = body.split("==", 1)
                layout.setdefault(parameter, {"indicators": {}})
                layout[parameter]["indicators"][float(value)] = index
            else:
                layout[body] = {"column": index}
        return layout

    def _load_gated(self, payload: Mapping[str, Any], nn) -> None:
        import torch

        self.thresholds = np.asarray(payload["thresholds"], dtype=np.float64)
        self.log_centre = np.asarray(payload["log_centre"], dtype=np.float64)
        self.log_scale = np.asarray(payload["log_scale"], dtype=np.float64)
        clamp = float(payload.get("decade_clamp", 2.0))
        self.excess_low = np.asarray(payload["excess_low"], dtype=np.float64) - clamp
        self.excess_high = np.asarray(payload["excess_high"], dtype=np.float64) + clamp
        # A column with no firing threshold answers from zero decades, so its
        # magnitude is read straight off the log-magnitude head.
        self.floor = np.where(np.isfinite(self.thresholds), self.thresholds, 0.0)

        size = int(payload["features"])
        layers: list[Any] = []
        for _ in range(int(payload["depth"])):
            layers += [nn.Linear(size, int(payload["hidden"])), nn.SiLU()]
            size = int(payload["hidden"])
        trunk = nn.Sequential(*layers)
        targets = int(payload["targets"])
        heads = {name: nn.Linear(size, targets) for name in ("significance", "sign", "magnitude")}
        module = nn.Module()
        module.trunk = trunk
        for name, head in heads.items():
            setattr(module, name, head)
        module.load_state_dict(payload["state_dict"])
        module.eval()
        self.net = module
        self._torch = torch

    @staticmethod
    def _slices(names: list[str], arguments: list[str]) -> dict[str, slice]:
        out: dict[str, slice] = {}
        for argument in arguments:
            columns = [index for index, name in enumerate(names)
                       if name == argument or name.startswith(argument + "[")]
            out[argument] = slice(columns[0], columns[-1] + 1)
        return out

    def features(self, column: Mapping[str, Any],
                 parameters: Mapping[str, Any] | None = None) -> np.ndarray:
        row = np.zeros(len(self.x_names), dtype=np.float64)
        for argument, where in self._x_slices.items():
            value = np.asarray(column[argument], dtype=np.float64).reshape(-1)
            row[where] = value if value.size == (where.stop - where.start) else value.item()
        given = dict(parameters or {})
        for name, where in self._parameters.items():
            value = float(given.get(name, self.parameter_defaults.get(name, 0.0)))
            if "column" in where:
                row[where["column"]] = value
            else:
                indicators = where["indicators"]
                nearest = min(indicators, key=lambda admitted: abs(admitted - value))
                row[indicators[nearest]] = 1.0
        return row

    def __call__(self, column: Mapping[str, Any],
                 parameters: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
        row = self.features(column, parameters)
        x = self.torch.from_numpy(np.arcsinh(row / self.x_scale).astype(np.float32)[None, :])
        with self.torch.no_grad():
            if self.kind == "gated":
                values = self._gated_answer(x)
            else:
                answer = self.net(x).numpy()[0]
                # the targets were scaled linearly, so undoing it is a
                # multiply: no sinh, which would turn the network's error into
                # an exponential one
                values = answer.astype(np.float64) * self.y_scale

        out: dict[str, np.ndarray] = {}
        for argument, where in self._y_slices.items():
            piece = values[where]
            if argument in self.delta_columns:
                # the network learned the change, so the answer is the state
                # it was given plus what the network says it becomes
                piece = piece + np.asarray(column[argument], dtype=np.float64).reshape(-1)
            out[argument] = piece if where.stop - where.start > 1 else piece.reshape(())
        return self._close(out, column)

    def _close(self, answer: dict[str, np.ndarray],
               column: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Fill in the answers the routine derives, then keep them physical.

        Order matters: the identities are what the routine did, so they come
        first and are exact.  The floor is a repair on top -- a tendency the
        network overshoots would take condensate below zero, which the routine
        never does and which stops a model run.
        """

        if not self.identities and not self.non_negative:
            return answer                    # a function with no table claims none
        dt = float(np.asarray(column["dt"], dtype=np.float64).reshape(-1)[0])
        for identity in self.identities:
            answer[identity.target] = identity(column, answer, dt)
        for name in self.non_negative:
            if name in answer:
                answer[name] = np.maximum(answer[name], 0.0)
        return answer

    def _gated_answer(self, x) -> np.ndarray:
        """Fire or not, which way, how big -- in that order."""

        logit_fire, logit_sign, magnitude = self.net.significance, self.net.sign, self.net.magnitude
        hidden = self.net.trunk(x)
        fires = logit_fire(hidden).numpy()[0] > 0.0
        positive = logit_sign(hidden).numpy()[0] > 0.0
        excess = magnitude(hidden).numpy()[0].astype(np.float64) * self.log_scale + self.log_centre
        excess = np.clip(excess, self.excess_low, self.excess_high)
        size = np.power(10.0, excess + self.floor)
        return np.where(fires, np.where(positive, size, -size), 0.0)

    def predict_rows(self, rows: np.ndarray) -> np.ndarray:
        """Predictions for a matrix of assembled feature rows.

        The same arithmetic as a single call, batched.  An evaluator that
        re-implemented it would be checking its own copy rather than what a
        model run executes, so there is one implementation and both use it.
        """

        rows = np.asarray(rows, dtype=np.float64)
        x = self.torch.from_numpy(np.arcsinh(rows / self.x_scale).astype(np.float32))
        with self.torch.no_grad():
            if self.kind == "gated":
                hidden = self.net.trunk(x)
                fires = self.net.significance(hidden).numpy() > 0.0
                positive = self.net.sign(hidden).numpy() > 0.0
                excess = (self.net.magnitude(hidden).numpy().astype(np.float64)
                          * self.log_scale[None, :] + self.log_centre[None, :])
                excess = np.clip(excess, self.excess_low[None, :], self.excess_high[None, :])
                size = np.power(10.0, excess + self.floor[None, :])
                values = np.where(fires, np.where(positive, size, -size), 0.0)
            else:
                values = self.net(x).numpy().astype(np.float64) * self.y_scale[None, :]
        for argument, where in self._y_slices.items():
            if argument in self.delta_columns:
                values[:, where] += rows[:, self._x_slices[argument]]
        return values

    def batched_answer(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Every answer for a matrix of rows, derived ones included."""

        values = self.predict_rows(rows)
        column = {name: rows[:, where] if where.stop - where.start > 1
                  else rows[:, where.start]
                  for name, where in self._x_slices.items()}
        answer = {name: values[:, where] if where.stop - where.start > 1
                  else values[:, where.start]
                  for name, where in self._y_slices.items()}
        if not self.identities and not self.non_negative:
            return answer
        dt = column["dt"].reshape(-1, 1)
        for identity in self.identities:
            answer[identity.target] = identity(column, answer, dt)
        for name in self.non_negative:
            if name in answer:
                answer[name] = np.maximum(answer[name], 0.0)
        return answer

    def as_method(self):
        """The same thing shaped as a method, for ``MethodType`` binding."""

        kernel = self

        def mmacro_pcond(self, inputs, parameters=None):
            from freecam.physics.result import FunctionResult

            answer = kernel(inputs, parameters)
            updated = set(kernel.delta_columns)
            return FunctionResult(
                outputs={k: v for k, v in answer.items() if k not in updated},
                updated_inputs={k: v for k, v in answer.items() if k in updated})

        return mmacro_pcond


def load_surrogate(path: str | Path) -> SurrogateKernel:
    import torch

    return SurrogateKernel(torch.load(Path(path), map_location="cpu", weights_only=False))


__all__ = ["SurrogateKernel", "load_surrogate"]
