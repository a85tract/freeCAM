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

Three kinds of trained model load through the same class.  The first is one
regression head over linearly scaled targets.  The second is *gated*: three
heads per target -- does this term fire, which way, and how big in decades
above its firing threshold -- which is what it takes to answer "nothing
happened here" for a routine whose answers are exactly zero most of the
time.  A gated model also reads the nine tunable parameters as features, so
it is a function of the namelist rather than of one point in it; a column
arrives without a namelist, so the run's values are supplied by the caller
or taken from the case defaults recorded at training.

The third is *compiled*: the exporter serialised the trained module itself
as TorchScript instead of a state dict, so its architecture travels with
it and this class never has to know what shape it has inside.  That is how
a model whose layers this file could not rebuild -- a transformer over the
column's levels, say -- arrives without teaching this file about tokens:
it is handed the same flat feature vector as the others and slices out
whatever it wants.  Such a checkpoint carries its own normalisation
(standardised features, clipped; standardised targets) rather than the
arcsinh scaling the two rebuilt kinds use, which is why the transform
lives behind :meth:`SurrogateKernel._encode` and
:meth:`SurrogateKernel._decode`.

This module deliberately holds no science: it assembles the feature
vector in the order the training set recorded, runs the network, and
undoes the transform.  Whether the answer is any good is what the model
run measures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import PhysicsError
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

        if self.kind == "compiled":
            self._load_compiled(payload, torch)
        elif self.kind == "gated":
            self.x_scale = np.asarray(payload["x_scale"])
            self._load_gated(payload, nn)
        else:
            self.x_scale = np.asarray(payload["x_scale"])
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

    def _load_compiled(self, payload: Mapping[str, Any], torch) -> None:
        """A module the exporter compiled to TorchScript, with its own scaling.

        Nothing is rebuilt: the archive the payload carries is the trained
        module, so whatever it is inside -- a transformer over the column's
        thirty levels, in the checkpoint this was written for -- arrives
        intact and takes the same flat feature vector as the rebuilt kinds,
        slicing its own profile, scalar and parameter columns out of it.

        Its normalisation is its own: features standardised and clipped to
        keep a floored standard deviation from turning a quiet channel into
        a huge input, targets standardised.  Undoing the target transform is
        an affine move, as it is for the linear kind -- never a sinh, which
        would make the network's error an exponential one.
        """

        import io

        self.x_mean = np.asarray(payload["x_mean"], dtype=np.float64)
        self.x_std = np.asarray(payload["x_std"], dtype=np.float64)
        clip = payload.get("x_clip")
        self.x_clip = None if clip is None else float(clip)
        self.y_mean = np.asarray(payload["y_mean"], dtype=np.float64)
        self.y_std = np.asarray(payload["y_std"], dtype=np.float64)
        # A soft-gated module answers a pair -- the value and, per target, a
        # logit for whether the term fires at all.  The threshold is on the
        # probability, so the shipped 0.5 is the logit's own sign.
        self.gate_threshold = float(payload.get("gate_threshold", 0.5))
        module = torch.jit.load(io.BytesIO(payload["torchscript"]), map_location="cpu")
        module.eval()
        self.net = module

    def _net_answer(self, x) -> np.ndarray:
        """The network's answers for encoded rows, in the routine's units.

        A gated module answers a target it says does not fire as **exactly**
        zero, which is what the routine answers most of the time and what a
        regression head alone cannot say.  Substituting the module's
        ``zero_norm`` before undoing the target scaling would be the other
        way to write this, and it is worse: it is zero only to float32
        rounding, and for condensate the residue lands negative -- the
        quantity CAM's own bounds check stops a run over.
        """

        answer = self.net(x)
        if not isinstance(answer, tuple):
            return self._decode(answer.numpy())
        value, gate = answer
        fires = self.torch.sigmoid(gate).numpy() >= self.gate_threshold
        return np.where(fires, self._decode(value.numpy()), 0.0)

    def _encode(self, rows: np.ndarray):
        """Feature rows in the units the network was trained on."""

        if self.kind == "compiled":
            scaled = (rows - self.x_mean) / self.x_std
            if self.x_clip is not None:
                scaled = np.clip(scaled, -self.x_clip, self.x_clip)
        else:
            scaled = np.arcsinh(rows / self.x_scale)
        return self.torch.from_numpy(scaled.astype(np.float32))

    def _decode(self, answer: Any) -> np.ndarray:
        """The network's answer in the routine's own units.

        Both scalings are affine, so undoing one is a multiply and an add
        over the last axis -- one column or a batch of them alike.
        """

        values = np.asarray(answer, dtype=np.float64)
        if self.kind == "compiled":
            return values * self.y_std + self.y_mean
        return values * self.y_scale

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
        self._parameter_columns(row, parameters)
        return row

    def _parameter_columns(self, rows: np.ndarray,
                           parameters: Mapping[str, Any] | None) -> None:
        """Write the namelist's values into every row's parameter columns."""

        given = dict(parameters or {})
        for name, where in self._parameters.items():
            value = float(given.get(name, self.parameter_defaults.get(name, 0.0)))
            if "column" in where:
                rows[..., where["column"]] = value
            else:
                indicators = where["indicators"]
                nearest = min(indicators, key=lambda admitted: abs(admitted - value))
                rows[..., indicators[nearest]] = 1.0

    def features_batch(self, columns: Mapping[str, Any],
                       parameters: Mapping[str, Any] | None = None) -> np.ndarray:
        """One feature row per column of a chunk, in the training set's order.

        The layout is the one :meth:`features` builds, assembled for many
        columns at once.  The model's unit is still the column -- nothing in
        the feature vector comes from a neighbour -- so the rows are
        independent and the batch is only about how many go through the
        network in one call.
        """

        count = next((np.asarray(columns[argument]).shape[0]
                      for argument in self._x_slices
                      if np.asarray(columns[argument]).ndim == 2), None)
        if count is None:
            raise PhysicsError(
                "a batch of columns must carry at least one profile argument "
                f"shaped (columns, levels); got {sorted(self._x_slices)}")
        rows = np.zeros((count, len(self.x_names)), dtype=np.float64)
        for argument, where in self._x_slices.items():
            value = np.asarray(columns[argument], dtype=np.float64)
            width = where.stop - where.start
            if value.ndim == 0:
                rows[:, where] = value                      # one value for the chunk
            elif value.ndim == 1:
                # one number per column, or one profile every column shares
                rows[:, where] = value[:count, None] if width == 1 else value[None, :]
            else:
                rows[:, where] = value[:count, :width]      # a profile per column
        self._parameter_columns(rows, parameters)
        return rows

    def batched(self, columns: Mapping[str, Any],
                parameters: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
        """Every answer for a chunk's live columns, in one pass of the network.

        The same arithmetic as calling this kernel once per column, with the
        columns stacked: one matrix multiply of ``ncol`` rows rather than
        ``ncol`` of one row, which is the difference between a network used
        well and used badly.  Floating point makes the two agree closely
        rather than exactly -- a batched matrix multiply blocks differently
        -- so a run that swaps one form for the other is a different run,
        which is true of any model in this slot anyway.
        """

        return self.batched_answer(self.features_batch(columns, parameters))

    def __call__(self, column: Mapping[str, Any],
                 parameters: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
        row = self.features(column, parameters)
        x = self._encode(row[None, :])
        with self.torch.no_grad():
            if self.kind == "gated":
                values = self._gated_answer(x)
            else:
                values = self._net_answer(x)[0]

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
        x = self._encode(rows)
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
                values = self._net_answer(x)
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
