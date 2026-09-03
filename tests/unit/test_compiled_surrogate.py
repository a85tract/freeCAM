"""A surrogate whose module travels compiled, not rebuilt.

The two rebuilt kinds are described by their layer sizes and a state dict,
so this file can only load an architecture it already knows how to
construct.  A *compiled* checkpoint carries the trained module itself as
TorchScript instead, which is how a model this file could not rebuild --
the transformer emulator over a column's thirty levels -- reaches a
stage's kernel slot.  What is checked here: that such a payload loads,
that its own normalisation is the one applied (standardised and clipped,
not the arcsinh scaling the rebuilt kinds use), that the feature vector it
is handed is still assembled in the training set's order, and that the
answers come back under the routine's names.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from freecam.physics.surrogate import SurrogateKernel, load_surrogate  # noqa: E402

#: The supervisor's exported transformer, if this checkout has it.  It is a
#: trained model rather than source, so it is not committed; the tests that
#: need it skip without it.
TRANSFORMER = REPO / "mmacro_pcond_transformer.pt"
SOFT_GATED = REPO / "mmacro_pcond_soft_gated.pt"
CAPTURE = REPO / "examples/mmacro_pcond_training.nc"


class _Doubler(torch.nn.Module):
    """Answers each target as the sum of the features it was given.

    Deliberately trivial: what is on trial is the transform around the
    module, so the module itself has to be something whose answer can be
    predicted by hand.
    """

    def __init__(self, targets: int) -> None:
        super().__init__()
        self.targets = targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = x.sum(dim=1, keepdim=True)
        return total.expand(x.shape[0], self.targets)


class _GatedDoubler(torch.nn.Module):
    """Answers the same value everywhere, with a per-target gate logit.

    ``gates`` are the logits to answer, so a test can say which targets
    fire and which do not.
    """

    def __init__(self, gates: list[float]) -> None:
        super().__init__()
        self.register_buffer("gates", torch.tensor(gates))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total = x.sum(dim=1, keepdim=True)
        value = total.expand(x.shape[0], self.gates.shape[0])
        return value, self.gates.expand(x.shape[0], self.gates.shape[0])


def _compiled_payload(*, x_mean, x_std, y_mean, y_std, clip=None) -> dict:
    """A compiled payload over one input profile and one output profile."""

    targets = len(y_mean)
    module = torch.jit.script(_Doubler(targets))
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    return {
        "kind": "compiled",
        "x_names": [f"a[{k}]" for k in range(len(x_mean))],
        "y_names": [f"b[{k}]" for k in range(targets)],
        "x_arguments": ["a"], "y_arguments": ["b"],
        "x_mean": np.asarray(x_mean, dtype=np.float64),
        "x_std": np.asarray(x_std, dtype=np.float64),
        "x_clip": clip,
        "y_mean": np.asarray(y_mean, dtype=np.float64),
        "y_std": np.asarray(y_std, dtype=np.float64),
        "torchscript": buffer.getvalue(),
        "delta_columns": {}, "delta_inputs": {}, "parameter_defaults": {},
        "levels": targets,
    }


def test_a_compiled_module_loads_without_being_rebuilt() -> None:
    kernel = SurrogateKernel(_compiled_payload(
        x_mean=[0.0, 0.0], x_std=[1.0, 1.0], y_mean=[0.0], y_std=[1.0]))

    assert kernel.kind == "compiled"
    # the module is the archive's, not one this file constructed
    assert isinstance(kernel.net, torch.jit.ScriptModule)
    # and nothing read the rebuilt kinds' scaling, which such a payload lacks
    assert not hasattr(kernel, "x_scale")
    assert not hasattr(kernel, "y_scale")


def test_the_features_are_standardised_and_the_answer_is_not() -> None:
    """(x - mean)/std in, y*std + mean out -- affine both ways."""

    kernel = SurrogateKernel(_compiled_payload(
        x_mean=[10.0, 100.0], x_std=[2.0, 4.0], y_mean=[7.0], y_std=[3.0]))
    # standardised features are (14-10)/2 = 2 and (120-100)/4 = 5, summing
    # to 7; the answer is then 7*3 + 7
    answer = float(np.asarray(kernel({"a": np.array([14.0, 120.0])})["b"]))

    assert answer == pytest.approx(7.0 * 3.0 + 7.0, rel=1e-6)


def test_the_clip_keeps_a_floored_channel_from_dominating() -> None:
    """A tiny std would turn a quiet channel into a huge input.

    The exporter floors the standard deviation per family and clips what
    comes out, so a feature far outside training cannot swamp the network.
    Without the clip this column would arrive as 1000 standard deviations.
    """

    payload = _compiled_payload(
        x_mean=[0.0], x_std=[1e-3], y_mean=[0.0], y_std=[1.0], clip=20.0)
    kernel = SurrogateKernel(payload)
    answer = float(np.asarray(kernel({"a": np.array([1.0])})["b"]))
    assert answer == pytest.approx(20.0, rel=1e-6)          # not 1000

    unclipped = SurrogateKernel(_compiled_payload(
        x_mean=[0.0], x_std=[1e-3], y_mean=[0.0], y_std=[1.0]))
    assert float(np.asarray(unclipped({"a": np.array([1.0])})["b"])) \
        == pytest.approx(1000.0, rel=1e-6)


def test_a_batch_answers_as_the_single_column_does() -> None:
    """``predict_rows`` and ``__call__`` are one transform, not two."""

    kernel = SurrogateKernel(_compiled_payload(
        x_mean=[1.0, 2.0], x_std=[0.5, 4.0], y_mean=[5.0, -1.0], y_std=[2.0, 0.25]))
    rows = np.array([[3.0, 10.0], [1.5, 2.5]])
    batched = kernel.predict_rows(rows)
    one = np.concatenate([np.asarray(kernel({"a": row})["b"]).reshape(-1)
                          for row in rows])

    assert batched.shape == (2, 2)
    assert np.allclose(batched.reshape(-1), one, rtol=1e-12, atol=0.0)


def test_the_rebuilt_kinds_still_use_their_own_scaling() -> None:
    """The arcsinh path is untouched: same numbers, one implementation."""

    weights = {"0.weight": torch.ones(1, 1), "0.bias": torch.zeros(1)}
    payload = {
        "kind": "linear", "features": 1, "targets": 1, "hidden": 1, "depth": 0,
        "x_names": ["a"], "y_names": ["b"], "x_arguments": ["a"], "y_arguments": ["b"],
        "x_scale": np.array([2.0]), "y_scale": np.array([10.0]),
        "delta_columns": {}, "delta_inputs": {}, "parameter_defaults": {},
        "levels": 1, "state_dict": weights,
    }
    kernel = SurrogateKernel(payload)
    answer = float(np.asarray(kernel({"a": np.array([3.0])})["b"]))

    assert answer == pytest.approx(float(np.arcsinh(1.5)) * 10.0, rel=1e-6)


def _gated_payload(*, gates, y_mean, y_std, threshold=None) -> dict:
    """A compiled payload whose module answers (value, gate logit)."""

    payload = _compiled_payload(
        x_mean=[0.0], x_std=[1.0], y_mean=y_mean, y_std=y_std)
    module = torch.jit.script(_GatedDoubler(list(gates)))
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    payload["torchscript"] = buffer.getvalue()
    payload["y_names"] = [f"b[{k}]" for k in range(len(gates))]
    payload["levels"] = len(gates)
    if threshold is not None:
        payload["gate_threshold"] = threshold
    return payload


def test_a_target_whose_gate_does_not_fire_is_answered_as_exactly_zero() -> None:
    """Not a small number -- zero.

    The routine's answers are exactly zero most of the time, and CAM's own
    bounds check stops a run over negative condensate, so a residue of the
    wrong sign is not a rounding detail.  Substituting the module's
    zero_norm before undoing the target scaling would leave one; this does
    not.
    """

    kernel = SurrogateKernel(_gated_payload(
        gates=[5.0, -5.0], y_mean=[100.0, 100.0], y_std=[3.0, 3.0]))
    answer = np.asarray(kernel({"a": np.array([2.0])})["b"], dtype=np.float64)

    assert answer[0] == pytest.approx(2.0 * 3.0 + 100.0, rel=1e-6)   # fires
    assert answer[1] == 0.0                                          # exactly
    assert not np.signbit(answer[1])                                 # not -0.0


def test_the_gate_threshold_is_read_from_the_checkpoint() -> None:
    """A logit of 1.0 is a probability of 0.73: it fires at 0.5, not at 0.9."""

    fires = SurrogateKernel(_gated_payload(
        gates=[1.0], y_mean=[0.0], y_std=[1.0], threshold=0.5))
    holds = SurrogateKernel(_gated_payload(
        gates=[1.0], y_mean=[0.0], y_std=[1.0], threshold=0.9))

    assert float(np.asarray(fires({"a": np.array([7.0])})["b"])) == pytest.approx(7.0)
    assert float(np.asarray(holds({"a": np.array([7.0])})["b"])) == 0.0
    assert fires.gate_threshold == 0.5 and holds.gate_threshold == 0.9


def test_a_gated_batch_answers_as_the_single_column_does() -> None:
    kernel = SurrogateKernel(_gated_payload(
        gates=[3.0, -3.0], y_mean=[1.0, 1.0], y_std=[2.0, 2.0]))
    rows = np.array([[4.0], [-1.0]])
    batched = kernel.predict_rows(rows)
    one = np.stack([np.asarray(kernel({"a": row})["b"]).reshape(-1) for row in rows])

    assert np.array_equal(batched, one)
    assert np.all(batched[:, 1] == 0.0)          # the closed gate, every row


# -- the real checkpoint, when this checkout has it ----------------------------


#: Every exported checkpoint this checkout has, transformer and soft-gated
#: alike: both are kind="compiled" and both must answer the same contract.
EXPORTED = tuple(path for path in (TRANSFORMER, SOFT_GATED) if path.is_file())


@pytest.mark.skipif(not EXPORTED, reason="no exported checkpoint in this checkout")
@pytest.mark.parametrize("checkpoint", EXPORTED, ids=lambda p: p.stem)
def test_an_exported_checkpoint_answers_every_value_the_stage_demands(
    checkpoint: Path,
) -> None:
    from freecam.physics.macrophysics import RETURNED

    kernel = load_surrogate(checkpoint)
    assert kernel.kind == "compiled"
    assert kernel.function == "mmacro_pcond"
    # the flat feature vector is the contract; the transformer slices its own
    # profile, scalar and parameter columns out of it
    assert len(kernel.y_names) == len(kernel.y_arguments) * kernel.levels
    assert set(RETURNED) == set(kernel.y_arguments)

    rng = np.random.default_rng(0)
    column = {name: rng.normal(size=kernel.levels)
              if slice_.stop - slice_.start > 1 else np.array(1800.0)
              for name, slice_ in kernel._x_slices.items()}
    answer = kernel(column)

    missing = [name for name in RETURNED if name not in answer]
    assert not missing, f"the transformer answered none of {missing}"
    for name in RETURNED:
        value = np.asarray(answer[name], dtype=np.float64)
        assert np.all(np.isfinite(value)), f"{name} is not finite"


@pytest.mark.skipif(not (EXPORTED and CAPTURE.is_file()),
                    reason="a checkpoint or the captured columns are absent")
@pytest.mark.parametrize("checkpoint", EXPORTED, ids=lambda p: p.stem)
def test_an_exported_checkpoint_runs_on_columns_the_routine_answered(
    checkpoint: Path,
) -> None:
    """Real captured inputs, not random ones: the shape the stage hands it."""

    netCDF4 = pytest.importorskip("netCDF4")
    from freecam.physics.macrophysics import RETURNED

    kernel = load_surrogate(checkpoint)
    dataset = netCDF4.Dataset(CAPTURE)
    inputs = {name[len("input__"):]: variable
              for name, variable in dataset.variables.items()
              if name.startswith("input__")}
    absent = [name for name in kernel.x_arguments
              if name not in inputs and name not in kernel.parameter_defaults]
    assert not absent, f"the capture does not carry {absent}"

    for sample in range(3):
        column = {name: np.asarray(variable[sample]).astype(np.float64)
                  for name, variable in inputs.items()}
        answer = kernel(column)
        for name in RETURNED:
            value = np.asarray(answer[name], dtype=np.float64)
            assert np.all(np.isfinite(value)), f"sample {sample}: {name}"
        # the routine's own identities close the state, so it is the input
        # state plus the answered tendency -- never the network's guess at it
        assert np.all(np.asarray(answer["ql0"]) >= 0.0)
