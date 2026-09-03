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


# -- the real checkpoint, when this checkout has it ----------------------------


@pytest.mark.skipif(not TRANSFORMER.is_file(),
                    reason=f"{TRANSFORMER.name} is not in this checkout")
def test_the_exported_transformer_answers_every_value_the_stage_demands() -> None:
    from freecam.physics.macrophysics import RETURNED

    kernel = load_surrogate(TRANSFORMER)
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


@pytest.mark.skipif(not (TRANSFORMER.is_file() and CAPTURE.is_file()),
                    reason="the transformer or the captured columns are absent")
def test_the_transformer_runs_on_columns_the_routine_itself_answered() -> None:
    """Real captured inputs, not random ones: the shape the stage hands it."""

    netCDF4 = pytest.importorskip("netCDF4")
    from freecam.physics.macrophysics import RETURNED

    kernel = load_surrogate(TRANSFORMER)
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
