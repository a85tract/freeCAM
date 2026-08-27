"""The gated surrogate: what fires, which way, how big -- and the inverse."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

torch = pytest.importorskip("torch")

from freecam.physics.surrogate import SurrogateKernel  # noqa: E402
from train_pi_cam_gated_surrogate import otsu_threshold, prepare  # noqa: E402


def test_otsu_finds_the_gap_between_physics_and_round_off() -> None:
    """The routine's answers are bimodal in log magnitude; the split is the gap."""

    rng = np.random.default_rng(0)
    residue = rng.normal(-27.0, 1.5, 4000)      # round-off left by the arithmetic
    physics = rng.normal(-8.0, 1.0, 2000)       # the term actually firing
    threshold, separation = otsu_threshold(np.concatenate([residue, physics]))
    assert -25.0 < threshold < -11.0            # inside the gap, not in either lobe
    assert separation > 15.0
    # One lobe alone has no gap to find, and must not be split.
    _, alone = otsu_threshold(rng.normal(-8.0, 1.0, 4000))
    assert alone < 5.0


def _training_set(path: Path, y: np.ndarray, x: np.ndarray) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "X.npy", x.astype(np.float32))
    np.save(path / "Y.npy", y.astype(np.float32))
    np.savez(path / "meta.npz",
             x_names=np.array([f"a[{k}]" for k in range(x.shape[1])]),
             y_names=np.array([f"b[{k}]" for k in range(y.shape[1])]),
             x_arguments=np.array(["a"]), y_arguments=np.array(["b"]),
             levels=np.int64(y.shape[1]))
    return path


def test_an_exact_zero_never_counts_as_firing(tmp_path: Path) -> None:
    """log10(0) is -inf and an ungated column's threshold is -inf.

    Comparing the two would call every zero a firing sample and hand the
    magnitude head a value of 10**0 -- one kilogram of condensate per
    kilogram of air, which is how this went wrong the first time.
    """

    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 1e-8, (500, 3))         # continuous: no gap, stays ungated
    y[::2, 0] = 0.0                             # ...but half of one column is zero
    x = rng.normal(0.0, 1.0, (500, 2))
    data = prepare(_training_set(tmp_path / "set", y, x))

    assert not np.isfinite(data["thresholds"]).any()          # nothing gated
    assert not data["fires"][data["target"] == 0.0].any()     # and no zero fires
    assert data["fires"][data["target"] != 0.0].all()


def _payload(thresholds, log_centre, log_scale, weights) -> dict:
    """A one-layer trunk with fixed heads, so the arithmetic is checkable."""

    targets = len(thresholds)
    payload = {
        "kind": "gated", "features": 1, "targets": targets, "hidden": 1, "depth": 1,
        "x_names": ["a"], "y_names": [f"b[{k}]" for k in range(targets)],
        "x_arguments": ["a"], "y_arguments": ["b"],
        "x_scale": np.ones(1),
        "thresholds": np.asarray(thresholds, dtype=np.float64),
        "log_centre": np.asarray(log_centre, dtype=np.float64),
        "log_scale": np.asarray(log_scale, dtype=np.float64),
        "excess_low": np.full(targets, -30.0), "excess_high": np.full(targets, 30.0),
        "decade_clamp": 0.0,
        "delta_columns": {}, "delta_inputs": {}, "parameter_defaults": {},
        "levels": targets,
        "state_dict": weights,
    }
    return payload


def test_the_gate_answers_exactly_zero_and_the_magnitude_inverts() -> None:
    thresholds = [-9.0, -9.0, -np.inf]
    weights = {
        "trunk.0.weight": torch.ones(1, 1), "trunk.0.bias": torch.zeros(1),
        # SiLU(1) = 0.731; the heads are read off that one hidden value.
        "significance.weight": torch.tensor([[1.0], [-1.0], [1.0]]),
        "significance.bias": torch.zeros(3),
        "sign.weight": torch.tensor([[1.0], [1.0], [-1.0]]),
        "sign.bias": torch.zeros(3),
        "magnitude.weight": torch.zeros(3, 1),
        "magnitude.bias": torch.tensor([2.0, 2.0, -4.0]),
    }
    kernel = SurrogateKernel(_payload(thresholds, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], weights))
    answer = np.asarray(kernel({"a": np.array([1.0])})["b"])

    # column 0 fires, positive: 10**(2 + (-9))
    assert answer[0] == pytest.approx(1e-7, rel=1e-6)
    # column 1's significance logit is negative, so it is exactly zero -- not
    # a small number, which is the whole point of the gate
    assert answer[1] == 0.0
    # column 2 is ungated (threshold -inf, floor 0) and negative: -10**-4
    assert answer[2] == pytest.approx(-1e-4, rel=1e-6)


def test_the_magnitude_is_clamped_to_the_band_training_saw() -> None:
    weights = {
        "trunk.0.weight": torch.ones(1, 1), "trunk.0.bias": torch.zeros(1),
        "significance.weight": torch.ones(1, 1), "significance.bias": torch.zeros(1),
        "sign.weight": torch.ones(1, 1), "sign.bias": torch.zeros(1),
        "magnitude.weight": torch.zeros(1, 1), "magnitude.bias": torch.tensor([50.0]),
    }
    payload = _payload([-9.0], [0.0], [1.0], weights)
    payload["excess_low"], payload["excess_high"] = np.array([-1.0]), np.array([3.0])
    # one column, so the kernel answers with a scalar, as the argument is
    answer = float(np.asarray(SurrogateKernel(payload)({"a": np.array([1.0])})["b"]))
    assert answer == pytest.approx(10.0 ** (3.0 - 9.0), rel=1e-6)   # not 10**41


def test_a_model_that_reads_the_namelist_is_given_it() -> None:
    """A kernel slot's contract is one argument; a surrogate wants two.

    The nine parameters are features of a gated model, so a stage that called
    it with the column alone would answer for the case defaults no matter what
    the caller asked -- a parameter study that silently returns the same
    numbers.  The stage passes them when, and only when, the kernel says it
    reads them, so a plain ``lambda column: {...}`` still works.
    """

    from freecam.physics.macrophysics import RETURNED, Macrophysics

    seen: list = []

    def plain(column):                                   # the old contract
        seen.append(None)
        return {name: np.zeros(30) for name in RETURNED}

    def reads_namelist(column, parameters=None):
        seen.append(parameters)
        return {name: np.zeros(30) for name in RETURNED}

    reads_namelist.takes_parameters = True

    column = {name: np.zeros(30) for name in ("t0", "qv0", "ql0", "qi0", "nl0", "ni0")}
    stage = Macrophysics(kernel=plain)
    stage.mmacro_pcond(column, {"cldfrc_rhminl": 0.95})
    assert seen == [None]                                # not offered, not passed

    stage.kernels["mmacro_pcond"] = reads_namelist
    stage.mmacro_pcond(column, {"cldfrc_rhminl": 0.95})
    assert seen[-1] == {"cldfrc_rhminl": 0.95}
