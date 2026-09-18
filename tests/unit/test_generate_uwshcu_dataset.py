"""The compute_uwshcu_inv dataset script: captured frames become one NetCDF dataset, one sample a column."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("generate_uwshcu", REPO / "examples/generate_compute_uwshcu_inv_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture(tmp_path: Path, module, ranks: int, calls: int, ncols) -> Path:
    from freecam.physics.spec import load_function_spec
    from freecam.pi_cam.hooks import load_hooks

    spec = load_function_spec(str(module.CONTRACT)); hook = load_hooks().hook(module.KERNEL)
    by_name = {a.name: a for a in spec.arguments}
    dims = dict(spec.dimensions)
    directory = tmp_path / "frame-capture"; directory.mkdir()
    (directory / "capture.json").write_text(json.dumps({"kernels": [module.KERNEL], "every": 25, "pbs_job_id": "1.test",
                                                          "run_tag": "test-capture", "native_library_sha256": "abc"}))
    rng = np.random.default_rng(0)
    for rank in range(ranks):
        arrays, meta = {}, []
        for call in range(calls):
            ncol = ncols[(rank * calls + call) % len(ncols)]
            meta.append({"step": 1 + 12 * call, "ncol": ncol, "token": 1 + 50 * call, "kernel": module.KERNEL})
            for side, names in (("in", hook.model_inputs), ("out", hook.model_outputs)):
                for name in names:
                    item = by_name[name]
                    shape = [ncol] + [int(dims[str(a)]) for a in item.native_shape[1:]]
                    value = np.asarray(1800.0) if item.rank == 0 else rng.uniform(0, 1, shape)
                    if name == "t0_inv":
                        value = np.full(shape, 250.0 + rank + call * 0.01)
                    arrays[f"{side}/{call}/{name}"] = value
        arrays["meta"] = np.asarray(json.dumps(meta))
        np.savez(directory / f"{module.KERNEL}.rank-{rank:04d}.npz", **arrays)
    return directory


def test_captured_frames_become_one_dataset_with_the_contracts_axes(tmp_path: Path) -> None:
    from netCDF4 import Dataset as NetCDF

    module = _load()
    capture = _capture(tmp_path, module, ranks=2, calls=2, ncols=(3, 2))
    out = tmp_path / "uwshcu.nc"
    assert module.main(["--capture", str(capture), "--output", str(out)]) == 0
    with NetCDF(str(out)) as handle:
        assert handle.dimensions["sample"].size == 3 + 2 + 3 + 2
        assert handle.dimensions["lev"].size == 30 and handle.dimensions["ilev"].size == 31 and handle.dimensions["cnst"].size == 57
        assert handle.variables["input__tr0_inv"].dimensions == ("sample", "lev", "cnst")
        assert handle.variables["output__umf_inv"].dimensions == ("sample", "ilev") and handle.variables["input__dt"].dimensions == ("sample",)
        assert set(handle.inputs.split(",")) == set(__import__("freecam.pi_cam.hooks", fromlist=["load_hooks"]).load_hooks().hook(module.KERNEL).model_inputs)
        assert len(handle.outputs.split(",")) == 30 and "output__cush" in handle.variables and "input__cush" in handle.variables
        assert list(handle.variables["constituent"][:3]) == ["Q", "CLDLIQ", "CLDICE"] and handle.variables["constituent"][9] == "H2OV"
        assert np.all(handle.variables["input__dt"][:] == 1800.0)
        # where each sample came from: rank 0 call 0 (3 columns), rank 0 call 1 (2), rank 1 call 0 (3), rank 1 call 1 (2)
        assert list(handle.variables["sample_rank"][:]) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        assert list(handle.variables["sample_column"][:5]) == [0, 1, 2, 0, 1] and list(handle.variables["sample_step"][:5]) == [1, 1, 1, 13, 13]
        assert np.allclose(handle.variables["input__t0_inv"][5:8, 0], 251.0) and np.allclose(handle.variables["input__t0_inv"][8:, 0], 251.01)
        assert json.loads(handle.captures)[0]["run_tag"] == "test-capture" and handle.function == "uwshcu::compute_uwshcu_inv"
        assert all(s == "ok" for s in handle.variables["status"][:])
    # the physics-function reader takes the file
    from freecam.physics import open_dataset

    dataset = open_dataset(out)
    assert len(dataset) == 10 and dataset.inputs["qv0_inv"].shape == (10, 30) and dataset.outputs["trten_inv"].shape == (10, 30, 57)


def test_the_selection_options_narrow_the_dataset(tmp_path: Path) -> None:
    from netCDF4 import Dataset as NetCDF

    module = _load()
    capture = _capture(tmp_path, module, ranks=4, calls=4, ncols=(2,))
    out = tmp_path / "small.nc"
    module.main(["--capture", str(capture), "--capture", str(capture), "--output", str(out), "--ranks", "0:4:2", "--every-call", "2", "--no-tracers"])
    with NetCDF(str(out)) as handle:
        assert handle.dimensions["sample"].size == 2 * 2 * 2 * 2           # two captures, ranks 0 and 2, calls 0 and 2, two columns
        assert "input__tr0_inv" not in handle.variables and "output__wtprec" not in handle.variables and "cnst" not in handle.dimensions
        assert sorted(set(handle.variables["sample_rank"][:])) == [0, 2] and sorted(set(handle.variables["sample_call"][:])) == [0, 2]
        assert sorted(set(handle.variables["sample_capture"][:])) == [0, 1] and handle.tracers == "left out"
    out2 = tmp_path / "few.nc"
    module.main(["--capture", str(capture), "--output", str(out2), "--max-samples", "3", "--arguments", "t0_inv,sten_inv"])
    with NetCDF(str(out2)) as handle:
        assert handle.dimensions["sample"].size == 3
        assert set(v for v in handle.variables if "__" in v) == {"input__t0_inv", "output__sten_inv"}
    with pytest.raises(SystemExit):
        module.main(["--capture", str(capture), "--output", str(tmp_path / "x.nc"), "--arguments", "nothing"])
