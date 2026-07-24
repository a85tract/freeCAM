from pathlib import Path
from types import SimpleNamespace

from netCDF4 import Dataset
import numpy as np

from pycam_sima.model.comm import SerialComm
from pycam_sima.model.contracts import FieldContract
from pycam_sima.model.history import HISTORY_FIELDS, HistoryWriter


class _HistoryPool:
    def __init__(self) -> None:
        self.dimensions = {"pver": 2, "pverp": 3}
        self.dynamic_fields = frozenset({"runtime_diagnostic"})
        self.contracts = {
            "runtime_diagnostic": FieldContract(
                "runtime_diagnostic",
                "float64",
                ("nphys_local", "pver"),
                "out",
                "plugin_state",
                "K",
                ccpp_standard_name="runtime_diagnostic",
                history=True,
                restart=True,
            ),
            # This reproduces the pre-existing static contract that must not
            # enter the dynamic-history path.
            "static_internal": FieldContract(
                "static_internal",
                "float64",
                ("nphys_local", "pver", "nconst"),
                "inout",
                "physics",
                history=True,
            ),
        }
        self.values = {
            "physics_global_column": np.array([1, 0], dtype=np.int32),
            "physics_latitude": np.array([0.1, 0.2]),
            "physics_longitude": np.array([0.3, 0.4]),
            "physics_cell_area": np.array([0.5, 0.5]),
            "hybrid_a_midpoint": np.array([0.1, 0.2]),
            "hybrid_b_midpoint": np.array([0.9, 0.8]),
            "hybrid_a_interface": np.array([0.0, 0.15, 0.3]),
            "hybrid_b_interface": np.array([1.0, 0.85, 0.7]),
            "history_sample_count": np.array(0, dtype=np.int32),
            "runtime_diagnostic": np.array(
                [[1.0, 2.0], [3.0, 4.0]], order="F"
            ),
        }
        for _output_name, state_name in HISTORY_FIELDS:
            self.values[state_name] = np.array([10.0, 20.0])

    def get(self, name: str) -> np.ndarray:
        return self.values[name]


def test_history_only_scans_opted_in_dynamic_fields(tmp_path: Path) -> None:
    pool = _HistoryPool()
    clock = SimpleNamespace(
        year=1,
        month=1,
        day=1,
        seconds=1800,
        nstep=1,
        dt_seconds=1800,
        yyyymmdd=10101,
    )
    path = HistoryWriter(tmp_path, "dynamic", SerialComm()).write(pool, clock)

    assert path is not None
    with Dataset(path) as dataset:
        assert "runtime_diagnostic" in dataset.variables
        assert "static_internal" not in dataset.variables
        assert dataset["runtime_diagnostic"].units == "K"
        assert dataset["runtime_diagnostic"].standard_name == "runtime_diagnostic"
