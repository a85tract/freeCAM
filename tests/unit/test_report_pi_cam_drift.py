import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from freecam.pi_cam.drift import cam_file, format_table, parse_variables, report, variable_drift

REPO = Path(__file__).resolve().parents[2]


def _write(path: Path, t, ps, *, q=None) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("column", len(t))
        dataset.createVariable("T", "f8", ("column",))[:] = np.asarray(t, dtype="f8")
        dataset.createVariable("PS", "f8", ("column",))[:] = np.asarray(ps, dtype="f8")
        if q is not None:
            dataset.createVariable("Q", "f8", ("column",))[:] = np.asarray(q, dtype="f8")


def _runs(tmp_path: Path) -> tuple[Path, Path]:
    reference, candidate = tmp_path / "oracle", tmp_path / "run"
    reference.mkdir(), candidate.mkdir()
    _write(reference / "case.cam.r.0001-01-03-00000.nc", [280.0, 290.0], [1e5, 1e5], q=[1e-3, 2e-3])
    _write(candidate / "case.cam.r.0001-01-03-00000.nc", [280.0, 293.0], [1e5, 1.01e5])
    return reference, candidate


def test_drift_is_measured_in_scaled_units_and_names_no_directory(tmp_path) -> None:
    reference, candidate = _runs(tmp_path)
    payload = report(reference, candidate)
    by_name = {row["name"]: row for row in payload["fields"]}
    # T differs by 3 K in one of two columns: rms sqrt(9/2), max 3, mean 1.5
    assert by_name["T"]["rms"] == pytest.approx(np.sqrt(4.5)) and by_name["T"]["max"] == 3.0
    assert by_name["T"]["mean"] == pytest.approx(1.5) and by_name["T"]["count"] == 2
    # PS is reported in hPa: 1000 Pa -> 10 hPa
    assert by_name["PS"]["max"] == pytest.approx(10.0) and by_name["PS"]["unit"] == "hPa"
    # a field one file lacks is a note, not an error; the fields neither has likewise
    assert by_name["Q"]["rms"] is None and by_name["Q"]["note"] == "absent from the candidate"
    assert by_name["U"]["note"] == "absent from the reference and candidate"
    assert payload["identical"] is False
    # the record names logical file names only: no site path reaches a committed record
    text = json.dumps(payload)
    assert str(tmp_path) not in text and payload["reference_file"] == "cam.r.0001-01-03-00000.nc"


def test_an_identical_pair_reads_as_identical_and_the_file_must_be_unique(tmp_path) -> None:
    reference, _ = _runs(tmp_path)
    payload = report(reference, reference)
    assert payload["identical"] is True and all(
        row["rms"] == 0.0 for row in payload["fields"] if row["rms"] is not None)
    _write(reference / "case.cam.r.0001-01-04-00000.nc", [1.0], [1.0])
    with pytest.raises(FileNotFoundError, match="expected one"):
        cam_file(reference)
    assert cam_file(reference, pick="last").name == "case.cam.r.0001-01-04-00000.nc"
    assert cam_file(reference, pick="first").name == "case.cam.r.0001-01-03-00000.nc"


def test_variables_parse_with_defaults_and_the_table_prints(tmp_path) -> None:
    assert parse_variables(["T", "Q:1e3:g/kg", "OMEGA:1:Pa/s", "PS:0.01"]) == (
        ("T", 1.0, "K"), ("Q", 1e3, "g/kg"), ("OMEGA", 1.0, "Pa/s"), ("PS", 0.01, ""))
    reference, candidate = _runs(tmp_path)
    rows = variable_drift(cam_file(reference), cam_file(candidate), (("T", 1.0, "K"), ("Q", 1e3, "g/kg")))
    table = format_table(rows)
    assert table.splitlines()[1].startswith("T ") and "absent from the candidate" in table


def test_the_tool_writes_a_record(tmp_path) -> None:
    reference, candidate = _runs(tmp_path)
    out = tmp_path / "drift.json"
    result = subprocess.run([sys.executable, str(REPO / "tools/report_pi_cam_drift.py"), "--reference", str(reference),
                             "--candidate", str(candidate), "--variables", "T", "PS", "--output", str(out)],
                            capture_output=True, text=True, check=True, cwd=REPO)
    assert "T " in result.stdout and out.exists()
    payload = json.loads(out.read_text())
    assert [row["name"] for row in payload["fields"]] == ["T", "PS"] and payload["schema_version"] == 1
