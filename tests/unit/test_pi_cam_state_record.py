"""The state recorder and the globe viewer's reading of it: frames, changes, columns, the page."""

from __future__ import annotations

import base64
import json
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from freecam.pi_cam import state_record
from freecam.pi_cam.state_record import START_OF_STEP, StateRecorder, field_spec, parse_steps
from freecam.pi_cam.state_view import StateData, snapshot_data, snapshot_html

PCOLS, PVER, PCNST = 4, 3, 5
NAMES = ["Q", "CLDLIQ", "CLDICE", "NUMLIQ", "NUMICE"]


class Comm:
    """Two ranks in one process: rank 1 contributes first, rank 0 then sees both."""

    def __init__(self) -> None:
        self.gathered: list = []
        self.reduced: list = []

    def for_rank(self, rank: int) -> "RankComm":
        return RankComm(self, rank)


class RankComm:
    def __init__(self, shared: Comm, rank: int) -> None:
        self.shared, self.rank = shared, rank

    def Barrier(self) -> None:  # noqa: N802 - the mpi4py name
        pass

    def gather(self, value, root=0):
        if self.rank == 1:
            self.shared.gathered.append(value)
            return None
        return [value, self.shared.gathered.pop(0)]

    def allreduce(self, value):
        if self.rank == 1:
            self.shared.reduced.append(np.asarray(value))
            return value
        return np.asarray(value) + self.shared.reduced.pop(0)


def pool(rank: int, ncol: list[int]) -> dict[str, np.ndarray]:
    chunks = len(ncol)
    lat = np.full((PCOLS, chunks), np.nan, order="F")
    lon = np.full((PCOLS, chunks), np.nan, order="F")
    t = np.zeros((PCOLS, PVER, chunks), order="F")
    q = np.zeros((PCOLS, PVER, PCNST, chunks), order="F")
    pmid = np.zeros((PCOLS, PVER, chunks), order="F")
    landfrac = np.zeros((PCOLS, chunks), order="F")
    column = 0
    for c, n in enumerate(ncol):
        for i in range(n):
            lat[i, c] = np.radians(-60.0 + 20.0 * (rank * 4 + column))
            lon[i, c] = np.radians(10.0 * (rank * 4 + column))
            t[i, :, c] = 250.0 + 10.0 * rank + column + np.arange(PVER)
            q[i, :, 1, c] = 1.0e-5 * (column + 1)
            pmid[i, :, c] = [10000.0, 50000.0, 90000.0]
            landfrac[i, c] = float(rank == 0)
            column += 1
    return {"phys_state.lat": lat, "phys_state.lon": lon, "phys_state.t": t, "phys_state.q": q,
            "phys_state.pmid": pmid, "phys_state.ncol": np.array(ncol, dtype=np.int32), "cam_in.landfrac": landfrac}


def record(directory: Path, monkeypatch, *, steps: int = 3, action_steps=(1,)) -> list[dict]:
    monkeypatch.setattr(state_record, "constituent_names", lambda library, count: NAMES[:count])
    comm = Comm()
    pools = [pool(0, [3, 1]), pool(1, [2, 0])]
    recorders = [StateRecorder(directory, rank=r, size=2, comm=comm.for_rank(r), fields=["T", "CLDLIQ"],
                               every=1, action_steps=action_steps, flush_every=2) for r in (0, 1)]
    for recorder in recorders:
        recorder.start()
    for r in (1, 0):
        recorders[r].describe(pools[r])
    for step in range(steps):
        for r in (1, 0):
            recorders[r].step_start(step, pools[r])
        for action, (dt, dq) in enumerate(((0.5, 0.0), (0.0, 2.0e-6))):
            for r in (1, 0):
                pools[r]["phys_state.t"] += dt
                pools[r]["phys_state.q"][:, :, 1, :] += dq
                recorders[r].after_action(step, f"cam_run1.process_{action}", pools[r],
                                          seconds=0.001 * (action + 1) * (r + 1),
                                          owner={"by": "fortran"} if action == 0 else
                                          {"by": "python-stage", "mode": "native-model",
                                           "kernels": {"k": {"by": "ml", "file": "m.pt", "device": "cpu"}}})
        for r in (1, 0):
            recorders[r].step_done(step, pools[r])
    for r in (1, 0):
        recorders[r].close()
    return pools


def test_every_rank_records_its_own_columns_and_rank_zero_describes_them(tmp_path: Path, monkeypatch) -> None:
    record(tmp_path, monkeypatch)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["complete"] is True and manifest["step_frames"] == [0, 1, 2]
    assert [frame[:2] for frame in manifest["action_frames"]] == [
        [1, START_OF_STEP], [1, "cam_run1.process_0"], [1, "cam_run1.process_1"]]
    assert [frame[2] for frame in manifest["action_frames"]][:2] == [None, {"by": "fortran"}]
    assert [f["name"] for f in manifest["fields"]] == ["T", "CLDLIQ"] and manifest["fields"][1]["levels"] == PVER
    assert manifest["dtype"] == {"steps": "<f4", "actions": "<f8"}
    columns = np.load(tmp_path / "columns.npz")
    assert list(columns["rank"]) == [0, 0, 0, 0, 1, 1]                 # 4 real columns on rank 0, 2 on rank 1
    np.testing.assert_allclose(columns["lat"], [-60, -40, -20, 0, 20, 40])
    np.testing.assert_allclose(columns["level_pressure_pa"], [10000.0, 50000.0, 90000.0])
    np.testing.assert_array_equal(np.load(tmp_path / "surface.npz")["landfrac"], [1, 1, 1, 1, 0, 0])
    # a rank's file: 3 step snapshots of (3 + 3 levels) x its columns
    assert (tmp_path / "ranks" / "rank-0001.steps.bin").stat().st_size == 3 * 6 * 2 * 4
    assert (tmp_path / "ranks" / "rank-0000.actions.bin").stat().st_size == 3 * 6 * 4 * 8


def test_the_viewer_reads_frames_changes_profiles_and_sections(tmp_path: Path, monkeypatch) -> None:
    pools = record(tmp_path, monkeypatch)
    data = StateData(tmp_path)
    t_end = data.values("steps", 2, "T")                               # after three steps of +0.5 K
    assert t_end.shape == (PVER, 6)
    np.testing.assert_allclose(t_end[:, 0], 250.0 + 1.5 + np.arange(PVER))
    np.testing.assert_allclose(t_end[0, 4:], 260.0 + 1.5 + np.array([0.0, 1.0]))
    np.testing.assert_allclose(data.values("steps", 1, "T", "change"), 0.5)
    np.testing.assert_allclose(data.values("steps", 0, "T", "change"), 0.0)   # no earlier frame
    # the process frames of step 1: the start, then each action's own change (CLDLIQ in mg/kg)
    changes = data.changes(1, "CLDLIQ")
    assert [row["name"] for row in changes] == [START_OF_STEP, "cam_run1.process_0", "cam_run1.process_1"]
    assert changes[0]["max"] == 0.0 and changes[1]["max"] == 0.0
    assert changes[2]["max"] == pytest.approx(2.0)                     # 2e-6 kg/kg = 2 mg/kg
    np.testing.assert_allclose(data.values("actions", 1, "T", "change"), 0.5)
    np.testing.assert_allclose(data.values("actions", 0, "T", "change"), 0.0)
    # each action's time: the mean over the ranks and the slowest rank (rank 1 took twice as long)
    assert [row["seconds_mean"] for row in changes] == pytest.approx([0.0, 0.0015, 0.003])
    assert [row["seconds_max"] for row in changes] == pytest.approx([0.0, 0.002, 0.004])
    profile = data.profile("actions", 1, "T", 0)
    assert profile["values"][0] == pytest.approx(251.0) and profile["before"][0] == pytest.approx(250.5)
    section = data.section("steps", 2, "T", lon=0.0, half_width_degrees=15.0)
    assert section["lat"] == [-60.0, -40.0]                            # the two columns near 0 E
    level = data.level("steps", 2, "CLDLIQ", 1)
    assert level.dtype == np.dtype("<f4") and level[0] == pytest.approx(10.0 + 3 * 2.0)
    with pytest.raises(ValueError):
        data.frame("steps", 3, "T")
    with pytest.raises(KeyError):
        data.frame("steps", 0, "Q")
    assert pools[0]["phys_state.t"][0, 0, 0] == pytest.approx(251.5)  # the model's arrays: only the model's changes


def test_the_page_embeds_quantized_frames_that_decode_to_the_data(tmp_path: Path, monkeypatch) -> None:
    record(tmp_path, monkeypatch)
    data = StateData(tmp_path)
    embedded = snapshot_data(data, fields=["T"], levels=[2])
    block = embedded["fields"]["T"]["data"]["2"]
    codes = np.frombuffer(zlib.decompress(base64.b64decode(block["steps"])), np.uint8).reshape(3, 6)
    decoded = block["lo"] + codes / 255.0 * (block["hi"] - block["lo"])
    truth = np.stack([data.values("steps", i, "T")[2] for i in range(3)])
    inside = (truth >= block["lo"]) & (truth <= block["hi"])
    np.testing.assert_allclose(decoded[inside], truth[inside], atol=(block["hi"] - block["lo"]) / 255)
    actions = embedded["action_steps"]["1"]["fields"]["T"]["levels"]["2"]
    changes = np.frombuffer(zlib.decompress(base64.b64decode(actions["changes"])), np.int8).reshape(3, 6)
    np.testing.assert_allclose(changes[1] / 127 * actions["scales"][1], 0.5)
    page = snapshot_html(data, fields=["T"], levels=[2])
    assert "/*STATE_DATA*/null" not in page and "Natural Earth" in page


def test_field_names_and_step_lists_are_checked() -> None:
    assert field_spec("CLDICE").constituent == "CLDICE" and field_spec("phys_state.q:3").constituent == 3
    assert field_spec("phys_state.omega").field == "phys_state.omega"
    with pytest.raises(ValueError):
        field_spec("temperature")
    assert parse_steps("3, 24,40-42") == (3, 24, 40, 41, 42)
    assert parse_steps("12-48/12") == (12, 24, 36, 48) == parse_steps("12;24;36;48")
    with pytest.raises(ValueError):
        parse_steps("12:24")
    with pytest.raises(ValueError):
        StateRecorder("/nonexistent", rank=0, size=1, fields=["T", "T"])


def test_an_unknown_constituent_fails_when_the_fields_are_bound(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(state_record, "constituent_names", lambda library, count: NAMES[:count])
    recorder = StateRecorder(tmp_path, rank=0, size=1, fields=["phys_state.q:RAINQM"])
    recorder.start()
    with pytest.raises(ValueError, match="RAINQM"):
        recorder.describe(pool(0, [2]))


class _Recorder:
    def __init__(self) -> None:
        self.after: list[tuple[int, str]] = []

    def wants_actions(self, step: int) -> bool:
        return step == 3

    def after_action(self, step: int, name: str, pool, seconds: float = 0.0, owner=None) -> None:
        assert seconds >= 0.0                       # the action's own time, measured by the driver
        assert owner == {"by": "python"} if name.endswith("_python") else owner == {"by": "fortran"}
        self.after.append((step, name))


class _Profiler:
    @contextmanager
    def region(self, name: str):
        yield


class _Action:
    def __init__(self, name: str, inner: "_Action | None" = None) -> None:
        self.qualified_name, self.operation, self.inner = name, name, inner
        self.name = name.split(".")[-1]
        self.kind = "python_process" if name.endswith("_python") else "scheme"


def test_the_driver_records_after_plan_actions_only_and_only_at_action_steps() -> None:
    from freecam.pi_cam.driver import PICAMDriver

    class Driver:
        kernel_counters = None
        profiler = _Profiler()
        timeline = None
        pool: dict = {}

        def _execute_action(self, action):
            if action.inner is not None:           # a Python stage running its native action inside its own
                PICAMDriver._execute(self, action.inner)
            return action.qualified_name

    driver = Driver()
    driver.state_recorder = _Recorder()
    driver._in_step = True
    driver.coupling_step = 3
    PICAMDriver._execute(driver, _Action("cam_run1.shallow_convection_python", _Action("cam_run1.shallow_convection")))
    PICAMDriver._execute(driver, _Action("cam_run1.deep_convection"))
    assert driver.state_recorder.after == [(3, "cam_run1.shallow_convection_python"), (3, "cam_run1.deep_convection")]
    driver.coupling_step = 4
    PICAMDriver._execute(driver, _Action("cam_run1.deep_convection"))
    assert len(driver.state_recorder.after) == 2                        # not an action step


def test_the_server_answers_only_with_its_token(tmp_path: Path, monkeypatch) -> None:
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from freecam.pi_cam import state_view

    record(tmp_path, monkeypatch)
    handler = type("H", (state_view._Handler,), {"data": StateData(tmp_path), "token": "tok", "lock": threading.Lock()})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(f"{base}/api/meta")
        assert refused.value.code == 403
        meta = json.loads(urllib.request.urlopen(f"{base}/api/meta?token=tok").read())
        assert meta["columns"] == 6 and meta["action_steps"]["1"][0] == START_OF_STEP
        level = np.frombuffer(urllib.request.urlopen(
            f"{base}/api/level?token=tok&kind=actions&i=1&field=T&lev=0&mode=change").read(), "<f4")
        np.testing.assert_allclose(level, 0.5)
        changes = json.loads(urllib.request.urlopen(f"{base}/api/changes?token=tok&step=1&field=T").read())
        assert [round(row["max"], 6) for row in changes] == [0.0, 0.5, 0.0]
        profile = json.loads(urllib.request.urlopen(f"{base}/api/profile?token=tok&kind=steps&i=2&field=T&col=5").read())
        assert profile["rank"] == 1 and len(profile["values"]) == PVER
        with pytest.raises(urllib.error.HTTPError) as bad:
            urllib.request.urlopen(f"{base}/api/level?token=tok&kind=steps&i=9&field=T&lev=0")
        assert bad.value.code == 400
        page = urllib.request.urlopen(f"{base}/?token=tok").read().decode()
        assert "freeCAM globe" in page and "/*COASTLINE*/null" not in page
    finally:
        server.shutdown()
        server.server_close()


def test_the_notebook_driver_passes_its_state_options_to_the_rank_workers(tmp_path: Path) -> None:
    import inspect

    from freecam.pi_cam import facade, session

    assert facade._state_options(False) is None
    assert facade._state_options(True) == {"dir": None}
    assert facade._state_options(tmp_path) == {"dir": tmp_path}
    options = facade._state_options({"fields": ["T", "CLDLIQ"], "action_steps": [24, 3, 24]})
    assert options == {"fields": ["T", "CLDLIQ"], "action_steps": [3, 24]}
    with pytest.raises(ValueError):
        facade._state_options({"feilds": ["T"]})                      # a misnamed option fails here
    with pytest.raises(ValueError):
        facade._state_options({"fields": ["temperature"]})
    driver = facade.Driver.__new__(facade.Driver)
    driver.record_state = None
    assert driver.state_dir is None
    driver.record_state = {"dir": tmp_path / "globe"}
    assert driver.state_dir == (tmp_path / "globe").resolve()
    assert "state_options" in inspect.getsource(facade.Driver._live_session)
    assert '"--state-dir", str(options["dir"])' in inspect.getsource(session.PICAMNotebookSession)
    worker = (Path(session.__file__).parent / "session_worker.py").read_text()
    assert worker.index("attach_state_recorder") < worker.index("driver.initialize()")


def test_a_column_is_traced_through_an_action_step(tmp_path: Path, monkeypatch) -> None:
    from freecam.pi_cam.state_view import owner_text, trace_text

    record(tmp_path, monkeypatch)
    data = StateData(tmp_path)
    column = data.nearest(-42.0, 11.0)
    assert column == 1                                                   # the column at 40 S, 10 E
    trace = data.trace(1, column, level=0)
    names = [row["name"] for row in trace["rows"]]
    assert names == [START_OF_STEP, "cam_run1.process_0", "cam_run1.process_1"]
    # T after each action at the top level; process_0 warms by 0.5 K, process_1 moistens by 2 mg/kg
    assert trace["rows"][0]["values"]["T"] == pytest.approx(251.0 + 0.5)
    assert trace["rows"][1]["change"]["T"] == pytest.approx(0.5) and trace["rows"][2]["change"]["T"] == 0.0
    assert trace["rows"][2]["change"]["CLDLIQ"] == pytest.approx(2.0)
    assert trace["rows"][1]["seconds_mean"] == pytest.approx(0.0015)
    assert owner_text(trace["rows"][1]["owner"]) == "original Fortran"
    assert "k by an ML model (m.pt, cpu)" in owner_text(trace["rows"][2]["owner"])
    # with no level: where the first field changed most over the step (all levels alike here: the first)
    assert data.trace(1, column)["level"] == 0
    text = trace_text(trace)
    assert "↓ cam_run1.process_0   (original Fortran, 1.5 ms)" in text and "ΔT +0.5 K" in text
    assert "over the step: ΔT +0.5 K   ΔCLDLIQ +2 mg/kg" in text
    with pytest.raises(ValueError):
        data.trace(0, column)                                            # step 0 has no action frames


def test_the_trace_command_and_route(tmp_path: Path, monkeypatch, capsys) -> None:
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from freecam.pi_cam import state_view

    record(tmp_path, monkeypatch)
    assert state_view.trace_main([str(tmp_path), "--lat", "-40", "--lon", "10", "--step", "1", "--level", "1"]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("step 1, the column at 40.0°S, 10.0°E (land), level 1")
    handler = type("H", (state_view._Handler,), {"data": StateData(tmp_path), "token": "tok", "lock": threading.Lock()})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/api/trace?token=tok&step=1&col=1&lev=0"
        trace = json.loads(urllib.request.urlopen(url).read())
        assert trace["column"] == 1 and trace["rows"][1]["owner"] == {"by": "fortran"}
    finally:
        server.shutdown()
        server.server_close()


def test_the_driver_names_what_computed_an_action() -> None:
    from pathlib import Path as _Path
    from types import SimpleNamespace

    from freecam.physics.native_model import NativeModel
    from freecam.pi_cam.driver import PICAMDriver

    model = NativeModel.__new__(NativeModel)
    model.path, model.device = _Path("/somewhere/uwshcu.pt"), "cuda"

    class Stage:
        execution = SimpleNamespace(mode="native-model")
        kernels = {"compute_uwshcu_inv": model, "fluxbelowinv": None}

        def replacements(self):
            return ("compute_uwshcu_inv",)

    stage = Stage()
    driver = SimpleNamespace(python_processes=SimpleNamespace(installed={
        "shallow_convection_python": SimpleNamespace(function=SimpleNamespace(__self__=stage))}))
    owner = PICAMDriver._action_owner(driver, SimpleNamespace(kind="python_process", name="shallow_convection_python"))
    assert owner == {"by": "python-stage", "mode": "native-model",
                     "kernels": {"compute_uwshcu_inv": {"by": "ml", "file": "uwshcu.pt", "device": "cuda"}}}
    assert PICAMDriver._action_owner(driver, SimpleNamespace(kind="scheme", name="x")) == {"by": "fortran"}
    assert PICAMDriver._action_owner(driver, SimpleNamespace(kind="boundary", name="x")) == {"by": "coupler"}
    assert PICAMDriver._action_owner(driver, SimpleNamespace(kind="python_process", name="gone")) == {"by": "python"}


def test_a_round_off_change_does_not_list_an_action() -> None:
    from freecam.pi_cam.state_view import trace_text

    def row(name, t, ice, dt=0.0, dice=0.0):
        return {"name": name, "owner": {"by": "fortran"}, "values": {"T": t, "CLDICE": ice},
                "change": {"T": dt, "CLDICE": dice}, "column_max_change": {"T": abs(dt), "CLDICE": abs(dice)}}

    trace = {"step": 1, "lat": 0.0, "lon": 0.0, "land": 0.0, "level": 0, "pressure_hpa": 900.0,
             "fields": [{"name": "T", "units": "K", "noise": 3e-10}, {"name": "CLDICE", "units": "mg/kg", "noise": 1e-11}],
             "rows": [row("start of step", 290.0, 0.0), row("a.gravity_wave_drag", 290.0, 4e-39, dice=4e-39),
                      row("a.deep_convection", 289.0, 4e-39, dt=-1.0)]}
    text = trace_text(trace)
    assert "gravity_wave_drag" not in text and "1 more changed this level" in text
    assert "ΔT -1 K   ΔCLDICE 0 mg/kg" in text and "over the step: ΔT -1 K   ΔCLDICE ≈0 mg/kg" in text


def record_with_faults(directory: Path, monkeypatch) -> None:
    """The recorder over three steps, step 1 recorded after every action, with faults put in:
    after process_0 of step 1 column 5 (rank 1) holds negative liquid; after process_1 column 2
    (rank 0) holds a NaN at its bottom level; at the end of step 2 column 4 is at 50 K."""

    monkeypatch.setattr(state_record, "constituent_names", lambda library, count: NAMES[:count])
    comm = Comm()
    pools = [pool(0, [3, 1]), pool(1, [2, 0])]
    recorders = [StateRecorder(directory, rank=r, size=2, comm=comm.for_rank(r), fields=["T", "CLDLIQ"],
                               every=1, action_steps=(1,), flush_every=2) for r in (0, 1)]
    for recorder in recorders:
        recorder.start()
    for r in (1, 0):
        recorders[r].describe(pools[r])
    for step in range(3):
        for r in (1, 0):
            recorders[r].step_start(step, pools[r])
        for action, (dt, dq) in enumerate(((0.5, 0.0), (0.0, 2.0e-6))):
            for r in (1, 0):
                pools[r]["phys_state.t"] += dt
                pools[r]["phys_state.q"][:, :, 1, :] += dq
                if step == 1 and action == 0 and r == 1:
                    pools[r]["phys_state.q"][1, 0, 1, 0] = -1.0e-9       # column 5, top level
                if step == 1 and action == 1 and r == 0:
                    pools[r]["phys_state.q"][2, 2, 1, 0] = np.nan        # column 2, bottom level
                if step == 2 and action == 1 and r == 1:
                    pools[r]["phys_state.t"][0, :, 0] = 50.0             # column 4
                recorders[r].after_action(step, f"cam_run1.process_{action}", pools[r], seconds=0.001,
                                          owner={"by": "fortran"})
        for r in (1, 0):
            recorders[r].step_done(step, pools[r])
    for r in (1, 0):
        recorders[r].close()


def test_anomaly_codes_follow_the_rules() -> None:
    from freecam.pi_cam.state_view import anomaly_codes, anomaly_rule

    values = np.array([[1.0, -1.0, np.nan, 2.0], [1.0, 1.0, 1.0, np.inf]])      # (levels, columns)
    assert list(anomaly_codes(values, "CLDLIQ")) == [0, 2, 1, 1]
    assert list(anomaly_codes(values, "U")) == [0, 0, 1, 1]                      # U may be negative
    assert list(anomaly_codes(np.array([[250.0, 50.0, 450.0]]), "T")) == [0, 3, 3]
    assert list(anomaly_codes(np.array([[-1.0]]), "phys_state.q:H2O2")) == [2]
    assert anomaly_rule("T") == "finite and within 100 to 400" and anomaly_rule("OMEGA") == "finite"


def test_the_action_that_made_a_column_anomalous_is_found(tmp_path: Path, monkeypatch) -> None:
    from freecam.pi_cam.state_view import anomalies_text, trace_text

    record_with_faults(tmp_path, monkeypatch)
    data = StateData(tmp_path)
    report = data.anomalies(1)
    first = {(f["field"], f["kind"]): f for f in report["first"]}
    assert set(first) == {("CLDLIQ", "negative"), ("CLDLIQ", "non-finite")}
    assert first[("CLDLIQ", "negative")]["name"] == "cam_run1.process_0" and first[("CLDLIQ", "negative")]["example_columns"] == [5]
    assert first[("CLDLIQ", "non-finite")]["name"] == "cam_run1.process_1" and first[("CLDLIQ", "non-finite")]["columns"] == 1
    frames = report["fields"]["CLDLIQ"]["frames"]
    assert [f["columns"] for f in frames] == [0, 1, 1] and [f["new"] for f in frames] == [0, 1, 1]
    assert report["fields"]["T"]["frames"][2]["columns"] == 0
    # the globe's mask: column 5 negative after process_0, column 2 non-finite after process_1
    after0, after1 = data.action_steps[1][1], data.action_steps[1][2]
    assert list(data.anomaly_mask("actions", after0, "CLDLIQ")) == [0, 0, 0, 0, 0, 2]
    assert list(data.anomaly_mask("actions", after1, "CLDLIQ")) == [0, 0, 1, 0, 0, 0]
    # at the ends of the steps: the NaN stays from step 1; the cold column is out of range at step 2
    steps = data.step_anomalies()
    assert steps["fields"]["CLDLIQ"]["columns"] == [0, 1, 1] and steps["fields"]["T"]["columns"] == [0, 0, 1]
    assert {(f["field"], f["step"]) for f in steps["first"]} == {("CLDLIQ", 1), ("T", 2)}
    # the trace of the broken column marks the action, and its non-finite values are None, not NaN
    trace = data.trace(1, 2)
    assert trace["level"] == 2                                   # the level that went non-finite
    assert trace["rows"][2]["became"] == {"CLDLIQ": "non-finite"} and trace["rows"][2]["values"]["CLDLIQ"] is None
    assert trace["rows"][1]["became"] == {}
    text = trace_text(trace)
    assert "⚠ the column became non-finite in CLDLIQ here" in text and "CLDLIQ = non-finite" in text
    summary = anomalies_text(data)
    assert "CLDLIQ became negative after cam_run1.process_0 (original Fortran): 1 columns" in summary
    assert "T out of range first at the end of step 2" in summary


def test_the_anomaly_routes_answer_strict_json(tmp_path: Path, monkeypatch, capsys) -> None:
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from freecam.pi_cam import state_view

    record_with_faults(tmp_path, monkeypatch)
    handler = type("H", (state_view._Handler,), {"data": StateData(tmp_path), "token": "tok", "lock": threading.Lock()})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    strict = lambda body: json.loads(body, parse_constant=lambda name: pytest.fail(f"{name} in the JSON"))
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        report = strict(urllib.request.urlopen(f"{base}/api/anomalies?token=tok&step=1").read())
        assert report["first"][0]["field"] == "CLDLIQ"
        assert strict(urllib.request.urlopen(f"{base}/api/anomalies?token=tok").read())["fields"]["T"]["columns"] == [0, 0, 1]
        mask = urllib.request.urlopen(f"{base}/api/anomaly_mask?token=tok&kind=actions&i=2&field=CLDLIQ").read()
        assert list(mask) == [0, 0, 1, 0, 0, 0]
        trace = strict(urllib.request.urlopen(f"{base}/api/trace?token=tok&step=1&col=2&lev=2").read())
        assert trace["rows"][2]["values"]["CLDLIQ"] is None
        assert strict(urllib.request.urlopen(f"{base}/api/meta?token=tok").read())["anomaly_rules"]["CLDLIQ"] == "finite and not negative"
    finally:
        server.shutdown()
        server.server_close()
    assert state_view.anomalies_main([str(tmp_path)]) == 0
    assert "CLDLIQ became non-finite after cam_run1.process_1" in capsys.readouterr().out


def test_the_page_carries_the_anomaly_codes(tmp_path: Path, monkeypatch) -> None:
    record_with_faults(tmp_path, monkeypatch)
    data = StateData(tmp_path)
    snap = snapshot_data(data)["anomalies"]
    unpack = lambda b64: np.frombuffer(zlib.decompress(base64.b64decode(b64)), np.uint8)
    np.testing.assert_array_equal(unpack(snap["action_steps"]["1"]["codes"]["CLDLIQ"]), data.action_codes(1)["CLDLIQ"].ravel())
    np.testing.assert_array_equal(unpack(snap["steps"]["codes"]["T"]).reshape(3, 6)[2], [0, 0, 0, 0, 3, 0])
    assert snap["action_steps"]["1"]["summary"]["first"][0]["name"] == "cam_run1.process_0"
    page = snapshot_html(data)
    start = page.index("const EMBEDDED = ") + len("const EMBEDDED = ")
    decoder = json.JSONDecoder(parse_constant=lambda name: pytest.fail(f"{name} in the embedded data"))
    embedded, _ = decoder.raw_decode(page, start)                      # strict JSON: no NaN, no Infinity
    assert embedded["snapshot"]["anomalies"]["steps"]["summary"]["first"][0]["field"] == "CLDLIQ"


def test_a_flush_waits_for_every_rank_before_the_manifest_names_the_frame(tmp_path: Path, monkeypatch) -> None:
    events: list = []
    monkeypatch.setattr(RankComm, "Barrier", lambda self: events.append(("barrier", self.rank)))
    write = StateRecorder._write_manifest
    monkeypatch.setattr(StateRecorder, "_write_manifest",
                        lambda self, *, complete: (events.append(("manifest", complete)), write(self, complete=complete)))
    record(tmp_path, monkeypatch)
    manifests = [k for k, event in enumerate(events) if event[0] == "manifest"]
    assert len(manifests) >= 3
    # after the description, every manifest rank 0 writes comes right after its wait for the other rank
    for k in manifests[1:]:
        if events[k] == ("manifest", False):
            assert events[k - 1] == ("barrier", 0), events[k - 3:k + 1]
    assert events[manifests[-1]] == ("manifest", True)


def test_a_failing_rank_keeps_what_it_recorded_and_waits_for_no_one(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(state_record, "constituent_names", lambda library, count: NAMES[:count])
    barriers: list = []
    monkeypatch.setattr(RankComm, "Barrier", lambda self: barriers.append(self.rank))
    comm = Comm()
    pools = [pool(0, [3, 1]), pool(1, [2, 0])]
    recorders = [StateRecorder(tmp_path, rank=r, size=2, comm=comm.for_rank(r), fields=["T"], every=1,
                               flush_every=24) for r in (0, 1)]
    for recorder in recorders:
        recorder.start()
    for r in (1, 0):
        recorders[r].describe(pools[r])
    for r in (1, 0):
        recorders[r].step_done(0, pools[r])
    barriers.clear()
    recorders[1].abandon()                              # rank 1 fails on its own
    assert barriers == []
    assert (tmp_path / "ranks" / "rank-0001.steps.bin").stat().st_size == 3 * 2 * 4
    assert json.loads((tmp_path / "manifest.json").read_text())["complete"] is False
