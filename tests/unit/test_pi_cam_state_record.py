"""The state recorder and the globe viewer's reading of it: frames, changes, columns, the page."""

from __future__ import annotations

import base64
import json
import os
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
    ts = np.where(np.isnan(lat), np.nan, 280.0 + 10.0 * rank)
    qbot = np.zeros((PCOLS, PCNST, chunks), order="F")
    qbot[:, 1, :] = q[:, PVER - 1, 1, :]
    dtdt = np.full((PCOLS, PVER, chunks), 1.0 / 86400.0, order="F")
    return {"phys_state.lat": lat, "phys_state.lon": lon, "phys_state.t": t, "phys_state.q": q,
            "phys_state.pmid": pmid, "phys_state.ncol": np.array(ncol, dtype=np.int32), "cam_in.landfrac": landfrac,
            "cam_in.ts": ts, "cam_out.qbot": qbot, "phys_tend.dtdt": dtdt}


def record(directory: Path, monkeypatch, *, steps: int = 3, action_steps=(1,), first: int = 0) -> list[dict]:
    return _record(directory, monkeypatch, steps=steps, action_steps=action_steps, first=first)[0]


def record_in_memory(monkeypatch, *, steps: int = 3, action_steps=(1,), **options) -> list[StateRecorder]:
    """The same run as ``record``, kept in the two ranks' memory (the model still up)."""

    return _record(None, monkeypatch, steps=steps, action_steps=action_steps, close=False, **options)[1]


def ask(recorders: list[StateRecorder], request: dict):
    """A state query as the rank workers answer it: every rank (rank 1 first, as the fake comm
    needs), and the first answer in rank order."""

    answers, errors = {}, []
    for r in (1, 0):
        try:
            answers[r] = recorders[r].query(request)
        except ValueError as error:
            errors.append(str(error))
    if errors:
        assert len(errors) == 2, errors                                # every rank refuses alike
        raise RuntimeError(f"PI-CAM command 'state_query' failed\nValueError: {errors[0]}")
    return next((answers[r] for r in (0, 1) if answers[r] is not None), None)


def _record(directory: Path | None, monkeypatch, *, steps: int = 3, action_steps=(1,), first: int = 0,
            close: bool = True, **options) -> tuple[list[dict], list[StateRecorder]]:
    monkeypatch.setattr(state_record, "constituent_names", lambda library, count: NAMES[:count])
    comm = Comm()
    pools = [pool(0, [3, 1]), pool(1, [2, 0])]
    recorders = [StateRecorder(directory, rank=r, size=2, comm=comm.for_rank(r), fields=["T", "CLDLIQ"],
                               every=1, action_steps=action_steps, flush_every=2, **options) for r in (0, 1)]
    for recorder in recorders:
        recorder.start()
    for r in (1, 0):
        recorders[r].describe(pools[r])
    for step in range(first, first + steps):
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
    if close:
        for r in (1, 0):
            recorders[r].close()
    return pools, recorders


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
    # rank 0 writes every rank's columns: 3 step snapshots of (3 + 3 levels) x 6 columns
    assert (tmp_path / "steps.bin").stat().st_size == 3 * 6 * 6 * 4
    assert (tmp_path / "actions.bin").stat().st_size == 3 * 6 * 6 * 8
    assert (tmp_path / "action_seconds.bin").stat().st_size == 3 * 2 * 8          # (frames, ranks)
    assert not (tmp_path / "ranks").exists()
    # one snapshot's level is every column, rank 0's then rank 1's
    first = np.fromfile(tmp_path / "steps.bin", dtype="<f4", count=6 * 6).reshape(6, 6)
    np.testing.assert_allclose(first[0], [250.5, 251.5, 252.5, 253.5, 260.5, 261.5])


def test_the_land_fraction_is_kept_from_the_first_recorded_step(tmp_path: Path, monkeypatch) -> None:
    record(tmp_path, monkeypatch, first=1, action_steps=(2,))          # a run counting steps from 1
    assert json.loads((tmp_path / "manifest.json").read_text())["step_frames"] == [1, 2, 3]
    np.testing.assert_array_equal(np.load(tmp_path / "surface.npz")["landfrac"], [1, 1, 1, 1, 0, 0])


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
    memory = facade._state_options({"store": "memory", "keep_steps": 50})
    assert memory == {"store": "memory", "keep_steps": 50, "action_steps": []}
    for wrong in ({"store": "disk"}, {"store": "memory", "dir": tmp_path}, {"keep_steps": 5},
                  {"store": "memory", "keep_actions": 0}):
        with pytest.raises(ValueError):
            facade._state_options(wrong)
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
    driver.record_state = memory
    assert driver.state_dir is None
    assert "state_options" in inspect.getsource(facade.Driver._live_session)
    source = inspect.getsource(session.PICAMNotebookSession)
    assert '"--state-dir", str(options["dir"])' in source and '"--state-memory"' in source
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


def test_the_manifest_never_names_a_frame_the_files_do_not_hold(tmp_path: Path, monkeypatch) -> None:
    named: list = []
    write = StateRecorder._write_manifest

    def checked(self, *, complete):
        # every rank's part of each frame the manifest is about to name is in the file already
        sizes = {kind: (tmp_path / f"{kind}.bin").stat().st_size for kind in ("steps", "actions", "action_seconds")}
        assert sizes["steps"] == self._written["steps"] * 6 * 6 * 4
        assert sizes["actions"] == self._written["actions"] * 6 * 6 * 8
        assert sizes["action_seconds"] == self._written["actions"] * 2 * 8
        named.append((self._written["steps"], complete))
        write(self, complete=complete)

    monkeypatch.setattr(StateRecorder, "_write_manifest", checked)
    record(tmp_path, monkeypatch)
    assert named[0] == (0, False) and (2, False) in named and named[-1] == (3, True)


def test_rank_zero_gathers_many_snapshots_a_few_at_a_time(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(state_record, "GATHER_CHUNK", 2)
    sent: list = []
    gather = RankComm.gather
    monkeypatch.setattr(RankComm, "gather", lambda self, value, root=0: (
        sent.append((value.dtype.str, len(value))) if self.rank == 1 and getattr(value, "ndim", 0) == 2 else None,
        gather(self, value, root))[1])
    record(tmp_path, monkeypatch, steps=5, action_steps=(1, 2))
    # flushes at steps 1 and 3 and at the close: 2, 2 and 1 step snapshots; 3 action snapshots
    # at each of the first two, sent as 2 and 1
    assert [count for dtype, count in sent if dtype == "<f4"] == [2, 2, 1]
    assert [count for dtype, count in sent if dtype == "<f8"] == [2, 1, 2, 1]
    data = StateData(tmp_path)
    np.testing.assert_allclose(data.values("steps", 4, "T")[0], [252.5, 253.5, 254.5, 255.5, 262.5, 263.5])
    np.testing.assert_allclose(data.values("actions", 4, "T", "change"), 0.5)


def test_a_directory_of_another_schema_is_refused(tmp_path: Path, monkeypatch) -> None:
    record(tmp_path, monkeypatch)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["schema_version"] = 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema 1"):
        StateData(tmp_path)


def test_a_failing_rank_keeps_what_it_recorded_and_waits_for_no_one(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(state_record, "constituent_names", lambda library, count: NAMES[:count])
    collectives: list = []
    monkeypatch.setattr(RankComm, "Barrier", lambda self: collectives.append(("barrier", self.rank)))
    gather = RankComm.gather
    monkeypatch.setattr(RankComm, "gather", lambda self, value, root=0: (
        collectives.append(("gather", self.rank)), gather(self, value, root))[1])
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
    for r in (1, 0):
        recorders[r].flush()                            # step 0 is in the file
    collectives.clear()
    for r in (1, 0):
        recorders[r].step_done(1, pools[r])
    recorders[1].abandon()                              # rank 1 fails on its own
    assert collectives == []
    assert recorders[1].describe_run()["step_snapshots"] == 1     # step 1 is dropped, not sent
    assert (tmp_path / "steps.bin").stat().st_size == 3 * 6 * 4
    assert json.loads((tmp_path / "manifest.json").read_text())["complete"] is False


def test_the_viewer_sees_a_manifest_rewritten_within_the_same_second(tmp_path: Path, monkeypatch) -> None:
    # several writes can fall in one second, all a file system may keep of the modification time
    record(tmp_path, monkeypatch)
    data = StateData(tmp_path)
    manifest_file = tmp_path / "manifest.json"
    stamp = manifest_file.stat().st_mtime_ns
    manifest = json.loads(manifest_file.read_text())
    manifest["step_frames"] = manifest["step_frames"][:1]
    temporary = manifest_file.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest))
    temporary.replace(manifest_file)
    os.utime(manifest_file, ns=(stamp, stamp))
    assert data.changed()
    data.reload()
    assert data.step_frames == [0]


def test_a_model_keeping_its_state_in_memory_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    recorders = record_in_memory(monkeypatch)
    assert list(tmp_path.iterdir()) == []
    assert [len(r._kept["steps"]) for r in recorders] == [3, 3] and [len(r._kept["actions"]) for r in recorders] == [3, 3]
    with pytest.raises(ValueError, match="writes its snapshots to files"):
        StateRecorder(tmp_path / "files", rank=0, size=1).query({"q": "layout"})


def test_the_live_viewer_answers_as_the_recorded_directory_does(tmp_path: Path, monkeypatch) -> None:
    from freecam.pi_cam.state_view import LiveStateData, respond

    record(tmp_path, monkeypatch)
    files = StateData(tmp_path)
    recorders = record_in_memory(monkeypatch)
    live = LiveStateData(lambda request: ask(recorders, request))
    assert live.step_frames == files.step_frames and live.action_frames == files.action_frames
    assert live.action_owners == files.action_owners
    np.testing.assert_array_equal(live.landfrac, files.landfrac)
    for kind, index, field, mode in (("steps", 2, "T", "value"), ("steps", 1, "CLDLIQ", "change"),
                                     ("actions", 2, "CLDLIQ", "change"), ("actions", 0, "T", "value")):
        np.testing.assert_array_equal(live.values(kind, index, field, mode), files.values(kind, index, field, mode))
    assert live.changes(1, "CLDLIQ") == files.changes(1, "CLDLIQ")
    assert live.changes(1, "T") == files.changes(1, "T")
    np.testing.assert_array_equal(live.action_seconds(), files.action_seconds())
    for column in (0, 3, 5):                                           # rank 0's and rank 1's columns
        assert live.trace(1, column) == files.trace(1, column)
        assert live.profile("steps", 2, "T", column) == files.profile("steps", 2, "T", column)
    assert live.section("steps", 2, "T", 20.0) == files.section("steps", 2, "T", 20.0)
    assert live.anomalies(1) == files.anomalies(1) and live.step_anomalies() == files.step_anomalies()
    meta = live.meta()
    assert meta["live"] is True and meta["complete"] is False          # the model is still up
    assert meta["kept_from"] == {"steps": 0, "actions": 0}
    same = ("live", "directory", "run", "complete")
    assert {key: value for key, value in meta.items() if key not in same} == {
        key: value for key, value in files.meta().items() if key not in same}
    status, body, _ = respond(live, "/api/level", {"kind": "steps", "i": "2", "field": "T", "lev": "1"})
    assert status == 200 and len(body) == 4 * 6
    fresh = LiveStateData(lambda request: ask(recorders, request))  # one level asked, not the field
    for kind, index, field, level, mode in (("steps", 2, "T", 1, "value"), ("steps", 1, "T", 2, "change"),
                                            ("actions", 2, "CLDLIQ", 0, "change"), ("steps", 0, "T", 0, "change")):
        np.testing.assert_array_equal(fresh.level(kind, index, field, level, mode),
                                      files.level(kind, index, field, level, mode))
    assert not any(len(key) == 3 for key in fresh._cache)


def test_the_live_viewer_follows_the_run_and_refuses_what_is_no_longer_kept(monkeypatch) -> None:
    from freecam.pi_cam.state_view import LiveStateData, respond

    recorders = record_in_memory(monkeypatch, steps=5, action_steps=(1, 3), keep_steps=2, keep_actions=3)
    live = LiveStateData(lambda request: ask(recorders, request))
    assert live.step_frames == [0, 1, 2, 3, 4] and live.kept_from == {"steps": 3, "actions": 3}
    np.testing.assert_allclose(live.values("steps", 4, "T")[0], [252.5, 253.5, 254.5, 255.5, 262.5, 263.5])
    with pytest.raises(ValueError, match="no longer kept"):
        live.values("steps", 2, "T")
    status, body, _ = respond(live, "/api/level", {"kind": "steps", "i": "0", "field": "T", "lev": "1"})
    assert status == 400 and b"newest 2" in body
    assert [row["name"] for row in live.changes(3, "T")][1:] == ["cam_run1.process_0", "cam_run1.process_1"]
    with pytest.raises(ValueError, match="no longer kept"):
        live.changes(1, "T")                                           # step 1's actions were dropped whole
    assert live.step_anomalies()["steps"] == [0, 1, 2, 3, 4]          # codes of the kept steps only
    # the run goes on: a reload takes the new frames only
    pools = [pool(0, [3, 1]), pool(1, [2, 0])]
    for r in (1, 0):
        recorders[r].step_done(5, pools[r])
    live.reload()
    assert live.step_frames == [0, 1, 2, 3, 4, 5] and live.kept_from["steps"] == 4


def test_a_state_query_goes_before_the_next_step() -> None:
    import threading

    from freecam.pi_cam.session import PICAMNotebookError, PICAMNotebookSession

    session = PICAMNotebookSession.__new__(PICAMNotebookSession)
    session._query_turn = threading.Condition()
    session._queries_waiting = 0
    session._ready = False
    session._connection = None
    session.request_timeout = 5.0
    with pytest.raises(PICAMNotebookError, match="still starting"):
        session.state_query({"q": "layout"})
    order: list[str] = []
    session._queries_waiting = 1                                       # a page's query is waiting
    stepping = threading.Thread(target=lambda: (session._wait_for_queries(), order.append("step")))
    stepping.start()
    stepping.join(0.2)
    assert stepping.is_alive() and order == []                         # the step waits for it
    order.append("query")
    with session._query_turn:
        session._queries_waiting = 0
        session._query_turn.notify_all()
    stepping.join(2.0)
    assert order == ["query", "step"]


def test_surface_fields_tendencies_and_a_surface_constituent_are_recorded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(state_record, "constituent_names", lambda library, count: NAMES[:count])
    pools = [pool(0, [3, 1]), pool(1, [2, 0])]
    recorder = StateRecorder(None, rank=0, size=1, fields=["TS", "DTDT", "cam_out.qbot:CLDLIQ", "phys_state.t"])
    recorder.start()
    recorder.describe(pools[0])
    recorder.step_done(0, pools[0])
    layout = recorder.query({"q": "layout"})
    fields = {field["name"]: field for field in layout["fields"]}
    assert fields["TS"]["levels"] == 1 and fields["TS"]["units"] == "K"
    assert fields["DTDT"]["levels"] == PVER and fields["DTDT"]["scale"] == 86400.0
    assert fields["cam_out.qbot:CLDLIQ"]["levels"] == 1
    np.testing.assert_allclose(recorder.query({"q": "frame", "kind": "steps", "index": 0, "field": "TS"}), [[280.0] * 4])
    np.testing.assert_allclose(recorder.query({"q": "frame", "kind": "steps", "index": 0, "field": "cam_out.qbot:CLDLIQ"}),
                               [[1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5]], rtol=1e-6)
    assert field_spec("cam_in.shf").group == "surface" and field_spec("phys_tend.dudt").group == "tendency"
    for wrong in ("pbuf.CLD", "cam_in", "cam_in.", "cam_out.x-y"):
        with pytest.raises(ValueError, match="owner"):
            field_spec(wrong)
    with pytest.raises(ValueError, match="no constituent axis"):
        StateRecorder(None, rank=0, size=1, fields=["cam_in.ts:Q"]).describe(pools[0])
