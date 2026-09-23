"""The optional action timeline: what the driver records, how it is written, and what the
viewer makes of it."""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from freecam.pi_cam import InMemoryBoundaryProvider, PICAMConfig, PICAMDriver, RecordingCAMBackend
from freecam.pi_cam.timeline import KIND_ACTION, KIND_PHASE, KIND_STEP, KIND_WAIT, RECORD_DTYPE, TimelineRecorder
from freecam.pi_cam.timeline_view import TimelineData, main as timeline_main, snapshot_html


def _driver(steps: int = 3) -> PICAMDriver:
    config = PICAMConfig(case_name="unit-timeline", source_root=Path("/tmp/source"), mpi_size=1, stop_n=steps)
    boundary = InMemoryBoundaryProvider({(s, 0): {"sst": np.full((2,), 280.0 + s)} for s in range(steps + 2)})
    return PICAMDriver(config, boundary, RecordingCAMBackend(), rank=0, size=1)


def _run(tmp_path: Path, steps: int = 3, flush_every: int = 2) -> Path:
    directory = tmp_path / "timeline"
    driver = _driver(steps)
    driver.attach_timeline(TimelineRecorder(directory, rank=0, size=1, flush_every=flush_every, run_label="unit"))
    driver.initialize()
    for _ in range(steps):
        driver.step()
    driver.finalize()
    return directory


def test_the_driver_records_nothing_unless_a_timeline_is_attached(tmp_path: Path) -> None:
    driver = _driver(1)
    assert driver.timeline is None
    driver.initialize()
    driver.step()
    assert not (tmp_path / "timeline").exists()


def test_every_action_and_step_is_recorded_and_written_to_the_rank_file(tmp_path: Path) -> None:
    directory = _run(tmp_path, steps=3, flush_every=2)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["ranks"] == 1 and manifest["complete"] is True and manifest["last_step_flushed"] == 2
    assert manifest["run"] == "unit" and manifest["record_dtype"][0] == ["step", "<i4"]
    records = np.fromfile(directory / "ranks" / "rank-0000.bin", dtype=RECORD_DTYPE)
    names = json.loads((directory / "ranks" / "rank-0000.names.json").read_text())
    kinds = records["kind"]
    # one step record per step, the initialization and finalization phases
    assert sorted(records["step"][kinds == KIND_STEP].tolist()) == [0, 1, 2]
    phases = {names[a] for a in records["action"][kinds == KIND_PHASE]}
    assert phases == {"initialize", "finalize"}
    # every plan action of every step, each inside its step's span
    per_step = [records[(kinds == KIND_ACTION) & (records["step"] == s)] for s in range(3)]
    assert len({len(p) for p in per_step}) == 1 and len(per_step[0]) >= 40
    action_names = [names[a] for a in per_step[0]["action"][np.argsort(per_step[0]["t0"])]]
    assert action_names[0] == "coupling.boundary_import" and action_names[-1] == "coupling.boundary_export"
    for s in range(3):
        span = records[(kinds == KIND_STEP) & (records["step"] == s)][0]
        acts = per_step[s]
        assert np.all(acts["t0"] >= span["t0"]) and np.all(acts["t1"] <= span["t1"]) and np.all(acts["t1"] >= acts["t0"])
    # the collective agreement calls are timed as waits inside their action
    waits = records[kinds == KIND_WAIT]
    in_steps = waits[waits["step"] >= 0]
    assert len(in_steps) >= 3 and {names[a] for a in in_steps["action"]} <= set(action_names)
    assert {names[a] for a in waits[waits["step"] < 0]["action"]} <= {"outside the plan"}     # initialization's


def test_a_timeline_is_attached_before_initialization_only(tmp_path: Path) -> None:
    driver = _driver(1)
    driver.initialize()
    with pytest.raises(Exception, match="before initialize"):
        driver.attach_timeline(TimelineRecorder(tmp_path / "t", rank=0, size=1))


def test_records_stay_in_memory_until_a_flush(tmp_path: Path) -> None:
    recorder = TimelineRecorder(tmp_path / "t", rank=0, size=1, flush_every=5)
    recorder.start()
    for step in range(4):
        t = recorder.clock()
        recorder.action(step, "cam_run1.deep_convection", t)
        recorder.step_done(step, t)
    assert (tmp_path / "t" / "ranks" / "rank-0000.bin").stat().st_size == 0        # nothing written yet
    recorder.step_done(4, recorder.clock())                                          # the fifth step flushes
    assert (tmp_path / "t" / "ranks" / "rank-0000.bin").stat().st_size == 9 * RECORD_DTYPE.itemsize
    with pytest.raises(ValueError):
        TimelineRecorder(tmp_path / "u", rank=0, size=1, flush_every=0)


def test_the_columns_of_every_rank_are_gathered_once_with_their_host(tmp_path: Path) -> None:
    class Comm:
        def __init__(self):
            self.barriers = 0

        def Barrier(self):
            self.barriers += 1

        def gather(self, value, root=0):
            host, local, lat, lon, phis = value
            return [value, ("node-b", 0, lat + 0.1, lon + 0.2, phis * 0)]

    comm = Comm()
    recorder = TimelineRecorder(tmp_path / "t", rank=0, size=2, comm=comm)
    recorder.start()
    assert comm.barriers == 1
    pool = {"phys_state.lat": np.radians(np.array([[10.0, -20.0], [11.0, 0.0], [np.inf, np.inf]])),
            "phys_state.lon": np.radians(np.array([[100.0, 200.0], [101.0, 0.0], [np.inf, np.inf]])),
            "phys_state.phis": np.array([[500.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            "phys_state.ncol": np.array([2, 1])}
    recorder.write_columns(pool)
    z = np.load(tmp_path / "t" / "columns.npz")
    assert z["rank"].tolist() == [0, 0, 0, 1, 1, 1]
    np.testing.assert_allclose(z["lat"][:3], [10.0, 11.0, -20.0])                 # chunk by chunk, ncol columns each
    np.testing.assert_allclose(z["lon"][:3], [100.0, 101.0, 200.0])
    assert z["phis"][:3].tolist() == [500.0, 0.0, 0.0] and z["host"].tolist()[1] == "node-b"


def test_the_viewer_aggregates_actions_steps_and_ranks(tmp_path: Path) -> None:
    directory = tmp_path / "t"
    for rank, slow in ((0, 1.0), (1, 3.0)):
        recorder = TimelineRecorder(directory, rank=rank, size=2, flush_every=100)
        recorder.start()
        recorder._origin = 0.0
        for step in range(2):
            base = 10.0 * step
            recorder.record(step, "coupling.boundary_import", KIND_ACTION, base + 0.0, base + 1.0)
            recorder.record(step, "cam_run1.deep_convection", KIND_ACTION, base + 1.0, base + 1.0 + slow)
            recorder.record(step, "coupling.boundary_export", KIND_ACTION, base + 1.0 + slow, base + 6.0)
            recorder.record(step, "coupling.boundary_export", KIND_WAIT, base + 1.0 + slow + 0.5, base + 6.0)
            recorder.record(step, "step", KIND_STEP, base, base + 6.0)
        recorder.close()
    data = TimelineData(directory)
    ov = data.overview()
    assert ov["steps"] == [0, 1] and ov["actions"] == ["coupling.boundary_import", "cam_run1.deep_convection", "coupling.boundary_export"]
    deep = ov["actions"].index("cam_run1.deep_convection")
    assert ov["max_ms"][deep] == [3000.0, 3000.0] and ov["mean_ms"][deep] == [2000.0, 2000.0]
    export = ov["actions"].index("coupling.boundary_export")
    assert ov["wait_ms"][export] == [3500.0, 3500.0]                                 # the fast rank waits longer
    assert ov["step_max_ms"] == [6000.0, 6000.0] and ov["total_s"][deep] == 6.0
    view = data.step_view(1)
    assert set(view["rank"]) == {0, 1} and min(view["t0"]) == 0.0 and view["wait"].count(1) == 2
    globe = data.globe_values(None)
    assert globe["ms"][deep] == [1000.0, 3000.0]
    assert data.default_steps() == [0, 1]
    with pytest.raises(KeyError):
        data.step_view(7)


def test_a_snapshot_embeds_the_data_in_one_page(tmp_path: Path) -> None:
    directory = _run(tmp_path, steps=3, flush_every=1)
    page = snapshot_html(TimelineData(directory), steps=[1])
    assert "/*TIMELINE_DATA*/null" not in page and '"steps":{"1":' in page
    out = tmp_path / "view.html"
    assert timeline_main([str(directory), "--html", str(out), "--steps", "0,2"]) == 0
    text = out.read_text()
    assert '"0":{"step":0' in text and '"2":{"step":2' in text and "<title>freeCAM Timeline</title>" in text


def test_the_server_answers_only_with_its_token(tmp_path: Path) -> None:
    from http.server import ThreadingHTTPServer

    from freecam.pi_cam import timeline_view

    directory = _run(tmp_path, steps=2, flush_every=1)
    handler = type("H", (timeline_view._Handler,), {"data": TimelineData(directory), "token": "tok", "lock": threading.Lock()})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(f"{base}/api/overview")
        assert refused.value.code == 403
        overview = json.loads(urllib.request.urlopen(f"{base}/api/overview?token=tok").read())
        assert overview["steps"] == [0, 1]
        step = json.loads(urllib.request.urlopen(f"{base}/api/step?token=tok&n=1").read())
        assert step["step"] == 1 and len(step["rank"]) > 40
        globe = json.loads(urllib.request.urlopen(f"{base}/api/globe?token=tok&step=all").read())
        assert len(globe["ms"]) == len(overview["actions"])
        page = urllib.request.urlopen(f"{base}/?token=tok").read().decode()
        assert "freeCAM timeline" in page
    finally:
        server.shutdown()
        server.server_close()


def test_the_notebook_driver_passes_its_timeline_to_the_rank_workers(tmp_path: Path) -> None:
    import inspect

    from freecam.pi_cam import facade, session

    # the session appends the flags to the worker command only when a timeline is asked for
    source = inspect.getsource(session.PICAMNotebookSession)
    assert '"--timeline-dir", str(self.timeline_dir)' in source and "if self.timeline_dir is not None" in source
    # the worker attaches the recorder before it initializes (read as text: it imports MPI)
    worker = (Path(session.__file__).parent / "session_worker.py").read_text()
    assert worker.index("attach_timeline") < worker.index("driver.initialize()")
    # the Driver resolves True to the run directory's timeline and passes it on
    driver = facade.Driver.__new__(facade.Driver)
    driver.timeline = False
    assert driver.timeline_dir is None
    driver.timeline = tmp_path / "elsewhere"
    assert driver.timeline_dir == (tmp_path / "elsewhere").resolve()
    assert "timeline_dir" in inspect.getsource(facade.Driver._live_session)
