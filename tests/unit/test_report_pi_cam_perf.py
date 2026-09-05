from pathlib import Path
import json

from tools import report_pi_cam_perf as rp


def test_stat_csv_counters_are_read_by_event_name() -> None:
    text = "# started on Thu\n\n12345.67,msec,task-clock:u,12345670,100.00,0.98,CPUs utilized\n" \
           "8765,,page-faults:u,12345670,100.00,0.7,K/sec\n<not supported>,,major-faults:u,0,100.00,,\n"
    counters = rp.parse_stat_csv(text)
    assert counters == {"task-clock": 12345.67, "page-faults": 8765.0}


def test_percent_lines_come_out_of_a_report_listing() -> None:
    text = ("# Overhead  Shared Object\n# ........  ............\n#\n"
            "    61.20%  libfreecam_pi_cam.so\n     9.03%  libpython3.11.so.1.0\n"
            "     0.40%  [unknown]\n")
    assert rp.parse_percent_lines(text) == [
        (61.2, "libfreecam_pi_cam.so"), (9.03, "libpython3.11.so.1.0"), (0.4, "[unknown]")]


def test_shared_objects_fall_into_the_step_s_buckets() -> None:
    assert rp.classify_dso("libfreecam_pi_cam.so") == "cam"
    assert rp.classify_dso("libpycesm_external_atm.so") == "coupler+components"
    assert rp.classify_dso("libpython3.11.so.1.0") == "python"
    assert rp.classify_dso("libmpi_intel.so.12") == "mpi"
    assert rp.classify_dso("libc.so.6") == "libc"
    assert rp.classify_dso("libimf.so") == "intel-math"
    assert rp.classify_dso("_multiarray_umath.cpython-311-x86_64-linux-gnu.so") == "numpy"


def test_the_record_aggregates_every_rank_s_counters(tmp_path, monkeypatch) -> None:
    perf = tmp_path / "perf"
    perf.mkdir()
    for rank, faults in ((0, 1000), (1, 3000)):
        (perf / f"stat.{rank}.csv").write_text(
            f"{1000 * (rank + 1)},msec,task-clock:u,1,100.00,1,CPUs\n{faults},,page-faults:u,1,100.00,1,K/sec\n")
    (perf / "record.1.data").write_bytes(b"")
    monkeypatch.setattr(rp, "perf_report", lambda data, sort, limit: (
        [(70.0, "libfreecam_pi_cam.so"), (20.0, "libpython3.11.so.1.0"), (5.0, "libc.so.6")]
        if sort == "dso" else [(3.0, "libc.so.6  [.] malloc")]))
    record = rp.build_record(perf, {"steps": 50, "timing": {"advance_seconds": 15.0}}, {"bfb": True}, "1", "abc")
    assert record["ranks_counted"] == 2
    assert record["page_faults_per_rank"]["max"] == 3000
    assert record["page_faults_per_rank_per_step"]["median"] == 40.0
    assert record["task_clock_seconds_per_rank"]["max"] == 2.0
    assert record["recorded_ranks"]["1"]["by_bucket"] == {"cam": 70.0, "python": 20.0, "libc": 5.0}
    assert record["recorded_ranks"]["1"]["top_symbols"][0]["symbol"] == "libc.so.6  [.] malloc"
    assert record["bfb"] is True and record["git_commit"] == "abc"
    json.dumps(record)


def test_callers_are_read_from_a_caller_graph() -> None:
    text = ("     4.10%  libfreecam_pi_cam.so  [.] __intel_avx_rep_memcpy\n"
            "            |\n"
            "            |--2.30%--physics_types_mp_physics_state_copy_\n"
            "            |          physpkg_mp_tphysbc_\n"
            "            |\n"
            "            |--1.10%--edge_mod_mp_edgevpack_\n"
            "             --0.70%--physics_types_mp_physics_state_copy_\n")
    assert rp.parse_callers(text, 5) == [
        {"percent": 3.0, "caller": "physics_types_mp_physics_state_copy_"},
        {"percent": 1.1, "caller": "edge_mod_mp_edgevpack_"}]
