from freecam.pi_cam.cli import _process_memory


def test_process_memory_reports_rank_local_bytes() -> None:
    sample = _process_memory("initialized", 7)

    assert sample["label"] == "initialized"
    assert sample["step"] == 7
    assert int(sample["ru_maxrss_bytes"]) > 0
    if "VmRSS_bytes" in sample:
        assert int(sample["VmRSS_bytes"]) > 0
