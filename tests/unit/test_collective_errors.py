from freecam.model.collective import collective_error_message


def test_collective_errors_group_identical_rank_tracebacks() -> None:
    errors = ["Traceback:\nKeyError: T"] * 512

    message = collective_error_message("Python process 'heating'", errors)

    assert message is not None
    assert "512/512 MPI ranks" in message
    assert "ranks 0-511" in message
    assert message.count("KeyError: T") == 1


def test_collective_errors_keep_distinct_failures_separate() -> None:
    message = collective_error_message(
        "state update",
        [None, "ValueError: shape", "ValueError: shape", "KeyError: field"],
    )

    assert message is not None
    assert "ranks 1-2" in message
    assert "rank 3" in message
    assert "2 distinct errors" in message
