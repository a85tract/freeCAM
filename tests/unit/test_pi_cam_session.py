import json
from pathlib import Path

from freecam.pi_cam import session as session_module
from freecam.pi_cam.session import PICAMNotebookSession, _authkey_argument


def _session_files(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    math_library = tmp_path / "libimf.so"
    math_library.touch()
    manifest = tmp_path / "native.json"
    manifest.write_text(json.dumps({"intel_math_library": str(math_library)}))
    config = tmp_path / "pi_cam.yaml"
    config.write_text(
        "\n".join(
            (
                "case_name: test",
                f"source_root: {tmp_path}",
                "mpi_size: 1",
                f"native_manifest: {manifest}",
            )
        )
        + "\n"
    )
    boundary = tmp_path / "boundary"
    boundary.mkdir()
    (boundary / "manifest.json").write_text("{}\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "atm_in").write_text("\n")
    env_script = tmp_path / "env.sh"
    env_script.write_text("true\n")
    return config, boundary, run_dir, env_script, math_library


def test_authkey_argument_is_unambiguous_when_base64_starts_with_dash() -> None:
    argument = _authkey_argument(bytes([248]) * 32)

    assert argument.startswith("--authkey=-")


def test_session_environment_preloads_manifest_math_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    config, boundary, run_dir, env_script, math_library = _session_files(tmp_path)
    session = PICAMNotebookSession(
        config,
        boundary=boundary,
        run_dir=run_dir,
        env_script=env_script,
    )
    monkeypatch.setattr(
        session_module.subprocess,
        "check_output",
        lambda *args, **kwargs: b"LD_LIBRARY_PATH=/mpi\0LD_PRELOAD=/other.so\0",
    )
    monkeypatch.setattr(
        session_module,
        "mpi_loader_environment",
        lambda environment: environment,
    )

    environment = session._environment()

    assert environment["LD_PRELOAD"].split(":") == [
        str(math_library),
        "/other.so",
    ]


def test_session_run_kernel_sends_explicit_worker_command(
    tmp_path: Path, monkeypatch
) -> None:
    config, boundary, run_dir, env_script, _ = _session_files(tmp_path)
    session = PICAMNotebookSession(
        config,
        boundary=boundary,
        run_dir=run_dir,
        env_script=env_script,
    )
    commands = []
    monkeypatch.setattr(
        session,
        "_request",
        lambda command: commands.append(command) or {"operation": "dadadj"},
    )

    result = session.run_kernel("dadadj")

    assert commands == [{"op": "run_kernel", "name": "dadadj"}]
    assert result["operation"] == "dadadj"


def test_session_run_action_sends_scheme_or_runtime_process_command(
    tmp_path: Path, monkeypatch
) -> None:
    config, boundary, run_dir, env_script, _ = _session_files(tmp_path)
    session = PICAMNotebookSession(
        config,
        boundary=boundary,
        run_dir=run_dir,
        env_script=env_script,
    )
    commands = []
    monkeypatch.setattr(
        session,
        "_request",
        lambda command: commands.append(command) or {"operation": "heating"},
    )

    result = session.run_action("heating", phase="cam_run1")

    assert commands == [
        {"op": "run_action", "name": "heating", "phase": "cam_run1"}
    ]
    assert result["operation"] == "heating"


def test_session_can_request_ordered_cam_run1_leaf_expansion(
    tmp_path: Path, monkeypatch
) -> None:
    config, boundary, run_dir, env_script, _ = _session_files(tmp_path)
    session = PICAMNotebookSession(
        config,
        boundary=boundary,
        run_dir=run_dir,
        env_script=env_script,
    )
    commands = []
    monkeypatch.setattr(
        session,
        "_request",
        lambda command: commands.append(command) or ({"name": "leaf"},),
    )

    result = session.expand_cam_run1_leaves()

    assert commands == [{"op": "expand_cam_run1_leaves"}]
    assert result == ({"name": "leaf"},)


def test_session_dynamic_field_and_python_process_commands(
    tmp_path: Path, monkeypatch
) -> None:
    config, boundary, run_dir, env_script, _ = _session_files(tmp_path)
    session = PICAMNotebookSession(
        config,
        boundary=boundary,
        run_dir=run_dir,
        env_script=env_script,
    )
    commands = []
    monkeypatch.setattr(
        session,
        "_request",
        lambda command: commands.append(command) or {"name": command.get("name", "p")},
    )

    session.create_field(
        "experiment_tracer",
        dimensions=("pcols", "pver"),
        aliases=("tracer",),
    )

    def callback(fields, context):
        fields["tracer"][...] += context.timestep_seconds

    session.install_python(
        callback,
        name="heating",
        phase="cam_run1",
        after="dadadj",
        writes=("tracer",),
    )

    assert commands[0]["op"] == "create_field"
    assert commands[0]["spec"]["dynamic"] is True
    assert commands[1]["op"] == "install_python"
    assert commands[1]["spec"]["group"] == "cam_run1"
    assert commands[1]["spec"]["after"] == "dadadj"
