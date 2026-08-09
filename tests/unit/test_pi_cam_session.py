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


def test_session_can_request_cam_run2_run4_leaf_expansion(
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

    run2 = session.expand_cam_run2_leaves()
    run4 = session.expand_cam_run4_leaves()
    combined = session.expand_cam_run2_run4_leaves()

    assert commands == [
        {"op": "expand_cam_run2_leaves"},
        {"op": "expand_cam_run4_leaves"},
        {"op": "expand_cam_run2_run4_leaves"},
    ]
    assert run2 == ({"name": "leaf"},)
    assert run4 == ({"name": "leaf"},)
    assert combined == ({"name": "leaf"},)


def test_session_exposes_pythonic_fields_physics_phases_and_kernels(
    tmp_path: Path, monkeypatch
) -> None:
    config, boundary, run_dir, env_script, _ = _session_files(tmp_path)
    session = PICAMNotebookSession(
        config,
        boundary=boundary,
        run_dir=run_dir,
        env_script=env_script,
    )
    session._status = {
        "fields": {
            "phys_state.t": {
                "shape": [4, 3],
                "dtype": "<f8",
                "aliases": ["temperature"],
                "standard_name": "air_temperature",
            }
        },
        "kernels": ("dadadj",),
        "step_plan": (
            {
                "phase": "cam_run1",
                "name": "dadadj",
                "operation": "dadadj",
                "kind": "scheme",
                "enabled": True,
            },
            {
                "phase": "cam_run2",
                "name": "rayleigh_friction",
                "operation": "rayleigh_friction_tend",
                "kind": "scheme",
                "enabled": True,
            },
        ),
    }
    commands = []

    def request(command):
        commands.append(command)
        if command["op"] == "field":
            return "rank-local-values"
        if command["op"] == "stats":
            return {"mean": 250.0}
        if command["op"] == "expand_cam_run2_leaves":
            return ({"name": "tracer_tendencies_leaf"},)
        if command["op"] == "step":
            return {**session._status, "step": command["count"]}
        return {"name": command.get("name", "dadadj")}

    monkeypatch.setattr(session, "_request", request)

    assert "temperature" in dir(session.fields)
    assert "dadadj" in dir(session.physics)
    assert "cam_run2" in dir(session.phases)
    assert "dadadj" in dir(session.kernels)
    assert session.fields.temperature.get(rank=2) == "rank-local-values"
    assert session.fields["air_temperature"].stats(rank="global") == {
        "mean": 250.0
    }
    session.physics.dadadj.run()
    session.physics.rayleigh_friction.enabled = False
    assert session.phases.cam_run2.expand() == (
        {"name": "tracer_tendencies_leaf"},
    )
    session.kernels.dadadj.run()
    assert session.advance(steps=2)["step"] == 2
    assert commands == [
        {"op": "field", "name": "phys_state.t", "rank": 2},
        {"op": "stats", "name": "phys_state.t", "rank": "global"},
        {"op": "run_action", "name": "dadadj", "phase": "cam_run1"},
        {
            "op": "set_action_enabled",
            "name": "rayleigh_friction",
            "phase": "cam_run2",
            "enabled": False,
        },
        {"op": "expand_cam_run2_leaves"},
        {"op": "run_kernel", "name": "dadadj"},
        {"op": "step", "count": 2},
    ]


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

    def request(command):
        commands.append(command)
        result = {"name": command.get("name", "p")}
        if command["op"] == "install_python":
            result.update(
                name=command["spec"]["name"],
                phase=command["spec"]["group"],
            )
        return result

    monkeypatch.setattr(session, "_request", request)

    tracer = session.fields.create(
        "experiment_tracer",
        dims=("pcols", "pver"),
        aliases=("tracer",),
    )

    def callback(fields, context):
        fields["tracer"][...] += context.timestep_seconds

    process = session.physics.install_python(
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
    assert tracer.name == "experiment_tracer"
    assert process.name == "heating"
    assert process.phase == "cam_run1"
