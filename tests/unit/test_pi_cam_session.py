import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from freecam.pi_cam import session as session_module
from freecam.pi_cam.plan import PICAMStepPlan
from freecam.pi_cam.session import (
    PICAMNotebookError,
    PICAMNotebookSession,
    _authkey_argument,
)


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


def test_pbs_submission_reports_qsub_stderr(
    tmp_path: Path, monkeypatch
) -> None:
    config, boundary, run_dir, env_script, _ = _session_files(tmp_path)
    session = PICAMNotebookSession(
        config,
        boundary=boundary,
        run_dir=run_dir,
        env_script=env_script,
        pbs_account="BAD_ACCOUNT",
    )

    def fail_qsub(*args, **kwargs):
        del kwargs
        raise subprocess.CalledProcessError(
            32,
            args[0],
            stderr="qsub: Invalid account for CPU usage",
        )

    monkeypatch.setattr(session_module.subprocess, "run", fail_qsub)

    with pytest.raises(PICAMNotebookError, match="Invalid account for CPU usage"):
        session._submit_pbs(
            ("mpiexec", "python"),
            {"LD_LIBRARY_PATH": "/mpi"},
        )


def test_pbs_submission_requires_an_account_before_qsub(
    tmp_path: Path, monkeypatch
) -> None:
    config, boundary, run_dir, env_script, _ = _session_files(tmp_path)
    session = PICAMNotebookSession(
        config,
        boundary=boundary,
        run_dir=run_dir,
        env_script=env_script,
    )
    called = False

    def unexpected_qsub(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("qsub must not run without an account")

    monkeypatch.setattr(session_module.subprocess, "run", unexpected_qsub)

    with pytest.raises(PICAMNotebookError, match="no PBS account is configured"):
        session._submit_pbs(("mpiexec", "python"), {"LD_LIBRARY_PATH": "/mpi"})
    assert not called


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


def test_session_calls_one_catalog_process_with_automatic_statepool_binding(
    tmp_path: Path, monkeypatch
) -> None:
    config, boundary, run_dir, env_script, _ = _session_files(tmp_path)
    session = PICAMNotebookSession(
        config,
        boundary=boundary,
        run_dir=run_dir,
        env_script=env_script,
    )
    session._status = {"step_plan": PICAMStepPlan.default().describe()}
    commands = []
    promoted = {
        "name": "cloud_fraction_fice",
        "qualified_name": "cloud_fraction::cldfrc_fice",
        "bindings": (
            {
                "argument": "t",
                "field": "phys_state.t",
                "dtype": "float64",
                "intent": "in",
                "local_rank": 2,
                "aggregate_dimensions": ("pcols", "pver", "chunks"),
                "source": "inferred",
            },
            {
                "argument": "fice",
                "field": "process_context.cloud_fraction_fice.fice",
                "dtype": "float64",
                "intent": "out",
                "local_rank": 2,
                "aggregate_dimensions": ("pcols", "pver", "chunks"),
                "source": "promoted",
            },
            {
                "argument": "fsnow",
                "field": "process_context.cloud_fraction_fice.fsnow",
                "dtype": "float64",
                "intent": "out",
                "local_rank": 2,
                "aggregate_dimensions": ("pcols", "pver", "chunks"),
                "source": "promoted",
            },
        ),
        "created_fields": (),
        "created_dimensions": (),
        "native_available": True,
    }

    def request(command):
        commands.append(command)
        if command["op"] == "promote_process":
            return promoted
        if command["op"] == "status":
            return {
                "step_plan": PICAMStepPlan.default().describe(),
                "promoted_processes": (promoted,),
            }
        if command["op"] == "run_promoted_process":
            return {"phase": "promoted_process", "operation": command["name"]}
        if command["op"] == "remove_promoted_process":
            return promoted
        raise AssertionError(command)

    monkeypatch.setattr(session, "_request", request)

    result = session.physics.cloud_fraction_fice()

    assert result.process.runnable is True
    assert result.trace["operation"] == "cloud_fraction_fice"
    assert tuple(result) == ("fice", "fsnow")
    assert result.fice.name == "process_context.cloud_fraction_fice.fice"
    assert commands[:3] == [
        {
            "op": "promote_process",
            "name": "cloud_fraction_fice",
            "bindings": {},
            "initials": {},
            "dimensions": {},
        },
        {"op": "status"},
        {"op": "run_promoted_process", "name": "cloud_fraction_fice"},
    ]


def test_session_process_call_accepts_a_python_field_object(
    tmp_path: Path,
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
                "shape": [16, 30, 1],
                "dtype": "<f8",
                "aliases": ["T"],
            }
        }
    }

    bindings, initials = session._process_call_arguments(
        {"t": session.fields.T}
    )

    assert bindings == {"t": "phys_state.t"}
    assert initials == {}


def test_session_trace_sends_worker_cursor(tmp_path: Path, monkeypatch) -> None:
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
        lambda command: commands.append(command) or ({"name": "dadadj"},),
    )

    assert session.trace(since=3) == ({"name": "dadadj"},)
    assert commands == [{"op": "trace", "since": 3}]


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


def test_session_exposes_pythonic_fields_processes_and_kernels(
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
                "name": "dry_adjustment",
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
    assert session.physics.names[:2] == ("dry_adjustment", "rayleigh_friction")
    assert session.physics.action_names == ("dry_adjustment", "rayleigh_friction")
    assert len(session.physics) > 270
    assert session.physics.coverage["runnable"] == 2
    assert session.physics.coverage["source_reachable"] == 372
    assert session.physics.dry_adjustment.operation == "dadadj"
    assert not hasattr(session.physics, "by_phase")
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
        {"op": "run_kernel", "name": "dadadj"},
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


def test_session_replaces_workflow_and_refreshes_the_live_view(
    tmp_path: Path, monkeypatch
) -> None:
    config, boundary, run_dir, env_script, _ = _session_files(tmp_path)
    session = PICAMNotebookSession(
        config,
        boundary=boundary,
        run_dir=run_dir,
        env_script=env_script,
    )
    session._status = {"step_plan": PICAMStepPlan.default().describe()}
    order = tuple(
        reversed(
            [
                row["phase"] + "." + row["name"]
                for row in session._status["step_plan"]
                if row["enabled"]
            ]
        )
    )
    replacement = tuple(reversed(session._status["step_plan"]))
    commands = []

    def request(command):
        commands.append(command)
        return {"plan": replacement}

    monkeypatch.setattr(session, "_request", request)

    result = session.replace_workflow(order)

    assert commands == [{"op": "replace_workflow", "order": order}]
    assert result["plan"] == replacement
    assert session.status["step_plan"] == tuple(dict(row) for row in replacement)


def test_session_ui_lists_every_supported_pi_cam_physics_interface(
    tmp_path: Path,
) -> None:
    config, boundary, run_dir, env_script, _ = _session_files(tmp_path)
    session = PICAMNotebookSession(
        config,
        boundary=boundary,
        run_dir=run_dir,
        env_script=env_script,
    )
    session._status = {"step_plan": PICAMStepPlan.default().describe()}

    order = session.workflow[:]
    radiation = session.workflow["radiation"]
    vertical_diffusion = session.workflow["vertical_diffusion"]
    assert radiation in order
    order.remove(radiation)
    order.insert(order.index(vertical_diffusion), radiation)
    assert order.index(radiation) < order.index(vertical_diffusion)

    assert len(session.physics) == 298
    assert len(set(session.physics.names)) == 298
    assert session.physics.coverage == {
        "interfaces": 298,
        "runnable": 36,
        "catalog_only": 262,
        "source_reachable": 372,
        "source_catalog": 371,
        "physical_processes": 276,
        "compiled_process_adapters": 276,
        "formerly_catalog_only_interfaces": 262,
        "catalog_adapters_compiled": 262,
        "catalog_current_case_loadable": 0,
        "current_case_loadable": 0,
        "configuration_specific": 276,
        "helper_routines": 95,
        "runtime_overlap": 14,
        "excluded_lifecycle": 1,
        "enabled": 31,
        "disabled": 5,
        "leaf": 15,
        "stage": 21,
    }
    assert session.physics.dadadj.operation == "dadadj"
    assert session.physics.dry_adjustment.operation == "dadadj"
    assert session.physics.leaf_cloud_diagnostics_calc.granularity == "leaf"
    assert sum(process.runnable for process in session.physics) == 36
    assert not hasattr(session.physics, "by_phase")
    assert session.physics.cloud_fraction_fice.runnable is False
    assert session.physics.cldfrc_fice.name == "cloud_fraction_fice"
    assert session.physics.zm_conv_evap.qualified_name == "zm_conv::zm_conv_evap"
    html = session.physics._repr_html_()
    assert "298 flat physics interfaces" in html
    assert "convect_deep_tend" in html
    assert "leaf_cloud_diagnostics_calc" in html


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
    session._status = {"step_plan": PICAMStepPlan.default().describe()}

    tracer = session.fields.create(
        "experiment_tracer",
        dims=("pcols", "pver"),
        aliases=("tracer",),
    )
    relative_humidity = session.fields.create_array("rh", np.zeros(30))

    def callback(fields, context):
        fields["tracer"][...] += context.timestep_seconds

    process = session.physics.install_python(
        callback,
        name="heating",
        after="dadadj",
        writes=("tracer",),
    )

    assert commands[0]["op"] == "create_field"
    assert commands[0]["spec"]["dynamic"] is True
    assert commands[1]["op"] == "create_array"
    assert commands[1]["name"] == "rh"
    assert np.array_equal(commands[1]["values"], np.zeros(30))
    assert commands[2]["op"] == "install_python"
    assert commands[2]["spec"]["group"] == "cam_run1"
    assert commands[2]["spec"]["after"] == "dadadj"
    assert tracer.name == "experiment_tracer"
    assert relative_humidity.name == "rh"
    assert process.name == "heating"
    assert process.phase == "cam_run1"
