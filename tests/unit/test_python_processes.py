from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pycam_sima import (
    InstalledPythonProcess,
    InstallPythonProcess,
    ObserveFields,
    PythonProcessContext,
    PythonProcessSpec,
    RemovePythonProcess,
    RunScheme,
    SegmentPlan,
)
from pycam_sima.model import (
    CCPPSuitePlan,
    ModelConfig,
    ModelSnapshot,
    execute_segment_plan,
)
from pycam_sima.model.checkpoint import (
    _rebind_runtime_local_fields,
    deserialize_snapshot,
    read_checkpoint,
    serialize_snapshot,
    write_checkpoint,
)
from pycam_sima.model.clock import ModelClock
from pycam_sima.model.comm import SerialComm
from pycam_sima.model.devices import DeviceRegistry
from pycam_sima.model.errors import (
    PythonProcessContractError,
    PythonProcessExecutionError,
    PythonProcessTaintedError,
)
from pycam_sima.model.grid import dimensions_for_rank
from pycam_sima.model.processes import ProcessRouter
from pycam_sima.model.python_processes import PythonProcessRegistry
from pycam_sima.model.state import StatePool
from pycam_sima.model.user_api import PhysicsCollection


ROOT = Path(__file__).resolve().parents[2]
KESSLER_SUITE = (
    ROOT / "external/CAM-SIMA/src/physics/ncar_ccpp/suites/suite_kessler.xml"
)


class _Driver:
    def __init__(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty-devices"
        empty.mkdir(parents=True)
        self.pool = StatePool(dimensions_for_rank(0, 24))
        self.pool.seal_static()
        self.comm = SerialComm()
        self.clock = ModelClock()
        self.state = SimpleNamespace(value="RUNNING")
        self.config = ModelConfig()
        self.scheme_plan = CCPPSuitePlan.from_xml(KESSLER_SUITE)
        self.backend = SimpleNamespace(
            devices=DeviceRegistry(empty),
            call_count=0,
        )
        self._last_phase = "physics_timestep_initial"
        self._last_scheme = None
        self._last_scheme_group = None
        self._boundary_index = 0
        self._native_call_depth = 0
        self._after_coupler_prepared = False
        self.python_processes = PythonProcessRegistry(self)
        self.processes = ProcessRouter(
            devices=self.backend.devices,
            native_invoke=lambda _name, _pool: None,
            runtime_processes=self.python_processes,
        )
        self.physics = PhysicsCollection(self)

    @property
    def execution_cursor(self):
        return (
            self.state.value,
            self.clock.nstep,
            self._last_phase,
            self._last_scheme,
            self._last_scheme_group,
            self._boundary_index,
            self._native_call_depth,
        )

    def install_python_process(self, spec, *, unsafe=False):
        return self.python_processes.install(spec, unsafe=unsafe)

    def remove_python_process(self, name):
        return self.python_processes.remove(name)

    def set_python_process_parameters(self, name, parameters):
        return self.python_processes.set_parameters(name, parameters)

    def run_scheme(self, name, *, group=None, parameters=None):
        scheme = self.scheme_plan.scheme(name, group=group)
        if parameters is None:
            self.processes.invoke(scheme, self.pool)
        else:
            self.python_processes.invoke(scheme, self.pool, parameters=parameters)
        self._last_scheme = scheme.key
        self._last_scheme_group = self.scheme_plan.execution_group(scheme.key)
        self._boundary_index += 1


def _add_one(fields, context):
    assert context.rank == 0
    assert context.size == 1
    assert context.timestep_seconds == 1800
    fields["air_temperature"][...] += 1.0


def test_spec_round_trip_and_segment_plan_serialization() -> None:
    offset = 1.25

    def closure(fields, context):
        fields["air_temperature"][...] += offset

    spec = PythonProcessSpec.from_callable(
        closure,
        name="closure_heating",
        after="kessler",
        writes=("air_temperature",),
    )
    restored = PythonProcessSpec.from_mapping(spec.as_dict())
    plan = SegmentPlan(
        "python-process",
        (
            InstallPythonProcess(restored),
            RemovePythonProcess(restored.name),
        ),
        unsafe=True,
    )

    assert restored.payload == spec.payload
    assert restored.payload_hash == spec.payload_hash
    assert SegmentPlan.from_mapping(plan.as_dict()).as_dict() == plan.as_dict()


def test_segment_plan_installs_runs_observes_and_removes_python_process(
    tmp_path: Path,
) -> None:
    driver = _Driver(tmp_path)
    before = driver.pool.get("physics_air_temperature").copy()
    spec = PythonProcessSpec.from_callable(
        _add_one,
        name="planned_heating",
        after="kessler",
        writes=("air_temperature",),
    )
    plan = SegmentPlan(
        "planned-python-process",
        (
            InstallPythonProcess(spec),
            RunScheme("planned_heating", "physics_before_coupler"),
            ObserveFields(("physics_air_temperature",)),
            RemovePythonProcess("planned_heating"),
        ),
        unsafe=True,
    )

    trace = execute_segment_plan(driver, plan)

    assert tuple(row["type"] for row in trace) == (
        "install_python_process",
        "run_scheme",
        "observe_fields",
        "remove_python_process",
    )
    assert np.array_equal(
        driver.pool.get("physics_air_temperature"),
        np.add(before, 1.0),
    )
    assert trace[2]["observations"][0]["field"] == ("physics_air_temperature")
    assert not driver.python_processes.installed


def test_spec_rejects_bad_signature_and_large_capture() -> None:
    with pytest.raises(PythonProcessContractError, match="exactly"):
        PythonProcessSpec.from_callable(lambda fields: None, name="bad")

    def extra_positional(fields, context, scale=1.0):
        del fields, context, scale

    with pytest.raises(PythonProcessContractError, match="exactly"):
        PythonProcessSpec.from_callable(extra_positional, name="extra-option")

    captured = b"x" * 2048

    def oversized(fields, context):
        del fields, context
        assert captured

    with pytest.raises(PythonProcessContractError, match="limit"):
        PythonProcessSpec.from_callable(
            oversized,
            name="oversized",
            max_payload_bytes=256,
        )


def test_keyword_parameters_support_defaults_updates_and_call_overrides(
    tmp_path: Path,
) -> None:
    driver = _Driver(tmp_path)
    values = driver.pool.get("physics_air_temperature")
    values[...] = 240.0

    def parameter_heating(fields, context, *, increment, scale=1.0):
        del context
        fields["air_temperature"][...] += increment * scale

    process = driver.physics.install_python(
        parameter_heating,
        name="parameter_heating",
        writes=("air_temperature",),
        parameters={"increment": 1.0, "scale": 2.0},
    )

    assert dict(process.parameters) == {"increment": 1.0, "scale": 2.0}
    process.run()
    assert np.array_equal(values, np.full_like(values, 242.0))

    process.run(increment=3.0)
    assert np.array_equal(values, np.full_like(values, 248.0))
    assert process.parameters["increment"] == 1.0

    process.parameters["increment"] = 0.5
    process.parameters.update(scale=4.0)
    assert dict(process.parameters) == {"increment": 0.5, "scale": 4.0}
    assert driver.python_processes.installed["parameter_heating"].spec.parameters == {
        "increment": 0.5,
        "scale": 4.0,
    }
    process.run()
    assert np.array_equal(values, np.full_like(values, 250.0))

    with pytest.raises(PythonProcessContractError, match="exactly"):
        process.run(unknown=1.0)
    with pytest.raises(TypeError, match="cannot be deleted"):
        del process.parameters["increment"]


def test_parameter_contract_rejects_missing_large_and_non_json_values() -> None:
    def parameter_heating(fields, context, *, increment):
        del fields, context, increment

    with pytest.raises(PythonProcessContractError, match="exactly"):
        PythonProcessSpec.from_callable(
            parameter_heating,
            name="missing-parameter",
        )
    with pytest.raises(PythonProcessContractError, match="unsupported type"):
        PythonProcessSpec.from_callable(
            parameter_heating,
            name="array-parameter",
            parameters={"increment": np.ones(2)},
        )
    with pytest.raises(PythonProcessContractError, match="limit"):
        PythonProcessSpec.from_callable(
            parameter_heating,
            name="large-parameter",
            parameters={"increment": "x" * 256},
            max_parameter_bytes=64,
        )


def test_process_context_is_read_only_and_complete() -> None:
    context = PythonProcessContext(
        process_name="custom_heating",
        group="physics_before_coupler",
        rank=3,
        size=24,
        step=7,
        timestep_seconds=1800,
        year=1,
        month=2,
        day=3,
        seconds=3600,
        calendar="NO_LEAP",
    )

    assert context.date == (1, 2, 3)
    assert context.process_name == "custom_heating"
    assert context.group == "physics_before_coupler"
    assert context.rank == 3
    assert context.size == 24
    with pytest.raises(AttributeError):
        context.step = 8


def test_install_run_control_and_remove_python_process(tmp_path: Path) -> None:
    driver = _Driver(tmp_path)
    before = driver.pool.get("physics_air_temperature").copy()

    process = driver.physics.install_python(
        _add_one,
        name="custom_heating",
        after="kessler",
        writes=("air_temperature",),
    )

    assert isinstance(process, InstalledPythonProcess)
    assert process.name == "custom_heating"
    assert driver.scheme_plan.scheme("custom_heating").implementation == (
        "python-runtime-process"
    )
    process.run()
    assert np.array_equal(
        driver.pool.get("physics_air_temperature"),
        np.add(before, 1.0),
    )
    process.disable()
    assert not driver.scheme_plan.scheme("custom_heating").enabled
    process.enable()
    assert driver.scheme_plan.scheme("custom_heating").enabled
    process.move(before="kessler")
    names = tuple(
        row["name"] for row in driver.scheme_plan.describe("physics_before_coupler")
    )
    assert names.index("custom_heating") < names.index("kessler")
    removed = driver.physics.remove_python("custom_heating")
    assert removed["name"] == "custom_heating"
    assert not driver.python_processes.installed
    with pytest.raises(ValueError, match="unknown scheme"):
        driver.scheme_plan.scheme("custom_heating")


def test_fields_resolve_canonical_alias_and_ccpp_names(
    tmp_path: Path,
) -> None:
    driver = _Driver(tmp_path)

    def add_to_declared_field(fields, context):
        del context
        only_name = next(iter(fields))
        fields[only_name][...] += 1.0

    cases = (
        ("canonical", "field:air_temperature", "air_temperature"),
        ("alias", "phys_t", "physics_air_temperature"),
        ("ccpp", "ccpp:air_temperature", "physics_air_temperature"),
    )
    for process_name, exposed_name, resolved_name in cases:
        before = driver.pool.get(resolved_name).copy()
        process = driver.physics.install_python(
            add_to_declared_field,
            name=process_name,
            writes=(exposed_name,),
        )
        process.run()
        assert np.array_equal(
            driver.pool.get(resolved_name),
            np.add(before, 1.0),
        )
        process.remove()


def test_read_fields_are_read_only_and_undeclared_fields_are_hidden(
    tmp_path: Path,
) -> None:
    driver = _Driver(tmp_path)

    def mutate_read(fields, context):
        del context
        fields["air_temperature"][...] = 0.0

    process = driver.physics.install_python(
        mutate_read,
        name="mutate_read",
        reads=("air_temperature",),
    )
    with pytest.raises(PythonProcessExecutionError, match="read-only"):
        process.run()

    def reenable_read(fields, context):
        del context
        values = fields["air_temperature"]
        values.flags.writeable = True

    protected = driver.physics.install_python(
        reenable_read,
        name="reenable_read",
        reads=("air_temperature",),
    )
    with pytest.raises(PythonProcessExecutionError, match="WRITEABLE"):
        protected.run()

    def undeclared(fields, context):
        del context
        fields["surface_geopotential"]

    hidden = driver.physics.install_python(
        undeclared,
        name="undeclared",
        reads=("air_temperature",),
    )
    with pytest.raises(PythonProcessExecutionError, match="was not declared"):
        hidden.run()

    def returns_value(fields, context):
        del fields, context
        return 1

    non_none = driver.physics.install_python(
        returns_value,
        name="returns_value",
    )
    with pytest.raises(PythonProcessExecutionError, match="must return None"):
        non_none.run()


def test_transactional_failure_restores_declared_writes(
    tmp_path: Path,
) -> None:
    driver = _Driver(tmp_path)
    values = driver.pool.get("physics_air_temperature")
    values[...] = 240.0

    def fail_after_write(fields, context):
        del context
        fields["air_temperature"][...] += 5.0
        raise RuntimeError("intentional")

    process = driver.physics.install_python(
        fail_after_write,
        name="transactional_failure",
        writes=("air_temperature",),
    )
    with pytest.raises(PythonProcessExecutionError, match="were restored"):
        process.run()
    assert np.array_equal(values, np.full_like(values, 240.0))


def test_nontransactional_failure_taints_state(tmp_path: Path) -> None:
    driver = _Driver(tmp_path)
    values = driver.pool.get("physics_air_temperature")
    values[...] = 240.0

    def fail_after_write(fields, context):
        del context
        fields["air_temperature"][...] += 5.0
        raise RuntimeError("intentional")

    spec = PythonProcessSpec.from_callable(
        fail_after_write,
        name="unsafe_failure",
        writes=("air_temperature",),
        transactional=False,
    )
    with pytest.raises(ValueError, match="unsafe=True"):
        driver.install_python_process(spec)
    with pytest.raises(ValueError, match="unsafe=True"):
        driver.physics.install_python(
            fail_after_write,
            name="unsafe_high_level",
            writes=("air_temperature",),
            transactional=False,
        )
    driver.install_python_process(spec, unsafe=True)
    with pytest.raises(PythonProcessTaintedError, match="tainted"):
        driver.run_scheme("unsafe_failure")
    assert np.array_equal(values, np.full_like(values, 245.0))


def test_snapshot_preserves_python_process_inventory(tmp_path: Path) -> None:
    driver = _Driver(tmp_path)

    def parameter_heating(fields, context, *, increment):
        del context
        fields["air_temperature"][...] += increment

    process = driver.physics.install_python(
        parameter_heating,
        name="checkpoint_heating",
        after="kessler",
        writes=("air_temperature",),
        parameters={"increment": 1.25},
    )
    process.parameters["increment"] = 0.5
    process.disable()

    snapshot = deserialize_snapshot(*serialize_snapshot(ModelSnapshot.capture(driver)))
    assert snapshot.python_process_inventory[0]["spec"]["name"] == (
        "checkpoint_heating"
    )
    assert snapshot.python_process_inventory[0]["spec"]["enabled"] is False
    assert snapshot.python_process_inventory[0]["spec"]["parameters"] == {
        "increment": 0.5
    }
    assert snapshot.python_process_inventory[0]["placement"]["group"] == (
        "physics_before_coupler"
    )

    restored = _Driver(tmp_path / "restored")
    restored.pool = snapshot.new_pool()
    restored.scheme_plan = CCPPSuitePlan.from_payload(snapshot.scheme_plan)
    restored.python_processes.restore_inventory(snapshot.python_process_inventory)
    assert not restored.scheme_plan.scheme("checkpoint_heating").enabled
    restored.scheme_plan.enable("checkpoint_heating")
    before = restored.pool.get("physics_air_temperature").copy()
    restored.run_scheme("checkpoint_heating")
    assert np.array_equal(
        restored.pool.get("physics_air_temperature"),
        np.add(before, 0.5),
    )

    checkpoint = write_checkpoint(driver, tmp_path / "disk-checkpoint")
    disk_snapshot = read_checkpoint(checkpoint, driver.comm)
    disk_spec = disk_snapshot.python_process_inventory[0]["spec"]
    assert disk_spec["name"] == "checkpoint_heating"
    assert disk_spec["payload_base64"] == (
        snapshot.python_process_inventory[0]["spec"]["payload_base64"]
    )
    assert disk_spec["payload_hash"] == (
        snapshot.python_process_inventory[0]["spec"]["payload_hash"]
    )
    assert disk_spec["parameters"] == {"increment": 0.5}


def test_restore_rebinds_process_local_mpi_fields(tmp_path: Path) -> None:
    del tmp_path
    values = {
        "mpi_communicator": 123,
        "mpi_root": 9,
        "flag_for_mpi_root": False,
    }
    pool = SimpleNamespace(
        ccpp_field_name=lambda name: name,
        set=lambda name, value: values.__setitem__(name, value),
    )
    driver = SimpleNamespace(
        pool=pool,
        comm=SimpleNamespace(rank=0, py2f=lambda: 456),
    )

    _rebind_runtime_local_fields(driver)

    assert values == {
        "mpi_communicator": 456,
        "mpi_root": 0,
        "flag_for_mpi_root": True,
    }
