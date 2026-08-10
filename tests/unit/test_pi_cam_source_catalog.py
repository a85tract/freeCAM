import ctypes
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import yaml

from freecam.core.fortran_adapter import PointerTableAdapter
from freecam.pi_cam.errors import PICAMConfigurationError
from freecam.pi_cam.source_catalog import (
    PICAMKernelRules,
    PICAMSourceCatalog,
)


def _write_rules(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind_map": {"real64": "float64"},
                "dimension_aliases": {"ncol": "local_columns"},
                "rules": [
                    {
                        "id": "heat-is-a-process",
                        "match": {"name_regex": "^heat$"},
                        "set": {"role": "process"},
                    }
                ],
                "overrides": {},
            },
            sort_keys=False,
        )
    )
    return path


def _write_source(path: Path) -> Path:
    path.parent.mkdir(parents=True)
    path.write_text(
        """module toy_physics
  use iso_fortran_env, only: real64
contains
  subroutine heat(temperature, ncol)
    real(kind=real64), intent(inout) :: temperature(ncol)
    integer, intent(in) :: ncol
    call limiter(temperature)
  end subroutine heat

  subroutine limiter(temperature)
    real(kind=real64), intent(inout) :: temperature(:)
  end subroutine limiter
end module toy_physics
"""
    )
    return path


def test_source_catalog_scans_every_procedure_and_resolves_calls(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    _write_source(physics / "toy_physics.F90")
    (physics / "constants.F90").write_text("module constants\nend module constants\n")
    rules = _write_rules(tmp_path / "rules.yaml")

    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=rules,
        scan_roots=(physics,),
    )

    assert catalog.summary()["source_files"] == 2
    assert catalog.summary()["parsed_files"] == 2
    assert catalog.summary()["procedures"] == 2
    procedures = {item.name: item for item in catalog.procedures}
    assert procedures["heat"].role == "process"
    assert procedures["heat"].arguments[0].dtype == "float64"
    assert procedures["heat"].arguments[0].dimensions == ("local_columns",)
    assert procedures["heat"].resolved_calls == ("toy_physics::limiter",)
    assert procedures["limiter"].qualified_name == "toy_physics::limiter"


def test_catalog_writes_one_descriptor_per_parsed_procedure(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    _write_source(physics / "toy_physics.F90")
    rules = _write_rules(tmp_path / "rules.yaml")
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=rules,
        scan_roots=(physics,),
    )

    outputs = catalog.write_descriptors(tmp_path / "catalog")

    assert len(outputs) == 2
    descriptor = yaml.safe_load(
        (tmp_path / "catalog/toy_physics__heat/kernel.yaml").read_text()
    )
    assert descriptor["name"] == "heat"
    assert descriptor["source"] == "components/cam/src/physics/toy_physics.F90"
    assert descriptor["arguments"][0]["dimensions"] == ["local_columns"]
    adapter = tmp_path / "catalog/toy_physics__heat/adapter.F90"
    manifest = tmp_path / "catalog/toy_physics__heat/adapter.json"
    assert adapter.is_file()
    assert manifest.is_file()
    source = adapter.read_text()
    assert "use toy_physics, only: heat" in source
    assert "call c_f_pointer" in source
    assert "call heat(arg_temperature, arg_ncol)" in source
    stale = tmp_path / "catalog/stale_kernel"
    stale.mkdir()
    (stale / "kernel.yaml").write_text("stale: true\n")
    (stale / "adapter.F90").write_text("stale\n")
    (stale / "adapter.json").write_text("{}\n")
    catalog.write_descriptors(tmp_path / "catalog", clean=True)
    assert not stale.exists()


def test_generated_pointer_adapter_compiles_with_original_module(tmp_path: Path) -> None:
    compiler = Path("/opt/cray/pe/gcc/12.2.0/bin/gfortran")
    executable = str(compiler) if compiler.is_file() else shutil.which("gfortran")
    if executable is None:
        pytest.skip("gfortran is unavailable")
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    original = _write_source(physics / "toy_physics.F90")
    rules = _write_rules(tmp_path / "rules.yaml")
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=rules,
        scan_roots=(physics,),
    )
    catalog.write_descriptors(tmp_path / "catalog")

    library = tmp_path / "libtoy_adapter.so"
    subprocess.run(
        [
            executable,
            "-fPIC",
            "-shared",
            "-ffree-line-length-none",
            str(original),
            str(tmp_path / "catalog/toy_physics__heat/adapter.F90"),
            "-o",
            str(library),
        ],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert library.is_file()


def test_generated_pointer_adapter_compiles_for_external_routine(
    tmp_path: Path,
) -> None:
    compiler = Path("/opt/cray/pe/gcc/12.2.0/bin/gfortran")
    executable = str(compiler) if compiler.is_file() else shutil.which("gfortran")
    if executable is None:
        pytest.skip("gfortran is unavailable")
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    physics.mkdir(parents=True)
    original = physics / "scale.F90"
    original.write_text(
        """subroutine scale(values, count)
  use iso_fortran_env, only: real64
  integer, intent(in) :: count
  real(kind=real64), intent(inout) :: values(count)
  values = values * 2.0_real64
end subroutine scale
"""
    )
    rules = _write_rules(tmp_path / "rules.yaml")
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=rules,
        scan_roots=(physics,),
    )
    catalog.write_descriptors(tmp_path / "catalog")
    adapter = next((tmp_path / "catalog").rglob("adapter.F90"))

    library = tmp_path / "libscale_adapter.so"
    subprocess.run(
        [
            executable,
            "-fPIC",
            "-shared",
            "-ffree-line-length-none",
            str(original),
            str(adapter),
            "-o",
            str(library),
        ],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert library.is_file()
    manifest = json.loads(next((tmp_path / "catalog").rglob("adapter.json")).read_text())
    adapter_runtime = PointerTableAdapter(
        ctypes.CDLL(str(library)),
        {
            "scale": {
                "symbol": manifest["symbol"],
                "action_id": manifest["action_id"],
                "arguments": (
                    {"field": "values", "dtype": "float64", "rank": 1},
                    {"field": "count", "dtype": "int32", "rank": 0},
                ),
            }
        },
        library_name=str(library),
    )
    values = np.asfortranarray(np.array([1.0, 2.0, 3.0]))
    count = np.asarray(3, dtype=np.int32)
    adapter_runtime.call(
        "scale", {"values": values, "count": count}, fcomm=0
    )
    assert np.array_equal(values, np.array([2.0, 4.0, 6.0]))


def test_complex_kind_maps_to_c_interoperable_complex_dtype(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    physics.mkdir(parents=True)
    (physics / "complex_kernel.F90").write_text(
        """module complex_kernel
  use iso_fortran_env, only: real64
contains
  subroutine rotate(value)
    complex(kind=real64), intent(inout) :: value
    value = value * (0.0_real64, 1.0_real64)
  end subroutine rotate
end module complex_kernel
"""
    )
    rules = _write_rules(tmp_path / "rules.yaml")
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=rules,
        scan_roots=(physics,),
    )

    procedure = catalog.procedure("complex_kernel::rotate")
    assert procedure.arguments[0].dtype == "complex128"
    catalog.write_descriptors(tmp_path / "catalog")
    assert next((tmp_path / "catalog").rglob("adapter.F90")).is_file()


def test_parse_failures_are_reported_instead_of_silently_skipped(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    physics.mkdir(parents=True)
    (physics / "broken.F90").write_text(
        "subroutine visible_name(value)\n this is not Fortran\n"
    )
    rules = _write_rules(tmp_path / "rules.yaml")

    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=rules,
        scan_roots=(physics,),
    )

    assert catalog.summary()["source_files"] == 1
    assert catalog.summary()["parsed_files"] == 0
    assert catalog.summary()["parse_failed_files"] == 1
    assert catalog.failures[0].fallback_procedures == ("visible_name",)


def test_cpp_recovery_keeps_inactive_branch_names_visible(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    physics.mkdir(parents=True)
    (physics / "conditional.F90").write_text(
        """module conditional
contains
#ifdef ENABLE_FIRST
subroutine first_path(value)
  real, USE_CONTIGUOUS intent(inout) :: value
end subroutine first_path
#else
subroutine second_path(value)
  real, USE_CONTIGUOUS intent(inout) :: value
end subroutine second_path
#endif
end module conditional
"""
    )
    rules = _write_rules(tmp_path / "rules.yaml")

    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=rules,
        scan_roots=(physics,),
    )

    procedures = {item.name: item for item in catalog.procedures}
    assert set(procedures) == {"first_path", "second_path"}
    assert procedures["second_path"].parser == "fparser-cpp"
    assert procedures["first_path"].parser == "fallback-regex"
    assert catalog.failures[0].fallback_procedures == ("first_path",)


def test_duplicate_module_procedures_from_alternative_packages_do_not_collide(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    _write_source(physics / "package_a/toy_physics.F90")
    _write_source(physics / "package_b/toy_physics.F90")
    rules = _write_rules(tmp_path / "rules.yaml")
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=rules,
        scan_roots=(physics,),
    )

    catalog.write_descriptors(tmp_path / "catalog")

    heat_descriptors = tuple(
        path
        for path in (tmp_path / "catalog").rglob("kernel.yaml")
        if yaml.safe_load(path.read_text())["name"] == "heat"
    )
    assert len(heat_descriptors) == 2
    assert heat_descriptors[0].parent != heat_descriptors[1].parent
    with pytest.raises(PICAMConfigurationError, match="ambiguous"):
        catalog.procedure("heat")


def test_rule_schema_rejects_unknown_set_keys(tmp_path: Path) -> None:
    rules_path = _write_rules(tmp_path / "rules.yaml")
    payload = yaml.safe_load(rules_path.read_text())
    payload["rules"][0]["set"] = {"invented_status": "wrong"}
    rules_path.write_text(yaml.safe_dump(payload))
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    _write_source(physics / "toy_physics.F90")

    with pytest.raises(PICAMConfigurationError, match="unknown values"):
        PICAMSourceCatalog.discover(
            tmp_path,
            source_root=source_root,
            rules_path=rules_path,
            scan_roots=(physics,),
        )
