import ctypes
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import yaml

import freecam.pi_cam.adapter_validation as adapter_validation
from freecam.core.fortran_adapter import PointerTableAdapter
from freecam.pi_cam.errors import PICAMConfigurationError
from freecam.pi_cam.source_catalog import (
    PICAMKernelRules,
    PICAMSourceCatalog,
)
from freecam.pi_cam.adapter_validation import (
    AdapterBuildContext,
    PICAMAdapterValidator,
    load_adapter_build_contexts,
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


def test_private_module_procedure_does_not_generate_external_adapter(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    physics.mkdir(parents=True)
    (physics / "visibility.F90").write_text(
        """module visibility
  implicit none
  private
  public :: public_kernel
contains
  subroutine public_kernel(value)
    real, intent(inout) :: value
    value = value + 1.0
  end subroutine public_kernel
  subroutine private_kernel(value)
    real, intent(inout) :: value
    value = value - 1.0
  end subroutine private_kernel
end module visibility
"""
    )
    rules = _write_rules(tmp_path / "rules.yaml")
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=rules,
        scan_roots=(physics,),
    )

    public = catalog.procedure("visibility::public_kernel")
    private = catalog.procedure("visibility::private_kernel")
    assert public.adapter_status == "candidate"
    assert "private_module_procedure" not in public.blockers
    assert private.adapter_status == "blocked"
    assert "private_module_procedure" in private.blockers

    output = tmp_path / "catalog"
    catalog.write_descriptors(output)
    assert (output / "visibility__public_kernel/adapter.F90").is_file()
    assert not (output / "visibility__private_kernel/adapter.F90").exists()


def test_rules_apply_build_default_real_and_ast_dimension_shape(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics/cosp"
    physics.mkdir(parents=True)
    (physics / "promoted.F90").write_text(
        """module promoted
contains
  subroutine update(values)
    real, dimension(:, :), intent(inout) :: values
    values = values + 1.0
  end subroutine update
end module promoted
"""
    )
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        """schema_version: 1
kind_map: {}
dimension_aliases: {}
rules:
  - id: promoted-default-real
    match:
      source_regex: /cosp/promoted\\.F90$
    set:
      default_real_dtype: float64
overrides: {}
"""
    )

    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=rules,
        scan_roots=(physics,),
    )
    argument = catalog.procedure("promoted::update").arguments[0]
    assert argument.dtype == "float64"
    assert argument.rank == 2
    assert argument.dimensions == (":", ":")
    output = tmp_path / "catalog"
    catalog.write_descriptors(output)
    assert "real(c_double), pointer :: arg_values(:,:)" in (
        output / "promoted__update/adapter.F90"
    ).read_text()


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


def test_generated_adapter_validator_compiles_resolves_and_smokes_abi_families(
    tmp_path: Path,
) -> None:
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
    descriptors = tmp_path / "catalog"
    catalog.write_descriptors(descriptors)
    heat_descriptor_path = descriptors / "toy_physics__heat/kernel.yaml"
    heat_descriptor = yaml.safe_load(heat_descriptor_path.read_text())
    heat_descriptor["active_plan_actions"] = ["test.heat"]
    heat_descriptor_path.write_text(yaml.safe_dump(heat_descriptor, sort_keys=False))

    module_dir = tmp_path / "modules"
    module_dir.mkdir()
    original_object = module_dir / "toy_physics.o"
    subprocess.run(
        [
            executable,
            "-c",
            "-fPIC",
            "-J",
            str(module_dir),
            str(original),
            "-o",
            str(original_object),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    archive = tmp_path / "libtoy.a"
    subprocess.run(
        ["ar", "rcs", str(archive), str(original_object)],
        check=True,
        capture_output=True,
        text=True,
    )

    report = PICAMAdapterValidator(
        descriptors,
        compiler=executable,
        module_dirs=(module_dir,),
        original_library=archive,
        work_root=tmp_path / "validation-build",
        workers=2,
    ).validate()

    assert report["generated_adapters"] == 2
    assert report["parse_status_counts"] == {"passed": 2}
    assert report["compile_status_counts"] == {"passed": 2}
    assert report["archive_symbol_status_counts"] == {"passed": 2}
    assert report["case_build_gate"]["passed"] is True
    assert report["case_build_gate"]["compiled_and_symbol_resolved"] == 2
    assert report["abi_signature_families"] == 2
    assert report["abi_smoke_status_counts"] == {"passed": 2}
    assert report["active_plan"]["procedures"] == 1
    assert report["active_plan"]["reachable_procedures"] == 2
    assert report["active_plan"]["reachable_generated_adapter_count"] == 2
    assert report["active_plan"]["reachable_generated_build"][
        "compile_status_counts"
    ] == {"passed": 2}
    assert report["scientific_bfb_scope"]["status_counts"] == {
        "required_pending_runtime_trace": 2
    }


def test_generated_adapter_validator_classifies_missing_case_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler = Path("/opt/cray/pe/gcc/12.2.0/bin/gfortran")
    executable = str(compiler) if compiler.is_file() else shutil.which("gfortran")
    if executable is None:
        pytest.skip("gfortran is unavailable")
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
    descriptors = tmp_path / "catalog"
    catalog.write_descriptors(descriptors)

    def fail_parser(path: Path) -> None:
        raise RuntimeError(f"parser tool failed for {path.name}")

    monkeypatch.setattr(adapter_validation, "_parse_fortran", fail_parser)

    report = PICAMAdapterValidator(
        descriptors,
        compiler=executable,
        work_root=tmp_path / "validation-build",
    ).validate()

    assert report["parse_status_counts"] == {"parser_tool_error": 2}
    assert report["compile_status_counts"] == {"failed": 2}
    assert report["failure_kind_counts"] == {"module_not_in_case_build": 2}
    assert report["full_compile_gate"]["passed"] is False
    assert report["full_compile_gate"]["compile_failures"] == 2
    assert report["case_build_gate"]["passed"] is False


def test_generated_adapters_select_matching_real_build_context(
    tmp_path: Path,
) -> None:
    compiler = Path("/opt/cray/pe/gcc/12.2.0/bin/gfortran")
    executable = str(compiler) if compiler.is_file() else shutil.which("gfortran")
    if executable is None:
        pytest.skip("gfortran is unavailable")
    source_root = tmp_path / "source"
    physics = source_root / "components/cam/src/physics"
    physics.mkdir(parents=True)
    sources = []
    for suffix in ("a", "b"):
        source = physics / f"kernel_{suffix}.F90"
        source.write_text(
            f"""module kernel_{suffix}
  use iso_fortran_env, only: real64
contains
  subroutine run_{suffix}(value)
    real(kind=real64), intent(inout) :: value(:)
    value = value + 1.0_real64
  end subroutine run_{suffix}
end module kernel_{suffix}
"""
        )
        sources.append(source)
    catalog = PICAMSourceCatalog.discover(
        tmp_path,
        source_root=source_root,
        rules_path=_write_rules(tmp_path / "rules.yaml"),
        scan_roots=(physics,),
    )
    descriptors = tmp_path / "catalog"
    catalog.write_descriptors(descriptors)

    contexts = []
    for suffix, source in zip(("a", "b"), sources):
        module_dir = tmp_path / f"modules-{suffix}"
        module_dir.mkdir()
        object_path = module_dir / f"kernel_{suffix}.o"
        subprocess.run(
            [
                executable,
                "-c",
                "-fPIC",
                "-J",
                str(module_dir),
                str(source),
                "-o",
                str(object_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        archive = module_dir / f"libkernel_{suffix}.a"
        subprocess.run(
            ["ar", "rcs", str(archive), str(object_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        contexts.append(
            AdapterBuildContext(
                name=f"context-{suffix}",
                module_dirs=(module_dir,),
                original_libraries=(archive,),
                selected_sources=frozenset(
                    {f"components/cam/src/physics/kernel_{suffix}.F90"}
                ),
            )
        )

    report = PICAMAdapterValidator(
        descriptors,
        compiler=executable,
        build_contexts=contexts,
        work_root=tmp_path / "validation-build",
        workers=2,
    ).validate()

    assert report["compile_status_counts"] == {"passed": 2}
    assert report["archive_symbol_status_counts"] == {"passed": 2}
    assert report["full_compile_gate"] == {
        "archive_symbol_failures": 0,
        "compile_failures": 0,
        "compiled_and_symbol_resolved": 2,
        "passed": True,
    }
    selected = {item["name"]: item["selected_context"] for item in report["adapters"]}
    assert selected == {
        "kernel_a::run_a": "context-a",
        "kernel_b::run_b": "context-b",
    }
    assert report["scientific_bfb_scope"]["status_counts"] == {
        "not_exercised": 2
    }


def test_build_context_file_resolves_selected_cam_sources(tmp_path: Path) -> None:
    source_dir = tmp_path / "source/components/cam/src/physics/cam"
    source_dir.mkdir(parents=True)
    (source_dir / "kernel.F90").write_text("module kernel\nend module kernel\n")
    cosp_dir = tmp_path / "source/components/cam/src/physics/cosp"
    llnl_dir = cosp_dir / "llnl"
    llnl_dir.mkdir(parents=True)
    (llnl_dir / "llnl_stats.F90").write_text(
        "module llnl_stats\nend module llnl_stats\n"
    )
    case_root = tmp_path / "case"
    camconf = case_root / "Buildconf/camconf"
    camconf.mkdir(parents=True)
    (camconf / "Filepath").write_text(str(source_dir) + "\n")
    build_root = tmp_path / "build"
    object_dir = build_root / "atm/obj"
    object_dir.mkdir(parents=True)
    (object_dir / "Srcfiles").write_text("kernel.F90\n")
    (object_dir / "kernel.mod").write_bytes(b"module-placeholder")
    (build_root / "lib").mkdir()
    (build_root / "lib/libatm.a").write_bytes(b"archive-placeholder")
    cosp_build = object_dir / "cosp"
    cosp_build.mkdir()
    (cosp_build / "Makefile").write_text(
        f"COSP_PATH := {cosp_dir}\nLLNL_PATH := {llnl_dir}\n"
    )
    member = cosp_build / "llnl_stats.o"
    member.write_bytes(b"object-placeholder")
    subprocess.run(
        ["ar", "rcs", str(cosp_build / "libcosp.a"), str(member)],
        check=True,
        capture_output=True,
        text=True,
    )
    matrix = tmp_path / "contexts.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "contexts": [
                    {
                        "name": "test",
                        "case_root": str(case_root),
                        "build_root": str(build_root),
                    }
                ],
            },
            sort_keys=False,
        )
    )

    (context,) = load_adapter_build_contexts(matrix)
    assert context.selected_sources == frozenset(
        {
            "components/cam/src/physics/cam/kernel.F90",
            "components/cam/src/physics/cosp/llnl/llnl_stats.F90",
        }
    )
    assert context.module_dirs == (object_dir.resolve(),)
    assert context.original_libraries == (
        cosp_build / "libcosp.a",
        (build_root / "lib/libatm.a").resolve(),
    )


def test_build_context_report_hides_personal_glade_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USER", "example_user")
    context = AdapterBuildContext(
        name="portable",
        module_dirs=(Path("/glade/work/example_user/case/bld/atm/obj"),),
        original_libraries=(
            Path("/glade/derecho/scratch/example_user/case/bld/lib/libatm.a"),
        ),
        build_root=Path("/glade/derecho/scratch/example_user/case/bld"),
        case_root=Path("/glade/work/example_user/case"),
    )

    record = context.as_dict()

    assert record["case_root"] == "/glade/work/$USER/case"
    assert record["build_root"] == "/glade/derecho/scratch/$USER/case/bld"
    assert record["module_dirs"] == ["/glade/work/$USER/case/bld/atm/obj"]
    assert record["original_libraries"] == [
        "/glade/derecho/scratch/$USER/case/bld/lib/libatm.a"
    ]


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
