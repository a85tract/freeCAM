"""Command-line entry points for the Python-owned CAM model."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .core.mpi import world_comm
from .core.runtime_env import ensure_mpi_loader_environment
from .model import (
    BranchSpec,
    CAMDriver,
    ModelConfig,
    SegmentPlan,
    execute_segment_plan,
    read_checkpoint,
    restore_driver,
)
from .model.checkpoint import CHECKPOINT_SCHEMA_VERSION
from .model.device_codegen import (
    DeviceDescription,
    build_device,
    build_device_bundle,
    resolve_source_closure,
)
from .model.device_catalog import DeviceCatalog
from .model.device_support import DeviceSupportMatrix
from .model.errors import DeviceBuildError
from .model.host_services import is_python_host_service_scheme
from .model.orbital_service import build_orbital_host_library
from .model.validation import compare_history_directories


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def command_run(args: argparse.Namespace) -> int:
    """Run the complete CAM model described by a configuration profile."""

    config = ModelConfig.from_yaml(args.config)
    ensure_mpi_loader_environment()
    comm = world_comm()
    if comm.size != config.mpi_size:
        raise SystemExit(
            f"configuration requires {config.mpi_size} MPI ranks, got {comm.size}"
        )
    history_dir = Path(args.history_dir).resolve()
    exists = history_dir.exists() if comm.rank == 0 else None
    if comm.bcast(exists, root=0):
        raise SystemExit(
            f"refusing to replace existing history directory: {history_dir}"
        )

    driver = CAMDriver(
        config,
        run_dir=args.run_dir,
        comm=comm,
        kernel_library=args.library,
        history_dir=history_dir,
    ).start()
    initialized_native_calls = driver.backend.call_count
    initialized_abi_checked = driver.backend._abi_checked
    steps = config.stop_n if args.steps is None else int(args.steps)
    if steps <= 0:
        raise SystemExit("steps must be positive")
    for _ in range(steps):
        driver.step()
    result = driver.stats()
    result.update(
        initialized_native_calls=initialized_native_calls,
        initialized_abi_checked=initialized_abi_checked,
    )
    driver.finalize()
    if comm.rank == 0:
        print(json.dumps(result, sort_keys=True))
    return 0


def command_run_segment(args: argparse.Namespace) -> int:
    """Run one restartable MPI segment without a persistent socket worker."""

    config = ModelConfig.from_yaml(args.config)
    ensure_mpi_loader_environment()
    comm = world_comm()
    if comm.size != config.mpi_size:
        raise SystemExit(
            f"configuration requires {config.mpi_size} MPI ranks, got {comm.size}"
        )

    if args.branch_spec and args.segment_plan:
        raise SystemExit("use --branch-spec or --segment-plan, not both")
    if args.segment_plan:
        payload = _read_json_collective(Path(args.segment_plan), comm)
        plan = SegmentPlan.from_mapping(payload)
        if args.steps is not None:
            raise SystemExit("use actions in --segment-plan or --steps, not both")
    elif args.branch_spec:
        payload = _read_json_collective(Path(args.branch_spec), comm)
        plan = BranchSpec.from_mapping(payload).to_segment_plan()
        if args.steps is not None:
            raise SystemExit("use steps in --branch-spec or --steps, not both")
    else:
        plan = BranchSpec(
            name=args.name,
            steps=1 if args.steps is None else args.steps,
        ).to_segment_plan()

    history_dir = Path(args.history_dir).resolve()
    if args.input_checkpoint:
        snapshot = read_checkpoint(args.input_checkpoint, comm)
        driver = restore_driver(
            snapshot,
            run_dir=args.run_dir,
            comm=comm,
            kernel_library=args.library,
            history_dir=history_dir,
            expected_config=config,
        )
    else:
        exists = history_dir.exists() if comm.rank == 0 else None
        if comm.bcast(exists, root=0):
            raise SystemExit(
                f"refusing to replace existing history directory: {history_dir}"
            )
        driver = CAMDriver(
            config,
            run_dir=args.run_dir,
            comm=comm,
            kernel_library=args.library,
            history_dir=history_dir,
        ).start()

    action_trace = execute_segment_plan(driver, plan)
    checkpoint = driver.write_checkpoint(args.output_checkpoint)
    result = driver.stats()
    result.update(
        branch=plan.name,
        segment_steps=plan.step_count,
        action_count=len(plan.actions),
        action_trace=action_trace,
        segment_plan=plan.as_dict(),
        checkpoint=str(checkpoint),
        checkpoint_schema=CHECKPOINT_SCHEMA_VERSION,
        pbs_job_id=os.environ.get("PBS_JOBID"),
    )
    driver.finalize()
    if comm.rank == 0:
        encoded = json.dumps(result, indent=2, sort_keys=True)
        if args.result_json:
            result_path = Path(args.result_json).resolve()
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(encoded)
        print(encoded)
    return 0


def _read_json_collective(path: Path, comm) -> dict:
    """Read one controller payload once and broadcast it to every MPI rank."""

    payload = None
    error = None
    if comm.rank == 0:
        try:
            payload = json.loads(path.read_text())
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error, root=0)
    if error:
        raise SystemExit(f"cannot read segment payload {path}: {error}")
    payload = comm.bcast(payload, root=0)
    if not isinstance(payload, dict):
        raise SystemExit(f"segment payload must be a JSON object: {path}")
    return payload


def command_build_kernels(args: argparse.Namespace) -> int:
    root = _repo_root()
    config_path = Path(
        getattr(args, "config", None)
        or root / "configs" / "fkessler_model.yaml"
    ).resolve()
    config = ModelConfig.from_yaml(config_path)
    target = Path(
        getattr(args, "target", None)
        or config.default_kernel_library(root)
    ).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    DeviceCatalog.discover(root).write_descriptors(
        root / "devices/generated", clean=True
    )
    subprocess.run(
        [
            "make",
            "-C",
            str(root / "native" / "kernels"),
            f"MODEL_CONFIG={config_path}",
            f"TARGET={target}",
            f"PYTHON={sys.executable}",
            "clean",
            "all",
        ],
        check=True,
    )
    print(
        json.dumps(
            {
                "config": str(config_path),
                "kernel_specialization": config.kernel_specialization,
                "kernel_specialization_id": config.kernel_specialization_id,
                "library": str(target),
            },
            sort_keys=True,
        )
    )
    return 0


def command_build_device(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root or _repo_root() / "build/devices")
    for descriptor in args.descriptors:
        manifest = build_device(
            descriptor,
            project_root=_repo_root(),
            output_root=output_root,
            compiler=args.compiler,
            fflags=shlex.split(args.fflags),
            ldflags=shlex.split(args.ldflags),
        )
        print(manifest)
    return 0


def command_build_device_bundle(args: argparse.Namespace) -> int:
    output_root = Path(
        args.output_root or _repo_root() / "build/catalog_devices"
    )
    manifests = build_device_bundle(
        args.descriptors,
        project_root=_repo_root(),
        output_root=output_root,
        compiler=args.compiler,
        fflags=shlex.split(args.fflags),
        ldflags=shlex.split(args.ldflags),
        bundle_name=args.name,
    )
    print(
        json.dumps(
            {
                "bundle": args.name,
                "device_count": len(manifests),
                "output_root": str(output_root.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


def command_audit_devices(args: argparse.Namespace) -> int:
    catalog = DeviceCatalog.discover(_repo_root())
    payload = catalog.machine_record()
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n")
    print(json.dumps(catalog.summary(), indent=2, sort_keys=True))
    return 0


def command_generate_devices(args: argparse.Namespace) -> int:
    catalog = DeviceCatalog.discover(_repo_root())
    output = Path(
        args.output_root or _repo_root() / "devices/generated"
    )
    descriptors = catalog.write_descriptors(output, clean=args.clean)
    print(
        json.dumps(
            {
                "suite_count": len(catalog.suites),
                "scheme_count": len(descriptors),
                "output_root": str(output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_build_catalog_devices(args: argparse.Namespace) -> int:
    """Build every framework-free connector using portable host providers."""

    root = _repo_root()
    catalog = DeviceCatalog.discover(root)
    descriptor_root = Path(
        args.descriptor_root or root / "devices/generated"
    ).resolve()
    catalog.write_descriptors(descriptor_root, clean=True)
    output_root = Path(
        args.output_root or root / "build/catalog_devices"
    ).resolve()
    selected: list[tuple[object, DeviceDescription]] = []
    dependency_blocked: list[dict[str, str]] = []
    candidates = []
    for entry in sorted(catalog.entries.values(), key=lambda item: item.name):
        if is_python_host_service_scheme(entry):
            continue
        descriptor = descriptor_root / entry.name / "device.yaml"
        try:
            description = DeviceDescription.from_yaml(
                descriptor, project_root=root
            )
            blockers = list(entry.blockers)
            if description.host_entrypoints:
                host_tables = {
                    endpoint.table
                    for endpoint in entry.entrypoints
                    if endpoint.phase in description.host_entrypoints
                }
                blockers = [
                    blocker
                    for blocker in blockers
                    if not any(
                        f"{table}." in blocker for table in host_tables
                    )
                ]
            unresolved = set(entry.unresolved_modules)
            unresolved -= set(description.external_modules)
            if blockers or unresolved:
                continue
            candidates.append(entry)
            resolve_source_closure(description)
            selected.append((entry, description))
        except DeviceBuildError as exc:
            dependency_blocked.append(
                {"name": entry.name, "reason": str(exc)}
            )
    results: list[dict[str, object]] = []
    core = [
        (entry, description)
        for entry, description in selected
        if not description.external_modules
    ]
    external = [
        (entry, description)
        for entry, description in selected
        if description.external_modules
    ]
    try:
        core_manifests = build_device_bundle(
            [
                descriptor_root / entry.name / "device.yaml"
                for entry, _description in core
            ],
            project_root=root,
            output_root=output_root,
            compiler=args.compiler,
            fflags=shlex.split(args.fflags),
            ldflags=shlex.split(args.ldflags),
            bundle_name="cam-sima-core",
        )
        manifest_by_name = {
            manifest.parent.name: manifest for manifest in core_manifests
        }
        for entry, _description in core:
            manifest = manifest_by_name[entry.name]
            results.append(
                {
                    "name": entry.name,
                    "status": "built",
                    "build_mode": "shared-module-state-bundle",
                    "manifest": str(manifest.relative_to(root)),
                }
            )
    except DeviceBuildError as exc:
        for entry, _description in core:
            results.append(
                {
                    "name": entry.name,
                    "status": "failed",
                    "build_mode": "shared-module-state-bundle",
                    "error": str(exc),
                }
            )

    for entry, _description in external:
        try:
            manifest = build_device(
                descriptor_root / entry.name / "device.yaml",
                project_root=root,
                output_root=output_root,
                compiler=args.compiler,
                fflags=shlex.split(args.fflags),
                ldflags=shlex.split(args.ldflags),
            )
            results.append(
                {
                    "name": entry.name,
                    "status": "built",
                    "build_mode": "external-library-device",
                    "manifest": str(manifest.relative_to(root)),
                }
            )
        except DeviceBuildError as exc:
            results.append(
                {
                    "name": entry.name,
                    "status": "failed",
                    "build_mode": "external-library-device",
                    "error": str(exc),
                }
            )
    orbital_library = build_orbital_host_library(
        root,
        compiler=args.compiler,
        fflags=shlex.split(args.fflags),
        ldflags=shlex.split(args.ldflags),
    )
    report = {
        "schema_version": 1,
        "source_revision": catalog.source_revision,
        "selection": (
            "ABI-compatible schemes with a recursively source-resolved, "
            "framework-free dependency closure"
        ),
        "abi_compatible_considered": len(candidates),
        "dependency_blocked": len(dependency_blocked),
        "dependency_blockers": dependency_blocked,
        "attempted": len(results),
        "bundled": len(core),
        "external_library_devices": len(external),
        "built": sum(item["status"] == "built" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "orbital_host_library": str(orbital_library.relative_to(root)),
        "results": results,
    }
    report_path = Path(
        args.report
        or root / "validation/portable_catalog_device_build.json"
    ).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "results"
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.strict and report["failed"]:
        return 1
    return 0


def command_scheme_status(args: argparse.Namespace) -> int:
    matrix = DeviceSupportMatrix.discover(_repo_root())
    payload = matrix.machine_record()
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(matrix.summary(), indent=2, sort_keys=True))
    return 0


def command_compare_history(args: argparse.Namespace) -> int:
    fields = (
        tuple(item.strip() for item in args.fields.split(",") if item.strip())
        if args.fields
        else None
    )
    compare_history_directories(
        args.reference_dir,
        args.candidate_dir,
        expected_files=args.files,
        expected_numeric_variables=(
            args.numeric_variables
            if args.numeric_variables is not None
            else (None if fields is not None else 26)
        ),
        **({"fields": fields} if fields is not None else {}),
    )
    print("BFB")
    return 0


def command_pool_worker(args: argparse.Namespace) -> int:
    """Serve one configurable MPI world partitioned into persistent slots."""

    from .notebook.pool_worker import serve_pool

    ensure_mpi_loader_environment()
    return serve_pool(args)


def main() -> int:
    parser = argparse.ArgumentParser(prog="pycam-sima")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the complete Python-owned CAM model")
    run.add_argument("config")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--history-dir", required=True)
    run.add_argument("--library")
    run.add_argument(
        "--steps",
        type=int,
        help="override ModelConfig.stop_n",
    )
    run.set_defaults(func=command_run)

    segment = sub.add_parser(
        "run-segment",
        help="run one checkpointed model segment for Dask fan-out",
    )
    segment.add_argument("config")
    segment.add_argument("--run-dir", required=True)
    segment.add_argument("--history-dir", required=True)
    segment.add_argument("--library")
    segment.add_argument("--input-checkpoint")
    segment.add_argument("--output-checkpoint", required=True)
    segment.add_argument("--branch-spec")
    segment.add_argument("--segment-plan")
    segment.add_argument("--name", default="segment")
    segment.add_argument("--steps", type=int)
    segment.add_argument("--result-json")
    segment.set_defaults(func=command_run_segment)

    build = sub.add_parser(
        "build-kernels",
        help="build main model kernels and generated Fortran devices",
    )
    build.add_argument(
        "--config",
        help="ModelConfig YAML used to specialize np/nc/pver/nconst",
    )
    build.add_argument(
        "--target",
        help="explicit output .so (defaults to the specialization cache)",
    )
    build.set_defaults(func=command_build_kernels)

    device = sub.add_parser(
        "build-device",
        help="generate and build source-preserving CCPP Fortran devices",
    )
    device.add_argument("descriptors", nargs="+")
    device.add_argument("--output-root")
    device.add_argument(
        "--compiler", default="/opt/cray/pe/gcc/12.2.0/bin/gfortran"
    )
    device.add_argument(
        "--fflags",
        default=(
            "-O2 -march=znver3 -fPIC -ffp-contract=off "
            "-ffree-line-length-none -cpp -DUSE_CONTIGUOUS="
        ),
    )
    device.add_argument(
        "--ldflags", default="-Wl,--as-needed -Wl,--no-undefined"
    )
    device.set_defaults(func=command_build_device)

    bundle = sub.add_parser(
        "build-device-bundle",
        help=(
            "link multiple generated connectors into one shared Fortran "
            "module namespace"
        ),
    )
    bundle.add_argument("descriptors", nargs="+")
    bundle.add_argument("--output-root")
    bundle.add_argument("--name", default="catalog")
    bundle.add_argument(
        "--compiler", default="/opt/cray/pe/gcc/12.2.0/bin/gfortran"
    )
    bundle.add_argument(
        "--fflags",
        default=(
            "-O2 -march=znver3 -fPIC -ffp-contract=off "
            "-ffree-line-length-none -cpp -DUSE_CONTIGUOUS="
        ),
    )
    bundle.add_argument(
        "--ldflags", default="-Wl,--as-needed -Wl,--no-undefined"
    )
    bundle.set_defaults(func=command_build_device_bundle)

    audit = sub.add_parser(
        "audit-devices",
        help="inventory every CCPP scheme referenced by pinned suite XML",
    )
    audit.add_argument("--output")
    audit.set_defaults(func=command_audit_devices)

    generate = sub.add_parser(
        "generate-devices",
        help="generate one source-preserving connector YAML per suite scheme",
    )
    generate.add_argument("--output-root")
    generate.add_argument("--clean", action="store_true")
    generate.set_defaults(func=command_generate_devices)

    catalog_build = sub.add_parser(
        "build-catalog-devices",
        help="build the automatically portable subset of catalog devices",
    )
    catalog_build.add_argument("--descriptor-root")
    catalog_build.add_argument("--output-root")
    catalog_build.add_argument("--report")
    catalog_build.add_argument("--strict", action="store_true")
    catalog_build.add_argument(
        "--compiler", default="/opt/cray/pe/gcc/12.2.0/bin/gfortran"
    )
    catalog_build.add_argument(
        "--fflags",
        default=(
            "-O2 -march=znver3 -fPIC -ffp-contract=off "
            "-ffree-line-length-none -cpp -DUSE_CONTIGUOUS="
        ),
    )
    catalog_build.add_argument(
        "--ldflags", default="-Wl,--as-needed -Wl,--no-undefined"
    )
    catalog_build.set_defaults(func=command_build_catalog_devices)

    status = sub.add_parser(
        "scheme-status",
        help="report connector/build status for every active suite scheme",
    )
    status.add_argument("--output")
    status.set_defaults(func=command_scheme_status)

    compare = sub.add_parser("compare-history", help="run the fixed BFB gate")
    compare.add_argument("reference_dir")
    compare.add_argument("candidate_dir")
    compare.add_argument("--files", type=int, default=51)
    compare.add_argument("--numeric-variables", type=int)
    compare.add_argument(
        "--fields",
        help=(
            "comma-separated scientific history fields; omitted keeps the "
            "fixed 26-field FKESSLER gate"
        ),
    )
    compare.set_defaults(func=command_compare_history)

    pool_worker = sub.add_parser(
        "pool-worker",
        help="serve multiple persistent CAM models in one partitioned MPI world",
    )
    from .notebook.pool_worker import add_arguments as add_pool_worker_arguments

    add_pool_worker_arguments(pool_worker)
    pool_worker.set_defaults(func=command_pool_worker)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
