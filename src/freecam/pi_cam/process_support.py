"""Machine-readable support accounting for flat PI-CAM processes.

The user-facing process namespace joins two sets: source procedures already
represented by the Python workflow and source procedures that previously had
only catalog entries.  This module audits the latter set without confusing
three different claims:

* an adapter was generated;
* that adapter compiled in the source file's owning CAM build context;
* the resulting device can be loaded by the selected PI-CAM executable.

An inactive physics configuration can satisfy the first two claims without
being scientifically runnable in the selected case.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .physics_catalog import PICAMPhysicsCatalog, merge_runtime_process_records


_LIVE_NATIVE_OWNERS = frozenset(
    {
        "type:physics_state",
        "type:physics_tend",
        "type:cam_in_t",
        "type:cam_out_t",
    }
)


def _configuration(source: str, error: str | None) -> str:
    token = f"{source} {error or ''}".lower()
    if "/cosp/" in token or "cosp_" in token or "radar_simulator" in token:
        return "cosp"
    if source.endswith("/radsw.F90") or "radconstants" in token:
        return "legacy_radiation"
    if "/carma/" in token or "sulfate_utils" in token:
        return "carma"
    return "selected_pi_cam"


def build_process_support_report(
    *,
    catalog: PICAMPhysicsCatalog,
    runtime_records: Sequence[Mapping[str, Any]],
    generation: Mapping[str, Any],
    compilation: Mapping[str, Any],
    loading: Mapping[str, Any],
    runtime_validation: Mapping[str, Any] | None = None,
    bfb_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Join source, compile, loader, StatePool, and BFB evidence.

    The returned ``processes`` table contains exactly the source processes
    that were not already represented by a workflow boundary.  In the pinned
    PI-CAM source this is the historical set of 262 catalog-only interfaces.
    """

    generated = {
        (str(item["name"]), str(item["original_source"])): dict(item)
        for item in generation.get("generated", ())
    }
    compiled_sources = {
        str(item["source"]): dict(item)
        for item in compilation.get("sources", ())
        if item.get("status") == "passed"
    }
    load_records = {
        str(item["name"]): dict(item) for item in loading.get("records", ())
    }
    merged = merge_runtime_process_records(runtime_records, catalog)
    source_only = tuple(row for row in merged if row["kind"] == "catalog_process")

    records: list[dict[str, Any]] = []
    for row in source_only:
        qualified_name = str(row["qualified_name"])
        source = str(row["source"])
        identity = f"{qualified_name}@{source}"
        adapter = generated.get((qualified_name, source))
        loader = load_records.get(identity, {})
        opaque_types = tuple(
            sorted(
                {
                    str(argument.get("fortran_type"))
                    for argument in row.get("arguments", ())
                    if not bool(argument.get("procedure", False))
                    and argument.get("dtype") is None
                }
            )
        )
        unknown_owners = tuple(
            item for item in opaque_types if item.lower() not in _LIVE_NATIVE_OWNERS
        )
        error = None if loader.get("error") is None else str(loader["error"])
        generated_ok = adapter is not None
        compiled_ok = source in compiled_sources
        loadable = bool(loader.get("loaded", False))
        records.append(
            {
                "name": str(row["api_name"]),
                "qualified_name": qualified_name,
                "source": source,
                "identity": identity,
                "adapter_generated": generated_ok,
                "adapter_compiled": compiled_ok,
                "adapter_symbol": None if adapter is None else adapter.get("symbol"),
                "statepool_pointer_contract": generated_ok and compiled_ok,
                "opaque_types": opaque_types,
                "binding_policy": (
                    "statepool_arrays"
                    if not opaque_types
                    else (
                        "statepool_arrays_and_live_native_owners"
                        if not unknown_owners
                        else "explicit_subsystem_owner_required"
                    )
                ),
                "current_case_loadable": loadable,
                "configuration": _configuration(source, error),
                "load_error": error,
                "supported": generated_ok and compiled_ok,
            }
        )

    configuration_counts = Counter(record["configuration"] for record in records)
    unloadable_configuration_counts = Counter(
        record["configuration"]
        for record in records
        if not record["current_case_loadable"]
    )
    generated_count = sum(record["adapter_generated"] for record in records)
    compiled_count = sum(record["adapter_compiled"] for record in records)
    statepool_count = sum(record["statepool_pointer_contract"] for record in records)
    loadable_count = sum(record["current_case_loadable"] for record in records)
    total = len(records)
    report: dict[str, Any] = {
        "schema_version": 1,
        "source_revision": catalog.source_revision,
        "catalog_physical_processes": len(catalog.physics_processes),
        "workflow_source_overlap": len(catalog.physics_processes) - total,
        "formerly_catalog_only_interfaces": total,
        "adapters_generated": generated_count,
        "adapters_compiled": compiled_count,
        "statepool_pointer_contracts": statepool_count,
        "current_case_loadable": loadable_count,
        "configuration_specific": total - loadable_count,
        "configuration_counts": dict(sorted(configuration_counts.items())),
        "unloadable_configuration_counts": dict(
            sorted(unloadable_configuration_counts.items())
        ),
        "all_catalog_only_interfaces_supported": bool(
            total > 0
            and generated_count == total
            and compiled_count == total
            and statepool_count == total
        ),
        "adapter_compile_job": compilation.get("pbs_job_id"),
        "loader_job": loading.get("pbs_job_id"),
        "representative_runtime_validation": dict(runtime_validation or {}),
        "representative_bfb_validation": dict(bfb_validation or {}),
        "processes": records,
    }
    return report


__all__ = ["build_process_support_report"]
