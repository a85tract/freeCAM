"""The physics kernel decoupling inventory: every action of the step, what covers it, and what is proven.

One record, built from the repository's own sources and never by hand: the
step plan (which actions run, in what order), the physics catalog (the
Fortran procedures under each action and whether an adapter can reach them),
the stage classes (which actions Python drives and which kernels they expose),
the segment-runner manifest (where the image can pause), the reviewed function
contracts, and the validation records (what has been captured, replayed,
replaced in the model and compared bit for bit).  Each action is classified
once; each kernel carries the state of its delivery loop; what automation
cannot decide is listed as unresolved rather than filled in.

The record is the acceptance ledger of the decoupling work: an action counts
as covered only when its kernels have closed the loop, and the summary is
computed from the rows, not typed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .tables import load_table

REPO = Path(__file__).resolve().parents[3]
FUNCTIONS = REPO / "native/pi_cam/functions"
VALIDATION = REPO / "validation"

SCHEMA_VERSION = 1

#: The action kinds that are not numerical physics and are out of the decoupling's scope.
NON_PHYSICS_KINDS = {"boundary": "boundary", "coupling": "boundary", "clock": "clock",
                     "dynamics": "dynamics", "io": "io", "service": "host_service"}

#: Scheme actions whose body is glue around the step rather than a scheme of its own,
#: and diagnostics computed for output only.  Reviewed by hand; the reason is recorded.
CONTROL_SCHEMES = {
    "cam_run1.state_initialize": "tphysbc's prologue: zeroes the tendencies and starts the energy check",
    "cam_run2.state_finalize": "tphysac's epilogue: the dry static energy update and the state checks",
    "cam_run2.surface_fluxes_and_emissions": "tphysac's opening: surface fluxes and emissions applied to the state",
}
DIAGNOSTIC_SCHEMES = {
    "cam_run1.state_and_convection_diagnostics_leaf": "diag_phys_writeout: history fields only",
    "cam_run1.cloud_diagnostics_leaf": "cloud_diagnostics_calc: history fields only",
    "cam_run1.tropopause_leaf": "tropopause_output: a diagnosed tropopause for history",
    "cam_run1.export_diagnostics_leaf": "diag_export: history fields only",
    "cam_run1.state_export_leaf": "cam_export: the atmosphere's surface exchange fields",
}

#: What the admitted configuration selects, read from the reference case's
#: namelist (values only).  These decide which scheme bodies do work.
CONFIGURATION = {
    "deep_scheme": "ZM", "shallow_scheme": "UW", "macrop_scheme": "park", "microp_scheme": "MG",
    "eddy_scheme": "diag_TKE", "do_tms": True, "cld_macmic_num_steps": 1, "use_gw_oro": True,
    "use_gw_convect_dp": False, "use_gw_convect_sh": False, "use_gw_front": False,
    "carma_model": "none", "do_clubb_sgs": False, "use_subcol_microp": False,
    "micro_do_icesupersat": False, "do_iss": True,
}

#: Enabled scheme actions whose body is expected to do no numerical work in
#: this configuration, and why.  The call still happens every step; the
#: expectation is from the source and the configuration, and stays
#: ``confirmed: false`` until a targeted test shows the state unchanged.
INERT_BY_CONFIGURATION = {
    "cam_run2.rayleigh_friction": "rayk0 is not set; the routine's tendency is zero and physics_update runs",
    "cam_run2.charge_neutrality": "no ionosphere: charge_fix reduces to mbar = mwdry",
    "cam_run2.qbo_relaxation": "qbo_use_forcing is off",
    "cam_run2.ion_drag": "no WACCM: iondrag_calc returns without a tendency",
    "cam_run2.carma_aerosol_tendencies_leaf": "carma_model is none",
    "cam_run1.carma_wet_deposition_leaf": "carma_model is none",
    "cam_run2.carma_statistics_leaf": "carma_model is none",
    "cam_run2.tracer_tendencies_leaf": "the test tracers are not enabled in this compset",
    "cam_run2.age_of_air_tendencies_leaf": "aoa_tracers_flag is off",
    "cam_run1.modal_aerosol_preparation_leaf": "modal_aero_prepare has no work outside the sub-column path",
    "cam_run1.sea_salt_rebin": "sslt_rebin_adv acts on bulk sea salt, which the modal aerosols do not carry",
}

#: The stage classes that drive an action from Python, and the kernels they expose.
STAGE_CLASSES = (
    "freecam.physics.cloud_macro_microphysics.CloudMacroMicrophysics",
    "freecam.physics.radiation.Radiation",
    "freecam.physics.pausable.DryAdjustment",
    "freecam.physics.pausable.ShallowConvection",
    "freecam.physics.pausable.RayleighFriction",
    "freecam.physics.pausable.ChargeNeutrality",
    "freecam.physics.pausable.QBORelaxation",
    "freecam.physics.pausable.IonDrag",
    "freecam.physics.pausable.SeaSaltRebin",
    "freecam.physics.pausable.ModalAerosolPreparation",
    "freecam.physics.pausable.CARMAWetDeposition",
    "freecam.physics.pausable.CARMAAerosolTendencies",
    "freecam.physics.pausable.CARMAStatistics",
    "freecam.physics.pausable.TracerTendencies",
    "freecam.physics.pausable.AgeOfAirTendencies",
)

#: Validation records per kernel, by the step of the delivery loop they prove.
#: ``{name}`` is the kernel's name; a pattern that names no file is a gap.
EVIDENCE_PATTERNS = {
    "capture": ("pi_cam_{name}_capture_50step.json",),
    "standalone_build": ("pi_cam_{name}_standalone_build.json", "pi_cam_{name}_standalone_manifest.json"),
    "replay_full_chunk": ("pi_cam_{name}_full_chunk_vs_capture.json",),
    "replay_single_column": ("pi_cam_{name}_single_column_vs_capture.json",),
    "replay_public_api": ("pi_cam_{name}_public_api_vs_capture.json",),
    "module_state": ("pi_cam_{name}_module_state.json",),
}
#: In-model replacement gates that are not named after the kernel: the record
#: and the bit-for-bit comparison, and what path they prove.
IN_MODEL_GATES = {
    "mmacro_pcond": (
        ("segmented", "pi_cam_stage7_segmented_original_50step.json",
         "pi_cam_stage7_segmented_original_vs_oracle_50step_bfb.json"),
    ),
    "micro_mg_tend": (
        ("legacy-python walk, core through its standalone image", "pi_cam_mm_core_standalone_50step.json",
         "pi_cam_mm_core_standalone_vs_oracle_50step_bfb.json"),
    ),
    "rad_rrtmg_sw": (
        ("legacy-python walk, original kernel", "pi_cam_rad_tend_python_50step.json",
         "pi_cam_rad_tend_python_vs_oracle_50step_bfb.json"),
    ),
    "rad_rrtmg_lw": (
        ("legacy-python walk, original kernel", "pi_cam_rad_tend_python_50step.json",
         "pi_cam_rad_tend_python_vs_oracle_50step_bfb.json"),
    ),
}
#: Performance evidence for a stage class with nothing replaced (paired months and the year).
PERFORMANCE_RECORDS = {
    "cam_run1.cloud_macro_microphysics": ("performance_overhead.md",
                                          "pi_cam_native_whole_1month_median.json"),
}
#: Execution evidence: per-leaf call counts recorded by the online gate and a month.
EXECUTION_RECORDS = (("online_50step", "pi_cam_exact_cesm_online_50step.json", 50),
                     ("month_1488step", "pi_cam_1month_stage_native-whole.json", 1488))


@dataclass(slots=True)
class KernelRow:
    """One kernel's delivery loop."""

    kernel: str
    stage_action: str
    owner_class: str
    routine: str | None
    contract: str                     # reviewed | draft | none
    contract_path: str | None
    bindable: bool
    validated_through_runner: bool
    evidence: dict[str, list[str]] = field(default_factory=dict)
    in_model_gates: list[dict[str, Any]] = field(default_factory=list)
    status: str = "open"
    missing: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ActionRow:
    """One action of the step, classified once."""

    id: str
    native_id: int | None
    phase: str
    kind: str
    granularity: str
    operation: str
    enabled: bool
    parent_stage: str | None
    classification: str
    activity: str
    activity_basis: str | None
    python_class: str | None
    kernels: list[str] = field(default_factory=list)
    kernel_candidates: list[dict[str, Any]] = field(default_factory=list)
    deeper_procedures: int = 0
    execution: dict[str, Any] = field(default_factory=dict)
    alternate_of: list[str] = field(default_factory=list)
    performance: list[str] = field(default_factory=list)
    coverage: str = "not-applicable"
    note: str | None = None


def _import(path: str) -> Any:
    module, _, name = path.rpartition(".")
    return getattr(__import__(module, fromlist=[name]), name)


def _contracts() -> dict[str, tuple[str, str, str | None]]:
    """routine or function name -> (status, relative path, routine)."""

    found: dict[str, tuple[str, str, str | None]] = {}
    for status, pattern in (("draft", "drafts/*.yaml"), ("reviewed", "*.yaml")):
        for path in sorted(FUNCTIONS.glob(pattern)):
            payload = load_table(path)
            if not isinstance(payload, Mapping):
                continue
            routine = payload.get("routine")
            relative = str(path.relative_to(REPO))
            for key in {str(payload.get("function")), str(routine)}:
                if key and key != "None":
                    found[key] = (status, relative, None if routine is None else str(routine))
    return found


def _record(name: str) -> Mapping[str, Any] | None:
    path = VALIDATION / name
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _kernel_rows(stage_classes: Iterable[str], runner_specs: Mapping[str, Any]) -> list[KernelRow]:
    contracts = _contracts()
    rows: list[KernelRow] = []
    for class_path in stage_classes:
        cls = _import(class_path)
        stage = cls()
        spec = runner_specs.get(cls.STAGE)
        for description in stage.describe_kernels():
            name = description["kernel"]
            contract = contracts.get(name)
            evidence = {step: [p.format(name=name) for p in patterns if (VALIDATION / p.format(name=name)).is_file()]
                        for step, patterns in EVIDENCE_PATTERNS.items()}
            gates = []
            for path, record, bfb_record in IN_MODEL_GATES.get(name, ()):
                bfb = _record(bfb_record)
                gates.append({
                    "path": path, "record": record, "bfb_record": bfb_record,
                    "present": (VALIDATION / record).is_file() and bfb is not None,
                    "bfb": None if bfb is None else bool(bfb.get("bfb")),
                    "compared_files": None if bfb is None else bfb.get("compared_files"),
                })
            missing = [step for step, files in evidence.items() if not files and step != "module_state"]
            if not any(g["present"] and g["bfb"] for g in gates):
                missing.append("in_model_replacement_bfb")
            if not description["bindable"]:
                missing.append("segment_runner")
            if contract is None or contract[0] != "reviewed":
                missing.append("reviewed_contract")
            status = "complete" if not missing else "open"
            rows.append(KernelRow(
                kernel=name, stage_action=cls.STAGE, owner_class=description["owner_class"],
                routine=None if contract is None else contract[2],
                contract="none" if contract is None else contract[0],
                contract_path=None if contract is None else contract[1],
                bindable=bool(description["bindable"]),
                validated_through_runner=bool(description["validated"]),
                evidence=evidence, in_model_gates=gates, status=status, missing=missing,
            ))
        close = getattr(stage, "close", None)
        if callable(close):
            close()
    return rows


def _execution_evidence() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """operation -> per-record call counts, and the records' own consistency facts."""

    by_operation: dict[str, dict[str, Any]] = {}
    facts: dict[str, Any] = {}
    for label, name, steps in EXECUTION_RECORDS:
        payload = _record(name)
        if payload is None:
            facts[label] = {"record": name, "present": False}
            continue
        counts = payload.get("leaf_operation_counts") or {}
        for operation, count in counts.items():
            by_operation.setdefault(str(operation), {})[label] = int(count)
        traces = payload.get("rank_trace_actions")
        plan_actions = payload.get("step_plan_actions")
        facts[label] = {
            "record": name, "present": True, "steps": steps, "pbs_job_id": payload.get("pbs_job_id"),
            "step_plan_actions": plan_actions,
            "rank_trace_actions_uniform": bool(traces) and len(set(traces)) == 1,
            "rank_trace_actions": traces[0] if traces else None,
            # every enabled action once a step, plus the actions of initialization
            "trace_covers_every_step": bool(traces) and isinstance(plan_actions, int)
            and traces[0] >= plan_actions * steps,
        }
    return by_operation, facts


def build_coverage() -> dict[str, Any]:
    """The inventory, from the repository's current sources."""

    from freecam.pi_cam.physics_catalog import PICAMPhysicsCatalog
    from freecam.pi_cam.plan import PICAMStepPlan
    from freecam.pi_cam.segment_runner import load_manifest

    plan_rows = PICAMStepPlan.default().describe()
    catalog = PICAMPhysicsCatalog.load_default()
    runner_specs = {spec.stage: spec for spec in load_manifest()}
    kernel_rows = _kernel_rows(STAGE_CLASSES, runner_specs)
    kernels_by_action: dict[str, list[KernelRow]] = {}
    for row in kernel_rows:
        kernels_by_action.setdefault(row.stage_action, []).append(row)
    class_by_action = {_import(path).STAGE: path for path in STAGE_CLASSES}
    by_operation, execution_facts = _execution_evidence()

    procedures = [p.as_dict() for p in catalog.processes]
    candidates_by_action: dict[str, list[dict[str, Any]]] = {}
    for procedure in procedures:
        actions = set(procedure.get("parent_actions") or ()) | set(procedure.get("workflow_actions") or ())
        for action in actions:
            candidates_by_action.setdefault(str(action), []).append(procedure)

    ids = [f'{row["phase"]}.{row["name"]}' for row in plan_rows]
    enabled_ids = {i for i, row in zip(ids, plan_rows) if row["enabled"]}
    groups: dict[str, list[str]] = {}
    for i, row in zip(ids, plan_rows):
        if row.get("parent_stage"):
            groups.setdefault(str(row["parent_stage"]), []).append(i)

    actions: list[ActionRow] = []
    for i, row in zip(ids, plan_rows):
        kind = str(row["kind"])
        enabled = bool(row["enabled"])
        alternate_of: list[str] = []
        if not enabled:
            if i in groups and any(leaf in enabled_ids for leaf in groups[i]):
                alternate_of = [leaf for leaf in groups[i] if leaf in enabled_ids]
            elif row.get("parent_stage") and str(row["parent_stage"]) in enabled_ids:
                alternate_of = [str(row["parent_stage"])]
        if kind in NON_PHYSICS_KINDS:
            classification = NON_PHYSICS_KINDS[kind]
        elif kind == "kernel":
            classification = "process_control"
        elif i in CONTROL_SCHEMES:
            classification = "process_control"
        elif i in DIAGNOSTIC_SCHEMES:
            classification = "diagnostics"
        elif kind == "scheme":
            classification = "numeric_scheme"
        else:
            classification = "unclassified"
        note = CONTROL_SCHEMES.get(i) or DIAGNOSTIC_SCHEMES.get(i)
        if not enabled:
            activity = "alternate-form" if alternate_of else "disabled"
            basis = ("the same work runs as " + ", ".join(alternate_of)) if alternate_of else None
        elif i in INERT_BY_CONFIGURATION:
            activity = "inert-by-configuration"
            basis = INERT_BY_CONFIGURATION[i] + " (unconfirmed: no targeted test yet)"
        elif classification in ("numeric_scheme", "process_control", "diagnostics"):
            activity = "active"
            basis = "enabled in the plan; called every step in the recorded runs"
        else:
            activity = "active"
            basis = "enabled in the plan"
        raw = sorted(candidates_by_action.get(i, ()), key=lambda p: (int(p.get("call_depth", 0)), str(p["name"])))
        shallow = [p for p in raw if int(p.get("call_depth", 0)) <= 2]
        candidates = [{
            "name": p["name"], "qualified_name": p["qualified_name"], "source": p["source"],
            "call_depth": p.get("call_depth"), "role": p.get("role"), "level": p.get("level"),
            "adapter_status": p.get("adapter_status"), "blockers": list(p.get("blockers") or ()),
        } for p in shallow] if classification == "numeric_scheme" else []
        execution = dict(by_operation.get(str(row["operation"]), {}))
        if not execution and enabled:
            execution = {label: "in the step plan; the trace count covers every step"
                         for label, facts in execution_facts.items() if facts.get("trace_covers_every_step")}
        kernels = [k.kernel for k in kernels_by_action.get(i, [])]
        if classification != "numeric_scheme":
            coverage = "not-applicable"
        elif activity != "active":
            coverage = "not-required-in-this-configuration" if activity == "inert-by-configuration" else "alternate"
        elif not kernels:
            coverage = "gap"
        elif all(k.status == "complete" for k in kernels_by_action[i]):
            coverage = "complete"
        else:
            coverage = "partial"
        actions.append(ActionRow(
            id=i, native_id=row.get("native_id"), phase=str(row["phase"]), kind=kind,
            granularity=str(row.get("granularity")), operation=str(row["operation"]), enabled=enabled,
            parent_stage=row.get("parent_stage"), classification=classification, activity=activity,
            activity_basis=basis, python_class=class_by_action.get(i), kernels=kernels,
            kernel_candidates=candidates, deeper_procedures=len(raw) - len(shallow) if classification == "numeric_scheme" else 0,
            execution=execution, alternate_of=alternate_of,
            performance=list(PERFORMANCE_RECORDS.get(i, ())), coverage=coverage, note=note,
        ))

    known = set(ids)
    unknown_parents = sorted(a for a in candidates_by_action if a not in known)
    closure = {
        "every_action_once": len(ids) == len(set(ids)),
        "every_enabled_action_classified": all(a.classification != "unclassified" for a in actions if a.enabled),
        "every_disabled_action_is_an_alternate_form": all(a.alternate_of for a in actions if not a.enabled),
        "catalog_actions_all_in_the_plan": not unknown_parents,
        "unknown_catalog_actions": unknown_parents,
        "kernel_owned_once": len({k.kernel for k in kernel_rows}) == len(kernel_rows),
        "execution_records": execution_facts,
    }
    unresolved = [
        {"what": a.id, "why": a.activity_basis}
        for a in actions if a.activity == "inert-by-configuration"
    ] + [
        {"what": f"{k.kernel}: {step}", "why": "no record in validation/ for this step of the loop"}
        for k in kernel_rows for step in k.missing
    ] + [
        {"what": a.id, "why": ("the physics catalog lists no procedure under this action; its active "
                               "call graph must be extended before this action's kernels can be chosen")}
        for a in actions
        if a.classification == "numeric_scheme" and a.activity == "active"
        and not a.kernel_candidates and not a.kernels
    ]
    summary = {
        "actions": len(actions),
        "enabled": sum(1 for a in actions if a.enabled),
        "by_classification": _counts(a.classification for a in actions),
        "numeric_schemes_active": sum(1 for a in actions if a.classification == "numeric_scheme" and a.activity == "active"),
        "numeric_scheme_coverage": _counts(a.coverage for a in actions if a.classification == "numeric_scheme"),
        "kernels": len(kernel_rows),
        "kernels_by_status": _counts(k.status for k in kernel_rows),
        "kernels_bindable": sum(1 for k in kernel_rows if k.bindable),
        "kernels_validated_through_runner": sum(1 for k in kernel_rows if k.validated_through_runner),
        "catalog_procedures": len(procedures),
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "generator": "tools/build_physics_kernel_coverage.py",
        "what": ("Every action of the PI-atm step classified once, with the Python class and kernels that "
                 "cover it, the catalog's procedures under it, the recorded execution counts, and the state "
                 "of each kernel's delivery loop; built from the repository's own records, never by hand."),
        "cam_source_revision": getattr(catalog, "cam_source_revision", None) or _catalog_revision(),
        "configuration": CONFIGURATION,
        "summary": summary,
        "closure": closure,
        "actions": [asdict(a) for a in actions],
        "kernels": [asdict(k) for k in kernel_rows],
        "unresolved": unresolved,
    }
    record["coverage_hash"] = coverage_hash(record)
    return record


def _catalog_revision() -> str | None:
    payload = _record("pi_cam_kernel_inventory.json")
    return None if payload is None else payload.get("cam_source_revision")


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def coverage_hash(record: Mapping[str, Any]) -> str:
    """A content hash of the record without its own hash."""

    body = {key: value for key, value in record.items() if key != "coverage_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def write_coverage(path: str | Path, record: Mapping[str, Any] | None = None) -> Path:
    target = Path(path)
    payload = build_coverage() if record is None else record
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return target


def check_closure(record: Mapping[str, Any]) -> list[str]:
    """The closure facts that must hold, as a list of failures (empty when it closes)."""

    closure = record["closure"]
    failures = []
    for key in ("every_action_once", "every_enabled_action_classified",
                "every_disabled_action_is_an_alternate_form", "catalog_actions_all_in_the_plan",
                "kernel_owned_once"):
        if not closure.get(key):
            failures.append(key)
    for label, facts in closure["execution_records"].items():
        if not facts.get("present"):
            failures.append(f"execution record missing: {label}")
        elif not facts.get("trace_covers_every_step"):
            failures.append(f"trace does not cover every step: {label}")
    return failures


__all__ = ["CONFIGURATION", "INERT_BY_CONFIGURATION", "STAGE_CLASSES", "ActionRow", "KernelRow",
           "build_coverage", "check_closure", "coverage_hash", "write_coverage"]
