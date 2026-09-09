"""The progress dashboard's snapshot: committed records joined, verified, sanitized.

The published page at ``/freeCAM/progress/`` renders exactly one file, exported
by ``tools/export_progress_snapshot.py`` from this module.  Each fact keeps its
own source of truth:

- process identity, default order, enabled state, display names and the
  additional callable catalog come from the Workflow Builder's committed
  snapshot (``web/public/catalog.json``, itself checked fresh in CI);
- dedicated Python classes and the per-action classification come from the
  decoupling ledger (``validation/physics_kernel_decoupling.json``);
- the numerical candidates and their call relationships come from the
  kernel-API closure inventory (``validation/pi_cam_kernel_api_closure.json``)
  -- its *identity and relations only*: the runtime-evidence blocks cached in
  that record are never read, because they go stale against the ledger;
- capability and validation status comes from the ledger's tracked kernels and
  from the underlying validation records themselves, which this module opens
  to verify the claim (bit-for-bit flag, run length, ranks, real pause counts);
- symbol-redirection feasibility comes from the relocation audit
  (``validation/pi_cam_kernel_api_redirectable_calls.json``);
- in-model replacement mechanisms and their gates come from the runner
  manifest (``native/pi_cam/segment_runners.yaml``) and the hook table.

Everything published is allowlisted: no absolute path, account, credential,
raw model array, capture file or job log reaches the output.  The snapshot is
deterministic; its ``content_hash`` covers everything except the volatile
build metadata (commit, branch, generation time).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import yaml

SCHEMA_VERSION = 1

CATALOG = "web/public/catalog.json"
LEDGER = "validation/physics_kernel_decoupling.json"
CLOSURE = "validation/pi_cam_kernel_api_closure.json"
AUDIT = "validation/pi_cam_kernel_api_redirectable_calls.json"
RUNNERS = "native/pi_cam/segment_runners.yaml"
HOOKS = "native/pi_cam/hooks.yaml"
WORK_ITEMS = "native/pi_cam/kernel_work_items.yaml"
#: Optional: the runtime coverage records of the counting image's observation
#: runs.  With one, every candidate carries per-process execution evidence;
#: without any there is no validated observation run, and the page says so
#: rather than passing static candidates off under the same labels.
OBSERVATION_RUNS = (
    ("50step", "50-step validation", "validation/pi_cam_kernel_runtime_coverage_50step.json"),
    ("1month", "One-month validation", "validation/pi_cam_kernel_runtime_coverage_1month.json"),
)
VALIDATION = "validation"
FUNCTIONS = "native/pi_cam/functions"

#: Build directories whose evidence counts as the default image; every other
#: image named by a record is a development image.
DEFAULT_IMAGE_BUILDS = {"pi_cam_promoted", "pi_cam_zero_copy"}

#: Substrings no published string may contain (personal paths, machines).
FORBIDDEN = ("/glade", "/home/", "/Users/", "desched", ".hsn.", "scratch")

#: ``stage_execution`` keys that do not follow the ``<operation>_python`` rule.
STAGE_EXECUTION_ALIASES = {"rad_tend": "radiation"}

CAPABILITY_EXPLANATIONS = {
    "contract": "Arguments, dimensions, types, and required module state are described in a reviewed contract.",
    "adapter_build": "A recorded build produced the standalone adapter image successfully. A successful build is not an execution test.",
    "independently_callable": "A recorded Python call executed this kernel through the public entrypoint without running its entire parent process.",
    "standalone_replay": "Inputs captured from the running model reproduce the original outputs bit-for-bit through the standalone function.",
    "in_model_replacement": "The model can intercept and replace this kernel at a supported execution point (a runner pause or a link-time hook).",
    "original_replacement_bfb": "Returning the original kernel's answer through the replacement mechanism preserved the validated model results, in the tested process only.",
}


class ProgressExportError(RuntimeError):
    """A source is missing, malformed, or joins ambiguously."""


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_schema(name: str, payload: Mapping[str, Any], supported: int) -> None:
    version = payload.get("schema_version")
    if version != supported:
        raise ProgressExportError(f"{name}: unsupported schema_version {version!r} (supported: {supported})")


def load_inputs(root: Path) -> dict[str, Any]:
    """Parse every source and record its content hash; reject unknown schemas."""

    inputs: dict[str, Any] = {"hashes": {}}
    for key, rel, loader in (
        ("catalog", CATALOG, json.loads),
        ("ledger", LEDGER, json.loads),
        ("closure", CLOSURE, json.loads),
        ("audit", AUDIT, json.loads),
        ("runners", RUNNERS, yaml.safe_load),
        ("hooks", HOOKS, yaml.safe_load),
    ):
        path = root / rel
        if not path.is_file():
            raise ProgressExportError(f"source is missing: {rel}")
        inputs[key] = loader(path.read_text())
        inputs["hashes"][rel] = _sha256_file(path)
    _require_schema(CATALOG, inputs["catalog"], 1)
    _require_schema(LEDGER, inputs["ledger"], 1)
    _require_schema(CLOSURE, inputs["closure"], 1)
    _require_schema(AUDIT, inputs["audit"], 1)
    _require_schema(RUNNERS, inputs["runners"], 1)
    _require_schema(HOOKS, inputs["hooks"], 1)
    work_items_path = root / WORK_ITEMS
    if work_items_path.is_file():
        payload = yaml.safe_load(work_items_path.read_text()) or {}
        _require_schema(WORK_ITEMS, payload, 1)
        inputs["work_items"] = payload.get("items") or []
        inputs["hashes"][WORK_ITEMS] = _sha256_file(work_items_path)
    else:
        inputs["work_items"] = []
    inputs["observation_runs"] = {}
    for key, _label, rel in OBSERVATION_RUNS:
        path = root / rel
        if path.is_file():
            payload = json.loads(path.read_text())
            _require_schema(rel, payload, 1)
            inputs["observation_runs"][key] = payload
            inputs["hashes"][rel] = _sha256_file(path)
    return inputs


def _read_record(root: Path, name: str) -> Mapping[str, Any] | None:
    """A validation record by repo-relative name; None when absent or unreadable."""

    path = root / VALIDATION / name
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


# --------------------------------------------------------------------------- #
# Sanitized evidence summaries
# --------------------------------------------------------------------------- #

def _clean(value: Any) -> Any:
    """Replace any string carrying a forbidden fragment; keep the structure intact.

    Structure matters: a ``parent: null`` tree edge or a ``routine: null`` field
    is data, so nothing is dropped -- a string that names a personal path or a
    site is replaced by the marker ``[withheld]``.
    """

    if isinstance(value, str):
        # a job id's server suffix is a site fact; the number alone stays legible
        value = re.sub(r"\.desched\d*", "", value)
        lowered = value.lower()
        return "[withheld]" if any(fragment in lowered for fragment in FORBIDDEN) else value
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in sorted(value.items())}
    return value


def _job_number(value: Any) -> str | None:
    """A PBS job id reduced to its number: the server name is a site fact."""

    if not isinstance(value, str) or not value:
        return None
    number = value.split(".", 1)[0]
    return number if number.isdigit() else None


def _image_of(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Which image produced a record: build-directory basename and hash, no path."""

    sha = record.get("native_library_sha256") or record.get("image_sha256") or record.get("library_sha256")
    manifest = record.get("native_manifest") or record.get("native_image") or ""
    build = ""
    if isinstance(manifest, str) and manifest:
        parts = [p for p in manifest.split("/") if p]
        for part in reversed(parts):
            if part.startswith("pi_cam_"):
                build = part
                break
    if not sha and not build:
        return None
    role = "default" if build in DEFAULT_IMAGE_BUILDS else ("development" if build else "unrecorded")
    return {"sha256": sha, "build": build or None, "role": role}


def _pauses_by_kernel(record: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Real per-kernel replacement calls, per stage-execution entry of a gate record."""

    out: dict[str, dict[str, int]] = {}
    stage_execution = record.get("stage_execution")
    if not isinstance(stage_execution, dict):
        return out
    for key, entry in stage_execution.items():
        if not isinstance(entry, dict):
            continue
        by_kernel = entry.get("python_model_calls_by_kernel")
        stage = STAGE_EXECUTION_ALIASES.get(key, key[: -len("_python")] if key.endswith("_python") else key)
        if isinstance(by_kernel, dict):
            out[stage] = {k: int(v) for k, v in sorted(by_kernel.items())}
        elif entry.get("active_replacements") and len(entry["active_replacements"]) == 1 \
                and isinstance(entry.get("python_model_calls"), int):
            # the oldest gate records count one active replacement without a by-kernel map
            out[stage] = {str(entry["active_replacements"][0]): int(entry["python_model_calls"])}
    return out


def summarize_evidence(name: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """One record's publishable summary: allowlisted fields only, then scrubbed."""

    summary: dict[str, Any] = {"record": name}
    if "compared_files" in record and "bfb" in record and "run_status" not in record and "calls" not in record:
        summary["kind"] = "bfb-comparison"
        summary["bfb"] = bool(record.get("bfb"))
        summary["compared_files"] = record.get("compared_files")
        first = record.get("first_difference")
        if isinstance(first, dict):
            summary["first_difference"] = {k: first.get(k) for k in ("step", "rank", "field", "index")}
    elif "samples" in record and "kernel" in record:
        summary["kind"] = "frame-replay" + ("-failure" if "failure" in name else "")
        for key in ("kernel", "layout", "binding", "calls", "samples", "compared_values",
                    "bfb", "statuses", "image_sha256", "module_state_digest", "shards"):
            if key in record:
                summary[key] = record[key]
        capture = record.get("capture")
        if isinstance(capture, dict):
            summary["capture_run_tag"] = capture.get("run_tag")
            summary["capture_job"] = _job_number(capture.get("pbs_job_id"))
        for key in ("outcome", "diagnosis", "consequence"):
            if key in record:
                summary[key] = record[key]
    elif "library_sha256" in record and ("wrapper" in record or "members" in record):
        summary["kind"] = "standalone-build"
        summary["library_sha256"] = record.get("library_sha256")
        summary["spec_sha256"] = record.get("spec_sha256")
        members = record.get("members")
        summary["archive_members"] = len(members) if isinstance(members, list) else None
        summary["original_call_proof"] = bool(record.get("original_call_proof"))
    elif "records_compared" in record and "passed" in record:
        summary["kind"] = "capture-replay"
        for key in ("function", "passed", "records_compared", "records_equal", "outputs_compared",
                    "image_sha256", "snapshot_digest", "driver_free"):
            if key in record:
                summary[key] = record[key]
    elif "entries" in record and "digest" in record and "function" in record:
        summary["kind"] = "module-state-snapshot"
        summary["digest"] = record.get("digest")
        summary["symbols"] = sorted(record.get("entries", {}))
        summary["mpi_ranks"] = record.get("mpi_ranks")
        summary["stable_after_one_step"] = record.get("stable_after_one_step")
    elif "failure" in name:
        summary["kind"] = "gate-failure"
        for key in ("what", "run_tag", "outcome", "diagnosis", "consequence", "steps_completed",
                    "ranks", "python_stages", "segmented_original_kernels", "hook_pauses_before_failure"):
            if key in record:
                summary[key] = record[key]
        first = record.get("first_difference")
        if isinstance(first, dict):
            summary["first_difference"] = {k: first.get(k) for k in ("step", "rank", "field", "index")}
    else:
        summary["kind"] = "run-summary"
        for key in ("run_status", "steps", "mpi_ranks", "execution_mode", "python_stages",
                    "segmented_original_kernels", "radiation_python", "cloud_macro_micro_python"):
            if key in record:
                summary[key] = record[key]
        hooks = record.get("hooks")
        if isinstance(hooks, dict):
            summary["hooks"] = hooks
        pauses = _pauses_by_kernel(record)
        if pauses:
            summary["replacement_calls"] = pauses
        capture = record.get("frame_capture")
        if isinstance(capture, dict):
            summary["frame_capture"] = {k: capture.get(k) for k in
                                        ("kernels", "calls_total_by_kernel", "run_tag") if k in capture}
        timing = record.get("timing")
        if isinstance(timing, dict) and isinstance(timing.get("advance_seconds"), (int, float)):
            summary["advance_seconds"] = round(float(timing["advance_seconds"]), 2)
    summary["pbs_job"] = _job_number(record.get("pbs_job_id"))
    image = _image_of(record)
    if image:
        summary["image"] = image
    cleaned = _clean(summary)
    assert isinstance(cleaned, dict)
    return cleaned


# --------------------------------------------------------------------------- #
# Kernels: identity, relations, capability states
# --------------------------------------------------------------------------- #

def _numeric_candidates(closure: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The inventory's candidates by qualified id; cached evidence is dropped here."""

    out = {}
    for procedure in closure["procedures"]:
        if procedure["category"] != "numeric_kernel" or not procedure["in_configuration"] \
                or procedure["inert_in_configuration"]:
            continue
        out[procedure["qualified"]] = procedure
    return out


def _procedures_by_qualified(closure: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {p["qualified"]: p for p in closure["procedures"]}


def _contract_index(root: Path) -> dict[str, str]:
    """qualified_name -> repo-relative contract path, from the committed contracts."""

    index: dict[str, str] = {}
    directory = root / FUNCTIONS
    if not directory.is_dir():
        return index
    for path in sorted(directory.glob("*.yaml")):
        try:
            document = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        qualified = (document or {}).get("qualified_name")
        if isinstance(qualified, str):
            index[qualified] = str(path.relative_to(root))
    return index


def resolve_tracked_kernel(ledger_kernel: Mapping[str, Any], candidates: Mapping[str, Mapping[str, Any]],
                           contracts: Mapping[str, str]) -> str:
    """The qualified id of a tracked ledger kernel; ambiguity is an error, not a guess."""

    routine = ledger_kernel.get("routine") or ledger_kernel["kernel"]
    matches = [qualified for qualified, p in candidates.items() if p["name"] == routine]
    if len(matches) == 1:
        return matches[0]
    stage = ledger_kernel.get("stage_action")
    scoped = [q for q in matches if stage in (candidates[q].get("parent_actions") or [])]
    if len(scoped) == 1:
        return scoped[0]
    contract_path = ledger_kernel.get("contract_path")
    if contract_path:
        by_contract = [q for q, path in contracts.items() if path == contract_path and q in matches]
        if len(by_contract) == 1:
            return by_contract[0]
    raise ProgressExportError(
        f"tracked kernel {routine!r} joins ambiguously: {sorted(matches)} (stage {stage!r})")


def _replay_names(evidence: Mapping[str, Any]) -> list[str]:
    names: set[str] = set()
    for step in ("replay_full_chunk", "replay_single_column", "replay_public_api"):
        names.update(evidence.get(step) or [])
    return sorted(names)


def _capabilities(tracked: Mapping[str, Any] | None, records: Mapping[str, Mapping[str, Any]],
                  adapter_hint: Mapping[str, Any] | None, replacement_contexts: list[dict],
                  failures: list[dict]) -> dict[str, Any]:
    """The six capability states, each backed by named evidence."""

    def state(value: str, **extra: Any) -> dict[str, Any]:
        return {"state": value, **extra}

    if tracked is None:
        blockers = list((adapter_hint or {}).get("blockers") or [])
        contract = state("not-implemented")
        callable_state = state("needs-binding", blockers=blockers) if blockers else state("not-assessed")
        return {
            "contract": contract,
            "adapter_build": state("not-implemented"),
            "independently_callable": callable_state,
            "standalone_replay": state("not-assessed"),
            "in_model_replacement": state("verified") if any(c["state"] == "verified" for c in replacement_contexts)
            else (state("available") if replacement_contexts else state("not-implemented")),
            "original_replacement_bfb": state("verified") if any(c["state"] == "verified" for c in replacement_contexts)
            else (state("not-verified") if replacement_contexts else state("not-assessed")),
        }

    evidence = tracked.get("evidence") or {}
    contract = state("available", path=tracked.get("contract_path"), review=tracked.get("contract")) \
        if tracked.get("contract") == "reviewed" else state("not-implemented")

    build_names = [n for n in (evidence.get("standalone_build") or []) if n in records]
    adapter = state("available", evidence=build_names) if build_names else state("not-implemented")

    def replay_executed(summary: Mapping[str, Any]) -> bool:
        if summary.get("kind") == "capture-replay":
            return int(summary.get("records_compared") or 0) > 0
        return int(summary.get("calls") or 0) > 0 and int((summary.get("statuses") or {}).get("ok") or 0) > 0

    def replay_bfb(summary: Mapping[str, Any]) -> bool:
        if summary.get("kind") == "capture-replay":
            compared = int(summary.get("records_compared") or 0)
            return bool(summary.get("passed")) and compared > 0 \
                and int(summary.get("records_equal") or 0) == compared
        return bool(summary.get("bfb"))

    replays = [n for n in _replay_names(evidence) if n in records]
    executed = [n for n in replays if replay_executed(records[n])]
    replay_failures = [f["record"] for f in failures if f.get("capability") == "standalone_replay"]
    if executed:
        callable_state = state("verified", evidence=executed)
    elif build_names:
        callable_state = state("not-verified", evidence=build_names,
                               explanation="The adapter compiled successfully, but no recorded execution "
                                           "through the public entrypoint exists.")
    else:
        callable_state = state("not-assessed")

    replayed = [n for n in executed if replay_bfb(records[n])]
    if replayed:
        replay = state("verified", evidence=replayed, historical_failures=replay_failures)
    elif replay_failures:
        replay = state("failed", evidence=replay_failures)
    elif executed:
        replay = state("not-verified", evidence=executed)
    else:
        replay = state("not-assessed")

    verified_contexts = [c for c in replacement_contexts if c["state"] == "verified"]
    in_model = state("available", contexts=sorted(c["process"] for c in replacement_contexts)) \
        if replacement_contexts else state("not-implemented")
    if replacement_contexts:
        if verified_contexts:
            bfb = state("verified", contexts=sorted(c["process"] for c in verified_contexts))
        elif any(c["state"] == "failed" for c in replacement_contexts):
            bfb = state("failed")
        else:
            bfb = state("not-verified")
    else:
        bfb = state("not-assessed")
    return {
        "contract": contract,
        "adapter_build": adapter,
        "independently_callable": callable_state,
        "standalone_replay": replay,
        "in_model_replacement": in_model,
        "original_replacement_bfb": bfb,
    }


# --------------------------------------------------------------------------- #
# Per-process membership and call relationships
# --------------------------------------------------------------------------- #

def process_edges(pid: str, members: list[str], procedures: Mapping[str, Mapping[str, Any]],
                  candidates: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each member kernel under its nearest numeric ancestor within the process.

    The parent is found by walking the caller graph restricted to procedures
    attributed to the process; non-numerical intermediaries on the path are
    kept so the page can explain it.  A kernel with no numeric ancestor is a
    top-level entry whose ``via`` names the chain up to the action's root.
    """

    member_set = set(members)
    in_process = {q for q, p in procedures.items() if pid in (p.get("parent_actions") or [])}

    edges: list[dict[str, Any]] = []
    for kernel in members:
        found: list[tuple[str, tuple[str, ...]]] = []
        seen = {kernel}
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(kernel, ())])
        root_chain: tuple[str, ...] | None = None
        while queue:
            current, via = queue.popleft()
            for caller in sorted(procedures.get(current, {}).get("callers") or []):
                if caller in seen or caller not in in_process:
                    continue
                seen.add(caller)
                if caller in member_set:
                    found.append((caller, via))
                    continue
                if procedures.get(caller, {}).get("action_root") and root_chain is None:
                    root_chain = via + (caller,)
                queue.append((caller, via + (caller,)))
        for parent, via in sorted(found):
            edges.append({"parent": parent, "kernel": kernel, "via": list(via)})
        if root_chain is not None or not found:
            # a path reached the action's root without crossing another kernel:
            # the kernel is also (or only) a top-level entry of the process
            edges.append({"parent": None, "kernel": kernel, "via": list(root_chain or ())})
    return sorted(edges, key=lambda e: (e["parent"] or "", e["kernel"]))


# --------------------------------------------------------------------------- #
# The snapshot
# --------------------------------------------------------------------------- #

def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _stage_map(record: Mapping[str, Any], actions_by_tail: Mapping[str, str]) -> dict[str, dict[str, int]]:
    """Gate record's per-process replacement counts, keyed by action id."""

    out: dict[str, dict[str, int]] = {}
    for tail, counts in _pauses_by_kernel(record).items():
        pid = actions_by_tail.get(tail)
        if pid:
            out[pid] = counts
    return out


def build_progress_snapshot(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    inputs = load_inputs(root)
    catalog, ledger, closure = inputs["catalog"], inputs["ledger"], inputs["closure"]
    audit, runners, hooks = inputs["audit"], inputs["runners"], inputs["hooks"]

    observation_runs = inputs["observation_runs"]
    work_items = inputs["work_items"]
    active_by_kernel = {item["kernel"]: item for item in work_items
                        if item.get("state") in ("in_progress", "blocked")}

    def development_of(qualified: str) -> dict[str, Any]:
        """Coordination state only: a claim never changes a scientific column."""

        item = active_by_kernel.get(qualified)
        if item is None:
            return {"state": "unclaimed"}
        return {"state": item["state"], "stage": item.get("stage"), "owner": item.get("owner"),
                "branch": item.get("branch"), "next_gate": item.get("next_gate"),
                "started_at": item.get("started_at"), "updated_at": item.get("updated_at"),
                "blocker": item.get("blocker"), "note": item.get("note"),
                "target_processes": item.get("target_processes") or [],
                "owner_class": item.get("owner_class")}

    def observation_of(qualified: str) -> dict[str, Any]:
        """Per-run execution evidence for one candidate; unknown is never success."""

        out: dict[str, Any] = {}
        for key, _label, _rel in OBSERVATION_RUNS:
            record = observation_runs.get(key)
            if record is None:
                continue
            row = next((r for r in record["kernels"] if r["qualified"] == qualified), None)
            if row is None:
                out[key] = {"status": "unknown", "status_reason": "not in the observation inventory"}
                continue
            out[key] = {k: row[k] for k in ("status", "status_reason", "calls_total", "calls_by_context",
                                            "first_step", "last_step", "ranks_with_calls",
                                            "count_meaning", "note", "coverage") if k in row}
        return out

    procedures = _procedures_by_qualified(closure)
    candidates = _numeric_candidates(closure)
    contracts = _contract_index(root)
    audit_by_q = {p["qualified"]: p for p in audit["procedures"]}
    hook_kernels = {h["kernel"] for h in hooks.get("hooks") or []}
    catalog_entries = {e["id"]: e for e in catalog.get("entries") or []}
    ledger_actions = {a["id"]: a for a in ledger["actions"]}
    actions_by_tail = {a["id"].split(".", 1)[1]: a["id"] for a in ledger["actions"]}

    # Adapter hints for untracked candidates, from the ledger's per-action catalog analysis.
    adapter_hints: dict[str, dict[str, Any]] = {}
    for action in ledger["actions"]:
        for hint in action.get("kernel_candidates") or []:
            qualified = hint.get("qualified_name")
            if qualified and qualified not in adapter_hints:
                adapter_hints[qualified] = {"adapter_status": hint.get("adapter_status"),
                                            "blockers": sorted(hint.get("blockers") or [])}

    # Tracked kernels resolved to qualified ids; the join must be unambiguous.
    tracked_by_q: dict[str, Mapping[str, Any]] = {}
    for kernel in ledger["kernels"]:
        tracked_by_q[resolve_tracked_kernel(kernel, candidates, contracts)] = kernel

    # Every record any status claim names is opened, verified present, summarized.
    evidence: dict[str, dict[str, Any]] = {}

    def record_summary(name: str) -> dict[str, Any] | None:
        if name in evidence:
            return evidence[name]
        payload = _read_record(root, name)
        if payload is None:
            return None
        evidence[name] = summarize_evidence(name, payload)
        return evidence[name]

    # Failure records, attributed to kernels by their own fields.
    failures: list[dict[str, Any]] = []
    for path in sorted((root / VALIDATION).glob("*failure*.json")):
        name = path.name
        summary = record_summary(name)
        if summary is None:
            continue
        payload = _read_record(root, name) or {}
        routine = payload.get("kernel") or payload.get("segmented_original_kernels")
        capability = "standalone_replay" if "replay" in name else "original_replacement_bfb"
        stages = payload.get("python_stages") or []
        failures.append({
            "record": name,
            "kernel_routine": routine if isinstance(routine, str) else None,
            "capability": capability,
            "contexts": [actions_by_tail.get(s) for s in stages if actions_by_tail.get(s)],
        })

    # Replacement contexts: runner manifest stage x kernel, verified through the
    # gate records themselves (per-process pause counts + the bit-for-bit companion).
    replacements: list[dict[str, Any]] = []
    for runner in runners["runners"]:
        stage = runner["stage"]
        for kernel_row in runner.get("kernels") or []:
            routine = kernel_row["name"]
            mechanism = "hook" if routine in hook_kernels else "pause"
            tracked = next((t for t in ledger["kernels"]
                            if (t.get("routine") or t["kernel"]) == routine), None)
            gate_rows: list[dict[str, Any]] = []
            seen_gates: set[tuple[str, str | None]] = set()
            for gate in (tracked or {}).get("in_model_gates") or []:
                if (gate["record"], gate.get("bfb_record")) in seen_gates:
                    continue
                seen_gates.add((gate["record"], gate.get("bfb_record")))
                summary = record_summary(gate["record"])
                bfb_summary = record_summary(gate["bfb_record"]) if gate.get("bfb_record") else None
                if summary is None:
                    continue
                counts = _stage_map(_read_record(root, gate["record"]) or {}, actions_by_tail)
                calls = (counts.get(stage) or {}).get(routine)
                row = {
                    "record": gate["record"],
                    "bfb_record": gate.get("bfb_record"),
                    "bfb": bool(gate.get("bfb")),
                    "replacement_calls": calls,
                }
                if calls is None:
                    row["limitation"] = ("This record predates per-kernel counting or ran the kernel in "
                                         "another process; the recorded result stands with that limitation.")
                gate_rows.append(row)
                if bfb_summary is None and gate.get("bfb_record"):
                    row["bfb"] = False
            verified_rows = [g for g in gate_rows if g["bfb"] and (g["replacement_calls"] or 0) > 0]
            historical = [g for g in gate_rows if g["bfb"] and g["replacement_calls"] is None]
            failed_here = [f["record"] for f in failures
                           if f.get("kernel_routine") == routine and stage in (f.get("contexts") or [])
                           and f["capability"] == "original_replacement_bfb"]
            if verified_rows:
                state = "verified"
            elif failed_here and not historical:
                state = "failed"
            elif historical:
                state = "historical-evidence"
            elif gate_rows:
                state = "not-verified"
            else:
                state = "not-verified"
            replacements.append({
                "kernel_routine": routine,
                "process": stage,
                "mechanism": mechanism,
                "within": kernel_row.get("within"),
                "state": state,
                "gates": gate_rows,
                "historical_failures": failed_here,
                "note": kernel_row.get("note"),
                "requires_development_image": mechanism == "hook",
            })

    # Every record the ledger's evidence names is opened before any status is judged.
    for tracked in ledger["kernels"]:
        for names in (tracked.get("evidence") or {}).values():
            for name in names or []:
                record_summary(name)

    # Kernel records.
    kernels: dict[str, dict[str, Any]] = {}
    routine_to_q = {}
    for qualified, tracked in tracked_by_q.items():
        routine_to_q[tracked.get("routine") or tracked["kernel"]] = qualified
    for qualified, procedure in sorted(candidates.items()):
        tracked = tracked_by_q.get(qualified)
        routine = procedure["name"]
        contexts = [r for r in replacements if r["kernel_routine"] == routine
                    and routine_to_q.get(routine) == qualified]
        kernel_failures = [f for f in failures if f.get("kernel_routine") == routine
                           and routine_to_q.get(routine) == qualified]
        audit_row = audit_by_q.get(qualified) or {}
        capabilities = _capabilities(tracked, evidence, adapter_hints.get(qualified), contexts, kernel_failures)
        if tracked is None and qualified in contracts:
            capabilities["contract"] = {"state": "available", "path": contracts[qualified], "review": "committed"}
        classification = audit_row.get("classification")
        redirect = {
            "classification": classification,
            "redirectable": bool(audit_row.get("redirectable")),
            "reading": "static: a relocation is a compiled call site, not an implemented hook",
        }
        if classification == "no-call-relocation":
            redirect["blocker"] = "inlined at every compiled call site; no symbol redirection can reach it"
        elif classification == "not-in-archive":
            redirect["blocker"] = "internal procedure absorbed into its host; it has no symbol of its own"
        callers = sorted(procedure.get("callers") or [])
        kernels[qualified] = {
            "id": qualified,
            "routine": routine,
            "module": procedure.get("module"),
            "kind": procedure.get("kind"),
            "host": procedure.get("host"),
            "public": bool(procedure.get("public")),
            "source": {"file": procedure.get("source"), "line_start": procedure.get("line_start"),
                       "line_end": procedure.get("line_end")},
            "processes": sorted(procedure.get("parent_actions") or []),
            "callers": callers,
            "tracked": tracked is not None,
            "status": (tracked or {}).get("status"),
            "missing": (tracked or {}).get("missing"),
            "owner_class": (tracked or {}).get("owner_class"),
            "note": (tracked or {}).get("note"),
            "module_state": sorted((tracked or {}).get("evidence", {}).get("module_state") or []),
            "observation": observation_of(qualified),
            "development": development_of(qualified),
            "capabilities": capabilities,
            "redirect": redirect,
            "adapter_hint": adapter_hints.get(qualified),
            "failures": [f["record"] for f in kernel_failures],
        }

    # Process records and membership.
    processes: list[dict[str, Any]] = []
    membership: dict[str, dict[str, Any]] = {}
    members_by_pid: dict[str, list[str]] = {}
    for qualified, procedure in candidates.items():
        for pid in procedure.get("parent_actions") or []:
            members_by_pid.setdefault(pid, []).append(qualified)
    for action in ledger["actions"]:
        pid = action["id"]
        entry = catalog_entries.get(pid) or {}
        members = sorted(members_by_pid.get(pid, []))
        # the core kernels: what the process's Python class exposes for replacement
        # (the ledger's per-action list), a far smaller set than the statically
        # reachable candidates below -- the page must never conflate the two
        core = []
        for routine in action.get("kernels") or []:
            tracked = next((k for k in ledger["kernels"]
                            if (k.get("routine") or k["kernel"]) == routine
                            and k.get("stage_action") == pid), None)
            core.append({
                "routine": routine,
                "id": routine_to_q.get(routine),
                "owner_class": (tracked or {}).get("owner_class"),
                "status": (tracked or {}).get("status"),
            })
        membership[pid] = {
            "kernels": members,
            "edges": process_edges(pid, members, procedures, candidates),
            "inventoried": bool(members) or pid in {a["id"] for a in closure["actions"] if a.get("procedures")},
        }
        processes.append({
            "id": pid,
            "native_id": action.get("native_id"),
            "operation": action.get("operation"),
            "display_name": entry.get("display_name") or pid.split(".", 1)[-1],
            "description": entry.get("description"),
            "phase": action.get("phase"),
            "kind": action.get("kind"),
            "classification": action.get("classification"),
            "granularity": action.get("granularity"),
            "parent_stage": action.get("parent_stage"),
            "enabled": bool(action.get("enabled")),
            "activity": action.get("activity"),
            "activity_basis": action.get("activity_basis"),
            "alternate_of": action.get("alternate_of") or [],
            "default_index": entry.get("default_index"),
            "in_default": bool(entry.get("in_default", True)),
            "python_api": "available",
            "python_class": action.get("python_class"),
            "class_kind": "dedicated" if action.get("python_class") else "generic",
            "core_kernels": core,
            "ledger_coverage": action.get("coverage"),
            "note": action.get("note"),
        })
    processes.sort(key=lambda p: (p["default_index"] is None, p["default_index"] or 0, p["id"]))

    additional_apis = sorted(
        ({"id": e["id"], "name": e.get("name"), "display_name": e.get("display_name"),
          "qualified_name": e.get("qualified_name"), "description": e.get("description"),
          "present": bool(e.get("present")), "addable": bool(e.get("addable")), "reason": e.get("reason")}
         for e in catalog.get("entries") or [] if e.get("kind") == "runtime_catalog_process"),
        key=lambda e: e["id"])

    unmapped = sorted(q for q, p in candidates.items() if not (p.get("parent_actions") or []))

    def cap_count(name: str, value: str) -> int:
        return sum(1 for k in kernels.values() if k["capabilities"][name]["state"] == value)

    totals = {
        "processes": len(processes),
        "processes_enabled": sum(1 for p in processes if p["enabled"]),
        "processes_with_dedicated_class": sum(1 for p in processes if p["class_kind"] == "dedicated"),
        "additional_apis": len(additional_apis),
        "additional_apis_runnable": sum(1 for e in additional_apis if not e["reason"]),
        "candidate_kernels": len(kernels),
        "tracked_kernels": sum(1 for k in kernels.values() if k["tracked"]),
        "tracked_by_status": {
            status: sum(1 for k in kernels.values() if k["tracked"] and k["status"] == status)
            for status in sorted({k["status"] for k in kernels.values() if k["tracked"]})},
        "independently_callable_verified": cap_count("independently_callable", "verified"),
        "standalone_replay_verified": cap_count("standalone_replay", "verified"),
        "in_model_replacement_available": sum(
            1 for k in kernels.values() if k["capabilities"]["in_model_replacement"]["state"] in ("available", "verified")),
        "original_replacement_bfb_verified": cap_count("original_replacement_bfb", "verified"),
        "redirectable_calls": sum(1 for k in kernels.values() if k["redirect"]["redirectable"]),
        "observed_by_run": {
            key: sum(1 for k in kernels.values() if (k["observation"].get(key) or {}).get("status") == "observed")
            for key, _label, _rel in OBSERVATION_RUNS if key in observation_runs
        },
        "unmapped_kernels": len(unmapped),
        "work_in_progress": sum(1 for i in work_items if i.get("state") == "in_progress"),
        "work_blocked": sum(1 for i in work_items if i.get("state") == "blocked"),
    }

    content = {
        "schema_version": SCHEMA_VERSION,
        "case": ledger.get("configuration"),
        "cam_source_revision": ledger.get("cam_source_revision"),
        "inputs": inputs["hashes"],
        "notes": {
            "banner": "Development snapshot. Some validated capabilities require an experimental image "
                      "and are not enabled in the default build.",
            "snapshot": "This is a published snapshot of committed records, not a live HPC job monitor.",
            "process_vs_kernel": "A process is a workflow operation. It can contain many numerical kernels. "
                                 "Calling the whole process does not mean every internal kernel is "
                                 "independently callable or replaceable.",
            "class_delegation": "A Python class may delegate numerical execution to the original Fortran; "
                                "class availability does not mean the calculations were translated into Python.",
            "candidates": "Candidate kernels come from a static reading of the configured call tree: they are "
                          "potentially reachable, not necessarily observed during execution.",
            "shared_kernels": "A kernel used by several processes is counted once globally; per-process totals "
                              "are not additive.",
            "observation": "Observed means the counting image recorded real calls in the selected validated "
                           "run, attributed to the process they ran in; a kernel observed in one process is "
                           "never marked observed in another. Zero calls count as not observed only where "
                           "instrumentation coverage is complete and the run finished bit-for-bit; everything "
                           "else stays unknown. Where coverage is partial, totals are lower bounds.",
            "no_observation_run": "No validated observation run available.",
            "development_vs_evidence": "A work item records that a kernel is being implemented -- by whom, "
                                       "on which branch, at which stage, toward which gate. It is development "
                                       "coordination only: In progress never renders as Replaceable, Verified "
                                       "or Done, and completion is always computed from the validation records.",
            "core_vs_candidates": "Replaceable kernels are the ones a process's Python class exposes for "
                                  "replacement today; an entry that exists but lacks a verified path shows its "
                                  "unverified state. Candidate numerical functions are everything the call-tree "
                                  "inventory reaches from the process recursively -- drivers, per-point helpers, "
                                  "saturation and packing libraries included. Exposing a process does not "
                                  "expose every candidate inside it.",
            "replacement_scope": "A replacement gate is scoped to the process and call site actually tested; "
                                 "it does not verify the same kernel's other callers.",
            "bfb_meaning": "Original-kernel replacement BFB verified does not prove that an arbitrary "
                           "replacement (for example a neural network) is scientifically correct.",
        },
        "capability_explanations": CAPABILITY_EXPLANATIONS,
        "work_items": work_items,
        "observation_runs": [
            {
                "key": key,
                "label": label,
                "run": record["run"],
                "image": record["image"],
                "summary": record["summary"],
                "validated": bool(record["run"].get("bfb")) and record["run"].get("run_status") == "passed",
            }
            for key, label, _rel in OBSERVATION_RUNS
            if (record := observation_runs.get(key)) is not None
        ],
        "process_observation": {
            key: observation_runs[key].get("process_calls", {})
            for key, _label, _rel in OBSERVATION_RUNS if key in observation_runs
        },
        "processes": processes,
        "additional_apis": additional_apis,
        "kernels": kernels,
        "process_membership": membership,
        "replacements": replacements,
        "evidence": dict(sorted(evidence.items())),
        "failure_records": sorted(failures, key=lambda f: f["record"]),
        "unmapped_kernels": unmapped,
        "totals": totals,
    }
    cleaned = _clean(content)
    assert isinstance(cleaned, dict)
    _assert_publishable(cleaned)
    content_hash = hashlib.sha256(json.dumps(cleaned, sort_keys=True).encode()).hexdigest()
    return {
        **cleaned,
        "content_hash": content_hash,
        "volatile": {
            "generated_at": None,   # filled by the exporter; never part of the hash
            "commit": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        },
    }


def _assert_publishable(content: Any, path: str = "$") -> None:
    """No published string may carry a personal path or site name."""

    if isinstance(content, str):
        lowered = content.lower()
        for fragment in FORBIDDEN:
            if fragment in lowered:
                raise ProgressExportError(f"forbidden fragment {fragment!r} at {path}: {content[:80]!r}")
    elif isinstance(content, list):
        for index, item in enumerate(content):
            _assert_publishable(item, f"{path}[{index}]")
    elif isinstance(content, dict):
        for key, item in content.items():
            _assert_publishable(key, f"{path}.{key}")
            _assert_publishable(item, f"{path}.{key}")


__all__ = ["ProgressExportError", "build_progress_snapshot", "load_inputs",
           "process_edges", "resolve_tracked_kernel", "summarize_evidence"]
