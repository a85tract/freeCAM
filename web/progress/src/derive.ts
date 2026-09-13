// Pure derivations over the snapshot: every count on the page comes from these,
// never from a number typed into a component.

import type { CapabilityState, KernelRecord, Membership, Replacement, Snapshot, TreeEdge } from "./types";

export const STATE_LABELS: Record<CapabilityState, string> = {
  available: "Available",
  "not-implemented": "Not implemented",
  "needs-binding": "Needs binding",
  "not-assessed": "Not assessed",
  verified: "Verified",
  "not-verified": "Not verified",
  failed: "Failed",
  "historical-evidence": "Historical evidence",
  "not-applicable": "Not applicable",
};

export const CAPABILITY_LABELS: Record<string, string> = {
  contract: "Contract available",
  adapter_build: "Adapter built",
  independently_callable: "Independently callable",
  standalone_replay: "Standalone replay verified",
  in_model_replacement: "In-model replacement available",
  original_replacement_bfb: "Original-kernel replacement BFB verified",
};

export function stateLabel(state: string): string {
  return STATE_LABELS[state as CapabilityState] ?? state;
}

/** Which browser view a process belongs to. */
export function processGroup(classification: string | null): "physics" | "dynamics" | "control" {
  if (classification === "numeric_scheme") return "physics";
  if (classification === "dynamics") return "dynamics";
  return "control";
}

/** The run the page selects by default: one month when validated, else 50 steps, else none. */
export function defaultRunKey(snapshot: Snapshot): string | null {
  const validated = (snapshot.observation_runs ?? []).filter((run) => run.validated);
  const month = validated.find((run) => run.key === "1month");
  return (month ?? validated[0])?.key ?? null;
}

export type ProcessKernelState = "observed" | "not-observed-here" | "gap";

/** One process's per-kernel execution state under the selected run.

    Observation is strictly per process: calls recorded in another process never
    mark a kernel observed here.  Zero calls mean "not observed here" only when
    the kernel's counting coverage is complete and the run finished cleanly;
    partial coverage and uninstrumented kernels stay gaps. */
export function processKernelStates(snapshot: Snapshot, pid: string, runKey: string): Map<string, ProcessKernelState> {
  const membership = snapshot.process_membership[pid];
  const members = membership?.kernels ?? [];
  const here = snapshot.process_observation?.[runKey]?.[pid] ?? {};
  const states = new Map<string, ProcessKernelState>();
  // calls observed in this process at run time that the static tree misses
  // are shown, never hidden -- they count as observed when this run saw them
  for (const kid of membership?.runtime_only ?? []) {
    if ((here[kid]?.calls ?? 0) > 0) states.set(kid, "observed");
  }
  for (const kid of members) {
    if ((here[kid]?.calls ?? 0) > 0) {
      states.set(kid, "observed");
      continue;
    }
    const global = snapshot.kernels[kid]?.observation?.[runKey];
    const settled = global && (global.status === "observed" || global.status === "not-observed-in-this-run");
    states.set(kid, settled && global.coverage === "full" ? "not-observed-here" : "gap");
  }
  return states;
}

export interface ExecutionInventory {
  candidates: number;
  observed: number;
  coveredNotObserved: number;
  gaps: number;
  runtimeOnly: number;
}

export function executionInventory(snapshot: Snapshot, pid: string, runKey: string): ExecutionInventory {
  const states = processKernelStates(snapshot, pid, runKey);
  const staticMembers = new Set(snapshot.process_membership[pid]?.kernels ?? []);
  let observed = 0;
  let covered = 0;
  let gaps = 0;
  let runtimeOnly = 0;
  for (const [kid, state] of states.entries()) {
    if (state === "observed") {
      observed += 1;
      if (!staticMembers.has(kid)) runtimeOnly += 1;
    } else if (state === "not-observed-here") covered += 1;
    else gaps += 1;
  }
  return { candidates: staticMembers.size, observed, coveredNotObserved: covered, gaps, runtimeOnly };
}

export interface ReplaceableCompletion {
  observedReplaceable: number;
  closed: number;
  unclassifiedObserved: number;
  excludedObserved: number;
}

/** The completion denominator of one process under one run: kernels observed
    executing here AND reviewed as replaceable_numeric.  Closed means the full
    loop -- standalone replay verified and the replacement gate bit-for-bit in
    this process.  Unclassified observed routines block completion. */
export function replaceableCompletion(snapshot: Snapshot, pid: string, runKey: string): ReplaceableCompletion {
  const states = processKernelStates(snapshot, pid, runKey);
  let observedReplaceable = 0;
  let closed = 0;
  let unclassifiedObserved = 0;
  let excludedObserved = 0;
  for (const [kid, state] of states.entries()) {
    if (state !== "observed") continue;
    const kernel = snapshot.kernels[kid];
    if (!kernel) continue;
    if (kernel.category === "unclassified") {
      unclassifiedObserved += 1;
      continue;
    }
    if (kernel.category !== "replaceable_numeric") {
      excludedObserved += 1;
      continue;
    }
    observedReplaceable += 1;
    const replaced = snapshot.replacements.some(
      (row) => row.kernel_routine === kernel.routine && row.process === pid && row.state === "verified");
    if (replaced && kernel.capabilities.standalone_replay?.state === "verified") closed += 1;
  }
  return { observedReplaceable, closed, unclassifiedObserved, excludedObserved };
}

export interface Bar {
  key: string;
  label: string;
  done: number;
  total: number;
  inventoried: boolean;
  denominator: string;
}

/** The four per-process capability bars.

    With a validated observation run selected the denominator is the kernels
    actually observed executing in this process during that run; candidates
    never observed here cannot be verified here and are counted separately by
    the execution inventory.  With no validated run the bars fall back to the
    static candidates under an explicitly static label -- never silently.
    Blocked and unassessed kernels stay in whichever denominator applies. */
export function processBars(snapshot: Snapshot, pid: string, runKey: string | null): Bar[] {
  const membership: Membership | undefined = snapshot.process_membership[pid];
  const states = runKey ? processKernelStates(snapshot, pid, runKey) : null;
  // the denominator: static members, plus the calls the selected run observed
  // here beyond the static tree; with no validated run, static members only
  const members = [
    ...(membership?.kernels ?? []),
    ...(membership?.runtime_only ?? []).filter((kid) => states?.get(kid) === "observed"),
  ];
  const inventoried = Boolean(membership?.inventoried && members.length > 0) || members.length > 0;
  const allKernels = members.map((k) => snapshot.kernels[k]).filter(Boolean) as KernelRecord[];
  const kernels = states ? allKernels.filter((k) => states.get(k.id) === "observed") : allKernels;
  const rows = snapshot.replacements.filter((r) => r.process === pid);
  const routineRows = new Map<string, Replacement>();
  for (const row of rows) routineRows.set(row.kernel_routine, row);
  const capCount = (name: string, wanted: string[]) =>
    kernels.filter((k) => wanted.includes(k.capabilities[name]?.state)).length;
  const excluded = allKernels.length - kernels.length;
  const denominator = states
    ? "of the kernels observed executing in this process in the selected run" +
      (excluded > 0
        ? `; observation coverage is incomplete: ${excluded} candidates remain unobserved or uninstrumented here`
        : "") +
      "; blocked and unassessed kernels stay in the denominator"
    : "of the process's static candidate kernels -- no validated observation run backs this denominator; " +
      "blocked and unassessed candidates stay in it";
  return [
    {
      key: "independently_callable",
      label: "Independently callable",
      done: capCount("independently_callable", ["verified"]),
      total: kernels.length,
      inventoried,
      denominator,
    },
    {
      key: "standalone_replay",
      label: "Standalone replay verified",
      done: capCount("standalone_replay", ["verified"]),
      total: kernels.length,
      inventoried,
      denominator,
    },
    {
      key: "in_model_replacement",
      label: "In-model replacement available",
      done: kernels.filter((k) => routineRows.has(k.routine) && k.processes.includes(pid)).length,
      total: kernels.length,
      inventoried,
      denominator: denominator + "; replacement availability is scoped to this process",
    },
    {
      key: "original_replacement_bfb",
      label: "Original-kernel replacement BFB verified",
      done: kernels.filter((k) => routineRows.get(k.routine)?.state === "verified").length,
      total: kernels.length,
      inventoried,
      denominator: denominator + "; a gate verifies only the process and call site it ran",
    },
  ];
}

export interface TreeNode {
  kernel: string;
  via: string[];
  children: TreeNode[];
  cycle: boolean;
}

/** The process's kernel tree from its edges; recursion is cut, never looped. */
export function buildTree(membership: Membership): TreeNode[] {
  const children = new Map<string, TreeEdge[]>();
  const roots: TreeEdge[] = [];
  for (const edge of membership.edges) {
    if (edge.parent === null) roots.push(edge);
    else {
      const bucket = children.get(edge.parent) ?? [];
      bucket.push(edge);
      children.set(edge.parent, bucket);
    }
  }
  const build = (edge: TreeEdge, path: Set<string>): TreeNode => {
    if (path.has(edge.kernel)) return { kernel: edge.kernel, via: edge.via, children: [], cycle: true };
    const next = new Set(path);
    next.add(edge.kernel);
    const below = (children.get(edge.kernel) ?? []).map((child) => build(child, next));
    return { kernel: edge.kernel, via: edge.via, children: below, cycle: false };
  };
  return roots.map((edge) => build(edge, new Set()));
}

export interface SearchHit {
  kind: "process" | "kernel" | "api";
  id: string;
  title: string;
  detail: string;
}

/** Search by readable name, Fortran name, module, and Python class. */
export function searchAll(snapshot: Snapshot, query: string, limit = 40): SearchHit[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return [];
  const hits: SearchHit[] = [];
  for (const process of snapshot.processes) {
    const haystack = [process.display_name, process.operation, process.id, process.python_class]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    if (haystack.includes(needle)) {
      hits.push({ kind: "process", id: process.id, title: process.display_name, detail: process.id });
    }
  }
  for (const kernel of Object.values(snapshot.kernels)) {
    const haystack = [kernel.id, kernel.routine, kernel.module, kernel.owner_class]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    if (haystack.includes(needle)) {
      hits.push({ kind: "kernel", id: kernel.id, title: kernel.id, detail: kernel.source.file ?? "" });
    }
    if (hits.length >= limit) break;
  }
  for (const api of snapshot.additional_apis) {
    if (hits.length >= limit) break;
    const haystack = [api.name, api.qualified_name].filter(Boolean).join(" ").toLowerCase();
    if (haystack.includes(needle)) {
      hits.push({ kind: "api", id: api.id, title: api.qualified_name ?? api.id, detail: api.reason ?? "" });
    }
  }
  return hits.slice(0, limit);
}

/** Overview tiles, every number derived from the snapshot's own records. */
export function overviewTiles(snapshot: Snapshot, runKey: string | null): { label: string; value: string; note: string }[] {
  const processes = snapshot.processes;
  const kernels = Object.values(snapshot.kernels);
  const enabled = processes.filter((p) => p.enabled).length;
  const dedicated = processes.filter((p) => p.class_kind === "dedicated").length;
  const tracked = kernels.filter((k) => k.tracked).length;
  const callable = kernels.filter((k) => k.capabilities.independently_callable?.state === "verified").length;
  const replayed = kernels.filter((k) => k.capabilities.standalone_replay?.state === "verified").length;
  const replaceable = kernels.filter((k) =>
    ["available", "verified"].includes(k.capabilities.in_model_replacement?.state)
  ).length;
  const bfb = kernels.filter((k) => k.capabilities.original_replacement_bfb?.state === "verified").length;
  return [
    {
      label: "Process APIs available",
      value: `${processes.length}`,
      note: `${enabled} enabled in the default workflow; every plan action is callable from Python`,
    },
    {
      label: "Dedicated Python classes",
      value: `${dedicated} of ${processes.length}`,
      note: "a class may delegate its numerics to the original Fortran",
    },
    {
      label: "Candidate numerical kernels",
      value: `${kernels.length}`,
      note: "unique, statically reachable in this configuration; shared kernels counted once",
    },
    {
      label: "Observed executing",
      value: runKey
        ? `${kernels.filter((k) => k.observation?.[runKey]?.status === "observed").length} of ${kernels.length}`
        : "no validated observation run",
      note: runKey
        ? "entered at least once in the selected validated run, counted in place by the counting image"
        : "an instrumented run of the case records which candidates actually execute",
    },
    {
      label: "Tracked kernel records",
      value: `${tracked}`,
      note: "kernels exposed for replacement by the stage classes, with implementation/validation records",
    },
    {
      label: "Independently callable",
      value: `${callable} of ${kernels.length}`,
      note: "a recorded Python call executed the kernel outside its parent process",
    },
    {
      label: "Standalone replay verified",
      value: `${replayed} of ${kernels.length}`,
      note: "captured inputs reproduced original outputs bit-for-bit",
    },
    {
      label: "In-model replacement available",
      value: `${replaceable} of ${kernels.length}`,
      note: "a supported pause or hook exists in at least one process",
    },
    {
      label: "Replacement BFB verified",
      value: `${bfb} of ${kernels.length}`,
      note: "the original's answer through the mechanism kept the model bit-for-bit, in the tested process",
    },
  ];
}
