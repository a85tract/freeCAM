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

export interface Bar {
  key: string;
  label: string;
  done: number;
  total: number;
  inventoried: boolean;
  denominator: string;
}

/** The four per-process bars. Blocked and unassessed kernels stay in the denominator. */
export function processBars(snapshot: Snapshot, pid: string): Bar[] {
  const membership: Membership | undefined = snapshot.process_membership[pid];
  const members = membership?.kernels ?? [];
  const inventoried = Boolean(membership?.inventoried && members.length > 0) || members.length > 0;
  const kernels = members.map((k) => snapshot.kernels[k]).filter(Boolean) as KernelRecord[];
  const rows = snapshot.replacements.filter((r) => r.process === pid);
  const routineRows = new Map<string, Replacement>();
  for (const row of rows) routineRows.set(row.kernel_routine, row);
  const capCount = (name: string, states: string[]) =>
    kernels.filter((k) => states.includes(k.capabilities[name]?.state)).length;
  const denominator =
    "of the process's candidate kernels (statically reachable in this configuration; " +
    "blocked and unassessed candidates stay in the denominator)";
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
export function overviewTiles(snapshot: Snapshot): { label: string; value: string; note: string }[] {
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
      label: "Tracked kernel records",
      value: `${tracked}`,
      note: "kernels with implementation/validation records in the ledger",
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
