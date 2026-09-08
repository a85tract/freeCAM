// The exported progress snapshot (tools/export_progress_snapshot.py). The page
// renders this one file and never contacts a runtime.

export type CapabilityState =
  | "available"
  | "not-implemented"
  | "needs-binding"
  | "not-assessed"
  | "verified"
  | "not-verified"
  | "failed"
  | "historical-evidence"
  | "not-applicable";

export interface Capability {
  state: CapabilityState;
  path?: string;
  review?: string;
  evidence?: string[];
  contexts?: string[];
  blockers?: string[];
  explanation?: string;
  historical_failures?: string[];
}

export interface KernelRecord {
  id: string;
  routine: string;
  module: string | null;
  kind: string | null;
  host: string | null;
  public: boolean;
  source: { file: string | null; line_start: number | null; line_end: number | null };
  processes: string[];
  callers: string[];
  tracked: boolean;
  status: string | null;
  missing: string[] | null;
  owner_class: string | null;
  note: string | null;
  module_state: string[];
  capabilities: Record<string, Capability>;
  redirect: { classification: string | null; redirectable: boolean; reading: string; blocker?: string };
  adapter_hint: { adapter_status: string | null; blockers: string[] } | null;
  failures: string[];
}

export interface CoreKernel {
  routine: string;
  id: string | null;
  owner_class: string | null;
  status: string | null;
}

export interface ProcessRecord {
  id: string;
  native_id: number | null;
  operation: string | null;
  display_name: string;
  description: string | null;
  phase: string | null;
  kind: string | null;
  classification: string | null;
  granularity: string | null;
  parent_stage: string | null;
  enabled: boolean;
  activity: string | null;
  activity_basis: string | null;
  alternate_of: string[];
  default_index: number | null;
  in_default: boolean;
  python_api: string;
  python_class: string | null;
  class_kind: "dedicated" | "generic";
  core_kernels: CoreKernel[];
  ledger_coverage: string | null;
  note: string | null;
}

export interface TreeEdge {
  parent: string | null;
  kernel: string;
  via: string[];
}

export interface Membership {
  kernels: string[];
  edges: TreeEdge[];
  inventoried: boolean;
}

export interface GateRow {
  record: string;
  bfb_record: string | null;
  bfb: boolean;
  replacement_calls: number | null;
  limitation?: string;
}

export interface Replacement {
  kernel_routine: string;
  process: string;
  mechanism: "pause" | "hook";
  within: string | null;
  state: string;
  gates: GateRow[];
  historical_failures: string[];
  note: string | null;
  requires_development_image: boolean;
}

export interface AdditionalApi {
  id: string;
  name: string | null;
  display_name: string | null;
  qualified_name: string | null;
  description: string | null;
  present: boolean;
  addable: boolean;
  reason: string | null;
}

export interface Snapshot {
  schema_version: number;
  content_hash: string;
  volatile: { generated_at: string | null; commit: string | null; branch: string | null };
  case: Record<string, unknown> | null;
  cam_source_revision: string | null;
  inputs: Record<string, string>;
  notes: Record<string, string>;
  capability_explanations: Record<string, string>;
  processes: ProcessRecord[];
  additional_apis: AdditionalApi[];
  kernels: Record<string, KernelRecord>;
  process_membership: Record<string, Membership>;
  replacements: Replacement[];
  evidence: Record<string, Record<string, unknown>>;
  failure_records: { record: string; kernel_routine: string | null; capability: string; contexts: string[] }[];
  unmapped_kernels: string[];
  totals: Record<string, number | Record<string, number>>;
}
