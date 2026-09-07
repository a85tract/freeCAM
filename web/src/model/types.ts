// The document and catalog as the Python side serialises them
// (src/freecam/pi_cam/workflow_builder).  Field names match the payloads
// exactly so a document round-trips between the page and the service.

export type Severity = "error" | "warning" | "info";

export type KernelBindingKind = "original" | "surrogate";

export interface KernelBinding {
  kind: KernelBindingKind;
  path: string | null;
}

export interface VariableDeclaration {
  name: string;
  like: string;
  units: string;
  output: boolean;
}

export type ParameterValue = number | string | boolean | null | ParameterValue[] | { [key: string]: ParameterValue };

export interface NodeConfiguration {
  parameters: Record<string, ParameterValue>;
  python_source: string | null;
  kernels: Record<string, KernelBinding>;
  variables: VariableDeclaration[];
}

export interface ParameterSpec {
  name: string;
  dtype: string;
  notes: string;
}

export interface KernelCapability {
  kernel: string;
  stage_action: string;
  stage_class: string;
  owner_class: string;
  bindable: boolean;
  validated: boolean;
  reason: string | null;
  evidence: string[];
}

export interface NodeMetadata {
  control_owner?: string;
  parameters?: ParameterSpec[];
  kernels?: KernelCapability[];
  adapter_status?: string;
  role?: string | null;
  parent_actions?: string[];
  arguments?: { name: string; intent: string; dtype: string; dimensions: string[] }[];
  current_case_loadable?: boolean;
  [key: string]: unknown;
}

export interface WorkflowNode {
  id: string;
  name: string;
  display_name: string;
  qualified_name: string;
  operation: string;
  phase: string;
  kind: string;
  implementation: string;
  enabled: boolean;
  movable: boolean;
  removable: boolean;
  locked: boolean;
  scientific: boolean;
  reads: string[];
  writes: string[];
  source: string | null;
  parent_stage: string | null;
  granularity: string;
  origin: "default" | "python" | "catalog";
  native_id: number | null;
  default_index: number | null;
  configuration: NodeConfiguration;
  metadata: NodeMetadata;
}

export interface WorkflowDocument {
  schema_version: number;
  revision: number;
  experimental: boolean;
  case: string;
  nsteps: number;
  namelist: Record<string, ParameterValue>;
  catalog_version: string;
  source_version: string;
  workflow_hash: string;
  nodes: WorkflowNode[];
}

export interface CatalogEntry extends WorkflowNode {
  category: string;
  addable: boolean;
  present: boolean;
  reason: string | null;
  in_default: boolean;
  description: string | null;
}

export interface CatalogRules {
  locked_operations: string[];
  control_skeleton: string[];
  parent_leaf_groups: Record<string, string[]>;
}

export interface CatalogSnapshot {
  schema_version: number;
  cases: Record<string, string>;
  default_nodes: WorkflowNode[];
  entries: CatalogEntry[];
  capabilities: KernelCapability[];
  parameters: Record<string, ParameterSpec[]>;
  rules: CatalogRules;
  source_revision: string;
  catalog_hash: string;
  default_document: WorkflowDocument;
  generated_at?: string;
  commit?: string | null;
}

export interface Issue {
  severity: Severity;
  code: string;
  message: string;
  node_id: string | null;
  field: string | null;
}

export interface ValidationReport {
  status: "valid" | "warning" | "error";
  level: "browser" | "local";
  revision: number;
  workflow_hash: string;
  issues: Issue[];
  error_count: number;
  warning_count: number;
  checks: Record<string, unknown>;
  disclaimer: string;
}

export const LOCKED_OPERATIONS = new Set(["boundary_import", "advance_timestep", "boundary_export"]);

export const SCIENTIFIC_KINDS = new Set([
  "scheme",
  "coupling",
  "dynamics",
  "python_process",
  "runtime_fortran_process",
  "runtime_catalog_process",
  "catalog_process",
]);

export const CONTROL_SKELETON_KINDS = new Set(["boundary", "clock", "io", "service", "kernel"]);

export const PYTHON_NODE_PREFIX = "python:";
