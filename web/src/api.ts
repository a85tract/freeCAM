// The page's two modes.  Locally, a Python service holds a Driver and the
// draft; the page edits, then hands it documents to check, generate and run.
// On the published preview there is no service: the page reads the catalog
// snapshot and does everything but run.

import type { CatalogSnapshot, ValidationReport, WorkflowDocument } from "./model/types";

export type Mode = "local" | "preview";

export interface RunStatus {
  state: "idle" | "initializing" | "queued" | "running" | "completed" | "error" | "stopping" | "closed";
  step: number | null;
  target_step: number | null;
  job_id: string | null;
  run_dir: string | null;
  workflow_hash: string | null;
  applied_hash: string | null;
  message: string | null;
  model_calls: Record<string, number>;
  started_at: string | null;
  finished_at: string | null;
}

export interface ServiceState {
  mode: "local";
  snapshot: CatalogSnapshot;
  draft: WorkflowDocument | null;
  run: RunStatus;
  case: string;
  nsteps: number;
  resources: { ranks: number; nodes: number; queue: string | null; walltime: string; account_set: boolean };
  driver_initialized: boolean;
  version: string;
}

export interface LogEvent {
  sequence: number;
  time: string;
  level: "info" | "warning" | "error";
  message: string;
}

export interface GeneratedFiles {
  directory: string;
  files: Record<string, string>;
  workflow_hash: string;
}

export class ServiceError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function tokenFromLocation(): string | null {
  const url = new URL(window.location.href);
  const token = url.searchParams.get("token") ?? new URLSearchParams(url.hash.replace(/^#/, "")).get("token");
  if (token) {
    try {
      sessionStorage.setItem("freecam-ui-token", token);
    } catch {
      // storage may be unavailable; the token still works for this load
    }
    url.searchParams.delete("token");
    window.history.replaceState(null, "", url.toString());
    return token;
  }
  try {
    return sessionStorage.getItem("freecam-ui-token");
  } catch {
    return null;
  }
}

export class ServiceClient {
  readonly token: string | null;

  constructor(token: string | null = tokenFromLocation()) {
    this.token = token;
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers: Record<string, string> = { "Content-Type": "application/json", ...(init.headers as Record<string, string> | undefined) };
    if (this.token) headers["X-FreeCAM-Token"] = this.token;
    const response = await fetch(path, { ...init, headers, credentials: "same-origin" });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
      } catch {
        // keep the status text
      }
      throw new ServiceError(response.status, detail);
    }
    return (await response.json()) as T;
  }

  state(): Promise<ServiceState> {
    return this.request<ServiceState>("api/state");
  }

  saveDraft(document: WorkflowDocument): Promise<{ workflow_hash: string }> {
    return this.request("api/draft", { method: "PUT", body: JSON.stringify({ document }) });
  }

  validate(document: WorkflowDocument): Promise<ValidationReport> {
    return this.request("api/validate", { method: "POST", body: JSON.stringify({ document }) });
  }

  generate(document: WorkflowDocument, artifacts: Record<string, string>): Promise<GeneratedFiles> {
    return this.request("api/generate", { method: "POST", body: JSON.stringify({ document, artifacts }) });
  }

  run(document: WorkflowDocument, steps: number, confirmResources: boolean): Promise<RunStatus> {
    return this.request("api/run", { method: "POST", body: JSON.stringify({ document, steps, confirm_resources: confirmResources }) });
  }

  stop(): Promise<RunStatus> {
    return this.request("api/stop", { method: "POST", body: "{}" });
  }

  close(): Promise<RunStatus> {
    return this.request("api/close", { method: "POST", body: "{}" });
  }

  status(): Promise<RunStatus> {
    return this.request("api/run");
  }

  events(since: number): Promise<{ events: LogEvent[]; run: RunStatus }> {
    return this.request(`api/events?since=${since}`);
  }
}

/** Decide the mode: a service answers api/state; the preview reads catalog.json. */
export async function detectMode(client: ServiceClient): Promise<{ mode: Mode; snapshot: CatalogSnapshot; service: ServiceState | null }> {
  try {
    const service = await client.state();
    return { mode: "local", snapshot: service.snapshot, service };
  } catch (error) {
    if (error instanceof ServiceError && error.status === 401) throw error;
    const response = await fetch("catalog.json");
    if (!response.ok) throw new Error("neither a local service nor a catalog snapshot is reachable");
    const snapshot = (await response.json()) as CatalogSnapshot;
    return { mode: "preview", snapshot, service: null };
  }
}
