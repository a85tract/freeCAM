import { useCallback, useState, type KeyboardEvent, type PointerEvent } from "react";

import type { LogEvent, Mode, RunStatus, ServiceState } from "../api";
import type { GeneratedArtifacts } from "../codegen/generate";
import type { Issue, ValidationReport, WorkflowDocument } from "../model/types";

export type PanelTab = "checks" | "code" | "run";

interface Props {
  mode: Mode;
  tab: PanelTab;
  onTab: (tab: PanelTab) => void;
  document: WorkflowDocument;
  browserReport: ValidationReport;
  localReport: ValidationReport | null;
  localReportPending: boolean;
  artifacts: { generated: GeneratedArtifacts; hash: string } | null;
  onSelectNode: (id: string) => void;
  onGenerate: () => void;
  onCopy: (text: string) => void;
  onDownload: (name: string, text: string, type: string) => void;
  service: ServiceState | null;
  run: RunStatus | null;
  events: LogEvent[];
  savedFiles: Record<string, string> | null;
  onRun: () => void;
  onStop: () => void;
  onClose: () => void;
  onEnableExperimental: () => void;
  height: number;
  onResize: (height: number) => void;
}

export const BOTTOM_MIN = 120;
export const BOTTOM_DEFAULT = 260;

/** The tallest the panel may be: leave the workflow at least this much room. */
export function bottomMax(): number {
  return Math.max(BOTTOM_MIN, (typeof window === "undefined" ? 900 : window.innerHeight) - 220);
}

export function clampHeight(height: number): number {
  return Math.min(bottomMax(), Math.max(BOTTOM_MIN, Math.round(height)));
}

const KEY_STEP = 24;

/** The bar between the workflow and the panel; drag it, use the arrow keys, or double-click to reset. */
function ResizeHandle({ height, onResize }: { height: number; onResize: (height: number) => void }) {
  const onPointerDown = useCallback((event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const startY = event.clientY;
    const startHeight = height;
    const target = event.currentTarget;
    target.setPointerCapture?.(event.pointerId);
    target.dataset.dragging = "true";
    const move = (moved: globalThis.PointerEvent) => onResize(clampHeight(startHeight + (startY - moved.clientY)));
    const stop = () => {
      delete target.dataset.dragging;
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
  }, [height, onResize]);
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "ArrowUp") onResize(clampHeight(height + KEY_STEP));
    else if (event.key === "ArrowDown") onResize(clampHeight(height - KEY_STEP));
    else if (event.key === "Home") onResize(bottomMax());
    else if (event.key === "End") onResize(BOTTOM_MIN);
    else return;
    event.preventDefault();
  };
  return (
    <div
      className="resize-handle"
      role="separator"
      aria-label="Resize the details panel"
      aria-orientation="horizontal"
      aria-valuemin={BOTTOM_MIN}
      aria-valuemax={bottomMax()}
      aria-valuenow={height}
      tabIndex={0}
      title="Drag to resize; double-click to reset"
      onPointerDown={onPointerDown}
      onDoubleClick={() => onResize(BOTTOM_DEFAULT)}
      onKeyDown={onKeyDown}
    />
  );
}

function IssueList({ issues, onSelect, onEnableExperimental }: { issues: Issue[]; onSelect: (id: string) => void; onEnableExperimental?: () => void }) {
  if (!issues.length) return <p className="muted">No findings.</p>;
  return (
    <ul className="issues">
      {issues.map((issue, index) => (
        <li key={index} className={issue.severity}>
          <span className="sev">{issue.severity}</span>
          <span>
            {issue.message}
            {issue.node_id && <> — <button className="link" onClick={() => onSelect(issue.node_id as string)}>{issue.node_id}</button></>}
            {issue.code === "experimental-required" && onEnableExperimental && (
              <> — <button className="link" onClick={onEnableExperimental}>Enable Experimental</button></>
            )}
          </span>
        </li>
      ))}
    </ul>
  );
}

const STATUS_BADGE: Record<ValidationReport["status"], string> = { valid: "ok", warning: "warn", error: "err" };

type CodeTab = "script" | "notebook" | "setup" | "workflow";

const CODE_TABS: { key: CodeTab; label: string; hint: string }[] = [
  { key: "script", label: "Script", hint: "A complete program. Save it on the machine that has the model and run `uv run python <file>` from the freeCAM checkout; it starts the model, applies this workflow, runs the steps and closes." },
  { key: "notebook", label: "Notebook", hint: "The same program as notebook cells, without outputs; open it in Jupyter on the machine that has the model." },
  { key: "setup", label: "Setup only", hint: "Configuration only: the classes and configure(driver). Paste it into a session that already has a driver and call configure(driver) after driver.initialize(). It does not start or run anything by itself." },
  { key: "workflow", label: "workflow.json", hint: "The document itself. Import it into any builder page, local or preview, to continue editing." },
];

export function BottomPanel(props: Props) {
  const [codeTab, setCodeTab] = useState<CodeTab>("script");
  const stale = props.artifacts !== null && props.artifacts.hash !== props.document.workflow_hash;
  return (
    <section className="bottom" aria-label="Details" style={{ height: props.height }}>
      <ResizeHandle height={props.height} onResize={props.onResize} />
      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={props.tab === "checks"} onClick={() => props.onTab("checks")}>
          Checks <span className={`badge ${STATUS_BADGE[props.browserReport.status]}`}>{props.browserReport.status}</span>
        </button>
        <button role="tab" aria-selected={props.tab === "code"} onClick={() => props.onTab("code")}>Code{stale ? " (stale)" : ""}</button>
        <button role="tab" aria-selected={props.tab === "run"} onClick={() => props.onTab("run")}>Run</button>
      </div>
      <div className="content">
        {props.tab === "checks" && (
          <div>
            <h3 style={{ margin: "4px 0" }}>Browser check</h3>
            <p className="muted" style={{ margin: "0 0 6px" }}>
              Moving, removing or replacing a physical process leaves the validated default behind; tick Experimental to say so, and the check lets it through as exploratory.
            </p>
            <IssueList issues={props.browserReport.issues} onSelect={props.onSelectNode} onEnableExperimental={props.onEnableExperimental} />
            {(props.browserReport.checks.not_verified as string[])?.length > 0 && (
              <>
                <h3 style={{ margin: "8px 0 4px" }}>Not verified here</h3>
                <ul className="muted">{(props.browserReport.checks.not_verified as string[]).map((item) => <li key={item}>{item}</li>)}</ul>
              </>
            )}
            <h3 style={{ margin: "8px 0 4px" }}>Local check {props.mode === "preview" && <span className="badge">needs the local service</span>}</h3>
            {props.mode === "preview" && <p className="muted">The published preview cannot parse Python, see model files or the native build; run <code>fc.Driver(...).ui()</code> locally for the authoritative check.</p>}
            {props.mode === "local" && props.localReportPending && <p className="muted">Checking…</p>}
            {props.mode === "local" && !props.localReportPending && !props.localReport && <p className="muted">Press Validate to check Python syntax, model files and the catalog on this machine.</p>}
            {props.localReport && (
              <>
                {props.localReport.workflow_hash !== props.document.workflow_hash && <p className="muted">This result is for an earlier version of the draft; validate again.</p>}
                <IssueList issues={props.localReport.issues} onSelect={props.onSelectNode} onEnableExperimental={props.onEnableExperimental} />
              </>
            )}
            <p className="muted" style={{ marginTop: 8 }}>{props.browserReport.disclaimer}</p>
          </div>
        )}

        {props.tab === "code" && (
          <div>
            <div className="code-tabs">
              <button className="primary" onClick={props.onGenerate}>{props.artifacts ? "Generate again" : "Generate"}</button>
              {props.artifacts && (
                <>
                  <span className="muted">workflow {props.artifacts.hash.slice(0, 12)}{stale ? " — the draft has changed since" : ""}</span>
                  {CODE_TABS.map((entry) => (
                    <button key={entry.key} className="secondary" aria-pressed={codeTab === entry.key} onClick={() => setCodeTab(entry.key)}>{entry.label}</button>
                  ))}
                  <button className="secondary" onClick={() => props.onCopy(props.artifacts!.generated[codeTab])}>Copy</button>
                  <button className="secondary" onClick={() => {
                    const generated = props.artifacts!.generated;
                    const stem = `freecam_workflow_${props.artifacts!.hash.slice(0, 12)}`;
                    if (codeTab === "setup") props.onDownload(`${stem}_setup.py`, generated.setup, "text/x-python");
                    if (codeTab === "script") props.onDownload(`${stem}.py`, generated.script, "text/x-python");
                    if (codeTab === "notebook") props.onDownload(`${stem}.ipynb`, generated.notebook, "application/x-ipynb+json");
                    if (codeTab === "workflow") props.onDownload("workflow.json", generated.workflow, "application/json");
                  }}>Download</button>
                </>
              )}
            </div>
            {!props.artifacts && <p className="muted">Generate freezes the current draft and produces a complete script, a notebook, a setup-only snippet for a session with a live driver, and workflow.json. The code uses the ordinary freeCAM interface; edit the workflow here rather than the code, or download and change it freely.</p>}
            {props.artifacts && <p className="muted" style={{ margin: "0 0 6px" }}>{CODE_TABS.find((entry) => entry.key === codeTab)!.hint}</p>}
            {props.artifacts && props.artifacts.generated.external_files.length > 0 && (
              <p className="muted">Files you provide: {props.artifacts.generated.external_files.join(", ")}</p>
            )}
            {props.artifacts && props.savedFiles && (
              <p className="muted">Saved by the local service: {Object.values(props.savedFiles).join(", ")}</p>
            )}
            {props.artifacts && <pre className="code" aria-label="Generated code">{props.artifacts.generated[codeTab]}</pre>}
          </div>
        )}

        {props.tab === "run" && (
          <div>
            {props.mode === "preview" && (
              <div>
                <p><strong>Preview — no CAM execution.</strong> This page edits and generates; it does not run the model.</p>
                <p>To run this workflow, open the builder from a Python session on the machine that has the model:</p>
                <pre className="code">{`import freecam as fc\n\nui = fc.Driver(case=${JSON.stringify(props.document.case)}, nsteps=${props.document.nsteps}).ui()\nprint(ui.url)   # open it, then Import the workflow.json you downloaded here`}</pre>
                <p className="muted">Or from a shell: <code>freecam ui --case {props.document.case} --port 8765</code></p>
              </div>
            )}
            {props.mode === "local" && (
              <div>
                <div className="code-tabs">
                  <button className="primary" onClick={props.onRun} disabled={!!props.run && (props.run.state === "running" || props.run.state === "initializing" || props.run.state === "queued")}>Run {props.document.nsteps} step{props.document.nsteps === 1 ? "" : "s"}</button>
                  <button className="secondary" onClick={props.onStop} disabled={!props.run || props.run.state !== "running"}>Stop</button>
                  <button className="secondary" onClick={props.onClose} disabled={!props.service?.driver_initialized || props.run?.state === "running"}>Close model</button>
                </div>
                <dl className="status-grid">
                  <dt>State</dt><dd>{props.run?.state ?? "idle"}{props.run?.message ? ` — ${props.run.message}` : ""}</dd>
                  <dt>Step</dt><dd>{props.run?.step ?? "–"}{props.run?.target_step ? ` of ${props.run.target_step}` : ""}</dd>
                  <dt>Job</dt><dd className="mono">{props.run?.job_id ?? "–"}</dd>
                  <dt>Run directory</dt><dd className="mono">{props.run?.run_dir ?? "–"}</dd>
                  <dt>Applied workflow</dt><dd className="mono">{props.run?.applied_hash ? props.run.applied_hash.slice(0, 12) : "–"}{props.run?.applied_hash && props.run.applied_hash !== props.document.workflow_hash ? " (the draft differs; the next Run applies the difference)" : ""}</dd>
                  <dt>Model calls</dt><dd className="mono">{props.run && Object.keys(props.run.model_calls).length ? Object.entries(props.run.model_calls).map(([k, v]) => `${k}: ${v}`).join(", ") : "–"}</dd>
                  <dt>Resources</dt><dd>{props.service ? `${props.service.resources.ranks} ranks on ${props.service.resources.nodes} nodes, queue ${props.service.resources.queue ?? "default"}, walltime ${props.service.resources.walltime}${props.service.resources.account_set ? "" : " — no allocation configured"}` : "–"}</dd>
                </dl>
                <div className="log" aria-label="Run log">
                  {props.events.length === 0 && <span className="muted">No events yet.</span>}
                  {props.events.map((event) => (
                    <div key={event.sequence} className={event.level}><span className="mono muted">{event.time.slice(11, 19)}</span> {event.message}</div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
