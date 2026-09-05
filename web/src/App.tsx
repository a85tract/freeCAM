import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DndContext, KeyboardSensor, PointerSensor, closestCenter, useSensor, useSensors, type DragEndEvent } from "@dnd-kit/core";
import { sortableKeyboardCoordinates } from "@dnd-kit/sortable";

import { detectMode, ServiceClient, ServiceError, type LogEvent, type Mode, type RunStatus, type ServiceState } from "./api";
import { generateAll, type GeneratedArtifacts } from "./codegen/generate";
import { BottomPanel, type PanelTab } from "./components/BottomPanel";
import { Canvas, visibleNodes } from "./components/Canvas";
import { Inspector } from "./components/Inspector";
import { Library } from "./components/Library";
import { Toolbar } from "./components/Toolbar";
import { validateDocument } from "./model/validate";
import type { CatalogSnapshot, ValidationReport } from "./model/types";
import { isDefault, useEditor } from "./store";

type Theme = "light" | "dark";

function initialTheme(): Theme {
  try {
    const stored = localStorage.getItem("freecam-ui-theme");
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    // no storage
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function readText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error("could not read the file"));
    reader.readAsText(file);
  });
}

export function download(name: string, text: string, type: string): void {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function App() {
  const [boot, setBoot] = useState<{ mode: Mode; snapshot: CatalogSnapshot; service: ServiceState | null } | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const client = useMemo(() => new ServiceClient(), []);
  useEffect(() => {
    detectMode(client).then(setBoot).catch((error: Error) => setBootError(error.message));
  }, [client]);
  if (bootError) return <div style={{ padding: 24 }}><h1>freeCAM Workflow Builder</h1><p>{bootError}</p></div>;
  if (!boot) return <div style={{ padding: 24 }}>Loading the catalog…</div>;
  return <Editor mode={boot.mode} snapshot={boot.snapshot} service={boot.service} client={client} />;
}

function Editor({ mode, snapshot, service: initialService, client }: { mode: Mode; snapshot: CatalogSnapshot; service: ServiceState | null; client: ServiceClient }) {
  const [state, dispatch] = useEditor(snapshot, initialService?.draft ?? null);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [panel, setPanel] = useState<PanelTab>("checks");
  const [artifacts, setArtifacts] = useState<{ generated: GeneratedArtifacts; hash: string } | null>(null);
  const [savedFiles, setSavedFiles] = useState<Record<string, string> | null>(null);
  const [localReport, setLocalReport] = useState<ValidationReport | null>(null);
  const [localPending, setLocalPending] = useState(false);
  const [service, setService] = useState<ServiceState | null>(initialService);
  const [run, setRun] = useState<RunStatus | null>(initialService?.run ?? null);
  const [events, setEvents] = useState<LogEvent[]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [previewRun, setPreviewRun] = useState(false);
  const sinceRef = useRef(0);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("freecam-ui-theme", theme);
    } catch {
      // no storage
    }
  }, [theme]);

  const browserReport = useMemo(
    () => validateDocument(state.document, snapshot.default_document, state.catalog, snapshot.capabilities, snapshot.catalog_hash),
    [state.document, state.catalog, snapshot],
  );

  // the draft lives on the service while it runs, so a refresh comes back to it
  useEffect(() => {
    if (mode !== "local") return;
    const handle = setTimeout(() => {
      client.saveDraft(state.document).catch((error: Error) => setNotice(`draft not saved: ${error.message}`));
    }, 400);
    return () => clearTimeout(handle);
  }, [mode, client, state.document]);

  // poll the run while it is doing something; nothing here touches the model's ranks per step
  const active = run !== null && ["initializing", "queued", "running", "stopping"].includes(run.state);
  useEffect(() => {
    if (mode !== "local") return;
    let cancelled = false;
    const tick = async () => {
      try {
        const result = await client.events(sinceRef.current);
        if (cancelled) return;
        if (result.events.length) {
          sinceRef.current = result.events[result.events.length - 1].sequence + 1;
          setEvents((previous) => [...previous, ...result.events].slice(-400));
        }
        setRun(result.run);
      } catch (error) {
        if (!cancelled) setNotice((error as Error).message);
      }
    };
    tick();
    const handle = setInterval(tick, active ? 1500 : 6000);
    return () => {
      cancelled = true;
      clearInterval(handle);
    };
  }, [mode, client, active]);

  const selectedNode = state.selected ? state.document.nodes.find((node) => node.id === state.selected) ?? null : null;
  const addable = useMemo(() => {
    const present = new Set(state.document.nodes.map((node) => node.id));
    return snapshot.entries.filter((entry) => entry.addable && !present.has(entry.id));
  }, [snapshot.entries, state.document]);

  const moveBy = useCallback((id: string, delta: number) => {
    const rows = visibleNodes(state.document, state.showControl).filter((node) => node.movable || node.id === id);
    const position = rows.findIndex((node) => node.id === id);
    const target = position + delta;
    if (position < 0 || target < 0 || target >= rows.length) return;
    dispatch({ type: "move", id, placement: delta < 0 ? { before: rows[target].id } : { after: rows[target].id } });
  }, [state.document, state.showControl, dispatch]);

  const sensors = useSensors(useSensor(PointerSensor), useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }));
  const onDragEnd = (event: DragEndEvent) => {
    const { active: dragged, over } = event;
    if (!over) return;
    const activeId = String(dragged.id);
    const overId = String(over.id);
    if (activeId.startsWith("library:")) {
      const entryId = (dragged.data.current as { entryId: string }).entryId;
      const entry = state.catalog.get(entryId);
      if (!entry) return;
      dispatch({ type: entry.in_default ? "restore" : "add", ...(entry.in_default ? { id: entryId } : { entryId }), placement: { before: overId } } as never);
      return;
    }
    if (activeId === overId) return;
    const rows = visibleNodes(state.document, state.showControl).map((node) => node.id);
    const from = rows.indexOf(activeId);
    const to = rows.indexOf(overId);
    if (from < 0 || to < 0) return;
    dispatch({ type: "move", id: activeId, placement: from < to ? { after: overId } : { before: overId } });
  };

  const generate = useCallback(async () => {
    const generated = generateAll(state.document, snapshot);
    setArtifacts({ generated, hash: state.document.workflow_hash });
    setSavedFiles(null);
    setPanel("code");
    if (mode === "local") {
      try {
        const saved = await client.generate(state.document, { setup: generated.setup, script: generated.script, notebook: generated.notebook, workflow: generated.workflow });
        setSavedFiles(saved.files);
      } catch (error) {
        setNotice(`generated in the browser; not saved by the service: ${(error as Error).message}`);
      }
    }
  }, [state.document, snapshot, mode, client]);

  const validate = useCallback(async () => {
    setPanel("checks");
    if (mode !== "local") return;
    setLocalPending(true);
    try {
      setLocalReport(await client.validate(state.document));
    } catch (error) {
      setNotice(`local check failed: ${(error as Error).message}`);
    } finally {
      setLocalPending(false);
    }
  }, [mode, client, state.document]);

  const startRun = useCallback(async (confirmed: boolean) => {
    if (mode !== "local") {
      setPreviewRun(true);
      setPanel("run");
      return;
    }
    if (browserReport.status === "error") {
      setPanel("checks");
      setNotice("the draft has structural errors; fix them before running");
      return;
    }
    if (!confirmed && !(service?.driver_initialized ?? false)) {
      setConfirming(true);
      return;
    }
    setConfirming(false);
    setPanel("run");
    try {
      const status = await client.run(state.document, state.document.nsteps, confirmed);
      setRun(status);
      const fresh = await client.state();
      setService(fresh);
    } catch (error) {
      if (error instanceof ServiceError && error.status === 409) setNotice(error.message);
      else setNotice(`run refused: ${(error as Error).message}`);
    }
  }, [mode, client, state.document, browserReport.status, service]);

  const copy = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setNotice("copied");
    } catch {
      setNotice("the browser refused clipboard access; use Download instead");
    }
  }, []);

  const importFile = useCallback(async (file: File) => {
    try {
      const payload = JSON.parse(await readText(file));
      dispatch({ type: "import", payload });
    } catch (error) {
      setNotice(`not a workflow.json: ${(error as Error).message}`);
    }
  }, [dispatch]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.closest(".cm-editor"))) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        dispatch({ type: event.shiftKey ? "redo" : "undo" });
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [dispatch]);

  return (
    <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
      <div className="app">
        {mode === "preview" && <div className="banner" role="status">Preview — no CAM execution. Edit, check, generate and download here; run it from a Python session with the model.</div>}
        <Toolbar
          mode={mode}
          snapshot={snapshot}
          document={state.document}
          canUndo={state.undo.length > 0}
          canRedo={state.redo.length > 0}
          isDefault={isDefault(state)}
          showControl={state.showControl}
          theme={theme}
          running={active}
          onCase={(value) => dispatch({ type: "set_case", case: value })}
          onSteps={(value) => dispatch({ type: "set_nsteps", nsteps: value })}
          onUndo={() => dispatch({ type: "undo" })}
          onRedo={() => dispatch({ type: "redo" })}
          onReset={() => dispatch({ type: "reset" })}
          onExperimental={(value) => dispatch({ type: "set_experimental", experimental: value })}
          onToggleControl={() => dispatch({ type: "toggle_control" })}
          onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
          onValidate={validate}
          onGenerate={generate}
          onRun={() => startRun(false)}
          onImport={importFile}
          onExport={() => download("workflow.json", JSON.stringify(state.document, null, 2) + "\n", "application/json")}
        />
        <Library
          entries={snapshot.entries}
          document={state.document}
          onAdd={(entryId) => dispatch({ type: "add", entryId, placement: state.selected ? { after: state.selected } : undefined })}
          onRestore={(id) => dispatch({ type: "restore", id })}
          onAddPython={() => dispatch({ type: "add_python", placement: state.selected ? { after: state.selected } : undefined })}
        />
        <Canvas
          document={state.document}
          selected={state.selected}
          showControl={state.showControl}
          onSelect={(id) => dispatch({ type: "select", id })}
          onMoveBy={moveBy}
          onRemove={(id) => dispatch({ type: "remove", id })}
          onSetEnabled={(id, enabled) => dispatch({ type: "set_enabled", id, enabled })}
        />
        <Inspector
          node={selectedNode}
          entry={selectedNode ? state.catalog.get(selectedNode.id) ?? null : null}
          addable={addable}
          theme={theme}
          onConfigure={(id, changes) => dispatch({ type: "configure", id, changes })}
          onSetEnabled={(id, enabled) => dispatch({ type: "set_enabled", id, enabled })}
          onRemove={(id) => dispatch({ type: "remove", id })}
          onReplaceWithPython={(id) => dispatch({ type: "replace", id })}
          onReplaceWithEntry={(id, entryId) => dispatch({ type: "replace", id, entryId })}
        />
        <BottomPanel
          mode={mode}
          tab={panel}
          onTab={setPanel}
          document={state.document}
          browserReport={browserReport}
          localReport={localReport}
          localReportPending={localPending}
          artifacts={artifacts}
          onSelectNode={(id) => dispatch({ type: "select", id })}
          onGenerate={generate}
          onCopy={copy}
          onDownload={download}
          service={service}
          run={run}
          events={events}
          savedFiles={savedFiles}
          onRun={() => startRun(false)}
          onStop={() => client.stop().then(setRun).catch((error: Error) => setNotice(error.message))}
          onClose={() => client.close().then((status) => { setRun(status); client.state().then(setService).catch(() => undefined); }).catch((error: Error) => setNotice(error.message))}
        />
        {(state.error || notice) && (
          <div className="toast" role="alert">
            <span>{state.error ?? notice}</span>
            <button onClick={() => { dispatch({ type: "dismiss_error" }); setNotice(null); }}>Dismiss</button>
          </div>
        )}
        {confirming && service && (
          <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
            <div className="box">
              <h3 id="confirm-title">Start the model?</h3>
              <p>The first Run initializes the model: {service.resources.ranks} MPI ranks on {service.resources.nodes} nodes through PBS (queue {service.resources.queue ?? "default"}, walltime {service.resources.walltime}), then applies this workflow and runs {state.document.nsteps} step{state.document.nsteps === 1 ? "" : "s"}.</p>
              {!service.resources.account_set && <p style={{ color: "var(--danger)" }}>No allocation is configured (FREECAM_ACCOUNT); the launch will fail.</p>}
              {((browserReport.checks.replaced_kernels as string[] | undefined)?.length ?? 0) > 0 && <p>Model files named in the workflow are loaded on every rank at the first kernel call.</p>}
              <div className="buttons">
                <button className="secondary" onClick={() => setConfirming(false)}>Cancel</button>
                <button className="primary" onClick={() => startRun(true)}>Start</button>
              </div>
            </div>
          </div>
        )}
        {previewRun && (
          <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="preview-title">
            <div className="box">
              <h3 id="preview-title">Preview — no CAM execution</h3>
              <p>This page has no model behind it. Download <code>workflow.json</code> (Export), then on the machine with the model:</p>
              <pre className="code">{`import freecam as fc\nui = fc.Driver(case=${JSON.stringify(state.document.case)}, nsteps=${state.document.nsteps}).ui()\nprint(ui.url)`}</pre>
              <p>Open the address, Import the file, and Run.</p>
              <div className="buttons">
                <button className="primary" onClick={() => setPreviewRun(false)}>Close</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </DndContext>
  );
}
