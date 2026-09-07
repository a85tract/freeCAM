import type { Mode } from "../api";
import type { CatalogSnapshot, WorkflowDocument } from "../model/types";

interface Props {
  mode: Mode;
  snapshot: CatalogSnapshot;
  document: WorkflowDocument;
  canUndo: boolean;
  canRedo: boolean;
  isDefault: boolean;
  showControl: boolean;
  theme: "light" | "dark";
  running: boolean;
  onCase: (value: string) => void;
  onSteps: (value: number) => void;
  onUndo: () => void;
  onRedo: () => void;
  onReset: () => void;
  onExperimental: (value: boolean) => void;
  onToggleControl: () => void;
  onToggleTheme: () => void;
  onValidate: () => void;
  onGenerate: () => void;
  onRun: () => void;
  onImport: (file: File) => void;
  onExport: () => void;
}

export function Toolbar(props: Props) {
  const { document } = props;
  return (
    <header className="toolbar" role="toolbar" aria-label="Workflow">
      <strong>freeCAM Workflow Builder</strong>
      <label>
        Case
        <select aria-label="Case" value={document.case} onChange={(event) => props.onCase(event.target.value)}>
          {Object.keys(props.snapshot.cases).map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </label>
      <label>
        Steps
        <input
          aria-label="Steps"
          type="number"
          min={1}
          value={document.nsteps}
          onChange={(event) => {
            const value = Number(event.target.value);
            if (Number.isInteger(value) && value >= 1) props.onSteps(value);
          }}
        />
      </label>
      <label>
        <input type="checkbox" checked={document.experimental} onChange={(event) => props.onExperimental(event.target.checked)} />
        Experimental
      </label>
      <label title="Show the control, clock and output actions that run every step">
        <input type="checkbox" checked={props.showControl} onChange={props.onToggleControl} />
        Full step
      </label>
      <span className="spacer" />
      <button className="secondary" onClick={props.onUndo} disabled={!props.canUndo} title="Undo (Ctrl+Z)">Undo</button>
      <button className="secondary" onClick={props.onRedo} disabled={!props.canRedo} title="Redo (Ctrl+Shift+Z)">Redo</button>
      <button className="secondary" onClick={props.onReset} disabled={props.isDefault} title="Back to the validated default; a running model is not touched">Reset</button>
      <button className="secondary" onClick={props.onValidate}>Validate</button>
      <button className="secondary" onClick={props.onGenerate}>Generate</button>
      <button className="primary" onClick={props.onRun} disabled={props.running} title={props.mode === "preview" ? "Preview: no CAM execution here" : "Run the declared steps on the model"}>
        {props.mode === "preview" ? "Run…" : props.running ? "Running…" : "Run"}
      </button>
      <label className="secondary" style={{ cursor: "pointer" }}>
        Import
        <input
          type="file"
          accept="application/json,.json"
          className="sr-only"
          aria-label="Import workflow.json"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) props.onImport(file);
            event.target.value = "";
          }}
        />
      </label>
      <button className="secondary" onClick={props.onExport} title="Download workflow.json">Export</button>
      <button className="secondary" onClick={props.onToggleTheme} aria-label="Toggle dark mode">{props.theme === "dark" ? "Light" : "Dark"}</button>
    </header>
  );
}
