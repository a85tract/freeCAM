import { useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";

import type { CatalogEntry, KernelBinding, NodeConfiguration, ParameterValue, VariableDeclaration, WorkflowNode } from "../model/types";

interface Props {
  node: WorkflowNode | null;
  entry: CatalogEntry | null;
  addable: CatalogEntry[];
  theme: "light" | "dark";
  onConfigure: (id: string, changes: Partial<NodeConfiguration>) => void;
  onSetEnabled: (id: string, enabled: boolean) => void;
  onRemove: (id: string) => void;
  onReplaceWithPython: (id: string) => void;
  onReplaceWithEntry: (id: string, entryId: string) => void;
}

type Tab = "about" | "parameters" | "python" | "kernels" | "variables";

function parseValue(text: string): ParameterValue | undefined {
  const trimmed = text.trim();
  if (!trimmed) return undefined;
  if (/^-?\d+(\.\d+)?([eE][-+]?\d+)?$/.test(trimmed)) return Number(trimmed);
  if (trimmed === "true" || trimmed === "false") return trimmed === "true";
  try {
    return JSON.parse(trimmed) as ParameterValue;
  } catch {
    return trimmed;
  }
}

function showValue(value: ParameterValue | undefined): string {
  if (value === undefined) return "";
  return typeof value === "string" ? value : JSON.stringify(value);
}

export function Inspector(props: Props) {
  const [tab, setTab] = useState<Tab>("about");
  const [replacement, setReplacement] = useState("");
  const { node } = props;
  if (!node) {
    return (
      <aside className="inspector" aria-label="Inspector">
        <h2>Inspector</h2>
        <p className="muted">Select a process to see what it is, its parameters, its code or its kernels.</p>
      </aside>
    );
  }
  const capabilities = node.metadata.kernels ?? [];
  const parameters = node.metadata.parameters ?? [];
  const tabs: Tab[] = ["about"];
  if (parameters.length) tabs.push("parameters");
  if (node.origin === "python") tabs.push("python", "variables");
  if (capabilities.length) tabs.push("kernels");
  const active: Tab = tabs.includes(tab) ? tab : "about";

  const setParameter = (name: string, text: string) => {
    const next = { ...node.configuration.parameters };
    const value = parseValue(text);
    if (value === undefined) delete next[name];
    else next[name] = value;
    props.onConfigure(node.id, { parameters: next });
  };
  const setBinding = (kernel: string, binding: KernelBinding) => {
    props.onConfigure(node.id, { kernels: { ...node.configuration.kernels, [kernel]: binding } });
  };
  const setVariables = (variables: VariableDeclaration[]) => props.onConfigure(node.id, { variables });

  return (
    <aside className="inspector" aria-label="Inspector">
      <h2>{node.display_name}</h2>
      <div>
        <span className={`badge ${node.origin === "python" ? "python" : ""}`}>{node.origin === "python" ? "Python process" : node.origin === "catalog" ? "catalog process" : node.kind}</span>{" "}
        {!node.enabled && <span className="badge warn">disabled</span>}{" "}
        {node.locked && <span className="badge control">required</span>}
      </div>
      {node.scientific && (
        <div className="actions">
          <button className="action" disabled={node.locked} onClick={() => props.onSetEnabled(node.id, !node.enabled)}>{node.enabled ? "Disable" : "Enable"}</button>
          <button className="action" disabled={!node.removable} onClick={() => props.onRemove(node.id)}>Remove</button>
          <button className="action" disabled={!node.removable} onClick={() => props.onReplaceWithPython(node.id)}>Replace with Python</button>
          <select aria-label="Replace with" value={replacement} onChange={(event) => setReplacement(event.target.value)} disabled={!node.removable}>
            <option value="">Replace with…</option>
            {props.addable.map((entry) => <option key={entry.id} value={entry.id}>{entry.display_name}</option>)}
          </select>
          <button className="action" disabled={!replacement || !node.removable} onClick={() => { props.onReplaceWithEntry(node.id, replacement); setReplacement(""); }}>Go</button>
        </div>
      )}
      <div className="tabs" role="tablist">
        {tabs.map((name) => (
          <button key={name} role="tab" aria-selected={active === name} onClick={() => setTab(name)}>{name[0].toUpperCase() + name.slice(1)}</button>
        ))}
      </div>

      {active === "about" && (
        <div>
          <p>{props.entry?.description ?? (node.origin === "python" ? "A Python process defined in this workflow." : "")}</p>
          <table>
            <tbody>
              <tr><th>Phase</th><td className="mono">{node.phase}</td></tr>
              <tr><th>Operation</th><td className="mono">{node.operation}</td></tr>
              {node.native_id !== null && <tr><th>Native id</th><td className="mono">{node.native_id}</td></tr>}
              {node.source && <tr><th>Source</th><td className="mono">{node.source}</td></tr>}
              {node.parent_stage && <tr><th>Leaf of</th><td className="mono">{node.parent_stage}</td></tr>}
              {(node.reads.length || node.writes.length) ? <tr><th>Fields</th><td className="mono">reads {node.reads.join(", ") || "–"}; writes {node.writes.join(", ") || "–"}</td></tr> : null}
              <tr><th>Implementation</th><td>{node.implementation}</td></tr>
            </tbody>
          </table>
          {node.origin === "python" && <p className="muted">Which fields it reads and writes is inferred when it runs; the check cannot verify it here.</p>}
          {node.origin === "catalog" && <p className="muted">Its field bindings are made on the live model at insertion.</p>}
        </div>
      )}

      {active === "parameters" && (
        <div>
          <p className="muted">Audited runtime tunables of this process. Leave a value empty to keep the namelist's.</p>
          <table>
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody>
              {parameters.map((spec) => (
                <tr key={spec.name}>
                  <td><span className="mono">{spec.name}</span><div className="muted">{spec.notes}</div></td>
                  <td>
                    <input aria-label={spec.name} value={showValue(node.configuration.parameters[spec.name])} placeholder={spec.dtype} onChange={(event) => setParameter(spec.name, event.target.value)} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {active === "python" && (
        <div>
          <div className="field">
            <label>Class source (an fc.Physics subclass; its <code>name</code> must stay <code>{node.name}</code>)</label>
            <div className="editor">
              <CodeMirror
                value={node.configuration.python_source ?? ""}
                height="280px"
                theme={props.theme}
                extensions={[python()]}
                onChange={(value) => props.onConfigure(node.id, { python_source: value })}
                aria-label="Python source"
              />
            </div>
          </div>
          <h3>Properties</h3>
          <p className="muted">Values assigned to the class's <code>fc.Property</code> attributes when the process is created.</p>
          <PropertyTable values={node.configuration.parameters} onChange={(parameters) => props.onConfigure(node.id, { parameters })} />
        </div>
      )}

      {active === "variables" && (
        <div>
          <p className="muted">Fields this process asks the model to create, written to history unless output is off.</p>
          <VariableTable variables={node.configuration.variables} onChange={setVariables} />
        </div>
      )}

      {active === "kernels" && (
        <div>
          <p className="muted">Kernels of this stage that a model may stand in for. A binding is offered only where the image's runner can pause at the kernel; “validated” means that path has passed a bit-for-bit gate with the original kernel answering.</p>
          {capabilities.map((capability) => {
            const binding = node.configuration.kernels[capability.kernel] ?? { kind: "original", path: null };
            return (
              <div key={capability.kernel} className="field" style={{ borderBottom: "1px solid var(--border)", paddingBottom: 8 }}>
                <div>
                  <span className="mono">{capability.kernel}</span>{" "}
                  <span className={`badge ${capability.bindable ? "ok" : "err"}`}>{capability.bindable ? "bindable" : "not bindable"}</span>{" "}
                  <span className={`badge ${capability.validated ? "ok" : "warn"}`}>{capability.validated ? "validated" : "not validated"}</span>
                </div>
                {capability.reason && <div className="muted">{capability.reason}</div>}
                <label>
                  <input type="radio" name={`binding-${capability.kernel}`} checked={binding.kind === "original"} onChange={() => setBinding(capability.kernel, { kind: "original", path: null })} /> original routine
                </label>
                <label>
                  <input type="radio" name={`binding-${capability.kernel}`} disabled={!capability.bindable} checked={binding.kind === "surrogate"} onChange={() => setBinding(capability.kernel, { kind: "surrogate", path: binding.path ?? "models/model.pt" })} /> trained network, by path
                </label>
                {binding.kind === "surrogate" && (
                  <input aria-label={`${capability.kernel} model path`} value={binding.path ?? ""} onChange={(event) => setBinding(capability.kernel, { kind: "surrogate", path: event.target.value })} placeholder="path to the model file" />
                )}
                {capability.evidence.length > 0 && <div className="muted">Evidence: {capability.evidence.join(", ")}</div>}
              </div>
            );
          })}
        </div>
      )}
    </aside>
  );
}

function PropertyTable({ values, onChange }: { values: Record<string, ParameterValue>; onChange: (values: Record<string, ParameterValue>) => void }) {
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  return (
    <table>
      <thead><tr><th>Property</th><th>Value</th><th></th></tr></thead>
      <tbody>
        {Object.keys(values).sort().map((key) => (
          <tr key={key}>
            <td className="mono">{key}</td>
            <td><input aria-label={`property ${key}`} value={showValue(values[key])} onChange={(event) => { const value = parseValue(event.target.value); const next = { ...values }; if (value === undefined) delete next[key]; else next[key] = value; onChange(next); }} /></td>
            <td><button onClick={() => { const next = { ...values }; delete next[key]; onChange(next); }} aria-label={`remove property ${key}`}>×</button></td>
          </tr>
        ))}
        <tr>
          <td><input aria-label="new property name" placeholder="name" value={name} onChange={(event) => setName(event.target.value)} /></td>
          <td><input aria-label="new property value" placeholder="value" value={text} onChange={(event) => setText(event.target.value)} /></td>
          <td><button disabled={!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)} onClick={() => { const value = parseValue(text); if (value !== undefined) onChange({ ...values, [name]: value }); setName(""); setText(""); }}>Add</button></td>
        </tr>
      </tbody>
    </table>
  );
}

function VariableTable({ variables, onChange }: { variables: VariableDeclaration[]; onChange: (variables: VariableDeclaration[]) => void }) {
  const [draft, setDraft] = useState<VariableDeclaration>({ name: "", like: "T", units: "1", output: true });
  const update = (index: number, changes: Partial<VariableDeclaration>) => onChange(variables.map((item, i) => (i === index ? { ...item, ...changes } : item)));
  return (
    <table>
      <thead><tr><th>Name</th><th>Like</th><th>Units</th><th>Output</th><th></th></tr></thead>
      <tbody>
        {variables.map((variable, index) => (
          <tr key={variable.name}>
            <td className="mono">{variable.name}</td>
            <td><input aria-label={`${variable.name} like`} value={variable.like} onChange={(event) => update(index, { like: event.target.value })} /></td>
            <td><input aria-label={`${variable.name} units`} value={variable.units} onChange={(event) => update(index, { units: event.target.value })} /></td>
            <td><input type="checkbox" aria-label={`${variable.name} output`} checked={variable.output} onChange={(event) => update(index, { output: event.target.checked })} /></td>
            <td><button aria-label={`remove variable ${variable.name}`} onClick={() => onChange(variables.filter((_, i) => i !== index))}>×</button></td>
          </tr>
        ))}
        <tr>
          <td><input aria-label="new variable name" placeholder="name" value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} /></td>
          <td><input aria-label="new variable like" value={draft.like} onChange={(event) => setDraft({ ...draft, like: event.target.value })} /></td>
          <td><input aria-label="new variable units" value={draft.units} onChange={(event) => setDraft({ ...draft, units: event.target.value })} /></td>
          <td><input type="checkbox" aria-label="new variable output" checked={draft.output} onChange={(event) => setDraft({ ...draft, output: event.target.checked })} /></td>
          <td><button disabled={!/^[A-Za-z_][A-Za-z0-9_]*$/.test(draft.name) || variables.some((v) => v.name === draft.name)} onClick={() => { onChange([...variables, draft]); setDraft({ name: "", like: "T", units: "1", output: true }); }}>Add</button></td>
        </tr>
      </tbody>
    </table>
  );
}
