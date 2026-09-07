import { useMemo, useState } from "react";
import { useDraggable } from "@dnd-kit/core";

import type { CatalogEntry, WorkflowDocument } from "../model/types";

interface Props {
  entries: CatalogEntry[];
  document: WorkflowDocument;
  onAdd: (entryId: string) => void;
  onRestore: (id: string) => void;
  onAddPython: () => void;
}

type Availability = "present" | "addable" | "unavailable" | "control";

function availability(entry: CatalogEntry, present: Set<string>): Availability {
  if (!entry.scientific) return "control";
  if (present.has(entry.id)) return "present";
  return entry.addable ? "addable" : "unavailable";
}

function LibraryItem({ entry, state, onAdd, onRestore }: { entry: CatalogEntry; state: Availability; onAdd: () => void; onRestore: () => void }) {
  const draggable = state === "addable";
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({ id: `library:${entry.id}`, data: { entryId: entry.id }, disabled: !draggable });
  return (
    <div ref={setNodeRef} className={`entry ${state}`} style={{ opacity: isDragging ? 0.5 : undefined }} {...attributes} {...listeners} role="listitem">
      <div className="title">
        <span className="name">{entry.display_name}</span>
        <span className={`badge ${entry.origin === "python" ? "python" : entry.scientific ? "" : "control"}`}>{entry.category}</span>
      </div>
      <div className="reason">
        {state === "present" && "In the workflow"}
        {state === "addable" && (entry.in_default ? "Removed from the default; drag or add to restore" : entry.description ?? "Can be added")}
        {state === "unavailable" && (entry.reason ?? "Not available in this configuration")}
        {state === "control" && "Runs every step; shown in the full-step view"}
      </div>
      {state === "addable" && (
        <div style={{ marginTop: 6 }}>
          <button onClick={entry.in_default ? onRestore : onAdd}>{entry.in_default ? "Restore" : "Add"}</button>
        </div>
      )}
    </div>
  );
}

export function Library(props: Props) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [show, setShow] = useState<"all" | "addable">("addable");
  const present = useMemo(() => new Set(props.document.nodes.map((node) => node.id)), [props.document]);
  const categories = useMemo(() => Array.from(new Set(props.entries.map((entry) => entry.category))).sort(), [props.entries]);
  const visible = props.entries.filter((entry) => {
    if (category !== "all" && entry.category !== category) return false;
    const state = availability(entry, present);
    if (show === "addable" && state !== "addable") return false;
    const needle = query.trim().toLowerCase();
    if (needle && !`${entry.display_name} ${entry.qualified_name} ${entry.operation} ${entry.description ?? ""}`.toLowerCase().includes(needle)) return false;
    return true;
  });
  return (
    <aside className="library" aria-label="Process library">
      <h2>Process library</h2>
      <div className="filters">
        <input type="search" placeholder="Search processes" aria-label="Search processes" value={query} onChange={(event) => setQuery(event.target.value)} />
        <select aria-label="Category" value={category} onChange={(event) => setCategory(event.target.value)}>
          <option value="all">All categories</option>
          {categories.map((name) => <option key={name} value={name}>{name}</option>)}
        </select>
        <select aria-label="Availability" value={show} onChange={(event) => setShow(event.target.value as "all" | "addable")}>
          <option value="addable">Can be added</option>
          <option value="all">Everything, with reasons</option>
        </select>
        <button className="secondary" onClick={props.onAddPython}>New Python process</button>
      </div>
      <div role="list">
        {visible.map((entry) => (
          <LibraryItem key={entry.id} entry={entry} state={availability(entry, present)} onAdd={() => props.onAdd(entry.id)} onRestore={() => props.onRestore(entry.id)} />
        ))}
        {!visible.length && <p className="muted">Nothing matches.</p>}
      </div>
    </aside>
  );
}
