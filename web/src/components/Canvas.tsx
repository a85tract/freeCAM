import { useCallback } from "react";
import { SortableContext, useSortable, verticalListSortingStrategy } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";

import type { WorkflowDocument, WorkflowNode } from "../model/types";

interface Props {
  document: WorkflowDocument;
  selected: string | null;
  showControl: boolean;
  onSelect: (id: string | null) => void;
  onMoveBy: (id: string, delta: number) => void;
  onRemove: (id: string) => void;
  onSetEnabled: (id: string, enabled: boolean) => void;
}

export function visibleNodes(document: WorkflowDocument, showControl: boolean): WorkflowNode[] {
  return showControl ? document.nodes : document.nodes.filter((node) => node.scientific);
}

function Row({ node, index, selected, movable, onSelect, onKeyDown, onMoveBy, onRemove, onSetEnabled }: {
  node: WorkflowNode;
  index: number;
  selected: boolean;
  movable: boolean;
  onSelect: () => void;
  onKeyDown: (event: React.KeyboardEvent) => void;
  onMoveBy: (delta: number) => void;
  onRemove: () => void;
  onSetEnabled: (enabled: boolean) => void;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: node.id, disabled: !movable });
  const style = { transform: CSS.Transform.toString(transform), transition };
  const classes = ["row", selected ? "selected" : "", node.enabled ? "" : "disabled", node.scientific ? "" : "control", isDragging ? "dragging" : ""].join(" ");
  const replaced = Object.values(node.configuration.kernels).filter((binding) => binding.kind !== "original").length;
  return (
    <li ref={setNodeRef} style={style} className={classes} tabIndex={0} role="option" aria-selected={selected} aria-label={`${index}. ${node.display_name}`} onClick={onSelect} onFocus={onSelect} onKeyDown={onKeyDown} data-testid={`row-${node.id}`}>
      <span className="handle" aria-hidden={!movable} {...(movable ? { ...attributes, ...listeners } : {})} title={movable ? "Drag to reorder" : "Fixed"}>{movable ? "⋮⋮" : "·"}</span>
      <span className="index">{index}</span>
      <span>
        <span className="name">{node.display_name}</span>
        <span className="detail">
          {node.origin === "python" ? "Python" : node.origin === "catalog" ? "catalog" : node.kind}
          {node.parent_stage ? " · leaf" : ""}
          {replaced ? ` · ${replaced} kernel${replaced > 1 ? "s" : ""} replaced` : ""}
          {Object.keys(node.configuration.parameters).length ? " · tuned" : ""}
        </span>
      </span>
      <span className="actions">
        {node.scientific && (
          <>
            <button aria-label={`Move ${node.display_name} up`} disabled={!movable} onClick={(event) => { event.stopPropagation(); onMoveBy(-1); }}>↑</button>
            <button aria-label={`Move ${node.display_name} down`} disabled={!movable} onClick={(event) => { event.stopPropagation(); onMoveBy(1); }}>↓</button>
            <button aria-label={`${node.enabled ? "Disable" : "Enable"} ${node.display_name}`} disabled={node.locked} onClick={(event) => { event.stopPropagation(); onSetEnabled(!node.enabled); }}>{node.enabled ? "On" : "Off"}</button>
            <button aria-label={`Remove ${node.display_name}`} disabled={!node.removable} onClick={(event) => { event.stopPropagation(); onRemove(); }}>×</button>
          </>
        )}
      </span>
    </li>
  );
}

export function Canvas(props: Props) {
  const nodes = visibleNodes(props.document, props.showControl);
  const ids = nodes.map((node) => node.id);
  const keyHandler = useCallback((node: WorkflowNode) => (event: React.KeyboardEvent) => {
    if (event.altKey && (event.key === "ArrowUp" || event.key === "ArrowDown")) {
      event.preventDefault();
      if (node.movable) props.onMoveBy(node.id, event.key === "ArrowUp" ? -1 : 1);
      return;
    }
    if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      event.preventDefault();
      const position = ids.indexOf(node.id) + (event.key === "ArrowUp" ? -1 : 1);
      if (position >= 0 && position < ids.length) {
        props.onSelect(ids[position]);
        const element = event.currentTarget.closest(".canvas")?.querySelector<HTMLElement>(`[data-testid="row-${ids[position]}"]`);
        element?.focus();
      }
      return;
    }
    if ((event.key === "Delete" || event.key === "Backspace") && node.removable) {
      event.preventDefault();
      props.onRemove(node.id);
      return;
    }
    if (event.key === " " && node.scientific && !node.locked) {
      event.preventDefault();
      props.onSetEnabled(node.id, !node.enabled);
    }
  }, [ids, props]);

  let lastPhase: string | null = null;
  return (
    <main className="canvas" aria-label="Workflow">
      <SortableContext items={ids} strategy={verticalListSortingStrategy}>
        <ul className="rows" role="listbox" aria-label="Step order">
          {nodes.map((node, position) => {
            const marker = !props.showControl && node.phase !== lastPhase ? node.phase : null;
            lastPhase = node.phase;
            return (
              <li key={node.id} style={{ listStyle: "none" }}>
                {marker && <div className="phase-marker" aria-hidden="true">{marker.replace("cam_", "CAM ").replace("coupling", "coupling").replace("clock", "clock")}</div>}
                <ul className="rows">
                  <Row
                    node={node}
                    index={position + 1}
                    selected={props.selected === node.id}
                    movable={node.movable}
                    onSelect={() => props.onSelect(node.id)}
                    onKeyDown={keyHandler(node)}
                    onMoveBy={(delta) => props.onMoveBy(node.id, delta)}
                    onRemove={() => props.onRemove(node.id)}
                    onSetEnabled={(enabled) => props.onSetEnabled(node.id, enabled)}
                  />
                </ul>
              </li>
            );
          })}
        </ul>
      </SortableContext>
      <p className="muted" style={{ marginTop: 12 }}>
        Drag a row, or use the arrows; with a row focused, Alt+↑/↓ moves it, Space toggles it, Delete removes it.
        Control, clock and output actions run every step; tick “Full step” to see them.
      </p>
    </main>
  );
}
