// The editor's state: the document, its history, the selection, and the mode
// the page runs in.  Edits are the pure functions of model/document.ts;
// this file only sequences them and keeps undo/redo.

import { useReducer } from "react";

import {
  configure,
  defaultAnchor,
  importDocument,
  insertNode,
  move,
  nodeFromEntry,
  nodeOf,
  pythonNode,
  pythonTemplate,
  rehash,
  remove,
  reorder,
  replaceNode,
  setEnabled,
  uniquePythonName,
  WorkflowEditError,
  type Placement,
} from "./model/document";
import type { CatalogEntry, CatalogSnapshot, NodeConfiguration, ParameterValue, WorkflowDocument } from "./model/types";

export const HISTORY_LIMIT = 200;

export interface EditorState {
  snapshot: CatalogSnapshot;
  catalog: Map<string, CatalogEntry>;
  document: WorkflowDocument;
  undo: WorkflowDocument[];
  redo: WorkflowDocument[];
  selected: string | null;
  error: string | null;
  showControl: boolean;
}

export type EditorAction =
  | { type: "select"; id: string | null }
  | { type: "move"; id: string; placement: Placement }
  | { type: "reorder"; order: string[] }
  | { type: "remove"; id: string }
  | { type: "set_enabled"; id: string; enabled: boolean }
  | { type: "add"; entryId: string; placement?: Placement }
  | { type: "restore"; id: string; placement?: Placement }
  | { type: "add_python"; name?: string; placement?: Placement; source?: string }
  | { type: "replace"; id: string; entryId?: string; name?: string }
  | { type: "configure"; id: string; changes: Partial<NodeConfiguration> }
  | { type: "set_experimental"; experimental: boolean }
  | { type: "set_case"; case: string }
  | { type: "set_nsteps"; nsteps: number }
  | { type: "set_namelist"; namelist: Record<string, ParameterValue> }
  | { type: "import"; payload: unknown }
  | { type: "load"; document: WorkflowDocument }
  | { type: "undo" }
  | { type: "redo" }
  | { type: "reset" }
  | { type: "toggle_control" }
  | { type: "dismiss_error" };

export function initialState(snapshot: CatalogSnapshot, draft?: WorkflowDocument | null): EditorState {
  const catalog = new Map(snapshot.entries.map((entry) => [entry.id, entry]));
  const document = rehash(draft ?? snapshot.default_document);
  return { snapshot, catalog, document, undo: [], redo: [], selected: null, error: null, showControl: false };
}

function commit(state: EditorState, next: WorkflowDocument): EditorState {
  const document = { ...next, revision: state.document.revision + 1 };
  const undo = [...state.undo, state.document].slice(-HISTORY_LIMIT);
  return { ...state, document, undo, redo: [], error: null };
}

export function reducer(state: EditorState, action: EditorAction): EditorState {
  try {
    switch (action.type) {
      case "select":
        return { ...state, selected: action.id };
      case "toggle_control":
        return { ...state, showControl: !state.showControl };
      case "dismiss_error":
        return { ...state, error: null };
      case "move":
        return commit(state, move(state.document, action.id, action.placement));
      case "reorder":
        return commit(state, reorder(state.document, action.order));
      case "remove": {
        const next = commit(state, remove(state.document, action.id));
        return state.selected === action.id ? { ...next, selected: null } : next;
      }
      case "set_enabled":
        return commit(state, setEnabled(state.document, action.id, action.enabled));
      case "add": {
        const entry = state.catalog.get(action.entryId);
        if (!entry) throw new WorkflowEditError(`catalog has no process '${action.entryId}'`);
        const node = nodeFromEntry(entry, state.document);
        const placement = action.placement ?? { before: defaultAnchor(node, state.document) };
        return { ...commit(state, insertNode(state.document, node, placement)), selected: node.id };
      }
      case "restore": {
        const entry = state.catalog.get(action.id);
        if (!entry) throw new WorkflowEditError(`catalog has no process '${action.id}'`);
        const node = nodeFromEntry(entry, state.document);
        const placement = action.placement ?? { before: defaultAnchor(node, state.document) };
        return { ...commit(state, insertNode(state.document, node, placement)), selected: node.id };
      }
      case "add_python": {
        const name = action.name ?? uniquePythonName(state.document);
        const placement = action.placement ?? { before: state.document.nodes[state.document.nodes.length - 1].id };
        const anchorId = placement.after ?? placement.before ?? null;
        const anchor = anchorId ? nodeOf(state.document, anchorId) : null;
        const after = placement.after && anchor ? anchor.display_name : null;
        const node = pythonNode(name, action.source ?? pythonTemplate(name, after));
        if (state.document.nodes.some((item) => item.name === name)) throw new WorkflowEditError(`a process named '${name}' is already in the workflow`);
        return { ...commit(state, insertNode(state.document, node, placement)), selected: node.id };
      }
      case "replace": {
        const old = nodeOf(state.document, action.id);
        let node;
        if (action.entryId) {
          const entry = state.catalog.get(action.entryId);
          if (!entry) throw new WorkflowEditError(`catalog has no process '${action.entryId}'`);
          node = nodeFromEntry(entry, state.document);
        } else {
          const name = action.name ?? uniquePythonName(state.document, `${old.display_name}_python`);
          if (state.document.nodes.some((item) => item.name === name)) throw new WorkflowEditError(`a process named '${name}' is already in the workflow`);
          const before = state.document.nodes.slice(0, state.document.nodes.indexOf(old)).reverse().find((n) => n.scientific && n.enabled);
          node = pythonNode(name, pythonTemplate(name, before ? before.display_name : null));
        }
        return { ...commit(state, replaceNode(state.document, action.id, node)), selected: node.id };
      }
      case "configure":
        return commit(state, configure(state.document, action.id, action.changes));
      case "set_experimental":
        return commit(state, rehash({ ...state.document, experimental: action.experimental }));
      case "set_case":
        return commit(state, rehash({ ...state.document, case: action.case }));
      case "set_nsteps":
        if (!Number.isInteger(action.nsteps) || action.nsteps < 1) throw new WorkflowEditError("nsteps must be a positive integer");
        return commit(state, rehash({ ...state.document, nsteps: action.nsteps }));
      case "set_namelist":
        return commit(state, rehash({ ...state.document, namelist: action.namelist }));
      case "import":
        return { ...commit(state, importDocument(action.payload, state.catalog, state.snapshot.default_document)), selected: null };
      case "load":
        return { ...state, document: rehash(action.document), undo: [], redo: [], selected: null, error: null };
      case "undo": {
        if (!state.undo.length) return state;
        const previous = state.undo[state.undo.length - 1];
        return { ...state, document: { ...previous, revision: state.document.revision + 1 }, undo: state.undo.slice(0, -1), redo: [...state.redo, state.document], error: null };
      }
      case "redo": {
        if (!state.redo.length) return state;
        const following = state.redo[state.redo.length - 1];
        return { ...state, document: { ...following, revision: state.document.revision + 1 }, undo: [...state.undo, state.document], redo: state.redo.slice(0, -1), error: null };
      }
      case "reset":
        return { ...commit(state, rehash({ ...state.snapshot.default_document, experimental: state.document.experimental })), selected: null };
      default:
        return state;
    }
  } catch (error) {
    if (error instanceof WorkflowEditError) return { ...state, error: error.message };
    throw error;
  }
}

export function useEditor(snapshot: CatalogSnapshot, draft?: WorkflowDocument | null) {
  return useReducer(reducer, initialState(snapshot, draft));
}

export function isDefault(state: EditorState): boolean {
  return state.document.workflow_hash === rehash(state.snapshot.default_document).workflow_hash;
}
