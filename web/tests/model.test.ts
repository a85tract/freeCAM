import { describe, expect, it } from "vitest";

import { canonical, sha256Hex, workflowHash } from "../src/model/canonical";
import { configure, importDocument, move, pythonNode, pythonTemplate, remove, setEnabled, WorkflowEditError } from "../src/model/document";
import { validateDocument } from "../src/model/validate";
import { initialState, reducer } from "../src/store";
import { catalogOf, loadSnapshot, pythonExecutable, pythonHash, pythonValidationCodes } from "./helpers";

const snapshot = loadSnapshot();
const catalog = catalogOf(snapshot);
const withPython = pythonExecutable() ? it : it.skip;

describe("hashing", () => {
  it("computes SHA-256 as everyone else does", () => {
    expect(sha256Hex("")).toBe("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    expect(sha256Hex("abc")).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    expect(sha256Hex("héating °C")).toBe(sha256Hex("héating °C"));
  });

  it("writes canonical JSON with sorted keys and JavaScript numbers", () => {
    expect(canonical({ b: 1, a: [1.5, 1e-7, 1e21, true, null, "x"] })).toBe('{"a":[1.5,1e-7,1e+21,true,null,"x"],"b":1}');
  });

  it("gives the snapshot's default document the hash the Python side stored", () => {
    expect(workflowHash(snapshot.default_document)).toBe(snapshot.default_document.workflow_hash);
  });

  withPython("agrees with Python on an edited document", () => {
    let document = snapshot.default_document;
    document = move(document, "cam_run1.radiation", { before: "cam_run1.dry_adjustment" });
    document = configure(document, "cam_run1.deep_convection", { parameters: { zmconv_c0_lnd: 0.0075, zmconv_ke: 1e-6 } });
    document = configure(document, "cam_run1.cloud_macro_microphysics", { kernels: { mmacro_pcond: { kind: "surrogate", path: "models/m.pt" } } });
    const python = pythonNode("heating", pythonTemplate("heating", "dry_adjustment") + "# °C\n", { parameters: { rate: 0.5 } });
    document = { ...document, nodes: [...document.nodes.slice(0, 40), python, ...document.nodes.slice(40)], nsteps: 7, namelist: { cldfrc_rhminl: 0.9 } };
    document = { ...document, workflow_hash: workflowHash(document) };
    expect(pythonHash(document)).toBe(document.workflow_hash);
  });
});

describe("edits", () => {
  it("keeps control actions where they are", () => {
    expect(() => move(snapshot.default_document, "coupling.boundary_import", { index: 3 })).toThrow(WorkflowEditError);
    expect(() => remove(snapshot.default_document, "clock.advance_timestep")).toThrow(WorkflowEditError);
    expect(() => setEnabled(snapshot.default_document, "coupling.boundary_export", false)).toThrow(WorkflowEditError);
  });

  it("moves a process and changes only the hash of what runs", () => {
    const moved = move(snapshot.default_document, "cam_run1.radiation", { before: "cam_run1.dry_adjustment" });
    const ids = moved.nodes.map((node) => node.id);
    expect(ids.indexOf("cam_run1.radiation")).toBeLessThan(ids.indexOf("cam_run1.dry_adjustment"));
    expect(moved.workflow_hash).not.toBe(snapshot.default_document.workflow_hash);
    const flagged = { ...moved, experimental: true };
    expect(workflowHash(flagged)).toBe(moved.workflow_hash);
  });

  it("imports what it exported, refreshing catalog nodes and keeping Python ones", () => {
    let state = initialState(snapshot);
    state = reducer(state, { type: "add_python", placement: { after: "cam_run1.dry_adjustment" } });
    state = reducer(state, { type: "configure", id: "python:notebook_process", changes: { parameters: { rate: 0.25 } } });
    state = reducer(state, { type: "set_nsteps", nsteps: 5 });
    const payload = JSON.parse(JSON.stringify(state.document));
    const imported = importDocument(payload, catalog, snapshot.default_document);
    expect(imported.workflow_hash).toBe(state.document.workflow_hash);
    expect(imported.nodes.find((n) => n.id === "python:notebook_process")?.configuration.parameters).toEqual({ rate: 0.25 });
    expect(() => importDocument({ nodes: [{ id: "cam_run1.nonesuch", name: "x", origin: "default" }] }, catalog, snapshot.default_document)).toThrow(/does not have/);
  });
});

describe("store", () => {
  it("undoes and redoes order, configuration, enablement and bindings together", () => {
    let state = initialState(snapshot);
    const start = state.document.workflow_hash;
    state = reducer(state, { type: "move", id: "cam_run1.radiation", placement: { before: "cam_run1.dry_adjustment" } });
    state = reducer(state, { type: "set_enabled", id: "cam_run1.shallow_convection", enabled: false });
    state = reducer(state, { type: "configure", id: "cam_run1.deep_convection", changes: { parameters: { zmconv_ke: 2e-6 } } });
    state = reducer(state, { type: "configure", id: "cam_run1.cloud_macro_microphysics", changes: { kernels: { mmacro_pcond: { kind: "surrogate", path: "m.pt" } } } });
    const edited = state.document.workflow_hash;
    expect(state.undo).toHaveLength(4);
    for (let i = 0; i < 4; i++) state = reducer(state, { type: "undo" });
    expect(state.document.workflow_hash).toBe(start);
    for (let i = 0; i < 4; i++) state = reducer(state, { type: "redo" });
    expect(state.document.workflow_hash).toBe(edited);
    state = reducer(state, { type: "reset" });
    expect(state.document.workflow_hash).toBe(start);
    expect(state.document.nodes.find((n) => n.id === "cam_run1.deep_convection")?.configuration.parameters).toEqual({});
  });

  it("reports an impossible edit instead of throwing", () => {
    let state = initialState(snapshot);
    state = reducer(state, { type: "remove", id: "coupling.boundary_import" });
    expect(state.error).toMatch(/required control action/);
    expect(state.document.workflow_hash).toBe(snapshot.default_document.workflow_hash);
  });

  it("adds a catalog entry back into its default slot", () => {
    let state = initialState(snapshot);
    const slot = state.document.nodes.findIndex((n) => n.id === "cam_run1.radiation");
    state = reducer(state, { type: "remove", id: "cam_run1.radiation" });
    state = reducer(state, { type: "restore", id: "cam_run1.radiation" });
    expect(state.document.nodes.findIndex((n) => n.id === "cam_run1.radiation")).toBe(slot);
  });

  it("replaces a process in place with a fresh Python process", () => {
    let state = initialState(snapshot);
    const slot = state.document.nodes.findIndex((n) => n.id === "cam_run1.radiation");
    state = reducer(state, { type: "replace", id: "cam_run1.radiation", name: "my_radiation" });
    expect(state.document.nodes[slot].id).toBe("python:my_radiation");
    expect(state.document.nodes.some((n) => n.id === "cam_run1.radiation")).toBe(false);
  });
});

describe("browser check", () => {
  it("passes the default and refuses a parent with its leaves", () => {
    const ok = validateDocument(snapshot.default_document, snapshot.default_document, catalog, snapshot.capabilities, snapshot.catalog_hash);
    expect(ok.status).toBe("valid");
    const both = setEnabled(snapshot.default_document, "cam_run2.tracers_and_chemistry", true);
    const report = validateDocument(both, snapshot.default_document, catalog, snapshot.capabilities, snapshot.catalog_hash);
    expect(report.status).toBe("error");
    expect(report.issues.map((i) => i.code)).toContain("parent-and-leaf");
  });

  it("asks for Experimental when the order changes, and warns once it is on", () => {
    const moved = move(snapshot.default_document, "cam_run1.radiation", { before: "cam_run1.dry_adjustment" });
    expect(validateDocument(moved, snapshot.default_document, catalog, snapshot.capabilities).issues.map((i) => i.code)).toContain("experimental-required");
    const allowed = { ...moved, experimental: true };
    const report = validateDocument(allowed, snapshot.default_document, catalog, snapshot.capabilities);
    expect(report.status).toBe("warning");
    expect(report.checks.order_changed).toBe(true);
  });

  it("offers a binding only where the runner covers the kernel", () => {
    const bad = configure(snapshot.default_document, "cam_run1.radiation", { kernels: { rad_rrtmg_sw: { kind: "surrogate", path: "m.pt" } } });
    expect(validateDocument(bad, snapshot.default_document, catalog, snapshot.capabilities).issues.map((i) => i.code)).toContain("kernel-not-bindable");
    // the runner pauses at micro_mg_tend and that pause has passed its gate: a binding there is only informational
    const proven = configure(snapshot.default_document, "cam_run1.cloud_macro_microphysics", { kernels: { micro_mg_tend: { kind: "surrogate", path: "m.pt" } } });
    const codes = validateDocument(proven, snapshot.default_document, catalog, snapshot.capabilities).issues.map((i) => i.code);
    expect(codes).not.toContain("kernel-not-validated");
    expect(codes).not.toContain("kernel-not-bindable");
    const good = configure(snapshot.default_document, "cam_run1.cloud_macro_microphysics", { kernels: { mmacro_pcond: { kind: "surrogate", path: "m.pt" } } });
    const report = validateDocument(good, snapshot.default_document, catalog, snapshot.capabilities);
    expect(report.status).toBe("valid");
    expect(report.issues.map((i) => i.code)).toEqual(expect.arrayContaining(["kernel-replaced", "model-file"]));
  });

  withPython("finds what the Python check finds", () => {
    let document = move(snapshot.default_document, "cam_run1.radiation", { before: "cam_run1.dry_adjustment" });
    document = configure(document, "cam_run1.deep_convection", { parameters: { zmconv_c0_lnd: "fast", nonesuch: 1 } });
    document = configure(document, "cam_run1.cloud_macro_microphysics", { kernels: { mmacro_pcond: { kind: "surrogate", path: "m.pt" } } });
    document = setEnabled(document, "cam_run2.tracers_and_chemistry", true);
    const ours = validateDocument(document, snapshot.default_document, catalog, snapshot.capabilities, snapshot.catalog_hash).issues.map((i) => i.code).sort();
    expect(ours).toEqual(pythonValidationCodes(document, "browser"));
  });
});
