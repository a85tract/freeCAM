import { describe, expect, it } from "vitest";

import { configure, move, pythonNode, pythonTemplate, setEnabled } from "../src/model/document";
import { workflowHash } from "../src/model/canonical";
import { describeChanges, generateAll, pyLiteral } from "../src/codegen/generate";
import type { WorkflowDocument } from "../src/model/types";
import { loadSnapshot, nbformatProblem, pythonExecutable, pythonSyntaxError } from "./helpers";

const snapshot = loadSnapshot();
const withPython = pythonExecutable() ? it : it.skip;

function edited(): WorkflowDocument {
  let document = snapshot.default_document;
  document = move(document, "cam_run1.radiation", { before: "cam_run1.dry_adjustment" });
  document = setEnabled(document, "cam_run1.shallow_convection", false);
  document = configure(document, "cam_run1.deep_convection", { parameters: { zmconv_c0_lnd: 0.0075 } });
  document = configure(document, "cam_run1.cloud_macro_microphysics", { kernels: { mmacro_pcond: { kind: "surrogate", path: "models/mmacro_pcond.pt" } } });
  const heating = pythonNode("heating", pythonTemplate("heating", "dry_adjustment"), {
    parameters: { rate: 0.5 },
    variables: [{ name: "heating_rate", like: "T", units: "K s-1", output: true }],
  });
  const slot = document.nodes.findIndex((n) => n.id === "cam_run1.dry_adjustment") + 1;
  const nodes = [...document.nodes.slice(0, slot), heating, ...document.nodes.slice(slot)];
  document = { ...document, nodes, nsteps: 4, namelist: { cldfrc_rhminl: 0.9 }, experimental: true };
  return { ...document, workflow_hash: workflowHash(document) };
}

describe("Python literals", () => {
  it("writes JSON values as Python", () => {
    expect(pyLiteral(true)).toBe("True");
    expect(pyLiteral(null)).toBe("None");
    expect(pyLiteral(0.0075)).toBe("0.0075");
    expect(pyLiteral("a \"b\"")).toBe('"a \\"b\\""');
    expect(pyLiteral({ b: [1, 2], a: "x" })).toBe('{"a": "x", "b": [1, 2]}');
  });
});

describe("the default workflow", () => {
  const artifacts = generateAll(snapshot.default_document, snapshot);

  it("configures nothing, so the run is the validated path", () => {
    expect(artifacts.changes).toEqual([]);
    expect(artifacts.script).toContain("the validated default: nothing to change");
    expect(artifacts.script).not.toContain("workflow.replace(");
    expect(artifacts.script).not.toContain("parameters[");
    expect(artifacts.script).toContain('fc.Driver(case="PI-atm", nsteps=2)');
    expect(artifacts.external_files).toEqual([]);
  });

  it("is deterministic", () => {
    const again = generateAll(snapshot.default_document, snapshot);
    expect(again).toEqual(artifacts);
    expect(artifacts.script).not.toMatch(/\d{4}-\d{2}-\d{2}T/);      // no timestamp inside the code
  });

  withPython("is valid Python and a valid notebook", () => {
    expect(pythonSyntaxError(artifacts.script)).toBeNull();
    expect(pythonSyntaxError(artifacts.setup)).toBeNull();
    expect(nbformatProblem(artifacts.notebook)).toBeNull();
  });
});

describe("an edited workflow", () => {
  const document = edited();
  const artifacts = generateAll(document, snapshot);

  it("names every change in the header", () => {
    const changes = describeChanges(document, snapshot.default_document);
    expect(changes).toEqual(expect.arrayContaining([
      "Python process heating",
      "shallow_convection disabled",
      "scientific order changed",
      "cloud_macro_microphysics.mmacro_pcond answered by models/mmacro_pcond.pt",
      "deep_convection parameters zmconv_c0_lnd",
      "namelist cldfrc_rhminl",
    ]));
    for (const change of changes) expect(artifacts.script).toContain(`#   - ${change}`);
    expect(artifacts.external_files).toEqual(["models/mmacro_pcond.pt"]);
  });

  it("uses the public interface in the order the service applies it", () => {
    const script = artifacts.script;
    const order = [
      'state.create("heating_rate", like="T", units="K s-1")',
      "heating = Heating()",
      "heating.rate = 0.5",
      'workflow.insert(heating, after="dry_adjustment")',
      'stage = CloudMacroMicrophysics(macro_surrogate="models/mmacro_pcond.pt")',
      "stage.attach(driver.cam)",
      "workflow.replace([",
      'driver.cam.parameters["zmconv_c0_lnd"] = 0.0075',
    ];
    let position = -1;
    for (const line of order) {
      const next = script.indexOf(line);
      expect(next, line).toBeGreaterThan(position);
      position = next;
    }
    expect(script).toContain("from freecam.physics.cloud_macro_microphysics import CloudMacroMicrophysics");
    expect(script).toContain('namelist={"cldfrc_rhminl": 0.9}');
    expect(script).not.toContain('"shallow_convection",');         // dropped from the order
    expect(script).not.toContain("torch");                            // weights stay a path
  });

  it("gives the setup snippet the same configure() as the script", () => {
    const configureOf = (text: string) => text.slice(text.indexOf("def configure(driver):"), text.indexOf("\n\n\n", text.indexOf("def configure")));
    expect(configureOf(artifacts.setup)).toBe(configureOf(artifacts.script));
  });

  withPython("is valid Python and a valid notebook", () => {
    expect(pythonSyntaxError(artifacts.script)).toBeNull();
    expect(pythonSyntaxError(artifacts.setup)).toBeNull();
    expect(nbformatProblem(artifacts.notebook)).toBeNull();
    const notebook = JSON.parse(artifacts.notebook);
    expect(notebook.cells.every((cell: { outputs?: unknown[]; execution_count?: unknown }) => !cell.outputs?.length && !cell.execution_count)).toBe(true);
    expect(notebook.metadata.freecam.workflow_hash).toBe(document.workflow_hash);
  });
});
