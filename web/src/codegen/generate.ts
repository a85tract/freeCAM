// From a workflow document to the freeCAM code that runs it.  One generator
// serves the page in the browser, the published preview, and -- bundled for
// Node -- the local service and the tests, so the same document yields the
// same script, notebook and setup snippet everywhere.
//
// The code uses the public interface as a notebook does: fc.Driver, the
// workflow list, process handles, fc.Physics and fc.Property, state.create,
// driver.cam.parameters, and a stage class attached to the model.  Nothing
// is hidden behind a loader; the reader sees every call the run makes.

import type { CatalogSnapshot, ParameterValue, WorkflowDocument, WorkflowNode } from "../model/types";

export interface GeneratedArtifacts {
  /** Process definitions and configure(driver): paste into a notebook with a live driver. */
  setup: string;
  /** A complete script: imports, configuration, initialization, run, cleanup. */
  script: string;
  /** An nbformat 4.5 notebook as JSON text, without outputs or execution counts. */
  notebook: string;
  /** The document itself, for continuing in the builder. */
  workflow: string;
  /** Files the user supplies that the code refers to by path. */
  external_files: string[];
  /** What differs from the validated default; empty for the default itself. */
  changes: string[];
}

const STAGE_IMPORTS: Record<string, { module: string; className: string; kwarg: Record<string, string> }> = {
  "cam_run1.cloud_macro_microphysics": {
    module: "freecam.physics.cloud_macro_microphysics",
    className: "CloudMacroMicrophysics",
    kwarg: { mmacro_pcond: "macro_surrogate" },
  },
};

export function pyLiteral(value: ParameterValue | undefined): string {
  if (value === null || value === undefined) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("only finite numbers can be written into Python");
    return String(value);
  }
  if (typeof value === "string") return pyString(value);
  if (Array.isArray(value)) return "[" + value.map(pyLiteral).join(", ") + "]";
  return "{" + Object.keys(value).sort().map((key) => `${pyString(key)}: ${pyLiteral(value[key])}`).join(", ") + "}";
}

export function pyString(value: string): string {
  return JSON.stringify(value);
}

function scientificOrder(document: WorkflowDocument): string[] {
  return document.nodes.filter((node) => node.scientific && node.enabled).map((node) => node.name);
}

function classNameOf(source: string, fallback: string): string {
  const match = /^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/m.exec(source);
  return match ? match[1] : fallback;
}

function previousScientific(document: WorkflowDocument, node: WorkflowNode): WorkflowNode | null {
  const index = document.nodes.indexOf(node);
  for (let i = index - 1; i >= 0; i--) {
    const candidate = document.nodes[i];
    if (candidate.scientific && candidate.enabled) return candidate;
  }
  return null;
}

function nextScientific(document: WorkflowDocument, node: WorkflowNode): WorkflowNode | null {
  const index = document.nodes.indexOf(node);
  for (let i = index + 1; i < document.nodes.length; i++) {
    const candidate = document.nodes[i];
    if (candidate.scientific && candidate.enabled) return candidate;
  }
  return null;
}

/** What the document changes against the default, in words. */
export function describeChanges(document: WorkflowDocument, defaults: WorkflowDocument): string[] {
  const changes: string[] = [];
  const defaultOrder = scientificOrder(defaults);
  const order = scientificOrder(document);
  const defaultByName = new Map(defaults.nodes.map((node) => [node.name, node]));
  for (const node of document.nodes) {
    if (node.origin === "python") changes.push(`Python process ${node.name}${node.enabled ? "" : " (disabled)"}`);
    else if (node.origin === "catalog") changes.push(`catalog process ${node.display_name}`);
  }
  for (const node of defaults.nodes) {
    if (!node.scientific) continue;
    const current = document.nodes.find((item) => item.id === node.id);
    if (!current) changes.push(`${node.display_name} removed`);
    else if (current.enabled !== node.enabled) changes.push(`${node.display_name} ${current.enabled ? "enabled" : "disabled"}`);
  }
  const sharedDefault = defaultOrder.filter((name) => order.includes(name));
  const sharedCurrent = order.filter((name) => defaultOrder.includes(name));
  if (sharedDefault.join(" ") !== sharedCurrent.join(" ")) changes.push("scientific order changed");
  for (const node of document.nodes) {
    for (const [kernel, binding] of Object.entries(node.configuration.kernels)) {
      if (binding.kind !== "original") changes.push(`${node.display_name}.${kernel} answered by ${binding.path}`);
    }
    const defaultNode = defaultByName.get(node.name);
    if (node.origin !== "python" && Object.keys(node.configuration.parameters).length) {
      changes.push(`${node.display_name} parameters ${Object.keys(node.configuration.parameters).sort().join(", ")}`);
    }
    if (node.origin === "python" && defaultNode === undefined && node.configuration.variables.length) {
      changes.push(`${node.name} declares ${node.configuration.variables.map((v) => v.name).join(", ")}`);
    }
  }
  if (Object.keys(document.namelist).length) changes.push(`namelist ${Object.keys(document.namelist).sort().join(", ")}`);
  if (document.case !== defaults.case) changes.push(`case ${document.case}`);
  return changes;
}

function header(document: WorkflowDocument, snapshot: CatalogSnapshot, lines: string[]): string[] {
  const external = externalFiles(document);
  return [
    `# freeCAM workflow ${document.workflow_hash.slice(0, 12)} -- generated by the Workflow Builder`,
    `# catalog ${snapshot.catalog_hash.slice(0, 12)}, source ${document.source_version || snapshot.source_revision}`,
    ...(lines.length ? ["# changes against the validated default:", ...lines.map((line) => `#   - ${line}`)] : ["# the validated default workflow, unchanged"]),
    ...(external.length ? ["# files you provide, referred to by path:", ...external.map((path) => `#   - ${path}`)] : []),
    ...(document.experimental ? ["# Experimental: the order or the set of processes is not the validated default."] : []),
  ];
}

export function externalFiles(document: WorkflowDocument): string[] {
  const files = new Set<string>();
  for (const node of document.nodes) {
    for (const binding of Object.values(node.configuration.kernels)) {
      if (binding.kind === "surrogate" && binding.path) files.add(binding.path);
    }
  }
  return [...files].sort();
}

/** The class sources of every Python process, each ending with one blank line. */
function processDefinitions(document: WorkflowDocument): string[] {
  const out: string[] = [];
  for (const node of document.nodes) {
    if (node.origin !== "python") continue;
    const source = (node.configuration.python_source ?? "").replace(/\s+$/, "");
    out.push(`# --- ${node.name} ---`, source, "", "");
  }
  return out;
}

/** The body of configure(driver): what the service applies, as code. */
export function configureBody(document: WorkflowDocument, defaults: WorkflowDocument): string[] {
  const lines: string[] = [];
  const indent = "    ";
  const add = (text = "") => lines.push(text ? indent + text : "");
  add("workflow = driver.cam.workflow");
  add("state = driver.cam.state");

  // 1. fields the Python processes declare
  const variables = document.nodes.filter((n) => n.origin === "python" && n.enabled).flatMap((n) => n.configuration.variables);
  if (variables.length) {
    add();
    add("# fields the Python processes declare");
    for (const variable of variables) {
      const args = [pyString(variable.name), `like=${pyString(variable.like)}`, `units=${pyString(variable.units)}`];
      if (!variable.output) args.push("output=False");
      add(`state.create(${args.join(", ")})`);
    }
  }

  // 2. Python processes, each placed after the process before it
  const pythonNodes = document.nodes.filter((n) => n.origin === "python");
  if (pythonNodes.length) {
    add();
    add("# Python processes, in their places");
    for (const node of pythonNodes) {
      const className = classNameOf(node.configuration.python_source ?? "", "Process");
      const properties = Object.entries(node.configuration.parameters);
      const variable = node.name;
      add(`${variable} = ${className}()`);
      for (const [key, value] of properties) add(`${variable}.${key} = ${pyLiteral(value)}`);
      const before = previousScientific(document, node);
      const after = nextScientific(document, node);
      const placement = before ? `after=${pyString(before.name)}` : after ? `before=${pyString(after.name)}` : "";
      add(`workflow.insert(${variable}${placement ? ", " + placement : ""})`);
      if (!node.enabled) add(`workflow[${pyString(node.name)}].disable()`);
    }
  }

  // 3. catalog processes
  const catalogNodes = document.nodes.filter((n) => n.origin === "catalog" && n.enabled);
  if (catalogNodes.length) {
    add();
    add("# original routines from the catalog, bound to the live state and inserted");
    for (const node of catalogNodes) {
      const before = previousScientific(document, node);
      const after = nextScientific(document, node);
      const placement = before ? `after=${pyString(before.name)}` : after ? `before=${pyString(after.name)}` : "";
      add(`driver.cam.physics.process(${pyString(node.name)}).insert(${placement})`);
    }
  }

  // 4. stage classes with a kernel replaced
  const bound = document.nodes.filter((n) => Object.values(n.configuration.kernels).some((b) => b.kind !== "original"));
  if (bound.length) {
    add();
    add("# a stage class in the stage's place, with a model in a kernel's slot");
    for (const node of bound) {
      const spec = STAGE_IMPORTS[node.id];
      if (!spec) {
        add(`raise NotImplementedError(${pyString(`no stage class is wired for ${node.display_name}`)})`);
        continue;
      }
      const kwargs: string[] = [];
      for (const [kernel, binding] of Object.entries(node.configuration.kernels)) {
        if (binding.kind === "original") continue;
        const kwarg = spec.kwarg[kernel];
        if (kwarg) kwargs.push(`${kwarg}=${pyString(binding.path ?? "")}`);
        else add(`raise NotImplementedError(${pyString(`${kernel} has no binding through ${spec.className}`)})`);
      }
      add(`stage = ${spec.className}(${kwargs.join(", ")})`);
      add(`stage.attach(driver.cam)`);
    }
  }

  // 5. enabled and disabled original processes, and the order
  const defaultOrder = scientificOrder(defaults);
  const order = scientificOrder(document);
  if (order.join(" ") !== defaultOrder.join(" ")) {
    add();
    add("# the step's scientific processes, in this order; anything not listed stops running");
    add("workflow.replace([");
    for (const name of order) add(`    ${pyString(name)},`);
    add("])");
  }

  // 6. runtime parameters of original processes
  const tuned = document.nodes.filter((n) => n.origin === "default" && Object.keys(n.configuration.parameters).length);
  if (tuned.length) {
    add();
    add("# audited runtime tunables, applied on every rank before the next step");
    for (const node of tuned) {
      for (const key of Object.keys(node.configuration.parameters).sort()) {
        add(`driver.cam.parameters[${pyString(key)}] = ${pyLiteral(node.configuration.parameters[key])}`);
      }
    }
  }

  if (lines.length === 2) add("# the validated default: nothing to change");
  return lines;
}

function driverCall(document: WorkflowDocument): string {
  const args = [`case=${pyString(document.case)}`, `nsteps=${document.nsteps}`];
  if (Object.keys(document.namelist).length) args.push(`namelist=${pyLiteral(document.namelist)}`);
  return `fc.Driver(${args.join(", ")})`;
}

function stageImports(document: WorkflowDocument): string[] {
  const out: string[] = [];
  for (const node of document.nodes) {
    if (!Object.values(node.configuration.kernels).some((b) => b.kind !== "original")) continue;
    const spec = STAGE_IMPORTS[node.id];
    if (spec) out.push(`from ${spec.module} import ${spec.className}`);
  }
  return out;
}

export function generateSetup(document: WorkflowDocument, snapshot: CatalogSnapshot): string {
  const defaults = snapshot.default_document;
  const changes = describeChanges(document, defaults);
  return [
    ...header(document, snapshot, changes),
    "import freecam as fc",
    ...stageImports(document),
    "",
    "",
    ...processDefinitions(document),
    "def configure(driver):",
    '    """Apply this workflow to a live model: call once, after driver.initialize()."""',
    "",
    ...configureBody(document, defaults),
    "",
  ].join("\n");
}

export function generateScript(document: WorkflowDocument, snapshot: CatalogSnapshot): string {
  const defaults = snapshot.default_document;
  const changes = describeChanges(document, defaults);
  return [
    ...header(document, snapshot, changes),
    "import freecam as fc",
    ...stageImports(document),
    "",
    `CASE = ${pyString(document.case)}`,
    `NSTEPS = ${document.nsteps}`,
    "",
    "",
    ...processDefinitions(document),
    "def configure(driver):",
    '    """Apply this workflow to a live model: call once, after driver.initialize()."""',
    "",
    ...configureBody(document, defaults),
    "",
    "",
    "def main():",
    `    # Nothing starts here; the first live operation launches the MPI session.`,
    `    with ${driverCall(document)} as driver:`,
    "        driver.initialize()",
    "        configure(driver)",
    "        result = driver.run(progress=True)",
    "        print(result)",
    "        print(driver.cam.history.latest())",
    "    # leaving the block closes the model and releases its ranks",
    "",
    "",
    'if __name__ == "__main__":',
    "    main()",
    "",
  ].join("\n");
}

interface Cell {
  cell_type: "markdown" | "code";
  id: string;
  metadata: Record<string, unknown>;
  source: string[];
  execution_count?: null;
  outputs?: unknown[];
}

function cell(kind: Cell["cell_type"], id: string, text: string): Cell {
  const lines = text.split("\n");
  const source = lines.map((line, index) => (index < lines.length - 1 ? line + "\n" : line)).filter((line, index, all) => !(index === all.length - 1 && line === ""));
  const base: Cell = { cell_type: kind, id, metadata: {}, source };
  if (kind === "code") {
    base.execution_count = null;
    base.outputs = [];
  }
  return base;
}

export function generateNotebook(document: WorkflowDocument, snapshot: CatalogSnapshot): string {
  const defaults = snapshot.default_document;
  const changes = describeChanges(document, defaults);
  const external = externalFiles(document);
  const intro = [
    `# freeCAM workflow \`${document.workflow_hash.slice(0, 12)}\``,
    "",
    `Generated by the Workflow Builder from catalog \`${snapshot.catalog_hash.slice(0, 12)}\`` +
      ` (source \`${document.source_version || snapshot.source_revision}\`).`,
    "",
    changes.length ? "Changes against the validated default:" : "This is the validated default workflow, unchanged.",
    ...changes.map((line) => `- ${line}`),
    ...(external.length ? ["", "Files you provide, referred to by path:", ...external.map((path) => `- \`${path}\``)] : []),
    ...(document.experimental ? ["", "**Experimental**: the order or the set of processes is not the validated default."] : []),
    "",
    "Constructing the `Driver` starts nothing; `initialize()` requests the ranks the case needs.",
  ].join("\n");
  const definitions = ["import freecam as fc", ...stageImports(document), "", "", ...processDefinitions(document),
    "def configure(driver):",
    '    """Apply this workflow to a live model: call once, after driver.initialize()."""',
    "",
    ...configureBody(document, defaults)].join("\n");
  const start = [`driver = ${driverCall(document)}`, "driver.initialize()", "configure(driver)", "driver.cam.workflow"].join("\n");
  const run = ["result = driver.run(progress=True)", "result", "", "driver.cam.history.latest()"].join("\n");
  const cells: Cell[] = [
    cell("markdown", "intro", intro),
    cell("markdown", "definitions-heading", "## Definitions and configuration"),
    cell("code", "definitions", definitions),
    cell("markdown", "start-heading", "## Initialize and arrange"),
    cell("code", "start", start),
    cell("markdown", "run-heading", "## Run and observe"),
    cell("code", "run", run),
    cell("markdown", "close-heading", "## Close the model\n\nReleases the MPI ranks and writes the timing reports."),
    cell("code", "close", "driver.close()"),
  ];
  const notebook = {
    cells,
    metadata: {
      kernelspec: { display_name: "Python 3", language: "python", name: "python3" },
      language_info: { name: "python" },
      freecam: { workflow_hash: document.workflow_hash, catalog_hash: snapshot.catalog_hash },
    },
    nbformat: 4,
    nbformat_minor: 5,
  };
  return JSON.stringify(notebook, null, 1) + "\n";
}

export function generateAll(document: WorkflowDocument, snapshot: CatalogSnapshot): GeneratedArtifacts {
  return {
    setup: generateSetup(document, snapshot),
    script: generateScript(document, snapshot),
    notebook: generateNotebook(document, snapshot),
    workflow: JSON.stringify(document, null, 2) + "\n",
    external_files: externalFiles(document),
    changes: describeChanges(document, snapshot.default_document),
  };
}
