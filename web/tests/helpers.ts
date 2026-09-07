import { existsSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";

import type { CatalogEntry, CatalogSnapshot, WorkflowDocument } from "../src/model/types";

export const REPO = resolve(__dirname, "..", "..");

export function loadSnapshot(): CatalogSnapshot {
  return JSON.parse(readFileSync(resolve(REPO, "web", "public", "catalog.json"), "utf8")) as CatalogSnapshot;
}

export function catalogOf(snapshot: CatalogSnapshot): Map<string, CatalogEntry> {
  return new Map(snapshot.entries.map((entry) => [entry.id, entry]));
}

/** The checkout's Python, when it is there; the cross-language checks skip otherwise. */
export function pythonExecutable(): string | null {
  const candidate = resolve(REPO, ".venv", "bin", "python");
  return existsSync(candidate) ? candidate : null;
}

export function runPython(code: string, stdin = ""): string {
  const python = pythonExecutable();
  if (!python) throw new Error("no Python");
  const result = spawnSync(python, ["-c", code], { input: stdin, encoding: "utf8", env: { ...process.env, PYTHONPATH: resolve(REPO, "src") } });
  if (result.status !== 0) throw new Error(result.stderr || `python exited ${result.status}`);
  return result.stdout;
}

export function pythonHash(document: WorkflowDocument): string {
  return runPython(
    "import json,sys\nfrom freecam.pi_cam.workflow_builder import WorkflowDocument\n" +
      "print(WorkflowDocument.from_payload(json.load(sys.stdin)).workflow_hash)",
    JSON.stringify(document),
  ).trim();
}

export function pythonSyntaxError(source: string): string | null {
  const out = runPython(
    "import ast,sys\nsrc=sys.stdin.read()\n" +
      "try:\n    ast.parse(src)\n    print('ok')\nexcept SyntaxError as e:\n    print(f'{e.msg} line {e.lineno}')",
    source,
  ).trim();
  return out === "ok" ? null : out;
}

export function nbformatProblem(notebook: string): string | null {
  const out = runPython(
    "import json,sys\nimport nbformat\nnb=nbformat.reads(sys.stdin.read(), as_version=4)\n" +
      "try:\n    nbformat.validate(nb)\n    print('ok')\nexcept nbformat.ValidationError as e:\n    print(str(e)[:400])",
    notebook,
  ).trim();
  return out === "ok" ? null : out;
}

export function pythonValidationCodes(document: WorkflowDocument, level = "browser"): string[] {
  const out = runPython(
    "import json,sys\nfrom freecam.pi_cam.workflow_builder import WorkflowDocument, load_catalog, validate_document\n" +
      "default, entries, snapshot = load_catalog()\n" +
      "doc = WorkflowDocument.from_payload(json.load(sys.stdin))\n" +
      `report = validate_document(doc, default=default, catalog=entries, level=${JSON.stringify(level)}, catalog_version=snapshot['catalog_hash'])\n` +
      "print(json.dumps(sorted(i.code for i in report.issues)))",
    JSON.stringify(document),
  );
  return JSON.parse(out) as string[];
}
