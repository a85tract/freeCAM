// Node entry for the generator: the Python service and the tests call it so
// they produce exactly what the browser produces.
//
//   node codegen.mjs <document.json> <snapshot.json>       -> artifacts as JSON on stdout
//   node codegen.mjs --hash <document.json>                -> the document's hash
//   node codegen.mjs --validate <document.json> <snapshot.json> -> the browser-level report

import { readFileSync } from "node:fs";

import { generateAll } from "./generate";
import { workflowHash } from "../model/canonical";
import { validateDocument } from "../model/validate";
import type { CatalogEntry, CatalogSnapshot, WorkflowDocument } from "../model/types";

function read<T>(path: string): T {
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

const args = process.argv.slice(2);
if (args[0] === "--hash") {
  const document = read<WorkflowDocument>(args[1]);
  process.stdout.write(workflowHash(document) + "\n");
} else if (args[0] === "--validate") {
  const document = read<WorkflowDocument>(args[1]);
  const snapshot = read<CatalogSnapshot>(args[2]);
  const catalog = new Map<string, CatalogEntry>(snapshot.entries.map((entry) => [entry.id, entry]));
  const report = validateDocument(document, snapshot.default_document, catalog, snapshot.capabilities, snapshot.catalog_hash);
  process.stdout.write(JSON.stringify(report) + "\n");
} else {
  const document = read<WorkflowDocument>(args[0]);
  const snapshot = read<CatalogSnapshot>(args[1]);
  process.stdout.write(JSON.stringify(generateAll(document, snapshot)) + "\n");
}
