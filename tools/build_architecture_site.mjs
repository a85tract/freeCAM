// Build the checked-in architecture HTML as a ChatGPT Sites Worker.

import {
  copyFile,
  mkdir,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const project = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const output = resolve(project, process.argv[2] || "dist");
const html = await readFile(
  resolve(project, "docs/pycam_sima_architecture.html"),
  "utf8",
);
const worker = `const HTML = ${JSON.stringify(html)};

export default {
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(HTML, {
        headers: {
          "content-type": "text/html; charset=utf-8",
          "cache-control": "public, max-age=300",
          "x-content-type-options": "nosniff"
        }
      });
    }
    return new Response("Not found", {
      status: 404,
      headers: {"content-type": "text/plain; charset=utf-8"}
    });
  }
};
`;

await rm(output, { recursive: true, force: true });
await mkdir(resolve(output, "server"), { recursive: true });
await mkdir(resolve(output, ".openai"), { recursive: true });
await writeFile(resolve(output, "server/index.js"), worker);
await copyFile(
  resolve(project, "docs/pycam_sima_architecture.html"),
  resolve(output, "index.html"),
);
await copyFile(
  resolve(project, ".openai/hosting.json"),
  resolve(output, ".openai/hosting.json"),
);
console.log(`Built architecture site in ${output}`);
