// Bundle the code generator for Node, so the Python service and the tests can
// generate the same script, notebook and setup snippet the browser does.
import { build } from "esbuild";
import { mkdirSync } from "node:fs";

mkdirSync("../src/freecam/pi_cam/workflow_builder/static", { recursive: true });
await build({
  entryPoints: ["src/codegen/cli.ts"],
  bundle: true,
  platform: "node",
  format: "esm",
  target: "node18",
  outfile: "../src/freecam/pi_cam/workflow_builder/static/codegen.mjs",
  logLevel: "warning",
});
