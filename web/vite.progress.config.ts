import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The progress dashboard: a separate static entry rooted at web/progress,
// published at /freeCAM/progress/.  It fetches one exported snapshot
// (progress/public/progress.json) and never contacts a runtime service.
export default defineConfig({
  plugins: [react()],
  root: "progress",
  base: "./",
  build: {
    outDir: "../dist-progress",
    emptyOutDir: true,
    sourcemap: false,
  },
});
