import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build lands inside the Python package, so a wheel carries the page and
// the local service serves it from there.  GitHub Pages builds with
// --base=/freeCAM/ --outDir dist-pages (see package.json).
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "../src/freecam/pi_cam/workflow_builder/static",
    emptyOutDir: true,
    sourcemap: false,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.ts", "tests/**/*.test.tsx"],
  },
});
