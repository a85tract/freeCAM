// The progress dashboard: rendering, navigation, and count integrity against a
// fixture snapshot, plus contract checks against the committed real snapshot.

import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "../progress/src/App";
import { buildTree, processBars, searchAll } from "../progress/src/derive";
import type { Snapshot } from "../progress/src/types";

const REAL_SNAPSHOT_PATH = resolve(process.cwd(), "progress/public/progress.json");

function fixture(): Snapshot {
  const capability = (state: string, extra: object = {}) => ({ state, ...extra }) as never;
  return {
    schema_version: 1,
    content_hash: "abc123def456",
    volatile: { generated_at: "2026-09-08T00:00:00+00:00", commit: "cafe0123456", branch: "dev-branch" },
    case: { case: "PI-atm" },
    cam_source_revision: "rev123456",
    inputs: {},
    notes: {
      core_vs_candidates:
        "Core kernels are exposed for replacement. Exposing a process does not expose every candidate inside it.",
      banner: "Some validated capabilities require an experimental image and are not enabled in the default build.",
      snapshot: "This is a published snapshot of committed records, not a live HPC job monitor.",
      process_vs_kernel: "A process is a workflow operation. It can contain many numerical kernels.",
      class_delegation: "A Python class may delegate numerical execution to original Fortran.",
      candidates: "Candidates are potentially reachable, not observed during execution.",
      shared_kernels: "A kernel used by several processes is counted once globally.",
      replacement_scope: "A replacement gate is scoped to the process and call site actually tested.",
      bfb_meaning: "BFB verified does not prove an arbitrary replacement is scientifically correct.",
    },
    capability_explanations: {
      contract: "Contract explanation.",
      adapter_build: "Build explanation.",
      independently_callable: "Callable explanation.",
      standalone_replay: "Replay explanation.",
      in_model_replacement: "Replacement explanation.",
      original_replacement_bfb: "BFB explanation.",
    },
    processes: [
      {
        id: "cam_run1.a", native_id: 1, operation: "a_tend", display_name: "Deep-ish convection",
        description: "Process A.", phase: "cam_run1", kind: "scheme", classification: "numeric_scheme",
        granularity: "stage", parent_stage: null, enabled: true, activity: "active",
        activity_basis: "enabled in the plan", alternate_of: [], default_index: 0, in_default: true,
        python_api: "available", python_class: "freecam.physics.pausable.A", class_kind: "dedicated",
        core_kernels: [{ routine: "alpha", id: "m::alpha", owner_class: "freecam.physics.sub.AlphaOwner",
                         status: "complete" }],
        ledger_coverage: "partial", note: null,
      },
      {
        id: "cam_run1.z", native_id: 3, operation: "z_tend", display_name: "Zeta scheme",
        description: "Process Z.", phase: "cam_run1", kind: "scheme", classification: "numeric_scheme",
        granularity: "stage", parent_stage: null, enabled: true, activity: "active",
        activity_basis: "enabled in the plan", alternate_of: [], default_index: 1, in_default: true,
        python_api: "available", python_class: null, class_kind: "generic", core_kernels: [],
        ledger_coverage: "gap", note: null,
      },
      {
        id: "coupling.io", native_id: 2, operation: "io", display_name: "History output",
        description: "I/O.", phase: "coupling", kind: "io", classification: "io", granularity: "stage",
        parent_stage: null, enabled: true, activity: "active", activity_basis: "enabled in the plan",
        alternate_of: [], default_index: 2, in_default: true, python_api: "available",
        python_class: null, class_kind: "generic", core_kernels: [],
        ledger_coverage: "not-applicable", note: null,
      },
    ],
    additional_apis: [
      { id: "catalog:x", name: "x", display_name: "x", qualified_name: "mx::x", description: "mx::x",
        present: false, addable: false, reason: "not independently runnable (context_required)" },
    ],
    kernels: {
      "m::alpha": {
        id: "m::alpha", routine: "alpha", module: "m", kind: "subroutine", host: null, public: true,
        source: { file: "components/m.F90", line_start: 10, line_end: 90 },
        processes: ["cam_run1.a"], callers: ["drv::a_tend"], tracked: true, status: "complete",
        missing: [], owner_class: "freecam.physics.pausable.A", note: null, module_state: ["state.json"],
        capabilities: {
          contract: capability("available", { path: "native/pi_cam/functions/alpha.yaml" }),
          adapter_build: capability("available", { evidence: ["alpha_build.json"] }),
          independently_callable: capability("verified", { evidence: ["alpha_replay.json"] }),
          standalone_replay: capability("verified", { evidence: ["alpha_replay.json"] }),
          in_model_replacement: capability("available", { contexts: ["cam_run1.a"] }),
          original_replacement_bfb: capability("verified", { contexts: ["cam_run1.a"] }),
        },
        redirect: { classification: "rename-references", redirectable: true, reading: "static" },
        adapter_hint: null, failures: ["alpha_failure.json"],
      },
      "m::shared": {
        id: "m::shared", routine: "shared", module: "m", kind: "subroutine", host: null, public: true,
        source: { file: "components/m.F90", line_start: 100, line_end: 150 },
        processes: ["cam_run1.a", "cam_run1.z"], callers: ["m::alpha", "drv::z_tend"], tracked: false,
        status: null, missing: null, owner_class: null, note: null, module_state: [],
        capabilities: {
          contract: capability("not-implemented"),
          adapter_build: capability("not-implemented"),
          independently_callable: capability("needs-binding", { blockers: ["derived_or_unknown_type"] }),
          standalone_replay: capability("not-assessed"),
          in_model_replacement: capability("not-implemented"),
          original_replacement_bfb: capability("not-assessed"),
        },
        redirect: { classification: "no-call-relocation", redirectable: false, reading: "static",
                    blocker: "inlined at every compiled call site; no symbol redirection can reach it" },
        adapter_hint: { adapter_status: "context_required", blockers: ["derived_or_unknown_type"] },
        failures: [],
      },
    },
    process_membership: {
      "cam_run1.a": {
        kernels: ["m::alpha", "m::shared"],
        edges: [
          { parent: null, kernel: "m::alpha", via: ["drv::a_tend"] },
          { parent: "m::alpha", kernel: "m::shared", via: [] },
        ],
        inventoried: true,
      },
      "cam_run1.z": { kernels: [], edges: [], inventoried: false },
      "coupling.io": { kernels: [], edges: [], inventoried: true },
    },
    replacements: [
      {
        kernel_routine: "alpha", process: "cam_run1.a", mechanism: "hook", within: "beta",
        state: "verified",
        gates: [{ record: "gate_a.json", bfb_record: "gate_a_bfb.json", bfb: true, replacement_calls: 100 }],
        historical_failures: [], note: null, requires_development_image: true,
      },
    ],
    evidence: {
      "gate_a.json": { kind: "run-summary", steps: 50, mpi_ranks: 512, pbs_job: "1234", bfb: true },
      "gate_a_bfb.json": { kind: "bfb-comparison", bfb: true, compared_files: 4 },
      "alpha_build.json": { kind: "standalone-build", library_sha256: "beef" },
      "alpha_replay.json": { kind: "frame-replay", calls: 10, samples: 10, bfb: true },
      "alpha_failure.json": { kind: "gate-failure", outcome: "not bit-for-bit" },
    },
    failure_records: [
      { record: "alpha_failure.json", kernel_routine: "alpha", capability: "original_replacement_bfb",
        contexts: ["cam_run1.a"] },
    ],
    unmapped_kernels: [],
    totals: {},
  };
}

const loadFixture = () => Promise.resolve(fixture());

beforeEach(() => {
  window.location.hash = "";
  localStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("the progress dashboard", () => {
  it("renders physics processes in the exported default order, with the banner", async () => {
    render(<App load={loadFixture} />);
    const browser = await screen.findByRole("navigation", { name: "Process browser" });
    const buttons = within(browser).getAllByRole("button").map((b) => b.textContent ?? "");
    const deep = buttons.findIndex((t) => t.includes("Deep-ish convection"));
    expect(buttons[deep]).toContain("1 core · 2 candidates");
    const zeta = buttons.findIndex((t) => t.includes("Zeta scheme"));
    expect(deep).toBeGreaterThan(-1);
    expect(zeta).toBeGreaterThan(deep);
    expect(buttons.join(" ")).not.toContain("History output");     // an I/O action is not a physics row
    expect(screen.getByRole("note").textContent).toContain("Development snapshot");
    expect(screen.getByRole("note").textContent).toContain("dev-branch");
  });

  it("selecting a process shows its bars and nested kernels; selecting a kernel shows capabilities and evidence", async () => {
    const user = userEvent.setup();
    render(<App load={loadFixture} />);
    await user.click(await screen.findByRole("button", { name: /Deep-ish convection/ }));
    // the two levels are never conflated: the class's core kernels first, with their
    // owner classes, then the recursive static candidates under their own heading
    expect(screen.getByText(/Core kernels exposed for replacement \(1\)/)).toBeInTheDocument();
    expect(screen.getAllByText("freecam.physics.sub.AlphaOwner").length).toBeGreaterThanOrEqual(2); // the facts row and the core list
    expect(screen.getByText(/All candidate numerical functions \(statically reachable/)).toBeInTheDocument();
    expect(screen.getByText(/Exposing a process does not expose every candidate/)).toBeInTheDocument();
    expect(screen.getByText("Candidate kernel coverage")).toBeInTheDocument();
    expect(screen.getAllByText("Independently callable").length).toBeGreaterThan(0);
    // counts come from the records: 1 of 2 kernels verified
    const barTexts = Array.from(document.querySelectorAll(".bar-head")).map((n) => n.textContent ?? "");
    expect(barTexts.some((t) => t.includes("Independently callable") && t.includes("1 / 2"))).toBe(true);
    // the tree nests shared under alpha, labeled as calls
    const tree = screen.getByRole("list", { name: /Internal numerical kernels/ });
    expect(within(tree).getByText("alpha")).toBeInTheDocument();
    expect(within(tree).getByText("shared")).toBeInTheDocument();
    await user.click(within(tree).getByText("shared"));
    expect(await screen.findByRole("heading", { name: /m::shared/ })).toBeInTheDocument();
    // needs-binding, with the blocker named and never rendered as completed
    expect(screen.getByText("Needs binding")).toBeInTheDocument();
    expect(screen.getAllByText(/derived_or_unknown_type/).length).toBeGreaterThan(0);
    expect(screen.getByText(/inlined at every compiled call site/)).toBeInTheDocument();
  });

  it("a kernel page shows scoped replacement evidence and collapsed historical failures", async () => {
    window.location.hash = "#/kernel/m::alpha";
    render(<App load={loadFixture} />);
    expect(await screen.findByRole("heading", { name: /m::alpha/ })).toBeInTheDocument();
    expect(screen.getAllByText("Verified").length).toBeGreaterThan(0);
    expect(screen.getByText(/100 replacement calls in this process/)).toBeInTheDocument();
    expect(screen.getByText(/requires an experimental image/)).toBeInTheDocument();
    const failures = screen.getByText(/Historical failed experiments \(1\)/);
    expect(failures).toBeInTheDocument();
    expect(screen.getByText(/scoped to the process and call site actually tested/)).toBeInTheDocument();
  });

  it("deep links select a process and kernel from the hash", async () => {
    window.location.hash = "#/process/cam_run1.a/kernel/m::shared";
    render(<App load={loadFixture} />);
    expect(await screen.findByRole("heading", { name: /m::shared/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /back to process/ })).toBeInTheDocument();
  });

  it("search finds kernels by module and processes by class name", async () => {
    const snapshot = fixture();
    const kernelHits = searchAll(snapshot, "m::sha");
    expect(kernelHits.map((h) => h.id)).toContain("m::shared");
    const classHits = searchAll(snapshot, "pausable.A");
    expect(classHits.map((h) => h.id)).toContain("cam_run1.a");
    const user = userEvent.setup();
    render(<App load={loadFixture} />);
    await screen.findByRole("navigation", { name: "Process browser" });
    await user.type(screen.getByRole("searchbox", { name: /Search processes/ }), "shared");
    const results = await screen.findByRole("list", { name: "Search results" });
    await user.click(within(results).getAllByRole("button")[0]);
    expect(await screen.findByRole("heading", { name: /m::shared/ })).toBeInTheDocument();
  });

  it("a process without an inventory shows Not inventoried, never 100%", async () => {
    const user = userEvent.setup();
    render(<App load={loadFixture} />);
    await user.click(await screen.findByRole("button", { name: /Zeta scheme/ }));
    expect(screen.getAllByText("Not inventoried").length).toBe(4);
    expect(screen.getAllByText(/absence of data is not completion/).length).toBe(4);
    const fills = Array.from(document.querySelectorAll(".bar-fill")) as HTMLElement[];
    expect(fills.every((f) => f.style.width === "0%")).toBe(true);
  });

  it("fetches only the exported snapshot and reports a load failure plainly", async () => {
    const fetchSpy = vi.fn(async (url: RequestInfo | URL) => {
      expect(String(url)).toBe("./progress.json");
      return { ok: true, json: async () => fixture() } as Response;
    });
    vi.stubGlobal("fetch", fetchSpy);
    render(<App />);
    await screen.findByRole("navigation", { name: "Process browser" });
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    cleanup();
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 404 }) as Response));
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/could not be loaded/);
  });

  it("applies and persists the shared theme preference", async () => {
    localStorage.setItem("freecam-ui-theme", "dark");
    render(<App load={loadFixture} />);
    await screen.findByRole("navigation", { name: "Process browser" });
    expect(document.documentElement.dataset.theme).toBe("dark");
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Light theme" }));
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(localStorage.getItem("freecam-ui-theme")).toBe("light");
  });
});

describe("derivations", () => {
  it("cuts recursion in the tree instead of looping", () => {
    const membership = {
      kernels: ["k::a", "k::b"],
      edges: [
        { parent: null, kernel: "k::a", via: [] },
        { parent: "k::a", kernel: "k::b", via: [] },
        { parent: "k::b", kernel: "k::a", via: [] },
      ],
      inventoried: true,
    };
    const roots = buildTree(membership);
    expect(roots).toHaveLength(1);
    const a = roots[0];
    expect(a.children[0].kernel).toBe("k::b");
    expect(a.children[0].children[0].cycle).toBe(true);
    expect(a.children[0].children[0].children).toHaveLength(0);
  });

  it("keeps blocked and unassessed kernels in every denominator", () => {
    const bars = processBars(fixture(), "cam_run1.a");
    expect(bars.every((bar) => bar.total === 2)).toBe(true);
    expect(bars.find((b) => b.key === "original_replacement_bfb")?.done).toBe(1);
  });
});

describe("the committed snapshot", () => {
  const snapshot = JSON.parse(readFileSync(REAL_SNAPSHOT_PATH, "utf8")) as Snapshot;

  it("has the schema and derivable counts the page relies on", () => {
    expect(snapshot.schema_version).toBe(1);
    expect(snapshot.processes.length).toBeGreaterThan(0);
    expect(Object.keys(snapshot.kernels).length).toBeGreaterThan(0);
    // totals in the file equal what the page derives from the records themselves
    const kernels = Object.values(snapshot.kernels);
    expect(snapshot.totals["candidate_kernels"]).toBe(kernels.length);
    expect(snapshot.totals["tracked_kernels"]).toBe(kernels.filter((k) => k.tracked).length);
    const verified = kernels.filter(
      (k) => k.capabilities.original_replacement_bfb?.state === "verified").length;
    expect(snapshot.totals["original_replacement_bfb_verified"]).toBe(verified);
  });

  it("names no personal path, account, or machine", () => {
    const text = JSON.stringify(snapshot).toLowerCase();
    for (const fragment of ["/glade", "desched", "/home/", "scratch"]) {
      expect(text).not.toContain(fragment);
    }
  });

  it("scopes a shared kernel's replacement verification to the tested process", () => {
    const fice = snapshot.kernels["cloud_fraction::cldfrc_fice"];
    expect(fice.processes.length).toBeGreaterThan(1);
    expect(fice.capabilities.original_replacement_bfb.contexts).toEqual(["cam_run1.deep_convection"]);
  });
});
