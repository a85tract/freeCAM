import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "../src/App";
import { loadSnapshot } from "./helpers";

// jsdom has no layout, which CodeMirror measures against; a textarea stands in for it here.
vi.mock("@uiw/react-codemirror", () => ({
  default: (props: { value: string; onChange?: (value: string) => void; "aria-label"?: string }) => (
    <textarea aria-label={props["aria-label"] ?? "Python source"} value={props.value} onChange={(event) => props.onChange?.(event.target.value)} />
  ),
}));

const snapshot = loadSnapshot();

function mockPreviewFetch() {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("api/state")) return new Response("no service", { status: 404 });
    if (url.endsWith("catalog.json")) return new Response(JSON.stringify(snapshot), { status: 200, headers: { "Content-Type": "application/json" } });
    return new Response("not found", { status: 404 });
  }));
}

describe("the page in preview mode", () => {
  beforeEach(() => {
    mockPreviewFetch();
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: vi.fn().mockReturnValue({ matches: false, addListener: vi.fn(), removeListener: vi.fn(), addEventListener: vi.fn(), removeEventListener: vi.fn() }),
    });
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
  });
  afterEach(() => vi.unstubAllGlobals());

  it("says it cannot run CAM, shows the default order, and edits with the keyboard", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByRole("status");
    expect(screen.getByRole("status")).toHaveTextContent(/Preview — no CAM execution/);
    const listbox = await screen.findByRole("listbox", { name: "Step order" });
    const rows = within(listbox).getAllByRole("option");
    expect(rows.length).toBe(snapshot.default_document.nodes.filter((n) => n.scientific).length);
    expect(rows[0]).toHaveTextContent("surface_fluxes_and_emissions");

    // select radiation, move it up with Alt+ArrowUp: the step order changes and the check asks for Experimental
    const labels = () => within(listbox).getAllByRole("option").map((row) => row.getAttribute("aria-label") ?? "");
    const before = labels().findIndex((label) => label.includes("radiation"));
    const radiation = screen.getByTestId("row-cam_run1.radiation");
    radiation.focus();
    await user.keyboard("{Alt>}{ArrowUp}{/Alt}");
    expect(labels().findIndex((label) => label.includes("radiation"))).toBe(before - 1);
    expect(screen.getByRole("tab", { name: /Checks/ })).toHaveTextContent("error");
    await user.click(screen.getByRole("tab", { name: /Checks/ }));
    expect(screen.getByText(/enable Experimental to run it/)).toBeInTheDocument();

    // Undo restores the default; the check is green again
    await user.click(screen.getByRole("button", { name: "Undo" }));
    expect(screen.getByRole("tab", { name: /Checks/ })).toHaveTextContent("valid");
  });

  it("adds a Python process, edits its property, generates code and downloads it", async () => {
    const user = userEvent.setup();
    const created: string[] = [];
    vi.stubGlobal("URL", Object.assign(URL, { createObjectURL: vi.fn(() => "blob:x"), revokeObjectURL: vi.fn() }));
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
      created.push(this.download);
    });
    render(<App />);
    await screen.findByRole("listbox", { name: "Step order" });
    await user.click(screen.getByRole("button", { name: "New Python process" }));
    expect(screen.getByTestId("row-python:notebook_process")).toBeInTheDocument();
    const inspector = screen.getByRole("complementary", { name: "Inspector" });
    expect(within(inspector).getByRole("heading", { level: 2 })).toHaveTextContent("notebook_process");
    await user.click(within(inspector).getByRole("tab", { name: "Python" }));
    await user.type(within(inspector).getByLabelText("new property name"), "rate");
    await user.type(within(inspector).getByLabelText("new property value"), "0.5");
    await user.click(within(inspector).getByRole("button", { name: "Add" }));
    expect(within(inspector).getByLabelText("property rate")).toHaveValue("0.5");

    await user.click(screen.getByRole("button", { name: "Generate" }));
    const code = await screen.findByLabelText("Generated code");
    expect(code).toHaveTextContent("notebook_process = NotebookProcess()");
    expect(code).toHaveTextContent("notebook_process.rate = 0.5");
    await user.click(screen.getByRole("button", { name: "script" }));
    expect(screen.getByLabelText("Generated code")).toHaveTextContent('with fc.Driver(case="PI-atm", nsteps=2) as driver:');
    await user.click(screen.getByRole("button", { name: "Download" }));
    expect(created.some((name) => name.endsWith(".py"))).toBe(true);
    click.mockRestore();
  });

  it("removes and restores a process, and imports a workflow.json", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByRole("listbox", { name: "Step order" });
    await user.click(screen.getByRole("button", { name: "Remove radiation" }));
    expect(screen.queryByTestId("row-cam_run1.radiation")).not.toBeInTheDocument();
    const library = screen.getByRole("complementary", { name: "Process library" });
    await user.click(within(library).getByRole("button", { name: "Restore" }));
    expect(screen.getByTestId("row-cam_run1.radiation")).toBeInTheDocument();

    const payload = { ...snapshot.default_document, nsteps: 9, nodes: snapshot.default_document.nodes.filter((n) => n.id !== "cam_run1.radiation") };
    const file = new File([JSON.stringify(payload)], "workflow.json", { type: "application/json" });
    await user.upload(screen.getByLabelText("Import workflow.json"), file);
    await waitFor(() => expect(screen.queryByTestId("row-cam_run1.radiation")).not.toBeInTheDocument());
    expect(screen.getByLabelText("Steps")).toHaveValue(9);
  });

  it("offers a model only for the kernel the runner covers", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByRole("listbox", { name: "Step order" });
    await user.click(screen.getByTestId("row-cam_run1.cloud_macro_microphysics"));
    const inspector = screen.getByRole("complementary", { name: "Inspector" });
    await user.click(within(inspector).getByRole("tab", { name: "Kernels" }));
    const radios = within(inspector).getAllByRole("radio", { name: /trained network/ });
    const enabled = radios.filter((radio) => !(radio as HTMLInputElement).disabled);
    expect(enabled).toHaveLength(1);
    await user.click(enabled[0]);
    expect(within(inspector).getByLabelText("mmacro_pcond model path")).toBeInTheDocument();
    expect(screen.getByTestId("row-cam_run1.cloud_macro_microphysics")).toHaveTextContent("1 kernel replaced");
  });

  it("switches theme and keeps the choice", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByRole("listbox", { name: "Step order" });
    await user.click(screen.getByRole("button", { name: "Toggle dark mode" }));
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(localStorage.getItem("freecam-ui-theme")).toBe("dark");
  });
});
