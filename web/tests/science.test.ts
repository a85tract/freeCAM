import { describe, expect, it } from "vitest";

import { texToHtml } from "../src/components/Science";
import { loadSnapshot } from "./helpers";

const snapshot = loadSnapshot();
const described = snapshot.entries.filter((entry) => entry.science);

function inlineTex(text: string | null): string[] {
  return (text ?? "").match(/\$[^$]+\$/g)?.map((part) => part.slice(1, -1)) ?? [];
}

describe("the science record the About tab shows", () => {
  it("covers every scientific process of the default workflow", () => {
    const missing = snapshot.default_nodes.filter((node) => node.scientific && !described.some((entry) => entry.id === node.id));
    expect(missing.map((node) => node.id)).toEqual([]);
  });

  it("has formulas KaTeX parses, display and inline", () => {
    let count = 0;
    for (const entry of described) {
      const science = entry.science!;
      for (const equation of science.equations) {
        expect(() => texToHtml(equation.tex, true, true), `${entry.id}: ${equation.tex}`).not.toThrow();
        count += 1;
      }
      const prose = [science.summary, science.configuration, ...science.equations.map((equation) => equation.caption)];
      for (const tex of prose.flatMap(inlineTex)) {
        expect(() => texToHtml(tex, false, true), `${entry.id}: ${tex}`).not.toThrow();
        count += 1;
      }
    }
    expect(count).toBeGreaterThan(50);
  });

  it("links every reference", () => {
    for (const entry of described) {
      for (const reference of entry.science!.references) {
        expect(reference.url, `${entry.id}: ${reference.key}`).toMatch(/^https:\/\//);
        if (reference.doi) expect(reference.url).toBe(`https://doi.org/${reference.doi}`);
      }
    }
  });

  it("names every phase of the step", () => {
    for (const node of snapshot.default_nodes) expect(snapshot.phases?.[node.phase]?.label, node.phase).toBeTruthy();
  });
});
