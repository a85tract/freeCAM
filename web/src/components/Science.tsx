import katex from "katex";
import "katex/dist/katex.min.css";

import type { PhaseInfo, ProcessScience } from "../model/types";

/**
 * LaTeX to KaTeX's HTML, typeset with its own fonts so a formula looks the
 * same in every browser (native MathML depends on the system's math fonts and
 * can drop accents such as the dot of a time derivative), with MathML beside
 * it for screen readers.  KaTeX escapes its input and, without `trust`, emits
 * no links, so the markup is safe to insert.
 */
export function texToHtml(tex: string, display: boolean, throwOnError = false): string {
  return katex.renderToString(tex, { displayMode: display, throwOnError, strict: "ignore" });
}

export function Tex({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = texToHtml(tex, display);
  return display
    ? <div className="equation" dangerouslySetInnerHTML={{ __html: html }} />
    : <span dangerouslySetInnerHTML={{ __html: html }} />;
}

/** Prose from the science record: `$...$` is inline LaTeX, backticks are code. */
export function Prose({ text }: { text: string }) {
  const parts = text.split(/(\$[^$]+\$|`[^`]+`)/);
  return (
    <>
      {parts.map((part, index) => {
        if (part.length > 2 && part.startsWith("$") && part.endsWith("$")) return <Tex key={index} tex={part.slice(1, -1)} />;
        if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) return <code key={index}>{part.slice(1, -1)}</code>;
        return part;
      })}
    </>
  );
}

/** The phase's informative name; the step plan's own name where the snapshot has none. */
export function phaseLabel(phases: Record<string, PhaseInfo> | undefined, phase: string): string {
  return phases?.[phase]?.label ?? phase;
}

/**
 * The About tab's science: what the process is, how it is set up in this
 * case, its equations and its literature.  `inherited` names the stage whose
 * description stands in for a catalogued sub-process without its own.
 */
export function ScienceSection({ science, inherited }: { science: ProcessScience; inherited?: string | null }) {
  return (
    <section className="science" aria-label="Science">
      <h3 className="science-title">{science.title}</h3>
      {inherited && <p className="muted">This process is part of the stage {inherited}; what follows describes that stage.</p>}
      {!science.active && <p><span className="badge warn">no effect in this case</span></p>}
      {science.summary && <p><Prose text={science.summary} /></p>}
      {science.configuration && (
        <>
          <h4>In this configuration</h4>
          <p><Prose text={science.configuration} /></p>
        </>
      )}
      {science.equations.length > 0 && (
        <>
          <h4>Equations</h4>
          {science.equations.map((equation, index) => (
            <figure key={index} className="equation-block">
              <Tex tex={equation.tex} display />
              {equation.caption && <figcaption className="muted"><Prose text={equation.caption} /></figcaption>}
            </figure>
          ))}
        </>
      )}
      {science.references.length > 0 && (
        <>
          <h4>References</h4>
          <ol className="references">
            {science.references.map((reference) => (
              <li key={reference.key}>
                {reference.citation}{" "}
                {reference.url && (
                  <a href={reference.url} target="_blank" rel="noreferrer">{reference.doi ? `doi:${reference.doi}` : "link"}</a>
                )}
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  );
}
