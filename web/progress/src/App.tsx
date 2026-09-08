import React, { useEffect, useMemo, useState } from "react";
import {
  Bar,
  buildTree,
  CAPABILITY_LABELS,
  overviewTiles,
  processBars,
  processGroup,
  searchAll,
  stateLabel,
  TreeNode,
} from "./derive";
import type { KernelRecord, ProcessRecord, Replacement, Snapshot } from "./types";

type View = "physics" | "dynamics" | "control" | "apis" | "unmapped";

interface Route {
  view: View;
  process?: string;
  kernel?: string;
}

function parseHash(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  if (parts[0] === "process" && parts[1]) return { view: "physics", process: parts[1], kernel: parts[3] };
  if (parts[0] === "kernel" && parts[1]) return { view: "physics", kernel: parts[1] };
  if (["physics", "dynamics", "control", "apis", "unmapped"].includes(parts[0])) {
    return { view: parts[0] as View };
  }
  return { view: "physics" };
}

function routeHash(route: Route): string {
  if (route.process) {
    return route.kernel
      ? `#/process/${encodeURIComponent(route.process)}/kernel/${encodeURIComponent(route.kernel)}`
      : `#/process/${encodeURIComponent(route.process)}`;
  }
  if (route.kernel) return `#/kernel/${encodeURIComponent(route.kernel)}`;
  return `#/${route.view}`;
}

type Theme = "light" | "dark";

function initialTheme(): Theme {
  try {
    const stored = localStorage.getItem("freecam-ui-theme");
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    /* storage may be unavailable */
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function App({ load }: { load?: () => Promise<Snapshot> }) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  const [query, setQuery] = useState("");
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("freecam-ui-theme", theme);
    } catch {
      /* storage may be unavailable */
    }
  }, [theme]);

  useEffect(() => {
    const onHash = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    const loader =
      load ??
      (async () => {
        const response = await fetch("./progress.json");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return (await response.json()) as Snapshot;
      });
    loader()
      .then((payload) => {
        if (payload?.schema_version !== 1 || !payload.kernels || !payload.processes) {
          throw new Error("the snapshot has an unsupported schema");
        }
        setSnapshot(payload);
      })
      .catch((cause) => setError(String(cause)));
  }, [load]);

  const navigate = (next: Route) => {
    window.location.hash = routeHash(next);
    setRoute(next);
  };

  if (error) {
    return (
      <div className="page">
        <main className="error" role="alert">
          <h1>freeCAM implementation progress</h1>
          <p>
            The progress snapshot could not be loaded: <code>{error}</code>. The page renders one exported
            file (<code>progress.json</code>) and shows nothing without it.
          </p>
          <p>
            <a href="../">Back to the Workflow Builder</a>
          </p>
        </main>
      </div>
    );
  }
  if (!snapshot) {
    return (
      <div className="page">
        <main className="loading">Loading the progress snapshot…</main>
      </div>
    );
  }

  const selectedProcess = route.process ? snapshot.processes.find((p) => p.id === route.process) : undefined;
  const selectedKernel = route.kernel ? snapshot.kernels[route.kernel] : undefined;
  const hits = searchAll(snapshot, query);

  return (
    <div className="page">
      <Header snapshot={snapshot} theme={theme} onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")} />
      <Overview snapshot={snapshot} />
      <div className="columns">
        <nav className="browser" aria-label="Process browser">
          <input
            type="search"
            placeholder="Search processes, kernels, classes…"
            aria-label="Search processes, kernels, classes"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          {query ? (
            <SearchResults hits={hits} onOpen={(hit) => {
              setQuery("");
              if (hit.kind === "process") navigate({ view: route.view, process: hit.id });
              else if (hit.kind === "kernel") navigate({ view: route.view, kernel: hit.id });
              else navigate({ view: "apis" });
            }} />
          ) : (
            <ProcessLists snapshot={snapshot} route={route} onNavigate={navigate} />
          )}
        </nav>
        <main className="detail">
          {selectedKernel ? (
            <KernelDetail
              snapshot={snapshot}
              kernel={selectedKernel}
              processId={selectedProcess?.id}
              onNavigate={navigate}
            />
          ) : selectedProcess ? (
            <ProcessDetail snapshot={snapshot} process={selectedProcess} onNavigate={navigate} />
          ) : route.view === "apis" ? (
            <AdditionalApis snapshot={snapshot} />
          ) : route.view === "unmapped" ? (
            <Unmapped snapshot={snapshot} onNavigate={navigate} />
          ) : (
            <section>
              <h2>Select a process</h2>
              <p className="muted">{snapshot.notes.process_vs_kernel}</p>
              <p className="muted">{snapshot.notes.class_delegation}</p>
            </section>
          )}
        </main>
      </div>
      <footer>
        <span>{snapshot.notes.snapshot}</span>
        <a href="../">Workflow Builder</a>
      </footer>
    </div>
  );
}

function Header({ snapshot, theme, onToggleTheme }: { snapshot: Snapshot; theme: Theme; onToggleTheme: () => void }) {
  const v = snapshot.volatile;
  return (
    <>
      <div className="banner" role="note">
        Development snapshot · {v.branch ?? "unknown branch"} · {(v.commit ?? "unknown").slice(0, 9)} —{" "}
        {snapshot.notes.banner}
      </div>
      <header className="masthead">
        <h1>freeCAM implementation progress</h1>
        <dl className="provenance">
          <div>
            <dt>Generated</dt>
            <dd>{v.generated_at ?? "unknown"}</dd>
          </div>
          <div>
            <dt>Snapshot</dt>
            <dd className="mono">{snapshot.content_hash.slice(0, 12)}</dd>
          </div>
          <div>
            <dt>Case</dt>
            <dd>PI-atm (ne16, CAM5, SE, 512 ranks)</dd>
          </div>
          <div>
            <dt>Source revision</dt>
            <dd className="mono">{(snapshot.cam_source_revision ?? "unknown").slice(0, 9)}</dd>
          </div>
        </dl>
        <button type="button" onClick={onToggleTheme}>
          {theme === "dark" ? "Light theme" : "Dark theme"}
        </button>
      </header>
    </>
  );
}

function Overview({ snapshot }: { snapshot: Snapshot }) {
  return (
    <section className="overview" aria-label="Coverage overview">
      <div className="tiles">
        {overviewTiles(snapshot).map((tile) => (
          <div className="tile" key={tile.label}>
            <div className="tile-value">{tile.value}</div>
            <div className="tile-label">{tile.label}</div>
            <div className="tile-note">{tile.note}</div>
          </div>
        ))}
      </div>
      <p className="muted">{snapshot.notes.process_vs_kernel} {snapshot.notes.class_delegation}</p>
      <p className="muted">{snapshot.notes.candidates} {snapshot.notes.shared_kernels}</p>
    </section>
  );
}

const VIEW_LABELS: Record<View, string> = {
  physics: "Physics (default order)",
  dynamics: "Dynamics",
  control: "Control, boundary, diagnostics and I/O",
  apis: "Additional callable APIs",
  unmapped: "Unmapped kernels",
};

function ProcessLists({
  snapshot,
  route,
  onNavigate,
}: {
  snapshot: Snapshot;
  route: Route;
  onNavigate: (route: Route) => void;
}) {
  const groups = useMemo(() => {
    const byGroup: Record<string, ProcessRecord[]> = { physics: [], dynamics: [], control: [] };
    for (const process of snapshot.processes) byGroup[processGroup(process.classification)].push(process);
    return byGroup;
  }, [snapshot]);
  return (
    <>
      <div className="tabs" role="tablist" aria-label="Views">
        {(Object.keys(VIEW_LABELS) as View[]).map((view) => (
          <button
            key={view}
            role="tab"
            aria-selected={route.view === view && !route.process && !route.kernel}
            onClick={() => onNavigate({ view })}
          >
            {VIEW_LABELS[view]}
          </button>
        ))}
      </div>
      {route.view === "apis" || route.view === "unmapped" ? (
        <p className="muted">Shown in the main panel.</p>
      ) : (
        <ul className="process-list">
          {groups[route.view].map((process) => (
            <li key={process.id}>
              <button
                className={route.process === process.id ? "selected" : ""}
                onClick={() => onNavigate({ view: route.view, process: process.id })}
              >
                <span className="name">{process.display_name}</span>
                <span className="chips">
                  {!process.enabled && <span className="chip off">disabled</span>}
                  {process.activity === "inert" && <span className="chip off">inactive here</span>}
                  {process.class_kind === "dedicated" ? (
                    <span className="chip ok">class</span>
                  ) : (
                    <span className="chip">generic</span>
                  )}
                  <span className="chip">
                    {(process.core_kernels ?? []).length} core · {snapshot.process_membership[process.id]?.kernels.length ?? 0} candidates
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

function SearchResults({
  hits,
  onOpen,
}: {
  hits: ReturnType<typeof searchAll>;
  onOpen: (hit: ReturnType<typeof searchAll>[number]) => void;
}) {
  if (!hits.length) return <p className="muted">No matches.</p>;
  return (
    <ul className="process-list" aria-label="Search results">
      {hits.map((hit) => (
        <li key={`${hit.kind}:${hit.id}`}>
          <button onClick={() => onOpen(hit)}>
            <span className="name">{hit.title}</span>
            <span className="chips">
              <span className="chip">{hit.kind}</span>
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}

function BarRow({ bar }: { bar: Bar }) {
  const width = bar.total > 0 ? Math.round((100 * bar.done) / bar.total) : 0;
  return (
    <div className="bar-row">
      <div className="bar-head">
        <span>{bar.label}</span>
        <span className="mono">
          {bar.inventoried ? `${bar.done} / ${bar.total}` : "Not inventoried"}
        </span>
      </div>
      <div className="bar-track" role="img" aria-label={`${bar.label}: ${bar.done} of ${bar.total}`}>
        <div className="bar-fill" style={{ width: `${bar.inventoried ? width : 0}%` }} />
      </div>
      <div className="bar-note">{bar.inventoried ? bar.denominator : "No call-tree inventory covers this process; absence of data is not completion."}</div>
    </div>
  );
}

function ProcessDetail({
  snapshot,
  process,
  onNavigate,
}: {
  snapshot: Snapshot;
  process: ProcessRecord;
  onNavigate: (route: Route) => void;
}) {
  const [flat, setFlat] = useState(false);
  const membership = snapshot.process_membership[process.id];
  const tree = useMemo(() => (membership ? buildTree(membership) : []), [membership]);
  const coreKernels = process.core_kernels ?? [];   // a cached pre-core snapshot degrades, never crashes
  const ownerClasses = [...new Set(coreKernels.map((c) => c.owner_class).filter(
    (cls): cls is string => Boolean(cls) && cls !== process.python_class))];
  const bars = processBars(snapshot, process.id);
  return (
    <section aria-label={`Process ${process.display_name}`}>
      <h2>{process.display_name}</h2>
      <dl className="facts">
        <div><dt>Operation</dt><dd className="mono">{process.operation ?? process.id}</dd></div>
        <div><dt>Plan action</dt><dd className="mono">{process.id}</dd></div>
        <div><dt>Default workflow</dt><dd>{process.enabled ? "enabled" : "disabled"}{process.alternate_of.length > 0 && ` (alternate of ${process.alternate_of.join(", ")})`}</dd></div>
        <div><dt>Python API</dt><dd>{process.python_api}</dd></div>
        <div>
          <dt>Python class</dt>
          <dd>{process.python_class ? <code>{process.python_class}</code> : "generic interface (plan control; no dedicated class)"}</dd>
        </div>
        {process.activity_basis && <div><dt>Activity</dt><dd>{process.activity}: {process.activity_basis}</dd></div>}
        {ownerClasses.length > 0 && (
          <div>
            <dt>Kernel owner classes</dt>
            <dd>
              {ownerClasses.map((cls) => <code key={cls}>{cls}</code>).reduce<React.ReactNode[]>(
                (out, node, i) => (i ? [...out, ", ", node] : [node]), [])}
              <span className="muted"> — composed into {process.display_name}; each owns its core kernel's contract</span>
            </dd>
          </div>
        )}
        {process.description && <div><dt>Description</dt><dd>{process.description}</dd></div>}
      </dl>
      <h3>Core kernels exposed for replacement ({coreKernels.length})</h3>
      {coreKernels.length === 0 ? (
        <p className="muted">This process's class exposes no replaceable kernel yet.</p>
      ) : (
        <ul className="kernel-list">
          {coreKernels.map((core) => (
            <li key={core.routine}>
              {core.id ? (
                <KernelLink snapshot={snapshot} kid={core.id} processId={process.id} onNavigate={onNavigate} />
              ) : (
                <code>{core.routine}</code>
              )}
              {core.owner_class && <span className="muted"> owned by <code>{core.owner_class}</code></span>}
            </li>
          ))}
        </ul>
      )}
      <p className="muted">{snapshot.notes.core_vs_candidates}</p>
      <h3>Candidate kernel coverage</h3>
      {bars.map((bar) => (
        <BarRow key={bar.key} bar={bar} />
      ))}
      <h3>
        All candidate numerical functions (statically reachable, recursive){" "}
        <button type="button" onClick={() => setFlat(!flat)} aria-pressed={flat}>
          {flat ? "Tree view" : "Flat list"}
        </button>
      </h3>
      {!membership || (!membership.kernels.length && !membership.inventoried) ? (
        <p className="muted">Not inventoried: the call-tree inventory does not cover this process. That is a
          gap in the inventory, not proof the process has no internal kernels.</p>
      ) : membership.kernels.length === 0 ? (
        <p className="muted">The inventory attributes no numerical kernel candidates to this process.</p>
      ) : flat ? (
        <ul className="kernel-list">
          {membership.kernels.map((kid) => (
            <li key={kid}>
              <KernelLink snapshot={snapshot} kid={kid} processId={process.id} onNavigate={onNavigate} />
            </li>
          ))}
        </ul>
      ) : (
        <ul className="tree" aria-label="Internal numerical kernels (calls)">
          {tree.map((node, index) => (
            <TreeItem key={`${node.kernel}:${index}`} snapshot={snapshot} node={node} processId={process.id} onNavigate={onNavigate} depth={0} />
          ))}
        </ul>
      )}
      <p className="muted">Tree relationships are “Calls.” {snapshot.notes.shared_kernels}</p>
    </section>
  );
}

function KernelLink({
  snapshot,
  kid,
  processId,
  onNavigate,
}: {
  snapshot: Snapshot;
  kid: string;
  processId: string;
  onNavigate: (route: Route) => void;
}) {
  const kernel = snapshot.kernels[kid];
  if (!kernel) return <code>{kid}</code>;
  return (
    <button className="kernel-link" onClick={() => onNavigate({ view: "physics", process: processId, kernel: kid })}>
      <code>{kernel.routine}</code>
      <span className="chips">
        {kernel.tracked && <span className={`chip ${kernel.status === "complete" ? "ok" : "info"}`}>{kernel.status}</span>}
        {kernel.processes.length > 1 && <span className="chip">shared ×{kernel.processes.length}</span>}
      </span>
    </button>
  );
}

function TreeItem({
  snapshot,
  node,
  processId,
  onNavigate,
  depth,
}: {
  snapshot: Snapshot;
  node: TreeNode;
  processId: string;
  onNavigate: (route: Route) => void;
  depth: number;
}) {
  const [open, setOpen] = useState(depth < 1);
  return (
    <li>
      <div className="tree-row">
        {node.children.length > 0 ? (
          <button className="expander" aria-expanded={open} onClick={() => setOpen(!open)}>
            {open ? "▾" : "▸"}
          </button>
        ) : (
          <span className="expander leaf" aria-hidden="true">·</span>
        )}
        <KernelLink snapshot={snapshot} kid={node.kernel} processId={processId} onNavigate={onNavigate} />
        {node.via.length > 0 && (
          <span className="via muted">via {node.via.map((v) => v.split("::").pop()).join(" → ")}</span>
        )}
        {node.cycle && <span className="chip off">recursive; not expanded again</span>}
      </div>
      {open && node.children.length > 0 && (
        <ul className="tree">
          {node.children.map((child, index) => (
            <TreeItem
              key={`${child.kernel}:${index}`}
              snapshot={snapshot}
              node={child}
              processId={processId}
              onNavigate={onNavigate}
              depth={depth + 1}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

function EvidenceEntry({ snapshot, name }: { snapshot: Snapshot; name: string }) {
  const record = snapshot.evidence[name];
  if (!record) {
    return (
      <li>
        <code>{name}</code> <span className="chip off">record not in this snapshot</span>
      </li>
    );
  }
  return (
    <li>
      <details>
        <summary>
          <code>{name}</code>
        </summary>
        <dl className="facts">
          {Object.entries(record)
            .filter(([key]) => key !== "record")
            .map(([key, value]) => (
              <div key={key}>
                <dt>{key}</dt>
                <dd className="mono">{typeof value === "string" ? value : JSON.stringify(value)}</dd>
              </div>
            ))}
        </dl>
      </details>
    </li>
  );
}

function KernelDetail({
  snapshot,
  kernel,
  processId,
  onNavigate,
}: {
  snapshot: Snapshot;
  kernel: KernelRecord;
  processId?: string;
  onNavigate: (route: Route) => void;
}) {
  const replacements = snapshot.replacements.filter(
    (row) => row.kernel_routine === kernel.routine && kernel.processes.includes(row.process)
  );
  const evidenceNames = new Set<string>();
  for (const capability of Object.values(kernel.capabilities)) {
    for (const name of capability.evidence ?? []) evidenceNames.add(name);
  }
  for (const row of replacements) {
    for (const gate of row.gates) {
      evidenceNames.add(gate.record);
      if (gate.bfb_record) evidenceNames.add(gate.bfb_record);
    }
  }
  return (
    <section aria-label={`Kernel ${kernel.id}`}>
      {processId && (
        <button className="back" onClick={() => onNavigate({ view: "physics", process: processId })}>
          ← back to process
        </button>
      )}
      <h2>
        <code>{kernel.id}</code>
      </h2>
      <dl className="facts">
        <div><dt>Shape</dt><dd>{kernel.kind ?? "unknown"}{kernel.host ? ` (internal to ${kernel.host})` : ""}{kernel.public ? "" : ", private"}</dd></div>
        <div>
          <dt>Source</dt>
          <dd className="mono">
            {kernel.source.file ?? "unknown"}
            {kernel.source.line_start != null && `:${kernel.source.line_start}–${kernel.source.line_end}`}
          </dd>
        </div>
        <div>
          <dt>Processes</dt>
          <dd>
            {kernel.processes.length === 0 && "none mapped"}
            {kernel.processes.map((pid) => (
              <button key={pid} className="linkish" onClick={() => onNavigate({ view: "physics", process: pid })}>
                {pid}
              </button>
            ))}
          </dd>
        </div>
        <div>
          <dt>Called by</dt>
          <dd className="mono">{kernel.callers.length ? kernel.callers.join(", ") : "no recorded caller"}</dd>
        </div>
        {kernel.owner_class && <div><dt>Owning class</dt><dd><code>{kernel.owner_class}</code></dd></div>}
        {kernel.module_state.length > 0 && (
          <div><dt>Hidden state</dt><dd>module state snapshotted: {kernel.module_state.join(", ")}</dd></div>
        )}
        {kernel.redirect.classification && (
          <div>
            <dt>Symbol redirection</dt>
            <dd>
              {kernel.redirect.classification}
              {kernel.redirect.blocker ? ` — ${kernel.redirect.blocker}` : " (potentially reachable by a hook; not itself an implemented hook)"}
            </dd>
          </div>
        )}
        {kernel.adapter_hint?.blockers?.length ? (
          <div><dt>Known blockers</dt><dd>{kernel.adapter_hint.blockers.join(", ")}</dd></div>
        ) : null}
        {kernel.note && <div><dt>Note</dt><dd>{kernel.note}</dd></div>}
      </dl>
      <h3>Capabilities</h3>
      <table className="capabilities">
        <tbody>
          {Object.entries(CAPABILITY_LABELS).map(([key, label]) => {
            const capability = kernel.capabilities[key];
            if (!capability) return null;
            return (
              <tr key={key}>
                <th scope="row">{label}</th>
                <td>
                  <span className={`chip state-${capability.state}`}>{stateLabel(capability.state)}</span>
                </td>
                <td className="muted">
                  {capability.explanation ?? snapshot.capability_explanations[key]}
                  {capability.blockers?.length ? ` Blockers: ${capability.blockers.join(", ")}.` : ""}
                  {capability.contexts?.length ? ` Scope: ${capability.contexts.join(", ")}.` : ""}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {replacements.length > 0 && (
        <>
          <h3>Replacement, by process</h3>
          {replacements.map((row) => (
            <ReplacementRow key={row.process} row={row} />
          ))}
          <p className="muted">{snapshot.notes.replacement_scope} {snapshot.notes.bfb_meaning}</p>
        </>
      )}
      <h3>Evidence</h3>
      {evidenceNames.size === 0 ? (
        <p className="muted">No validation record names this kernel yet.</p>
      ) : (
        <ul className="evidence">
          {[...evidenceNames].sort().map((name) => (
            <EvidenceEntry key={name} snapshot={snapshot} name={name} />
          ))}
        </ul>
      )}
      {kernel.failures.length > 0 && (
        <details className="failures">
          <summary>Historical failed experiments ({kernel.failures.length})</summary>
          <ul className="evidence">
            {kernel.failures.map((name) => (
              <EvidenceEntry key={name} snapshot={snapshot} name={name} />
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

function ReplacementRow({ row }: { row: Replacement }) {
  return (
    <div className="replacement">
      <div>
        <code>{row.process}</code> — mechanism: {row.mechanism}
        {row.within && (
          <span className="muted"> (inside the compiled {row.within})</span>
        )}{" "}
        <span className={`chip state-${row.state}`}>{stateLabel(row.state)}</span>
        {row.requires_development_image && (
          <span className="chip warn">requires an experimental image; not in the default build</span>
        )}
      </div>
      <ul>
        {row.gates.map((gate, index) => (
          <li key={index} className="muted">
            <code>{gate.record}</code>: bit-for-bit {gate.bfb ? "yes" : "no"}
            {gate.replacement_calls != null
              ? `, ${gate.replacement_calls} replacement calls in this process`
              : gate.limitation
                ? ` — ${gate.limitation}`
                : ""}
          </li>
        ))}
        {row.historical_failures.map((name) => (
          <li key={name} className="muted">
            historical failure: <code>{name}</code>
          </li>
        ))}
      </ul>
      {row.note && <p className="muted">{row.note}</p>}
    </div>
  );
}

function AdditionalApis({ snapshot }: { snapshot: Snapshot }) {
  const [filter, setFilter] = useState("");
  const needle = filter.trim().toLowerCase();
  const rows = snapshot.additional_apis.filter(
    (api) => !needle || (api.qualified_name ?? api.id).toLowerCase().includes(needle)
  );
  return (
    <section aria-label="Additional callable APIs">
      <h2>Additional callable APIs outside the default workflow</h2>
      <p className="muted">
        The runtime catalog lists {snapshot.additional_apis.length} discoverable procedures. Each entry
        records why it is or is not independently runnable today; none of this is part of the default step.
      </p>
      <input
        type="search"
        aria-label="Filter catalog"
        placeholder="Filter by name…"
        value={filter}
        onChange={(event) => setFilter(event.target.value)}
      />
      <ul className="api-list">
        {rows.slice(0, 200).map((api) => (
          <li key={api.id}>
            <code>{api.qualified_name ?? api.id}</code>
            <span className="muted"> {api.reason ?? "independently runnable"}</span>
          </li>
        ))}
        {rows.length > 200 && <li className="muted">…and {rows.length - 200} more; refine the filter.</li>}
      </ul>
    </section>
  );
}

function Unmapped({ snapshot, onNavigate }: { snapshot: Snapshot; onNavigate: (route: Route) => void }) {
  return (
    <section aria-label="Unmapped kernels">
      <h2>Candidates without a process mapping</h2>
      <p className="muted">
        These kernels are statically reachable in this configuration, but the inventory attributes them to
        no plan action. They stay visible here rather than disappearing from totals.
      </p>
      <ul className="kernel-list">
        {snapshot.unmapped_kernels.map((kid) => (
          <li key={kid}>
            <button className="kernel-link" onClick={() => onNavigate({ view: "unmapped", kernel: kid })}>
              <code>{kid}</code>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
