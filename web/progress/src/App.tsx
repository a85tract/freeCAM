import React, { useEffect, useMemo, useState } from "react";
import {
  Bar,
  buildTree,
  CAPABILITY_LABELS,
  defaultRunKey,
  executionInventory,
  overviewTiles,
  processBars,
  processGroup,
  processKernelStates,
  searchAll,
  stateLabel,
  TreeNode,
} from "./derive";
import type { KernelRecord, ProcessRecord, Replacement, Snapshot } from "./types";

type View = "physics" | "dynamics" | "control" | "apis" | "unmapped" | "kernels" | "inprogress" | "blocked";

interface Route {
  view: View;
  process?: string;
  kernel?: string;
}

function parseHash(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  if (parts[0] === "process" && parts[1]) return { view: "physics", process: parts[1], kernel: parts[3] };
  if (parts[0] === "kernel" && parts[1]) return { view: "physics", kernel: parts[1] };
  if (["physics", "dynamics", "control", "apis", "unmapped", "kernels", "inprogress", "blocked"].includes(parts[0])) {
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
  const [runKey, setRunKey] = useState<string | null>(null);

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
        setRunKey(defaultRunKey(payload));
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
      <Header snapshot={snapshot} theme={theme} onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
        runKey={runKey} onRunKey={setRunKey} />
      <Overview snapshot={snapshot} runKey={runKey} />
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
              runKey={runKey}
            />
          ) : selectedProcess ? (
            <ProcessDetail snapshot={snapshot} process={selectedProcess} onNavigate={navigate} runKey={runKey} />
          ) : route.view === "inprogress" || route.view === "blocked" ? (
            <WorkBoard snapshot={snapshot} state={route.view === "blocked" ? "blocked" : "in_progress"}
              onNavigate={navigate} />
          ) : route.view === "kernels" ? (
            <KernelIndex snapshot={snapshot} runKey={runKey} onNavigate={navigate} />
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

function Header({ snapshot, theme, onToggleTheme, runKey, onRunKey }: {
  snapshot: Snapshot;
  theme: Theme;
  onToggleTheme: () => void;
  runKey: string | null;
  onRunKey: (key: string | null) => void;
}) {
  const v = snapshot.volatile;
  const runs = snapshot.observation_runs ?? [];
  const selected = runs.find((run) => run.key === runKey);
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
        <div className="run-select">
          {runs.length === 0 ? (
            <span className="chip warn">{snapshot.notes.no_observation_run}</span>
          ) : (
            <label>
              Observation run{" "}
              <select
                aria-label="Observation run"
                value={runKey ?? ""}
                onChange={(event) => onRunKey(event.target.value || null)}
              >
                {runs.map((run) => (
                  <option key={run.key} value={run.key} disabled={!run.validated}>
                    {run.label}
                    {run.validated ? "" : " (not validated)"}
                  </option>
                ))}
              </select>
            </label>
          )}
          {selected && (
            <span className="muted">
              {String(selected.run["steps"] ?? "?")} steps
              {selected.run["final_date"] ? ` to ${selected.run["final_date"]}` : ""} · bit-for-bit:{" "}
              {selected.run["bfb"] ? "yes" : "no"} · image{" "}
              <span className="mono">{String(selected.image["native_library_sha256"] ?? "").slice(0, 10)}</span> ·
              job {String(selected.run["pbs_job"] ?? "?")}
            </span>
          )}
        </div>
        <button type="button" onClick={onToggleTheme}>
          {theme === "dark" ? "Light theme" : "Dark theme"}
        </button>
      </header>
    </>
  );
}

function Overview({ snapshot, runKey }: { snapshot: Snapshot; runKey: string | null }) {
  return (
    <section className="overview" aria-label="Coverage overview">
      <div className="tiles">
        {overviewTiles(snapshot, runKey).map((tile) => (
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

function viewLabels(snapshot: Snapshot): Record<View, string> {
  const items = snapshot.work_items ?? [];
  const inProgress = items.filter((i) => i.state === "in_progress").length;
  const blocked = items.filter((i) => i.state === "blocked").length;
  return {
    physics: "Physics (default order)",
    dynamics: "Dynamics",
    control: "Control, boundary, diagnostics and I/O",
    kernels: "Kernels",
    inprogress: `In progress (${inProgress})`,
    blocked: `Blocked (${blocked})`,
    apis: "Additional callable APIs",
    unmapped: "Unmapped kernels",
  };
}

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
        {(Object.entries(viewLabels(snapshot)) as [View, string][]).map(([view, label]) => (
          <button
            key={view}
            role="tab"
            aria-selected={route.view === view && !route.process && !route.kernel}
            onClick={() => onNavigate({ view })}
          >
            {label}
          </button>
        ))}
      </div>
      {["apis", "unmapped", "kernels", "inprogress", "blocked"].includes(route.view) ? (
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
                    {(process.core_kernels ?? []).length} replaceable · {snapshot.process_membership[process.id]?.kernels.length ?? 0} candidates
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

type KernelFilter = "observed" | "not-observed" | "gaps" | "all";

const FILTER_LABELS: Record<KernelFilter, string> = {
  observed: "Observed",
  "not-observed": "Not observed",
  gaps: "Instrumentation gaps",
  all: "Static candidates",
};

function ProcessDetail({
  snapshot,
  process,
  onNavigate,
  runKey,
}: {
  snapshot: Snapshot;
  process: ProcessRecord;
  onNavigate: (route: Route) => void;
  runKey: string | null;
}) {
  const [flat, setFlat] = useState(false);
  const [filter, setFilter] = useState<KernelFilter | null>(null);
  const membership = snapshot.process_membership[process.id];
  const tree = useMemo(() => (membership ? buildTree(membership) : []), [membership]);
  const coreKernels = process.core_kernels ?? [];   // a cached pre-core snapshot degrades, never crashes
  const ownerClasses = [...new Set(coreKernels.map((c) => c.owner_class).filter(
    (cls): cls is string => Boolean(cls) && cls !== process.python_class))];
  const bars = processBars(snapshot, process.id, runKey);
  const states = runKey ? processKernelStates(snapshot, process.id, runKey) : null;
  const inventory = runKey ? executionInventory(snapshot, process.id, runKey) : null;
  const activeFilter: KernelFilter = filter ?? (runKey ? "observed" : "all");
  const filtered = (want: "observed" | "not-observed-here" | "gap") =>
    (membership?.kernels ?? []).filter((kid) => states?.get(kid) === want);
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
      <CurrentImplementation snapshot={snapshot} pid={process.id} onNavigate={onNavigate} />
      <h3>Replaceable kernels ({coreKernels.length})</h3>
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
      <h3>Execution inventory{runKey ? "" : " (no validated observation run)"}</h3>
      {inventory ? (
        <>
          <dl className="facts">
            <div><dt>Candidate kernels</dt><dd>{inventory.candidates}</dd></div>
            <div><dt>Observed executing here</dt><dd>{inventory.observed}</dd></div>
            <div><dt>Fully covered, not observed here</dt><dd>{inventory.coveredNotObserved}</dd></div>
            <div><dt>Partially covered or uninstrumented</dt><dd>{inventory.gaps}</dd></div>
          </dl>
          <p className="muted">{snapshot.notes.observation}</p>
        </>
      ) : (
        <p className="muted">{snapshot.notes.no_observation_run} The counts below fall back to static
          candidates and say so; they are never presented as execution evidence.</p>
      )}
      <h3>Capability progress {runKey ? "(observed kernels)" : "(static candidates)"}</h3>
      {inventory && inventory.gaps + inventory.coveredNotObserved > 0 && (
        <p className="muted">
          Observation coverage is incomplete; {inventory.gaps} candidates remain uninstrumented or
          unknown here{inventory.coveredNotObserved > 0
            ? ` and ${inventory.coveredNotObserved} covered candidates were not observed in this process`
            : ""}. The bars below cover only the {inventory.observed} observed kernels, not the whole process.
        </p>
      )}
      {bars.map((bar) => (
        <BarRow key={bar.key} bar={bar} />
      ))}
      <h3>Internal kernels by evidence</h3>
      <div className="tabs filter-tabs" role="tablist" aria-label="Kernel filters">
        {(Object.keys(FILTER_LABELS) as KernelFilter[]).map((key) => (
          <button key={key} role="tab" aria-selected={activeFilter === key}
            disabled={!runKey && key !== "all"} onClick={() => setFilter(key)}>
            {FILTER_LABELS[key]}
            {states && key === "observed" && ` (${filtered("observed").length})`}
            {states && key === "not-observed" && ` (${filtered("not-observed-here").length})`}
            {states && key === "gaps" && ` (${filtered("gap").length})`}
            {key === "all" && ` (${membership?.kernels.length ?? 0})`}
          </button>
        ))}
      </div>
      {activeFilter !== "all" && states ? (
        <KernelStateList
          snapshot={snapshot}
          process={process}
          runKey={runKey!}
          kids={filtered(activeFilter === "observed" ? "observed"
            : activeFilter === "not-observed" ? "not-observed-here" : "gap")}
          filter={activeFilter}
          onNavigate={onNavigate}
        />
      ) : null}
      {activeFilter === "all" && (
      <h4 className="subhead">
        All candidate numerical functions (statically reachable, recursive){" "}
        <button type="button" onClick={() => setFlat(!flat)} aria-pressed={flat}>
          {flat ? "Tree view" : "Flat list"}
        </button>
      </h4>
      )}
      {activeFilter !== "all" ? null : !membership || (!membership.kernels.length && !membership.inventoried) ? (
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

function KernelStateList({
  snapshot,
  process,
  runKey,
  kids,
  filter,
  onNavigate,
}: {
  snapshot: Snapshot;
  process: ProcessRecord;
  runKey: string;
  kids: string[];
  filter: KernelFilter;
  onNavigate: (route: Route) => void;
}) {
  if (!kids.length) {
    return <p className="muted">No kernels in this category for the selected run.</p>;
  }
  const here = snapshot.process_observation?.[runKey]?.[process.id] ?? {};
  return (
    <ul className="kernel-list" aria-label={FILTER_LABELS[filter]}>
      {kids.map((kid) => {
        const kernel = snapshot.kernels[kid];
        const observation = kernel?.observation?.[runKey];
        return (
          <li key={kid}>
            <KernelLink snapshot={snapshot} kid={kid} processId={process.id} onNavigate={onNavigate} />
            {filter === "observed" && here[kid] && (
              <span className="muted">
                {" "}{here[kid].calls.toLocaleString()} calls in this process
                {here[kid].first_step >= 0 && ` (steps ${here[kid].first_step}–${here[kid].last_step})`}
                {observation?.count_meaning ? "; lower bound" : ""}
              </span>
            )}
            {filter === "not-observed" && (
              <span className="muted"> fully counted; zero calls in this process in this run</span>
            )}
            {filter === "gaps" && (
              <span className="muted">
                {" "}{observation?.status_reason ?? (observation?.coverage === "partial"
                  ? "partially counted; unobserved paths remain"
                  : "no counting entry")}
              </span>
            )}
          </li>
        );
      })}
    </ul>
  );
}

function CurrentImplementation({
  snapshot,
  pid,
  onNavigate,
}: {
  snapshot: Snapshot;
  pid: string;
  onNavigate: (route: Route) => void;
}) {
  const items = (snapshot.work_items ?? []).filter(
    (item) => item.state !== "closed" && item.target_processes.includes(pid));
  if (!items.length) return null;
  const byClass = new Map<string, typeof items>();
  for (const item of items) {
    const cls = item.owner_class ?? "unassigned";
    byClass.set(cls, [...(byClass.get(cls) ?? []), item]);
  }
  return (
    <>
      <h3>Current implementation</h3>
      <ul className="kernel-list implementation">
        {[...byClass.entries()].map(([cls, rows]) => (
          <li key={cls}>
            <code>{cls.split(".").pop()}</code>
            <ul>
              {rows.map((item) => (
                <li key={item.kernel}>
                  <button className="linkish" onClick={() => onNavigate({ view: "physics", process: pid, kernel: item.kernel })}>
                    <code>{item.kernel}</code>
                  </button>{" "}
                  <span className={`chip ${item.state === "blocked" ? "state-failed" : "info"}`}>
                    {item.state === "blocked" ? "Blocked" : "In progress"}
                  </span>
                  <span className="muted"> · {item.stage} · next: {item.next_gate}
                    {item.blocker ? ` · blocker: ${item.blocker}` : ""}</span>
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ul>
      <p className="muted">{snapshot.notes.development_vs_evidence}</p>
    </>
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
  runKey,
}: {
  snapshot: Snapshot;
  kernel: KernelRecord;
  processId?: string;
  onNavigate: (route: Route) => void;
  runKey: string | null;
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
      <h3>Development</h3>
      {kernel.development.state === "unclaimed" ? (
        <p className="muted"><span className="chip off">Unclaimed</span> no active work item for this kernel.</p>
      ) : (
        <div className="replacement">
          <span className={`chip ${kernel.development.state === "blocked" ? "state-failed" : "info"}`}>
            {kernel.development.state === "blocked" ? "Blocked" : "In progress"}
          </span>{" "}
          stage {kernel.development.stage} · owner {kernel.development.owner} · branch{" "}
          <code>{kernel.development.branch}</code> · {kernel.development.started_at} →{" "}
          {kernel.development.updated_at} · next gate: {kernel.development.next_gate}
          {kernel.development.blocker && <p className="muted">blocker: {kernel.development.blocker}</p>}
          {kernel.development.note && <p className="muted">{kernel.development.note}</p>}
          <p className="muted">{snapshot.notes.development_vs_evidence}</p>
        </div>
      )}
      <h3>Execution observation</h3>
      {(snapshot.observation_runs ?? []).length === 0 ? (
        <p className="muted">{snapshot.notes.no_observation_run}</p>
      ) : (
        (snapshot.observation_runs ?? []).map((run) => {
          const observation = kernel.observation?.[run.key];
          if (!observation) {
            return <p key={run.key} className="muted">{run.label}: no observation data.</p>;
          }
          return (
            <div key={run.key} className="replacement">
              <div>
                {run.label}{run.key === runKey ? " (selected)" : ""}{" "}
                <span className={`chip state-${observation.status === "observed" ? "verified"
                  : observation.status === "not-observed-in-this-run" ? "not-verified" : "not-assessed"}`}>
                  {observation.status === "observed" ? "Observed"
                    : observation.status === "not-observed-in-this-run" ? "Not observed in this run" : "Unknown"}
                </span>
                {!run.validated && <span className="chip warn">run not validated</span>}
              </div>
              {observation.status === "observed" && (
                <ul>
                  {Object.entries(observation.calls_by_context ?? {}).map(([context, calls]) => (
                    <li key={context} className="muted">
                      <code>{context}</code>: {calls.toLocaleString()} calls
                      {["initialization", "finalize", "run-unattributed"].includes(context) &&
                        " (lifecycle; not counted toward any process's progress)"}
                    </li>
                  ))}
                  <li className="muted">
                    steps {observation.first_step}–{observation.last_step};{" "}
                    {observation.ranks_with_calls} ranks called it
                    {observation.count_meaning ? `; ${observation.count_meaning}` : ""}
                    {observation.note ? `; ${observation.note}` : ""}
                  </li>
                </ul>
              )}
              {observation.status !== "observed" && observation.status_reason && (
                <p className="muted">{observation.status_reason}</p>
              )}
            </div>
          );
        })
      )}
      <h3>Scientific evidence</h3>
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

function WorkBoard({
  snapshot,
  state,
  onNavigate,
}: {
  snapshot: Snapshot;
  state: "in_progress" | "blocked";
  onNavigate: (route: Route) => void;
}) {
  const items = (snapshot.work_items ?? []).filter((item) => item.state === state);
  return (
    <section aria-label={state === "blocked" ? "Blocked work items" : "Work in progress"}>
      <h2>{state === "blocked" ? "Blocked" : "In progress"} ({items.length})</h2>
      <p className="muted">{snapshot.notes.development_vs_evidence}</p>
      {items.length === 0 ? (
        <p className="muted">No {state === "blocked" ? "blocked" : "active"} work items.</p>
      ) : (
        <div className="board-scroll">
        <table className="capabilities work-board">
          <thead>
            <tr>
              <th>Process</th><th>Python class</th><th>Kernel</th><th>Stage</th><th>Owner</th>
              <th>Branch</th><th>Started / updated</th><th>Next gate</th><th>Blocker</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.kernel}>
                <td>{item.target_processes.map((pid) => (
                  <button key={pid} className="linkish" onClick={() => onNavigate({ view: "physics", process: pid })}>
                    {pid}
                  </button>
                ))}</td>
                <td><code>{item.owner_class ?? "—"}</code></td>
                <td>
                  <button className="linkish" onClick={() => onNavigate({ view: "kernels", kernel: item.kernel })}>
                    <code>{item.kernel}</code>
                  </button>
                </td>
                <td>{item.stage}</td>
                <td>{item.owner}</td>
                <td><code>{item.branch}</code></td>
                <td>{item.started_at} / {item.updated_at}</td>
                <td>{item.next_gate}</td>
                <td>{item.blocker ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}
    </section>
  );
}

function KernelIndex({
  snapshot,
  runKey,
  onNavigate,
}: {
  snapshot: Snapshot;
  runKey: string | null;
  onNavigate: (route: Route) => void;
}) {
  const [filter, setFilter] = useState("");
  const needle = filter.trim().toLowerCase();
  const rows = Object.values(snapshot.kernels)
    .filter((k) => !needle || k.id.toLowerCase().includes(needle))
    .slice(0, 250);
  return (
    <section aria-label="All candidate kernels">
      <h2>Candidate kernels ({Object.keys(snapshot.kernels).length})</h2>
      <input type="search" aria-label="Filter kernels" placeholder="Filter by module::routine…"
        value={filter} onChange={(event) => setFilter(event.target.value)} />
      <ul className="kernel-list">
        {rows.map((kernel) => (
          <li key={kernel.id}>
            <button className="kernel-link" onClick={() => onNavigate({ view: "kernels", kernel: kernel.id })}>
              <code>{kernel.id}</code>
            </button>
            <span className="chips">
              {kernel.development.state !== "unclaimed" && (
                <span className="chip info">{kernel.development.state === "blocked" ? "blocked" : "in progress"}</span>
              )}
              {runKey && kernel.observation?.[runKey]?.status === "observed" && <span className="chip ok">observed</span>}
              {kernel.tracked && <span className={`chip ${kernel.status === "complete" ? "ok" : "info"}`}>{kernel.status}</span>}
            </span>
          </li>
        ))}
        {Object.keys(snapshot.kernels).length > rows.length && (
          <li className="muted">…filter to narrow the list.</li>
        )}
      </ul>
    </section>
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
