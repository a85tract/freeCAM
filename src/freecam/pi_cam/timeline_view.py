"""Read a timeline directory (freecam.pi_cam.timeline) and show it.

``freecam timeline DIR`` serves the viewer on loopback, with a token in the
address, and re-reads the directory as a running model appends to it;
``freecam timeline DIR --html FILE`` writes one self-contained page instead
(the overview, the globe for every action, and the timelines of a few steps).

The viewer answers three questions, one view each:

* where the time goes: actions x steps, coloured by the slowest rank's time
  (or the mean, the imbalance, or the time spent waiting in collectives);
* who is slow and who waits, in one step: ranks x time within the step;
* where on Earth: the chosen action's time on each rank, painted on the
  columns that rank computes.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from .timeline import KIND_ACTION, KIND_PHASE, KIND_STEP, KIND_WAIT, RECORD_DTYPE, STEP_NAME

MS = 1e3


def _round(values: np.ndarray, digits: int = 3) -> list:
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


class TimelineData:
    """Every rank's records of one timeline directory, and the aggregates the viewer draws."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        manifest = self.directory / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"{self.directory} holds no timeline (no manifest.json)")
        self._signature: tuple = ()
        self.reload()

    # -- loading -------------------------------------------------------------
    def _current_signature(self) -> tuple:
        ranks = self.directory / "ranks"
        files = sorted(ranks.glob("rank-*.bin")) if ranks.is_dir() else []
        manifest = self.directory / "manifest.json"
        return (manifest.stat().st_mtime_ns, tuple(f.stat().st_size for f in files))

    def changed(self) -> bool:
        return self._current_signature() != self._signature

    def reload(self) -> None:
        self._signature = self._current_signature()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.nranks = int(self.manifest["ranks"])
        names: list[str] = []
        index: dict[str, int] = {}
        parts = []
        for rank in range(self.nranks):
            path = self.directory / "ranks" / f"rank-{rank:04d}.bin"
            names_path = self.directory / "ranks" / f"rank-{rank:04d}.names.json"
            if not path.is_file() or not names_path.is_file():
                continue
            count = path.stat().st_size // RECORD_DTYPE.itemsize        # a record being appended is left out
            records = np.fromfile(path, dtype=RECORD_DTYPE, count=count)
            local = json.loads(names_path.read_text())
            mapping = np.empty(len(local), dtype=np.int32)
            for i, name in enumerate(local):
                if name not in index:
                    index[name] = len(names)
                    names.append(name)
                mapping[i] = index[name]
            keep = records["action"] < len(local)                      # names written after these records
            records = records[keep]
            parts.append((rank, records, mapping))
        self.names = names
        n = sum(len(r) for _, r, _ in parts)
        self.rank = np.empty(n, dtype=np.int32)
        self.step = np.empty(n, dtype=np.int32)
        self.action = np.empty(n, dtype=np.int32)
        self.kind = np.empty(n, dtype=np.int8)
        self.t0 = np.empty(n, dtype=np.float64)
        self.t1 = np.empty(n, dtype=np.float64)
        at = 0
        for rank, records, mapping in parts:
            k = len(records)
            self.rank[at:at + k] = rank
            self.step[at:at + k] = records["step"]
            self.action[at:at + k] = mapping[records["action"]]
            self.kind[at:at + k] = records["kind"]
            self.t0[at:at + k] = records["t0"]
            self.t1[at:at + k] = records["t1"]
            at += k
        self.ranks_present = sorted({rank for rank, _, _ in parts})
        self._columns = None
        self._aggregate()

    def _aggregate(self) -> None:
        step_mask = self.kind == KIND_STEP
        steps_all = self.step[step_mask]
        # a step is shown once every rank present has written it
        if steps_all.size:
            counts = np.bincount(steps_all - steps_all.min(), minlength=1)
            complete = np.flatnonzero(counts >= max(1, len(self.ranks_present))) + steps_all.min()
        else:
            complete = np.zeros(0, dtype=np.int64)
        self.steps = complete.astype(np.int64)
        step_pos = {int(s): i for i, s in enumerate(self.steps)}
        self._step_pos = step_pos
        nsteps = len(self.steps)
        # the actions, in the order they run in the first complete step on the lowest rank
        action_mask = self.kind == KIND_ACTION
        order: list[int] = []
        if nsteps:
            first = (self.step == self.steps[0]) & action_mask & (self.rank == (self.ranks_present[0] if self.ranks_present else 0))
            order = [int(a) for a in self.action[first][np.argsort(self.t0[first], kind="stable")]]
            seen = set()
            order = [a for a in order if not (a in seen or seen.add(a))]
        others = sorted(set(int(a) for a in np.unique(self.action[action_mask])) - set(order))
        self.action_order = order + others
        pos = {a: i for i, a in enumerate(self.action_order)}
        nact = len(self.action_order)
        ranks = np.asarray(self.ranks_present, dtype=np.int64)
        rank_pos = np.full(self.nranks, -1, dtype=np.int64)
        rank_pos[ranks] = np.arange(len(ranks))
        self._rank_pos = rank_pos
        shape = (len(ranks), max(nact, 1), max(nsteps, 1))
        self.cube = np.zeros(shape, dtype=np.float32)          # seconds of each action, rank x action x step
        self.wait_cube = np.zeros(shape, dtype=np.float32)     # of which waiting in collectives
        if nact and nsteps:
            lookup_step = np.full(int(self.step.max()) + 2 if self.step.size else 1, -1, dtype=np.int64)
            for s, i in step_pos.items():
                lookup_step[s] = i
            lookup_action = np.full(len(self.names) + 1, -1, dtype=np.int64)
            for a, i in pos.items():
                lookup_action[a] = i
            for kind, target in ((KIND_ACTION, self.cube), (KIND_WAIT, self.wait_cube)):
                m = (self.kind == kind) & (self.step >= 0)
                si = lookup_step[self.step[m]]
                ai = lookup_action[self.action[m]]
                ri = rank_pos[self.rank[m]]
                ok = (si >= 0) & (ai >= 0) & (ri >= 0)
                np.add.at(target, (ri[ok], ai[ok], si[ok]), (self.t1[m] - self.t0[m])[ok].astype(np.float32))
        # each step's duration: the slowest rank's; and its span, first start to last end
        self.step_max = np.zeros(nsteps)
        self.step_mean = np.zeros(nsteps)
        self.step_start = np.zeros(nsteps)
        if nsteps:
            m = step_mask & np.isin(self.step, self.steps)
            si = np.array([step_pos[int(s)] for s in self.step[m]], dtype=np.int64)
            dur = self.t1[m] - self.t0[m]
            np.maximum.at(self.step_max, si, dur)
            np.add.at(self.step_mean, si, dur)
            self.step_mean /= max(1, len(self.ranks_present))
            self.step_start[:] = np.inf
            np.minimum.at(self.step_start, si, self.t0[m])
        phase_mask = self.kind == KIND_PHASE
        self.phases = {}
        for a in np.unique(self.action[phase_mask]):
            sel = phase_mask & (self.action == a)
            self.phases[self.names[int(a)]] = float((self.t1[sel] - self.t0[sel]).max())

    # -- the three views -----------------------------------------------------
    def columns(self) -> dict[str, Any]:
        if self._columns is None:
            path = self.directory / "columns.npz"
            if not path.is_file():
                self._columns = {"rank": [], "lat": [], "lon": [], "land": [], "host": [], "local_rank": []}
            else:
                z = np.load(path, allow_pickle=False)
                phis = z["phis"] if "phis" in z.files else np.zeros_like(z["lat"])
                self._columns = {
                    "rank": z["rank"].astype(int).tolist(),
                    "lat": _round(z["lat"], 3), "lon": _round(z["lon"], 3),
                    "land": (np.abs(phis) > 1.0).astype(int).tolist(),
                    "host": [str(h) for h in z["host"]], "local_rank": z["local_rank"].astype(int).tolist(),
                }
        return self._columns

    def overview(self) -> dict[str, Any]:
        nact, nsteps = len(self.action_order), len(self.steps)
        if nact and nsteps:
            worst = self.cube.max(axis=0)
            mean = self.cube.mean(axis=0)
            wait = self.wait_cube.max(axis=0)
        else:
            worst = mean = wait = np.zeros((nact, nsteps))
        return {
            "run": self.manifest.get("run", ""),
            "ranks": self.nranks,
            "ranks_present": len(self.ranks_present),
            "complete": bool(self.manifest.get("complete")),
            "updated_utc": self.manifest.get("updated_utc", ""),
            "steps": self.steps.tolist(),
            "actions": [self.names[a] for a in self.action_order],
            "phases": {k: round(v, 3) for k, v in self.phases.items()},
            "step_max_ms": _round(self.step_max * MS),
            "step_mean_ms": _round(self.step_mean * MS),
            "max_ms": [_round(row * MS) for row in worst],
            "mean_ms": [_round(row * MS) for row in mean],
            "wait_ms": [_round(row * MS) for row in wait],
            "total_s": _round(self.cube.sum(axis=2).max(axis=0) if nact and nsteps else np.zeros(nact), 4),
        }

    def step_view(self, step: int) -> dict[str, Any]:
        """Every rank's actions and waits in one step, in ms from the step's first start."""

        if step not in self._step_pos:
            raise KeyError(f"step {step} is not in the timeline")
        origin = self.step_start[self._step_pos[step]]
        m = (self.step == step) & ((self.kind == KIND_ACTION) | (self.kind == KIND_WAIT))
        pos = {a: i for i, a in enumerate(self.action_order)}
        action = np.array([pos.get(int(a), -1) for a in self.action[m]], dtype=np.int64)
        order = np.lexsort((self.t0[m], self.rank[m]))
        return {
            "step": int(step),
            "rank": self.rank[m][order].tolist(),
            "action": action[order].tolist(),
            "wait": (self.kind[m][order] == KIND_WAIT).astype(int).tolist(),
            "t0": _round((self.t0[m][order] - origin) * MS),
            "t1": _round((self.t1[m][order] - origin) * MS),
        }

    def globe_values(self, step: int | None) -> dict[str, Any]:
        """Each rank's time in every action, ms: at one step, or the mean over all steps."""

        if step is None:
            values = self.cube.mean(axis=2) if len(self.steps) else self.cube[:, :, 0]
        else:
            values = self.cube[:, :, self._step_pos[step]]
        full = np.zeros((self.nranks, values.shape[1]))
        full[np.asarray(self.ranks_present, dtype=np.int64)] = values
        return {"step": step, "ms": [_round(row * MS) for row in full.T]}   # action x rank

    def default_steps(self, count: int = 4) -> list[int]:
        """Steps worth embedding in a snapshot: the first, the slowest, a typical one, the last."""

        if not len(self.steps):
            return []
        chosen = [int(self.steps[0]), int(self.steps[int(np.argmax(self.step_max))]),
                  int(self.steps[int(np.argsort(self.step_max)[len(self.steps) // 2])]), int(self.steps[-1])]
        out: list[int] = []
        for s in chosen:
            if s not in out:
                out.append(s)
        return out[:count]


# -- the page -----------------------------------------------------------------
def _page() -> str:
    return resources.files("freecam.pi_cam").joinpath("timeline_static/viewer.html").read_text()


def snapshot_html(data: TimelineData, steps: list[int] | None = None) -> str:
    """One self-contained page: the overview, the columns, the globe for every action over all
    steps and at the chosen steps, and those steps' timelines."""

    steps = data.default_steps() if steps is None else [s for s in steps if s in data._step_pos]
    embedded = {
        "overview": data.overview(),
        "columns": data.columns(),
        "globe": {"all": data.globe_values(None), **{str(s): data.globe_values(s) for s in steps}},
        "steps": {str(s): data.step_view(s) for s in steps},
    }
    payload = json.dumps(embedded, separators=(",", ":")).replace("</", "<\\/")
    return _page().replace("/*TIMELINE_DATA*/null", payload, 1)


class _Handler(BaseHTTPRequestHandler):
    data: TimelineData
    token: str
    lock: threading.Lock
    checked = 0.0

    def log_message(self, *args: Any) -> None:     # quiet
        pass

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any) -> None:
        self._send(json.dumps(payload, separators=(",", ":")).encode(), "application/json")

    def _refresh(self) -> None:
        cls = type(self)
        with cls.lock:
            now = time.monotonic()
            if now - cls.checked > 2.0:
                cls.checked = now
                if cls.data.changed():
                    cls.data.reload()

    def do_GET(self) -> None:  # noqa: N802 - the http.server interface
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if query.get("token", [""])[0] != self.token:
            self._send(b"forbidden: open the address the command printed", "text/plain", HTTPStatus.FORBIDDEN)
            return
        try:
            if url.path == "/":
                self._send(_page().encode(), "text/html; charset=utf-8")
                return
            self._refresh()
            data = type(self).data
            if url.path == "/api/overview":
                self._json(data.overview())
            elif url.path == "/api/columns":
                self._json(data.columns())
            elif url.path == "/api/step":
                self._json(data.step_view(int(query["n"][0])))
            elif url.path == "/api/globe":
                step = query.get("step", ["all"])[0]
                self._json(data.globe_values(None if step == "all" else int(step)))
            else:
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError) as error:
            self._send(str(error).encode(), "text/plain", HTTPStatus.BAD_REQUEST)


def serve(directory: str | Path, *, host: str = "127.0.0.1", port: int = 0, open_browser: bool = False) -> None:
    data = TimelineData(directory)
    token = secrets.token_urlsafe(16)
    handler = type("TimelineHandler", (_Handler,), {"data": data, "token": token, "lock": threading.Lock()})
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_address[1]}/?token={token}"
    print(f"freeCAM timeline of {data.directory} ({len(data.steps)} steps, {data.nranks} ranks): {url}", flush=True)
    print("forward the port to your machine if this is a remote host (ssh -L, or the editor's port forwarding)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="freecam timeline", description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="a timeline directory written with --timeline-dir")
    parser.add_argument("--html", type=Path, default=None, help="write one self-contained page to FILE and exit")
    parser.add_argument("--steps", default=None, help="comma-separated steps to embed in --html (default: first, slowest, typical, last)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="port to serve on (default: a free one)")
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    args = parser.parse_args(argv)
    if args.html is not None:
        data = TimelineData(args.directory)
        steps = None if args.steps is None else [int(s) for s in args.steps.split(",") if s.strip()]
        args.html.write_text(snapshot_html(data, steps))
        print(f"wrote {args.html} ({len(data.steps)} steps, {data.nranks} ranks)")
        return 0
    serve(args.directory, host=args.host, port=args.port, open_browser=args.open)
    return 0


if __name__ == "__main__":
    sys.exit(main())
