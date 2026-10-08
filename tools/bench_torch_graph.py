"""Time a hooked kernel's model on one GPU the three ways the image can run it, at N processes.

    tools/bench_torch_graph.py proc <mode> <model.pt> <anchors.npz> <barrier dir> <nprocs> <out.jsonl> \\
        [--package <model.pt2>] [--ftorch-root <prefix>] [--rows 27]
    tools/bench_torch_graph.py summarize <out.jsonl> --mode <mode> --procs <n>

``mode`` is how a batched forward is answered:

  ordinary  the TorchScript forward as FTorch runs it: the host arrays copied to the GPU, the
            forward op by op, the packed answer copied back (the run's H leg);
  graph     the image's CUDA graph runner (libpycam_torch_graph, the FTorch prefix's) over the
            same forward: captured once, then each call refills its inputs and replays (R);
  compiled  the same runner capturing an AOTInductor package of the model instead
            (tools/compile_torch_model.py): its element-wise work fused.

With ``--tf32`` the process lets cuBLAS multiply float32 matrices on the A100's tensor cores
(TF32: inputs rounded to a 10-bit mantissa, sums kept in float32), and records how far its
answer on the first batch is from the float32 TorchScript forward's, over each output column's
range.  ``--rows`` sets the batch: a rank's columns (27), or a GPU's 32 ranks' (864).

The runner is the library the image links, driven through ctypes as the hook drives it: the
model loaded by FTorch's own torch_jit_load, the arrays Fortran-ordered float64 in host memory,
the batch's rows of the anchor dataset.  Each process warms up, waits at a file barrier for the
others, then times its calls; one JSON line a process.  The processes of a phase share GPU 0
(through the site's MPS on a node that runs one), as a GPU's ranks do in a run.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CALLS = 80


def model_inputs(kernel: str) -> list[str]:
    import yaml

    table = yaml.safe_load((REPO / "native/pi_cam/hooks.yaml").read_text())
    for hook in table["hooks"]:
        if hook["kernel"] == kernel:
            return list(hook["model"]["inputs"])
    raise SystemExit(f"{kernel}: no hook with a model block")


def batches(anchors: Path, names: list[str], rows: int) -> list[list[np.ndarray]]:
    """Consecutive rows-column batches of the anchors, Fortran-ordered as the hook's arrays are."""

    z = np.load(anchors)
    total = int(z[names[1]].shape[0])
    out = []
    for start in range(0, total - rows + 1, rows):
        batch = [np.array([float(np.asarray(z[names[0]]).reshape(-1)[0])], dtype=np.float64)]
        batch += [np.asfortranarray(z[name][start:start + rows], dtype=np.float64) for name in names[1:]]
        out.append(batch)
    return out


def wait_at(barrier: Path, nprocs: int) -> None:
    (barrier / f"ready.{os.getpid()}").touch()
    start = time.time()
    while len(list(barrier.glob("ready.*"))) < nprocs and time.time() - start < 600:
        time.sleep(0.02)


class Runner:
    """The image's graph runner over one batch's host arrays, as the hook calls it."""

    def __init__(self, prefix: Path, model: Path, package: Path | None, host: list[np.ndarray], out: np.ndarray) -> None:
        lib = prefix / "lib64"
        self.ftorch = ctypes.CDLL(str(lib / "libftorch.so"), mode=ctypes.RTLD_GLOBAL)
        self.graph = ctypes.CDLL(str(lib / "libpycam_torch_graph.so"))
        self.ftorch.torch_jit_load.restype = ctypes.c_void_p
        self.ftorch.torch_jit_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
        self.module = self.ftorch.torch_jit_load(str(model).encode(), 1, 0, False, False)   # torch_kCUDA, device 0
        n = len(host)
        self.host, self.out = host, out
        self.ptrs = (ctypes.c_void_p * n)(*[a.ctypes.data for a in host])
        self.ndim = (ctypes.c_int * n)(*[a.ndim for a in host])
        self.shapes = (ctypes.c_int64 * sum(a.ndim for a in host))(*[e for a in host for e in a.shape])
        self.out_shape = (ctypes.c_int64 * 2)(*out.shape)
        g = self.graph
        g.pycam_torch_graph_open_v2.restype = ctypes.c_void_p
        g.pycam_torch_graph_open_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int64), ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        g.pycam_torch_graph_run.restype = ctypes.c_int
        g.pycam_torch_graph_run.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int64), ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
        g.pycam_torch_graph_compiled_gaps.restype = None
        g.pycam_torch_graph_compiled_gaps.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                                                      ctypes.POINTER(ctypes.c_double)]
        g.pycam_torch_graph_message.restype = ctypes.c_int
        g.pycam_torch_graph_message.argtypes = [ctypes.c_char_p, ctypes.c_int]
        status = ctypes.c_int(0)
        start = time.perf_counter()
        self.handle = g.pycam_torch_graph_open_v2(
            self.module, None if package is None else str(package).encode(), 0, n, self.ptrs, self.ndim, self.shapes,
            out.ctypes.data, 2, self.out_shape, 3, ctypes.byref(status))
        self.capture_seconds = time.perf_counter() - start
        self.status = status.value
        buffer = ctypes.create_string_buffer(512)
        g.pycam_torch_graph_message(buffer, 512)
        self.message = buffer.value.decode(errors="replace")
        median, largest = ctypes.c_double(-1.0), ctypes.c_double(-1.0)
        g.pycam_torch_graph_compiled_gaps(self.handle, ctypes.byref(median), ctypes.byref(largest))
        self.gap = (median.value, largest.value)

    def __call__(self) -> None:
        status = self.graph.pycam_torch_graph_run(self.handle, len(self.host), self.ptrs, self.ndim, self.shapes,
                                                  self.out.ctypes.data, 2, self.out_shape)
        if status != 0:
            raise SystemExit(f"replay failed: status {status}")


def proc(arguments: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(1)
    names = model_inputs(arguments.kernel)
    all_batches = batches(arguments.anchors, names, arguments.rows)
    host = [a.copy(order="F") for a in all_batches[0]]
    out = np.zeros((arguments.rows, arguments.width), dtype=np.float64, order="F")
    record: dict[str, object] = {"pid": os.getpid(), "mode": arguments.mode, "rows": arguments.rows,
                                 "tf32": bool(arguments.tf32)}
    reference = None
    if arguments.tf32:
        # the float32 answer first, then the tensor cores for everything after (the runner's
        # libtorch is this process's: the switch is one)
        device = torch.device("cuda:0")
        script = torch.jit.load(str(arguments.model), map_location=device).eval()
        with torch.no_grad():
            reference = script(*[torch.from_numpy(a).to(device) for a in host]).cpu()
        del script
        torch.backends.cuda.matmul.allow_tf32 = True
    if arguments.mode == "ordinary":
        device = torch.device("cuda:0")
        model = torch.jit.load(str(arguments.model), map_location=device).eval()

        def call() -> None:
            with torch.no_grad():
                answer = model(*[torch.from_numpy(a).to(device) for a in host])
                out[...] = answer.cpu().numpy()
        start = time.perf_counter()
        call()
        record["first_call_seconds"] = time.perf_counter() - start
    else:
        runner = Runner(arguments.ftorch_root, arguments.model, arguments.package if arguments.mode == "compiled" else None,
                        host, out)
        record.update({"status": runner.status, "capture_seconds": runner.capture_seconds})
        if runner.status != 0:
            record["message"] = runner.message
            with open(arguments.out, "a") as handle:
                handle.write(json.dumps(record) + "\n")
            return 1
        if runner.gap[0] >= 0.0:
            record["compiled_gap"] = {"median": runner.gap[0], "max": runner.gap[1]}
        call = runner
    for k in range(1, 9):                                      # warm, on other batches
        for a, b in zip(host, all_batches[k % len(all_batches)]):
            a[...] = b
        call()
    if reference is not None:
        from compile_torch_model import compare

        for a, b in zip(host, all_batches[0]):
            a[...] = b
        call()
        gap = compare(reference, torch.from_numpy(np.ascontiguousarray(out)))
        record["tf32_gap"] = {key: gap[key] for key in ("scaled_gap_median", "scaled_gap", "columns_over_1e-3", "columns")}
    wait_at(arguments.barrier, arguments.nprocs)
    times = []
    for k in range(CALLS):
        for a, b in zip(host, all_batches[(k + 9) % len(all_batches)]):
            a[...] = b                                         # the hook gathers each step's batch into the same arrays
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    record.update({"calls": CALLS, "ms_per_forward": float(np.median(times) * 1e3),
                   "ms_per_forward_mean": float(np.mean(times) * 1e3)})
    with open(arguments.out, "a") as handle:
        handle.write(json.dumps(record) + "\n")
    return 0


def summarize(arguments: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in arguments.out.read_text().splitlines()] if arguments.out.is_file() else []
    timed = [r for r in rows if "ms_per_forward" in r]
    summary: dict[str, object] = {"mode": arguments.mode, "processes": arguments.procs,
                                  "rows": timed[0].get("rows") if timed else None,
                                  "tf32": bool(timed and timed[0].get("tf32")), "finished": len(timed),
                                  "refused": [r for r in rows if r.get("status")][:1]}
    if timed:
        per = np.array([r["ms_per_forward"] for r in timed])
        summary.update({"ms_per_forward_median": float(np.median(per)), "ms_per_forward_slowest": float(per.max())})
        gaps = [r["compiled_gap"] for r in timed if "compiled_gap" in r]
        if gaps:
            summary["compiled_gap"] = {"median_largest": max(g["median"] for g in gaps),
                                       "max": max(g["max"] for g in gaps)}
        tf32 = [r["tf32_gap"] for r in timed if "tf32_gap" in r]
        if tf32:
            summary["tf32_gap"] = max(tf32, key=lambda g: g["scaled_gap"])
        captures = [r["capture_seconds"] for r in timed if "capture_seconds" in r]
        if captures:
            summary["capture_seconds_max"] = float(max(captures))
    print(json.dumps(summary))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("proc")
    one.add_argument("mode", choices=("ordinary", "graph", "compiled"))
    one.add_argument("model", type=Path)
    one.add_argument("anchors", type=Path)
    one.add_argument("barrier", type=Path)
    one.add_argument("nprocs", type=int)
    one.add_argument("out", type=Path)
    one.add_argument("--package", type=Path)
    one.add_argument("--ftorch-root", type=Path, default=REPO / "build" / "ftorch-cuda")
    one.add_argument("--kernel", default="compute_uwshcu_inv")
    one.add_argument("--rows", type=int, default=27)
    one.add_argument("--width", type=int, default=1190, help="the packed answer's columns")
    one.add_argument("--tf32", action="store_true", help="float32 matrix products on the tensor cores (TF32)")
    many = sub.add_parser("summarize")
    many.add_argument("out", type=Path)
    many.add_argument("--mode", required=True)
    many.add_argument("--procs", type=int, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "proc":
        if arguments.mode == "compiled" and arguments.package is None:
            raise SystemExit("compiled needs --package")
        return proc(arguments)
    return summarize(arguments)


if __name__ == "__main__":
    sys.exit(main())
