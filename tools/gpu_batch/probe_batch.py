"""Batched-GPU probe: every rank calls the batching plugin as the hook would (16 columns of
the 20 uwshcu inputs, one packed output) and times it; the group leader runs the forward on
the GPU for the whole group.  Also times the like-for-like host reference (each rank runs the
model itself on its 16 columns on the CPU) in the same process population.

  mpiexec -n 128 -ppn 128 bash gpu_rank_env.sh python probe_batch.py --plugin fcb_plugin.so \
      --model m.pt --gpus-per-node 4 --calls 200 --jitter-ms 0
"""
import argparse
import ctypes
import os
import sys
import time

import numpy as np
from mpi4py import MPI

D1 = [0, 31, 31, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 30, 30, 0, 0, 30]
D2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 0, 0, 0, 0, 0, 0]
NCOL, NOUT_W = 16, 1190


NAMES = ["dt", "ps0_inv", "zs0_inv", "p0_inv", "z0_inv", "dp0_inv", "u0_inv", "v0_inv", "qv0_inv", "ql0_inv", "qi0_inv",
         "t0_inv", "s0_inv", "tr0_inv", "tke_inv", "cldfrct_inv", "concldfrct_inv", "pblh", "cush", "dpdry0_inv"]


def make_inputs(rng: np.random.Generator, anchors: str | None, rank: int) -> list[np.ndarray]:
    """The 20 hook inputs for 16 columns: real captured columns from the anchor dataset when
    given (rows rank*16 ... in the file's order), random numbers otherwise."""
    if anchors:
        z = np.load(anchors, allow_pickle=True)
        n = z["t0_inv"].shape[0]
        start = (rank * NCOL) % (n - NCOL)
        arrays = [np.array([float(z["dt"][start])])]
        for name in NAMES[1:]:
            arrays.append(np.asfortranarray(np.asarray(z[name][start:start + NCOL], dtype=np.float64)))
        return arrays
    arrays = []
    for j in range(20):
        if j == 0:
            arrays.append(np.array([1800.0]))
            continue
        shape = [NCOL] + ([D1[j]] if D1[j] else []) + ([D2[j]] if D2[j] else [])
        arrays.append(np.asfortranarray(rng.standard_normal(shape)))
    return arrays


def busy_wait(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus-per-node", type=int, default=4)
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--jitter-ms", type=float, default=0.0, help="uniform random CPU busy time before each call, per rank")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--anchors", default=None, help="anchor dataset (.npz keyed by input name) for real columns")
    ap.add_argument("--device-index", type=int, default=0, help="-1 runs the leader's forward on the host (a login-node check)")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    rng = np.random.default_rng(1000 + rank)
    lib = ctypes.CDLL(args.plugin, mode=ctypes.RTLD_GLOBAL)
    lib.fcb_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.fcb_init.restype = ctypes.c_int
    lib.fcb_plugin.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64),
                               ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64)]
    lib.fcb_plugin.restype = ctypes.c_int
    lib.fcb_cpu_reference.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    lib.fcb_cpu_reference.restype = ctypes.c_int
    lib.fcb_stats.argtypes = [ctypes.POINTER(ctypes.c_double)]
    lib.fcb_group_size.restype = ctypes.c_int
    lib.fcb_group_rank.restype = ctypes.c_int

    arrays = make_inputs(rng, args.anchors, rank)
    packed = np.zeros((NCOL, NOUT_W), order="F")
    in_ptrs = (ctypes.c_void_p * 20)(*[a.ctypes.data for a in arrays])
    in_shapes = (ctypes.c_int64 * 60)()
    for j, a in enumerate(arrays):
        dims = list(a.shape) if j else [1]
        for k, d in enumerate(dims):
            in_shapes[3 * j + k] = d
    out_ptrs = (ctypes.c_void_p * 1)(packed.ctypes.data)
    out_shapes = (ctypes.c_int64 * 3)(NCOL, NOUT_W, 0)

    t_init0 = time.perf_counter()
    status = lib.fcb_init(comm.py2f(), args.gpus_per_node, args.model.encode(), args.device_index)
    t_init = time.perf_counter() - t_init0
    if status != 0:
        print(f"[r{rank}] fcb_init failed with status {status}", flush=True)
        comm.Abort(1)
    gsize, grank = lib.fcb_group_size(), lib.fcb_group_rank()

    def one_call() -> None:
        if args.jitter_ms > 0:
            busy_wait(rng.uniform(0, args.jitter_ms) * 1e-3)
        st = lib.fcb_plugin(20, in_ptrs, in_shapes, 1, out_ptrs, out_shapes)
        if st != 0:
            print(f"[r{rank}] plugin status {st}", flush=True)
            comm.Abort(2)

    for _ in range(args.warm):
        one_call()
    lib.fcb_reset_stats()
    comm.Barrier()
    walls = []
    t0 = time.perf_counter()
    for _ in range(args.calls):
        s = time.perf_counter()
        one_call()
        walls.append(time.perf_counter() - s)
    elapsed = time.perf_counter() - t0
    stats = (ctypes.c_double * 7)()
    lib.fcb_stats(stats)
    walls = np.array(walls) * 1e3
    finite = bool(np.isfinite(packed).all())
    row = dict(rank=rank, grank=grank, gsize=gsize, init_s=t_init, mean_ms=float(walls.mean()), p95_ms=float(np.percentile(walls, 95)),
               max_ms=float(walls.max()), gather=stats[0], wait_in=stats[1], forward=stats[2], wait_out=stats[3], scatter=stats[4],
               total=stats[5], loop_ms_per_call=elapsed / args.calls * 1e3, finite=finite)

    cpu_row = None
    if not args.skip_cpu:
        cpu_out = np.zeros((NCOL, NOUT_W), order="F")
        for _ in range(args.warm):
            lib.fcb_cpu_reference(20, in_ptrs, cpu_out.ctypes.data, args.model.encode())
        comm.Barrier()
        cw = []
        for _ in range(args.calls):
            s = time.perf_counter()
            lib.fcb_cpu_reference(20, in_ptrs, cpu_out.ctypes.data, args.model.encode())
            cw.append(time.perf_counter() - s)
        cw = np.array(cw) * 1e3
        diff = float(np.abs(cpu_out - packed).max()) if finite else float("nan")
        scale = float(np.abs(cpu_out).max()) or 1.0
        rel = float((np.abs(cpu_out - packed) / np.maximum(np.abs(cpu_out), 1e-3 * scale)).max()) if finite else float("nan")
        cpu_row = dict(mean_ms=float(cw.mean()), p95_ms=float(np.percentile(cw, 95)), max_abs_diff_vs_batched=diff, max_rel_diff=rel)

    rows = comm.gather((row, cpu_row), root=0)
    if rank == 0:
        rows_b = [r for r, _ in rows]
        means = np.array([r["mean_ms"] for r in rows_b])
        leaders = [r for r in rows_b if r["grank"] == 0]
        others = [r for r in rows_b if r["grank"] != 0]
        print(f"batched GPU plugin: {size} ranks, groups of {gsize} ({size // gsize} leaders), {args.calls} calls a rank, jitter {args.jitter_ms} ms")
        print(f"  init (model load, window): leader {max((r['init_s'] for r in leaders), default=0):.1f} s, others {max((r['init_s'] for r in others), default=0):.1f} s")
        print(f"  wall a call, all ranks: mean {means.mean():.2f} ms, min-rank {means.min():.2f}, max-rank {means.max():.2f}; "
              f"p95 over ranks {np.mean([r['p95_ms'] for r in rows_b]):.2f} ms; loop {np.mean([r['loop_ms_per_call'] for r in rows_b]):.2f} ms a call")
        for label, group in (("leaders", leaders), ("others", others)):
            if group:
                print(f"  {label:8s}: gather {np.mean([r['gather'] for r in group]):.3f}  wait-in {np.mean([r['wait_in'] for r in group]):.3f}  "
                      f"forward {np.mean([r['forward'] for r in group]):.3f}  wait-out {np.mean([r['wait_out'] for r in group]):.3f}  "
                      f"scatter {np.mean([r['scatter'] for r in group]):.3f}  total {np.mean([r['total'] for r in group]):.3f} ms")
        print(f"  outputs finite on all ranks: {all(r['finite'] for r in rows_b)}")
        if cpu_row is not None:
            cpu_means = np.array([c["mean_ms"] for _, c in rows])
            diffs = np.array([c["max_abs_diff_vs_batched"] for _, c in rows])
            rels = np.array([c["max_rel_diff"] for _, c in rows])
            print(f"host reference (each rank, its own 16 columns on the CPU): mean {cpu_means.mean():.2f} ms a call, "
                  f"min-rank {cpu_means.min():.2f}, max-rank {cpu_means.max():.2f}; batched vs host: max abs diff {np.nanmax(diffs):.3e}, "
                  f"max relative diff {np.nanmax(rels):.3e}")
    lib.fcb_finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
