"""Time a hooked kernel's TorchScript model on a GPU against one CPU thread, offline.

    tools/bench_model_on_device.py <kernel> <model.pt> <capture.npz> [--json out.json]

The hook hands the model one chunk of 16 columns a call with the arrays in host memory and
wants the packed output back there, so the first GPU number is that: copy in, run, copy out.
Larger batches (columns stacked from consecutive captured calls) show what batching across
chunks or ranks could buy; the device-resident column leaves the transfers out.  Inputs and
outputs are float64 as the hook hands them; the inputs are the model block's, read from
``native/pi_cam/hooks.yaml``.  Without a CUDA device only the CPU rows are printed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
ROWS = 16


def model_inputs(kernel: str) -> list[str]:
    table = yaml.safe_load((REPO / "native/pi_cam/hooks.yaml").read_text())
    for hook in table["hooks"]:
        if hook["kernel"] == kernel:
            return list(hook["model"]["inputs"])
    raise SystemExit(f"{kernel}: no hook with a model block")


def main(argv: list[str]) -> int:
    kernel, model_path, capture = argv[0], argv[1], argv[2]
    out_json = argv[argv.index("--json") + 1] if "--json" in argv else None
    names = model_inputs(kernel)
    z = np.load(capture, allow_pickle=True)
    calls = sorted({int(k.split("/")[1]) for k in z.files if k.startswith("in/")})

    def chunk(c: int) -> list[np.ndarray]:
        arrays = []
        for name in names:
            a = np.nan_to_num(np.asarray(z[f"in/{c}/{name}"], np.float64))
            if a.ndim == 0:
                arrays.append(a.reshape(1))
                continue
            if a.shape[0] < ROWS:
                a = np.concatenate([a, a[: ROWS - a.shape[0]]], 0)
            arrays.append(np.asfortranarray(a[:ROWS]))        # the hook's arrays are Fortran-ordered
        return arrays

    def batch(n_cols: int) -> list[np.ndarray]:
        parts = [chunk(c) for c in calls[: max(1, n_cols // ROWS)]]
        return [parts[0][0]] + [np.asfortranarray(np.concatenate([p[i] for p in parts], 0)[:n_cols]) for i in range(1, len(names))]

    def timed(fn, n: int = 200, warm: int = 20) -> float:
        for _ in range(warm):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    results = {"schema_version": 1, "what": __doc__.strip().splitlines()[0], "kernel": kernel, "model": Path(model_path).name,
               "torch": torch.__version__, "cuda": torch.version.cuda, "rows": {}}
    torch.set_num_threads(1)
    cpu = torch.jit.load(model_path, map_location="cpu").eval()
    ins16 = chunk(calls[0])
    t_cpu = [torch.from_numpy(a) for a in ins16]
    with torch.no_grad():
        ms = timed(lambda: cpu(*t_cpu))
    results["rows"][f"cpu 1 thread, {ROWS} columns"] = {"ms_per_call": ms, "us_per_column": ms * 1e3 / ROWS}
    print(f"CPU one thread, {ROWS} columns: {ms:.3f} ms a call")
    if not torch.cuda.is_available():
        print("no CUDA device here")
    else:
        dev = torch.device("cuda:0")
        results["gpu"] = torch.cuda.get_device_name(0)
        gpu = torch.jit.load(model_path, map_location=dev).eval()
        print("GPU:", results["gpu"])

        def hook_like(arrays):
            tensors = [torch.from_numpy(a).to(dev) for a in arrays]
            with torch.no_grad():
                return gpu(*tensors).cpu().numpy()

        for n_cols in (16, 64, 256, 1024, 4096, 16384):
            arrays = batch(n_cols)
            n = 100 if n_cols <= 1024 else 30
            ms = timed(lambda: hook_like(arrays), n=n)
            resident = [torch.from_numpy(a).to(dev) for a in arrays]
            with torch.no_grad():
                ms_res = timed(lambda: gpu(*resident), n=n)
            results["rows"][f"gpu, {n_cols} columns, host in and out"] = {"ms_per_call": ms, "us_per_column": ms * 1e3 / n_cols}
            results["rows"][f"gpu, {n_cols} columns, device resident"] = {"ms_per_call": ms_res, "us_per_column": ms_res * 1e3 / n_cols}
            print(f"GPU {n_cols:6d} columns: host in and out {ms:8.3f} ms a call ({ms * 1e3 / n_cols:7.2f} us a column) | device resident {ms_res:8.3f} ms")
        with torch.no_grad():
            diff = float(np.abs(cpu(*t_cpu).numpy() - hook_like(ins16)).max())
        results["max_abs_diff_cpu_gpu_one_chunk"] = diff
        print(f"max |cpu - gpu| over the packed output of one chunk: {diff:.3e}")
        for n_cols in (256, 4096):
            tensors = [torch.from_numpy(a) for a in batch(n_cols)]
            with torch.no_grad():
                ms = timed(lambda: cpu(*tensors), n=20)
            results["rows"][f"cpu 1 thread, {n_cols} columns"] = {"ms_per_call": ms, "us_per_column": ms * 1e3 / n_cols}
            print(f"CPU one thread, {n_cols} columns: {ms:.3f} ms a call ({ms * 1e3 / n_cols:.2f} us a column)")
    if out_json:
        Path(out_json).write_text(json.dumps(results, indent=1) + "\n")
        print("wrote", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
