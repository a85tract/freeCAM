"""Compile a hooked kernel's TorchScript model with AOTInductor: its forward as fused kernels.

    tools/compile_torch_model.py <kernel> <model.pt> <eager.py:function> <anchors.npz> \\
        --output <model.pt2> [--device cuda] [--rows 27] [--max-rows 4096]

A TorchScript forward runs op by op, one kernel each (the v4 shallow-convection model: 793 a
call on a GPU).  AOTInductor compiles an exported graph ahead of time into a package whose
element-wise work is fused into a few generated kernels, the matrix products left to the
vendor library; the image loads it beside the TorchScript model and replays it as a CUDA
graph (``NativeModel(..., graph=True, compiled=...)``).

TorchScript cannot be exported directly when its code has loops whose trip counts it cannot
see, so the model's eager form is given: ``function`` in ``eager.py`` takes the loaded
TorchScript module and returns an ``nn.Module`` with the same forward, built from its weights.
That form is checked against the TorchScript forward before anything is compiled, and the
package against the TorchScript forward after; both are recorded in ``<model.pt2>.json`` with
the archive's sha256, which the image's binding checks.  The batch dimension is dynamic (a
rank's batch is its live columns); the package takes C-contiguous inputs.

The fused kernels evaluate the same expressions in another order and with other library
routines (the GPU's libdevice for exp and pow, the transformer layer unfused rather than
through its fast path), so the package's answers are not the TorchScript forward's bit for
bit: the record gives how far apart they are, column by column, on the anchors.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]


def model_inputs(kernel: str) -> list[str]:
    table = yaml.safe_load((REPO / "native/pi_cam/hooks.yaml").read_text())
    for hook in table["hooks"]:
        if hook["kernel"] == kernel:
            return list(hook["model"]["inputs"])
    raise SystemExit(f"{kernel}: no hook with a model block")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_function(spec: str) -> Callable[[Any], torch.nn.Module]:
    path, _, name = spec.rpartition(":")
    if not path or not name:
        raise SystemExit(f"the eager form is <file.py>:<function>, not {spec!r}")
    module_spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    if module_spec is None or module_spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, name)


def anchors(path: Path, names: list[str], rows: int, device: torch.device) -> list[torch.Tensor]:
    """rows columns of the anchor dataset, float64 and C-contiguous on the device, dt as one value."""

    z = np.load(path)
    if int(z[names[1]].shape[0]) < rows:
        raise SystemExit(f"{rows} columns asked, {z[names[1]].shape[0]} in {path}")
    out = [torch.tensor([float(np.asarray(z[names[0]]).reshape(-1)[0])], dtype=torch.float64)]
    out += [torch.from_numpy(np.ascontiguousarray(z[name][:rows], dtype=np.float64)) for name in names[1:]]
    return [t.to(device) for t in out]


def padded(inputs: list[torch.Tensor], pad: int) -> list[torch.Tensor]:
    """The same batch with its last pad rows zero, as rows past a chunk's columns are."""

    out = [inputs[0]]
    for t in inputs[1:]:
        t = t.clone()
        t[-pad:] = 0.0
        out.append(t)
    return out


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    """How far candidate is from reference: bit for bit, or the gap over each output column's
    own range, the median column's and the largest (a column the reference leaves at zero counts
    its gap as it is).  The image's graph runner refuses a package whose median exceeds 1e-3: a
    rounded or gated output can flip in a few columns of the right model."""

    a, b = reference.double().cpu(), candidate.double().cpu()
    both = torch.isfinite(a) & torch.isfinite(b)
    gap = (a - b).abs().where(both, torch.zeros_like(a))
    scale = a.abs().where(both, torch.zeros_like(a)).amax(0)
    column = gap.amax(0)
    scaled = column.where(scale == 0, column / scale.clamp_min(1e-300))
    return {"bit_for_bit": bool(torch.equal(a.nan_to_num(7.0), b.nan_to_num(7.0))
                                and torch.equal(torch.isnan(a), torch.isnan(b))),
            "scaled_gap": float(scaled.max()), "scaled_gap_median": float(scaled.median()),
            "max_abs_diff": float(gap.max()),
            "values_differing": int((gap > 0).sum()), "values": int(a.numel()),
            "non_finite_mismatch": int((torch.isfinite(a) != torch.isfinite(b)).sum())}


def timed(fn: Callable[[], Any], device: torch.device, n: int = 200, warm: int = 20) -> float:
    for _ in range(warm):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(n):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / n * 1e3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kernel")
    parser.add_argument("model", type=Path, help="the TorchScript archive the hook binds")
    parser.add_argument("eager", help="<file.py>:<function>: the TorchScript module -> its eager nn.Module")
    parser.add_argument("anchors", type=Path, help="an anchor dataset: one row a column, keyed by input name")
    parser.add_argument("--output", type=Path, required=True, help="the AOTInductor package (.pt2)")
    parser.add_argument("--device", default="cuda", help="cuda (the image's GPUs) or cpu (a check of the export)")
    parser.add_argument("--rows", type=int, default=27, help="the batch the checks and timings use (a rank's columns)")
    parser.add_argument("--max-rows", type=int, default=4096, help="the largest batch the package takes")
    arguments = parser.parse_args(argv)

    device = torch.device(arguments.device)
    names = model_inputs(arguments.kernel)
    torch.backends.cuda.matmul.allow_tf32 = False                 # as libtorch runs in the image
    torch.backends.cudnn.allow_tf32 = False
    script = torch.jit.load(str(arguments.model), map_location=device).eval()
    eager = load_function(arguments.eager)(script).to(device).eval()
    record: dict[str, Any] = {
        "schema_version": 1, "what": __doc__.strip().splitlines()[0], "kernel": arguments.kernel,
        "model": arguments.model.name, "model_sha256": sha256(arguments.model), "eager": Path(arguments.eager).name,
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "inputs": names, "rows": {"min": 2, "max": arguments.max_rows}, "layout": "C-contiguous",
    }

    sample = anchors(arguments.anchors, names, arguments.rows, device)
    with torch.no_grad():
        reference = script(*sample)
        record["eager_vs_torchscript"] = compare(reference, eager(*sample))
    if not record["eager_vs_torchscript"]["bit_for_bit"]:
        print(json.dumps(record["eager_vs_torchscript"]), file=sys.stderr)
        raise SystemExit("the eager form does not answer as the TorchScript forward does: nothing compiled")

    rows = torch.export.Dim("rows", min=2, max=arguments.max_rows)
    dynamic = tuple(None if i == 0 else {0: rows} for i in range(len(names)))
    start = time.perf_counter()
    exported = torch.export.export(eager, tuple(sample), dynamic_shapes=dynamic, strict=False)
    record["export_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    package = torch._inductor.aoti_compile_and_package(
        exported, package_path=str(arguments.output),
        inductor_configs={"aot_inductor.metadata": {"pycam_kernel": arguments.kernel,
                                                    "pycam_model_sha256": record["model_sha256"]}})
    record["compile_seconds"] = time.perf_counter() - start
    record["package"] = Path(package).name
    record["package_sha256"] = sha256(Path(package))

    compiled = torch._inductor.aoti_load_package(str(package))
    checks: dict[str, Any] = {}
    for n in sorted({2, arguments.rows, min(arguments.max_rows, 64)}):
        batch = anchors(arguments.anchors, names, n, device)
        with torch.no_grad():
            checks[f"{n} rows"] = compare(script(*batch), compiled(*batch))
    batch = padded(anchors(arguments.anchors, names, arguments.rows, device), 3)
    with torch.no_grad():
        answer = compiled(*batch)
        checks[f"{arguments.rows} rows, last 3 padding"] = compare(script(*batch), answer)
        checks[f"{arguments.rows} rows, last 3 padding"]["padding_rows_zero"] = bool((answer[-3:] == 0).all())
    record["compiled_vs_torchscript"] = checks

    with torch.no_grad():
        record["ms_per_forward"] = {"torchscript": timed(lambda: script(*sample), device),
                                    "compiled": timed(lambda: compiled(*sample), device),
                                    "rows": arguments.rows, "processes": 1}
    Path(f"{arguments.output}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("compile_seconds", "ms_per_forward")}))
    print(json.dumps(checks[f"{arguments.rows} rows"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
