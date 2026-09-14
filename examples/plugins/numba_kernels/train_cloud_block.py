#!/usr/bin/env python3
"""Train an MLP emulator of one compute block of the cloud macro/microphysics stage from a block capture.

    train_cloud_block.py <capture-dir> <out-prefix> --block macro|micro [--hidden 512] [--epochs 20] [--threads 64]
                         [--val-ranks 64] [--max-ranks N] [--workers 32]

The capture (``--cloud-block-capture``, freecam.physics.cloud_block) holds each block's inputs by name in single
precision and its outputs exactly.  Features: every array input flattened per column -- a (pcols, n) field gives n
features, the constituent array only the constituents the block's tendency flags, the tracer detrainment all its
sets; a feature that is never negative and heavy-tailed (99th percentile over a hundred times its median) is taken as
log10(x + floor).  Targets: the tendency (``ptend_s``, the flagged ``ptend_q``), the detrainment integrals for the
macrophysics block, and every buffer field the block changed in at least one captured call; the rest of its buffer
fields are left alone by the model (the write-back keeps what a model does not return).  Everything is standardised;
predictions are clipped to the range seen in training.  Weights (transposed for the plugin), the layout and a report
go to ``<out-prefix>_weights.npz``, ``<out-prefix>_layout.json``, ``<out-prefix>.report.json``.
"""
import argparse, glob, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from freecam.physics.cloud_block import BLOCKS  # noqa: E402

LOG_FLOOR = 1e-30
SCALAR_META = ("nstep", "lchnk", "ncol", "dt")

#: --core: the physics the network has to learn, and no more.  Inputs: the state's thermodynamics and geometry, the
#: five bulk water constituents, the surface, the convection carries, the cloud fields the driver reads.  Outputs it
#: learns: the bulk tendencies, the detrainment integrals, the cloud fractions and in-cloud water, the equilibrium
#: fields.  Outputs derived exactly in the model module instead of learned: AST = max(ALST, AIST); the old-time-sample
#: copies of the updated state (TCWAT, QCWAT, LCWAT, ICCWAT, NLWAT, NIWAT); the water tracers' tendencies (left zero).
CORE_INPUTS = ("state_t", "state_pmid", "state_pdel", "state_zm", "state_omega", "state_ps", "state_phis", "state_q",
               "cam_in_landfrac", "cam_in_ocnfrac", "cam_in_snowhland", "cam_in_ts", "cam_in_sst",
               "dlf", "dlf2", "zdu", "cmfmc", "cmfmc2",
               "CLD", "CONCLD", "AST", "ALST", "AIST", "QLST", "QIST", "CMELIQ", "FICE", "QCWAT", "TCWAT", "LCWAT", "ICCWAT", "NLWAT", "NIWAT",
               "SH_FRAC", "DP_FRAC")
CORE_BULK = (0, 1, 2, 3, 4)                 # vapour, cloud liquid, cloud ice, their numbers: the constituents the network predicts
CORE_DERIVED = ("AST", "TCWAT", "QCWAT", "LCWAT", "ICCWAT", "NLWAT", "NIWAT")


def input_names(z):
    return sorted({k.split("/", 2)[2] for k in z.files if k.startswith("in/0/")})


def output_names(z):
    return sorted({k.split("/", 2)[2] for k in z.files if k.startswith("out/0/")})


def feature_layout(z, lq, core=False):
    """(name, source, index-or-None, lo, hi) per feature group, from the first record's shapes."""

    layout, lo = [], 0
    flagged = np.array(CORE_BULK) if core else np.nonzero(lq)[0]
    for name in input_names(z):
        if core and name not in CORE_INPUTS:
            continue
        a = z[f"in/0/{name}"]
        if a.ndim == 1:
            layout.append((name, name, None, lo, lo + 1)); lo += 1
        elif a.ndim == 2:
            layout.append((name, name, None, lo, lo + a.shape[1])); lo += a.shape[1]
        elif name == "state_q":
            for m in flagged:
                layout.append((f"state_q[{m}]", name, int(m), lo, lo + a.shape[1])); lo += a.shape[1]
        else:
            for m in range(a.shape[2]):
                layout.append((f"{name}[{m}]", name, int(m), lo, lo + a.shape[1])); lo += a.shape[1]
    return layout, lo


def target_layout(z, block, changed, core=False):
    layout, lo = [], 0
    layout.append(("ptend_s", lo, lo + 30)); lo += 30
    nflag = len(CORE_BULK) if core else z["out/0/ptend_q"].shape[2]
    layout.append(("ptend_q", lo, lo + 30 * nflag)); lo += 30 * nflag
    if block == "macro":
        layout.append(("det_s", lo, lo + 1)); lo += 1
        layout.append(("det_ice", lo, lo + 1)); lo += 1
    for name in changed:
        if core and name in CORE_DERIVED:
            continue
        a = z[f"out/0/{name}"]
        width = int(np.prod(a.shape[1:])) if a.ndim > 1 else 1
        layout.append((name, lo, lo + width)); lo += width
    return layout, lo


def changed_fields(paths, block):
    """Buffer fields the block changed in at least one record of the given files (inputs are single precision)."""

    changed = set()
    for path in paths:
        z = np.load(path, allow_pickle=True); meta = json.loads(str(z["meta"]))
        for i in range(len(meta)):
            ncol = int(meta[i]["ncol"])
            for name in BLOCKS[block].buffer_names:
                if f"in/{i}/{name}" in z.files and f"out/{i}/{name}" in z.files and name not in changed:
                    a = z[f"in/{i}/{name}"][:ncol].astype(np.float32); b = z[f"out/{i}/{name}"][:ncol].astype(np.float32)
                    if a.shape == b.shape and not np.array_equal(a, b):
                        changed.add(name)
            # fields named only by the image (cloud-borne aerosols, tracer precipitation) are outputs too
            for name in output_names(z):
                if name not in BLOCKS[block].outputs and name not in changed and f"in/{i}/{name}" in z.files:
                    a = z[f"in/{i}/{name}"][:ncol].astype(np.float32); b = z[f"out/{i}/{name}"][:ncol].astype(np.float32)
                    if a.shape == b.shape and not np.array_equal(a, b):
                        changed.add(name)
    return sorted(changed)


def load_rank(args):
    path, layout, nf, tlayout, nt, q_out = args
    z = np.load(path, allow_pickle=True); meta = json.loads(str(z["meta"]))
    Xs, Ys = [], []
    for i, m in enumerate(meta):
        ncol = int(m["ncol"]); X = np.empty((ncol, nf), np.float32); Y = np.empty((ncol, nt), np.float32)
        for name, src, idx, lo, hi in layout:
            a = np.asarray(z[f"in/{i}/{src}"])[:ncol]
            if idx is not None:
                a = a[..., idx]
            X[:, lo:hi] = a.reshape(ncol, -1)
        for name, lo, hi in tlayout:
            a = np.asarray(z[f"out/{i}/{name}"])[:ncol]
            if name == "ptend_q" and q_out is not None:
                a = a[:, :, q_out]              # the saved array holds the flagged constituents in flag order
            Y[:, lo:hi] = a.reshape(ncol, -1)
        Xs.append(X); Ys.append(Y)
    return np.concatenate(Xs), np.concatenate(Ys)


def r2(pred, true):
    ss = ((true - pred) ** 2).sum(); tot = ((true - true.mean(0)) ** 2).sum()
    return 1.0 - ss / max(tot, 1e-300)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture_dir"); ap.add_argument("out_prefix"); ap.add_argument("--block", choices=("macro", "micro"), required=True)
    ap.add_argument("--hidden", type=int, default=512); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--threads", type=int, default=64); ap.add_argument("--val-ranks", type=int, default=64)
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--max-ranks", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--core", action="store_true", help="learn the physics only: CORE_INPUTS in, the bulk tendencies and the non-derivable fields out")
    a = ap.parse_args()
    import torch

    torch.set_num_threads(a.threads); torch.manual_seed(0)
    t0 = time.time()
    files = sorted(glob.glob(f"{a.capture_dir}/cloud_{a.block}.rank-*.npz"))[: a.max_ranks]
    z0 = np.load(files[0], allow_pickle=True)
    lq = np.asarray(z0["out/0/ptend_lq"]).reshape(-1)
    flagged = np.nonzero(lq)[0].tolist()
    q_out = [flagged.index(m) for m in CORE_BULK] if a.core else None      # positions of the bulk constituents among the flagged
    layout, nf = feature_layout(z0, lq, core=a.core)
    changed = changed_fields(files[:8], a.block)
    tlayout, nt = target_layout(z0, a.block, changed, core=a.core)
    learned = [n for n, _, _ in tlayout]
    print(f"{len(files)} ranks; {nf} features in {len(layout)} groups; {nt} targets: {learned}"
          + (f"; derived, not learned: {[c for c in changed if c in CORE_DERIVED]}; tracer tendencies left zero" if a.core else ""), flush=True)
    with Pool(a.workers) as pool:
        parts = pool.map(load_rank, [(f, layout, nf, tlayout, nt, q_out) for f in files])
    ntrain = max(1, len(parts) - a.val_ranks)
    X = np.concatenate([p[0] for p in parts[:ntrain]]); Y = np.concatenate([p[1] for p in parts[:ntrain]])
    Xv = np.concatenate([p[0] for p in parts[ntrain:]]) if ntrain < len(parts) else X[:4096]
    Yv = np.concatenate([p[1] for p in parts[ntrain:]]) if ntrain < len(parts) else Y[:4096]
    del parts
    print(f"loaded {X.shape[0]:,} training columns, {Xv.shape[0]:,} validation ({time.time()-t0:.0f}s)", flush=True)
    # the transform of every feature: log10 for the never-negative heavy-tailed ones, decided on the training set
    kinds = np.zeros(nf, np.int8)
    for name, src, idx, lo, hi in layout:
        block_x = X[:, lo:hi]
        mins = block_x.min(0); med = np.median(np.abs(block_x), 0); p99 = np.percentile(np.abs(block_x), 99, axis=0)
        kinds[lo:hi] = ((mins >= 0) & (p99 > 100 * np.maximum(med, 1e-300))).astype(np.int8)
    def transform(M):
        out = M.astype(np.float64, copy=True)
        log = kinds != 0
        out[:, log] = np.log10(np.maximum(out[:, log], 0.0) + LOG_FLOOR)
        return out
    X = transform(X); Xv = transform(Xv)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1); okv = np.isfinite(Xv).all(1) & np.isfinite(Yv).all(1)
    X, Y, Xv, Yv = X[ok], Y[ok], Xv[okv], Yv[okv]
    x_mean = X.mean(0); x_std = np.maximum(X.std(0), 1e-6 * np.abs(X).max(0) + 1e-12)
    Y64 = Y.astype(np.float64); y_mean = Y64.mean(0)
    # a target that hardly varies (or never does) is not blown up by its standardisation
    y_std = np.maximum(Y64.std(0), 1e-3 * np.abs(Y64).max(0)); y_std[y_std == 0] = 1.0
    del Y64
    y_min = Y.min(0).astype(np.float64); y_max = Y.max(0).astype(np.float64)
    # a target a hundred standard deviations out is an outlier the clip at inference removes anyway
    Xt = torch.tensor((X - x_mean) / x_std, dtype=torch.float32).clamp_(-20, 20); Yt = torch.tensor((Y - y_mean) / y_std, dtype=torch.float32).clamp_(-100, 100)
    Xvt = torch.tensor((Xv - x_mean) / x_std, dtype=torch.float32).clamp_(-20, 20); Yvt = torch.tensor((Yv - y_mean) / y_std, dtype=torch.float32).clamp_(-100, 100)
    del X, Xv
    H = a.hidden; n = Xt.shape[0]
    net = torch.nn.Sequential(torch.nn.Linear(nf, H), torch.nn.ReLU(), torch.nn.Linear(H, H), torch.nn.ReLU(), torch.nn.Linear(H, nt))
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-5)
    steps = (n + a.batch - 1) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * steps, pct_start=0.1)
    print(f"MLP {nf}->{H}->{H}->{nt}: {nparam:,} parameters; {a.epochs} epochs of {steps} steps", flush=True)
    for epoch in range(a.epochs):
        net.train(); perm = torch.randperm(n); total = 0.0
        for start in range(0, n, a.batch):
            idx = perm[start:start + a.batch]; opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xt[idx]), Yt[idx]); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); total += float(loss.detach()) * len(idx)
        net.eval()
        with torch.no_grad():
            vloss = float(torch.nn.functional.mse_loss(net(Xvt), Yvt))
        print(f"epoch {epoch+1:3d}: train {total/n:.4f}  val {vloss:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    with torch.no_grad():
        pv = np.clip(net(Xvt).double().numpy() * y_std + y_mean, y_min, y_max)
    report = {"block": a.block, "hidden": H, "parameters": int(nparam), "train_columns": int(n), "val_columns": int(Xvt.shape[0]),
              "features": nf, "targets": nt, "epochs": a.epochs, "changed_fields": changed, "r2": {}, "rmse": {}}
    for name, lo, hi in tlayout:
        report["r2"][name] = float(r2(pv[:, lo:hi], Yv[:, lo:hi])); report["rmse"][name] = float(np.sqrt(((pv[:, lo:hi] - Yv[:, lo:hi]) ** 2).mean()))
    print("validation R2:", {k: round(v, 3) for k, v in report["r2"].items()}, flush=True)
    W = [m.weight.detach().numpy() for m in net if isinstance(m, torch.nn.Linear)]; B = [m.bias.detach().numpy() for m in net if isinstance(m, torch.nn.Linear)]
    np.savez(f"{a.out_prefix}_weights.npz", W1T=np.ascontiguousarray(W[0].T, np.float32), b1=B[0].astype(np.float32),
             W2T=np.ascontiguousarray(W[1].T, np.float32), b2=B[1].astype(np.float32), W3T=np.ascontiguousarray(W[2].T, np.float32), b3=B[2].astype(np.float32),
             x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, y_min=y_min, y_max=y_max, kinds=kinds, lq=lq.astype(np.int32))
    json.dump({"block": a.block, "features": [(n_, s, i, int(lo), int(hi)) for n_, s, i, lo, hi in layout], "targets": [(n_, int(lo), int(hi)) for n_, lo, hi in tlayout],
               "NF": nf, "NT": nt, "hidden": H, "log_floor": LOG_FLOOR, "pver": 30, "flagged": flagged, "pcnst": int(lq.shape[0]),
               "core": bool(a.core), "q_out": list(CORE_BULK) if a.core else flagged,
               "derived": [c for c in changed if c in CORE_DERIVED] if a.core else []},
              open(f"{a.out_prefix}_layout.json", "w"), indent=1)
    json.dump(report, open(f"{a.out_prefix}.report.json", "w"), indent=1)
    print(f"saved {a.out_prefix}_weights.npz ({nparam:,} parameters, {4*nparam/1e6:.1f} MB) and the layout ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
