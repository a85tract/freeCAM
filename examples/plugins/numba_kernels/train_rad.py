"""Train a per-column emulator of radiation_tend's computing branch on process captures, export it.

    train_rad.py <capture-dir> <out-prefix> [--hidden 256] [--epochs 40] [--threads 64] [--val-ranks 64] [--max-ranks N]

Inputs per column (the process contract's names): temperature, pressure, water vapour, cloud fraction,
in-cloud water and ice paths, the cloud optics' effective sizes, snow, ozone, the modal aerosols
(mixing ratios, wet diameters, aerosol water), the surface albedos and upward longwave, the zenith
angle.  Outputs: the two heating rates (energy units) and the ten fluxes.  Shortwave outputs are
learned on lit columns only and set to zero in the dark in postprocessing, where the physics has them
exactly zero.  Positive heavy-tailed inputs go through log10(x + floor).
"""
import argparse, glob, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

PVER = 30
# (name, source, kind): source is an input array name (and constituent index for state_q); kind 'lin' or 'log'
PROFILES = [
    ("t", "state_t", None, "lin"), ("pmid", "state_pmid", None, "log"), ("qv", "state_q", 0, "log"),
    ("cld", "cld", None, "lin"), ("iclwp", "iclwp", None, "log"), ("iciwp", "iciwp", None, "log"),
    ("dei", "dei", None, "lin"), ("mu", "mu", None, "lin"), ("lambdac", "lambdac", None, "log"),
    ("cldfsnow", "cldfsnow", None, "lin"), ("icswp", "icswp", None, "log"), ("des", "des", None, "lin"),
] + [(f"aer{c}", "state_q", c, "log") for c in range(42, 57)] + \
    [(f"dgnumwet{m}", "dgnumwet", m, "log") for m in range(3)] + [(f"qaerwat{m}", "qaerwat", m, "log") for m in range(3)]
PROFILES31 = [("o3vmr", "rstate_o3vmr", None, "log")]
SCALARS = ["coszrs", "cam_in_asdir", "cam_in_asdif", "cam_in_aldir", "cam_in_aldif", "cam_in_lwup", "clat"]
SCALAR_META = ["calday"]
OUT_PROFILES = ["qrs", "qrl"]
OUT_SCALARS = ["fsns", "fsnt", "flns", "flnt", "fsds", "sols", "soll", "solsd", "solld", "flwds"]
SW_OUTPUTS = {"qrs", "fsns", "fsnt", "fsds", "sols", "soll", "solsd", "solld"}
LOG_FLOOR = 1e-30

def feature_layout():
    layout = []; lo = 0
    for name, src, idx, kind in PROFILES:
        layout.append((name, src, idx, kind, lo, lo + PVER)); lo += PVER
    for name, src, idx, kind in PROFILES31:
        layout.append((name, src, idx, kind, lo, lo + PVER + 1)); lo += PVER + 1
    for name in SCALARS + SCALAR_META:
        layout.append((name, name, None, "lin", lo, lo + 1)); lo += 1
    return layout, lo
LAYOUT, NF = feature_layout()
def target_layout():
    layout = []; lo = 0
    for name in OUT_PROFILES: layout.append((name, lo, lo + PVER)); lo += PVER
    for name in OUT_SCALARS: layout.append((name, lo, lo + 1)); lo += 1
    return layout, lo
TLAYOUT, NT = target_layout()

def transform(values, kind):
    v = np.asarray(values, np.float64)
    return np.log10(np.maximum(v, 0.0) + LOG_FLOOR) if kind == "log" else v

def load_rank(path):
    z = np.load(path, allow_pickle=True); meta = json.loads(str(z["meta"]))
    Xs, Ys = [], []
    for i, m in enumerate(meta):
        ncol = int(m["ncol"]); X = np.empty((ncol, NF)); Y = np.empty((ncol, NT))
        for name, src, idx, kind, lo, hi in LAYOUT:
            if src in m: X[:, lo:hi] = float(m[src]); continue
            a = np.asarray(z[f"in/{i}/{src}"])[:ncol]
            if idx is not None: a = a[..., idx]
            X[:, lo:hi] = transform(a.reshape(ncol, -1), kind)
        for name, lo, hi in TLAYOUT:
            Y[:, lo:hi] = np.asarray(z[f"out/{i}/{name}"])[:ncol].reshape(ncol, -1)
        Xs.append(X); Ys.append(Y)
    return np.concatenate(Xs).astype(np.float32), np.concatenate(Ys)

def r2(pred, true):
    ss = ((true - pred) ** 2).sum(); tot = ((true - true.mean(0)) ** 2).sum()
    return 1.0 - ss / max(tot, 1e-300)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("capture_dir"); ap.add_argument("out_prefix")
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--threads", type=int, default=64); ap.add_argument("--val-ranks", type=int, default=64)
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--max-ranks", type=int, default=None)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    t0 = time.time()
    files = sorted(glob.glob(f"{a.capture_dir}/radiation_tend.rank-*.npz"))[: a.max_ranks]
    with ProcessPoolExecutor(a.workers) as pool: parts = list(pool.map(load_rank, files))
    ntrain = len(files) - a.val_ranks
    X = np.concatenate([p[0] for p in parts[:ntrain]]); Y = np.concatenate([p[1] for p in parts[:ntrain]])
    Xv = np.concatenate([p[0] for p in parts[ntrain:]]); Yv = np.concatenate([p[1] for p in parts[ntrain:]]); del parts
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1); X, Y = X[ok], Y[ok]
    okv = np.isfinite(Xv).all(1) & np.isfinite(Yv).all(1); Xv, Yv = Xv[okv], Yv[okv]
    print(f"loaded train {X.shape[0]:,} columns ({ntrain} ranks), validation {Xv.shape[0]:,} ({a.val_ranks}); {NF} features -> {NT} targets in {time.time()-t0:.0f}s", flush=True)
    cosz = X[:, [lo for n, _, _, _, lo, _ in LAYOUT if n == "coszrs"][0]]; coszv = Xv[:, [lo for n, _, _, _, lo, _ in LAYOUT if n == "coszrs"][0]]
    lit = cosz > 0; litv = coszv > 0
    print(f"lit columns: {100*lit.mean():.1f}% of train", flush=True)
    x_mean = X.mean(0, dtype=np.float64); x_std = X.std(0, dtype=np.float64); x_abs = np.abs(X).max(0)
    x_std = np.maximum(np.maximum(x_std, 1e-4 * x_abs), 1e-20)
    # target statistics on the columns where each target is live (SW on lit columns)
    sw_cols = np.zeros(NT, bool)
    for name, lo, hi in TLAYOUT:
        if name in SW_OUTPUTS: sw_cols[lo:hi] = True
    y_mean = np.where(sw_cols, Y[lit].mean(0), Y.mean(0)); y_std = np.where(sw_cols, Y[lit].std(0), Y.std(0))
    y_abs = np.abs(Y).max(0); y_std = np.maximum(y_std, 1e-3 * y_abs); y_std = np.where(y_abs > 1e-30, y_std, 1.0)
    y_min, y_max = Y.min(0), Y.max(0); span = y_max - y_min; y_min_c = y_min - 0.05 * span * (y_min < 0); y_max_c = y_max + 0.05 * span
    Xt = torch.tensor((X - x_mean) / x_std, dtype=torch.float32).clamp_(-20, 20); Yt = torch.tensor((Y - y_mean) / y_std, dtype=torch.float32)
    Xvt = torch.tensor((Xv - x_mean) / x_std, dtype=torch.float32).clamp_(-20, 20); Yvt = torch.tensor((Yv - y_mean) / y_std, dtype=torch.float32)
    # the loss mask: SW targets count on lit columns only
    mask = torch.ones((X.shape[0], NT), dtype=torch.float32); mask[:, sw_cols] = torch.tensor(lit, dtype=torch.float32)[:, None]
    maskv = torch.ones((Xv.shape[0], NT), dtype=torch.float32); maskv[:, sw_cols] = torch.tensor(litv, dtype=torch.float32)[:, None]
    H = a.hidden
    net = torch.nn.Sequential(torch.nn.Linear(NF, H), torch.nn.ReLU(), torch.nn.Linear(H, H), torch.nn.ReLU(), torch.nn.Linear(H, NT))
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-5)
    n = Xt.shape[0]; steps = (n + 2047) // 2048
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * steps, pct_start=0.1)
    for epoch in range(a.epochs):
        net.train(); perm = torch.randperm(n); total = 0.0
        for i in range(0, n, 2048):
            idx = perm[i:i + 2048]; opt.zero_grad()
            loss = (mask[idx] * (net(Xt[idx]) - Yt[idx]) ** 2).sum() / mask[idx].sum()
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); total += float(loss.detach()) * len(idx)
        net.eval()
        with torch.no_grad(): pv = net(Xvt); vloss = float((maskv * (pv - Yvt) ** 2).sum() / maskv.sum())
        print(f"epoch {epoch+1:3d}: train {total/n:.4f}  val {vloss:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    with torch.no_grad(): pv = net(Xvt).double().numpy() * y_std + y_mean
    pv = np.minimum(np.maximum(pv, y_min_c), y_max_c); pv[:, sw_cols] *= litv[:, None]
    report = {"hidden": H, "parameters": int(nparam), "train_columns": int(n), "val_columns": int(Xv.shape[0]), "features": NF, "targets": NT, "r2": {}, "rmse": {}}
    for name, lo, hi in TLAYOUT:
        rows = litv if name in SW_OUTPUTS else np.ones(len(Yv), bool)
        report["r2"][name] = float(r2(pv[rows, lo:hi], Yv[rows, lo:hi])); report["rmse"][name] = float(np.sqrt(((pv[rows, lo:hi] - Yv[rows, lo:hi]) ** 2).mean()))
    print("validation R2:", {k: round(v, 3) for k, v in report["r2"].items()}, flush=True)
    print("validation RMSE:", {k: f"{v:.3g}" for k, v in report["rmse"].items()}, flush=True)
    Path(f"{a.out_prefix}.report.json").write_text(json.dumps(report, indent=1))
    # export: folded float32 weights and the layouts, for the compiled forward
    W1 = net[0].weight.detach().double().numpy(); b1 = net[0].bias.detach().double().numpy()
    W3 = net[4].weight.detach().double().numpy(); b3 = net[4].bias.detach().double().numpy()
    np.savez(f"{a.out_prefix}_weights.npz",
             W1T=np.ascontiguousarray((W1 / x_std).T.astype(np.float32)), b1=(b1 - W1 @ (x_mean / x_std)).astype(np.float32),
             W2T=np.ascontiguousarray(net[2].weight.detach().numpy().T), b2=net[2].bias.detach().numpy(),
             W3T=np.ascontiguousarray((W3 * y_std[:, None]).T.astype(np.float32)), b3=(b3 * y_std + y_mean).astype(np.float32),
             x_lo=(x_mean - 20 * x_std).astype(np.float32), x_hi=(x_mean + 20 * x_std).astype(np.float32),
             y_min=y_min_c, y_max=y_max_c, sw_cols=sw_cols)
    json.dump({"features": [list(row) for row in LAYOUT], "targets": [list(row) for row in TLAYOUT], "NF": NF, "NT": NT, "PVER": PVER,
               "log_floor": LOG_FLOOR, "sw_outputs": sorted(SW_OUTPUTS)}, open(f"{a.out_prefix}_layout.json", "w"), indent=1)
    print(f"saved {a.out_prefix}_weights.npz ({nparam:,} parameters, {4*nparam/1e6:.1f} MB) and the layout", flush=True)

if __name__ == "__main__":
    main()
