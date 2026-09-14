"""Train an emulator of radiation_tend's whole compute block on process captures, and export it.
    train_rad_block.py <capture-dir> <out-prefix> [--hidden 256] [--epochs 40] [--threads 64] [--val-ranks 64] [--max-ranks N]

The block contract (freecam.physics.radiation_process.BLOCK_INPUTS): the model takes only what the driver
reads from memory before any arithmetic -- the state, the buffer's cloud fraction, optics inputs and ozone
mass mixing ratio, the surface albedos and upward longwave, the calendar day and the column's latitude and
longitude -- and returns the two heating rates and the ten fluxes.  The solar zenith angle and the RRTMG
gas profiles are inside the block: the network learns them.  A thirteenth output, a lit-column logit, learns
where the sun is up so the shortwave outputs can be zeroed exactly in the dark.
Training data is the same capture the slot emulator used: ozone mass mixing ratio is recovered from the
recorded RRTMG state (o3vmr / amdo, model levels 1..30 at RRTMG levels 2..31), lit columns from coszrs.
"""
import argparse, glob, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_rad as T
PVER = T.PVER
AMDO = 0.603428                     # rrtmg_state.F90: molecular weight of dry air / ozone; o3vmr = o3mmr * amdo
SCALARS = ["cam_in_asdir", "cam_in_asdif", "cam_in_aldir", "cam_in_aldif", "cam_in_lwup", "clat", "clon"]
SCALAR_META = ["calday"]

def feature_layout():
    layout = []; lo = 0
    for name, src, idx, kind in T.PROFILES:
        layout.append((name, src, idx, kind, lo, lo + PVER)); lo += PVER
    layout.append(("o3", "ozone", None, "log", lo, lo + PVER)); lo += PVER      # the buffer's ozone mass mixing ratio
    for name in SCALARS + SCALAR_META:
        layout.append((name, name, None, "lin", lo, lo + 1)); lo += 1
    return layout, lo
LAYOUT, NF = feature_layout()
TLAYOUT, NT = T.target_layout()

def load_rank(path):
    z = np.load(path, allow_pickle=True); meta = json.loads(str(z["meta"]))
    Xs, Ys, Ls = [], [], []
    for i, m in enumerate(meta):
        ncol = int(m["ncol"]); X = np.empty((ncol, NF)); Y = np.empty((ncol, NT))
        for name, src, idx, kind, lo, hi in LAYOUT:
            if src in m: X[:, lo:hi] = float(m[src]); continue
            if src == "ozone":
                a = np.asarray(z[f"in/{i}/rstate_o3vmr"])[:ncol, 1:PVER + 1] / AMDO
            else:
                a = np.asarray(z[f"in/{i}/{src}"])[:ncol]
                if idx is not None: a = a[..., idx]
            X[:, lo:hi] = T.transform(a.reshape(ncol, -1), kind)
        for name, lo, hi in TLAYOUT:
            Y[:, lo:hi] = np.asarray(z[f"out/{i}/{name}"])[:ncol].reshape(ncol, -1)
        Xs.append(X); Ys.append(Y); Ls.append(np.asarray(z[f"in/{i}/coszrs"])[:ncol] > 0)
    return np.concatenate(Xs).astype(np.float32), np.concatenate(Ys), np.concatenate(Ls)

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
    X = np.concatenate([p[0] for p in parts[:ntrain]]); Y = np.concatenate([p[1] for p in parts[:ntrain]]); lit = np.concatenate([p[2] for p in parts[:ntrain]])
    Xv = np.concatenate([p[0] for p in parts[ntrain:]]); Yv = np.concatenate([p[1] for p in parts[ntrain:]]); litv = np.concatenate([p[2] for p in parts[ntrain:]]); del parts
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1); X, Y, lit = X[ok], Y[ok], lit[ok]
    okv = np.isfinite(Xv).all(1) & np.isfinite(Yv).all(1); Xv, Yv, litv = Xv[okv], Yv[okv], litv[okv]
    print(f"loaded train {X.shape[0]:,} columns ({ntrain} ranks), validation {Xv.shape[0]:,} ({a.val_ranks}); {NF} features -> {NT} targets + lit in {time.time()-t0:.0f}s", flush=True)
    print(f"lit columns: {100*lit.mean():.1f}% of train", flush=True)
    x_mean = X.mean(0, dtype=np.float64); x_std = X.std(0, dtype=np.float64); x_abs = np.abs(X).max(0)
    x_std = np.maximum(np.maximum(x_std, 1e-4 * x_abs), 1e-20)
    sw_cols = np.zeros(NT, bool)
    for name, lo, hi in TLAYOUT:
        if name in T.SW_OUTPUTS: sw_cols[lo:hi] = True
    y_mean = np.where(sw_cols, Y[lit].mean(0), Y.mean(0)); y_std = np.where(sw_cols, Y[lit].std(0), Y.std(0))
    y_abs = np.abs(Y).max(0); y_std = np.maximum(y_std, 1e-3 * y_abs); y_std = np.where(y_abs > 1e-30, y_std, 1.0)
    y_min, y_max = Y.min(0), Y.max(0); span = y_max - y_min; y_min_c = y_min - 0.05 * span * (y_min < 0); y_max_c = y_max + 0.05 * span
    Xt = torch.tensor((X - x_mean) / x_std, dtype=torch.float32).clamp_(-20, 20); Yt = torch.tensor((Y - y_mean) / y_std, dtype=torch.float32)
    Xvt = torch.tensor((Xv - x_mean) / x_std, dtype=torch.float32).clamp_(-20, 20); Yvt = torch.tensor((Yv - y_mean) / y_std, dtype=torch.float32)
    litt = torch.tensor(lit, dtype=torch.float32); litvt = torch.tensor(litv, dtype=torch.float32)
    mask = torch.ones((X.shape[0], NT), dtype=torch.float32); mask[:, sw_cols] = litt[:, None]
    maskv = torch.ones((Xv.shape[0], NT), dtype=torch.float32); maskv[:, sw_cols] = litvt[:, None]
    H = a.hidden
    net = torch.nn.Sequential(torch.nn.Linear(NF, H), torch.nn.ReLU(), torch.nn.Linear(H, H), torch.nn.ReLU(), torch.nn.Linear(H, NT + 1))
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-5)
    n = Xt.shape[0]; steps = (n + 2047) // 2048
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * steps, pct_start=0.1)
    bce = torch.nn.BCEWithLogitsLoss()
    def loss_of(idx, Xx, Yx, Mx, Lx):
        out = net(Xx[idx]); y, logit = out[:, :NT], out[:, NT]
        return (Mx[idx] * (y - Yx[idx]) ** 2).sum() / Mx[idx].sum() + bce(logit, Lx[idx])
    for epoch in range(a.epochs):
        net.train(); perm = torch.randperm(n); total = 0.0
        for i in range(0, n, 2048):
            idx = perm[i:i + 2048]; opt.zero_grad()
            loss = loss_of(idx, Xt, Yt, mask, litt); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); total += float(loss.detach()) * len(idx)
        net.eval()
        with torch.no_grad(): vloss = float(loss_of(torch.arange(Xvt.shape[0]), Xvt, Yvt, maskv, litvt))
        print(f"epoch {epoch+1:3d}: train {total/n:.4f}  val {vloss:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    with torch.no_grad():
        out = net(Xvt); pv = out[:, :NT].double().numpy() * y_std + y_mean; plit = (out[:, NT] > 0).numpy()
    pv = np.minimum(np.maximum(pv, y_min_c), y_max_c); pv[:, sw_cols] *= plit[:, None]
    report = {"hidden": H, "parameters": int(nparam), "train_columns": int(n), "val_columns": int(Xv.shape[0]), "features": NF, "targets": NT,
              "epochs": a.epochs, "lit_accuracy": float((plit == litv).mean()), "r2": {}, "rmse": {}}
    for name, lo, hi in TLAYOUT:
        rows = litv if name in T.SW_OUTPUTS else np.ones(len(Yv), bool)
        report["r2"][name] = float(T.r2(pv[rows, lo:hi], Yv[rows, lo:hi])); report["rmse"][name] = float(np.sqrt(((pv[rows, lo:hi] - Yv[rows, lo:hi]) ** 2).mean()))
    print("lit accuracy:", round(report["lit_accuracy"], 4), flush=True)
    print("validation R2:", {k: round(v, 3) for k, v in report["r2"].items()}, flush=True)
    print("validation RMSE:", {k: f"{v:.3g}" for k, v in report["rmse"].items()}, flush=True)
    Path(f"{a.out_prefix}.report.json").write_text(json.dumps(report, indent=1))
    # export: folded float32 weights (the lit logit row unscaled) and the layouts, for the compiled forward
    W1 = net[0].weight.detach().double().numpy(); b1 = net[0].bias.detach().double().numpy()
    W3 = net[4].weight.detach().double().numpy(); b3 = net[4].bias.detach().double().numpy()
    y_std1 = np.concatenate([y_std, [1.0]]); y_mean1 = np.concatenate([y_mean, [0.0]])
    np.savez(f"{a.out_prefix}_weights.npz",
             W1T=np.ascontiguousarray((W1 / x_std).T.astype(np.float32)), b1=(b1 - W1 @ (x_mean / x_std)).astype(np.float32),
             W2T=np.ascontiguousarray(net[2].weight.detach().numpy().T), b2=net[2].bias.detach().numpy(),
             W3T=np.ascontiguousarray((W3 * y_std1[:, None]).T.astype(np.float32)), b3=(b3 * y_std1 + y_mean1).astype(np.float32),
             x_lo=(x_mean - 20 * x_std).astype(np.float32), x_hi=(x_mean + 20 * x_std).astype(np.float32),
             y_min=y_min_c, y_max=y_max_c, sw_cols=sw_cols)
    json.dump({"contract": "block", "features": [list(row) for row in LAYOUT], "targets": [list(row) for row in TLAYOUT], "NF": NF, "NT": NT,
               "lit_index": NT, "PVER": PVER, "log_floor": T.LOG_FLOOR, "sw_outputs": sorted(T.SW_OUTPUTS)}, open(f"{a.out_prefix}_layout.json", "w"), indent=1)
    print(f"saved {a.out_prefix}_weights.npz ({nparam:,} parameters, {4*nparam/1e6:.1f} MB) and the layout", flush=True)
if __name__ == "__main__":
    main()
