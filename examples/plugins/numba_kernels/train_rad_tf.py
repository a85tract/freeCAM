"""Train a level-token transformer emulator of radiation_tend's computing branch, export it as TorchScript for the slot.
    train_rad_tf.py <capture-dir> <out-prefix> [--d 64] [--layers 2] [--epochs 30] [--threads 64] [--val-ranks 64] [--max-ranks N]
Tokens: the 30 model levels, each carrying 34 features (the 12 profiles, ozone, 15 aerosol mixing ratios, 3 wet
diameters, 3 aerosol waters; heavy-tailed ones as log10) plus 8 column scalars broadcast; a learned position per
level.  Heads: per level qrs and qrl, from the mean token the ten fluxes.  Shortwave targets learned on lit columns
and zeroed in the dark.  The export is one TorchScript module with the slot's 46-tensor signature
(radiation_process.TABLE_INPUTS) returning the 12 outputs, float64 in and out, frozen and optimised.
"""
import argparse, glob, importlib.util, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import os
# this checkout: the file's place under examples/plugins/numba_kernels, else FREECAM_REPO, else the working directory
REPO = Path(__file__).resolve().parents[3] if (Path(__file__).resolve().parents[3] / "src").is_dir() else Path(os.environ.get("FREECAM_REPO", Path.cwd()))
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "examples/plugins/numba_kernels"))
import train_rad as T                                   # the capture loader and the flat feature layout
from freecam.physics.radiation_process import TABLE_INPUTS, TABLE_OUTPUTS
PVER = T.PVER; NPROF = 33; NLEV_FEAT = 34; NSCAL = 8; NF_TOK = NLEV_FEAT + NSCAL
assert T.NF == NPROF * PVER + (PVER + 1) + NSCAL
TABLE_NAMES = [n for n, _ in TABLE_INPUTS]

def tokens(X):
    """(n, 1029) flat features -> (n, 30, 34) level tokens and (n, 8) scalars, in the wrapper's feature order."""
    n = X.shape[0]
    prof = X[:, :NPROF * PVER].reshape(n, NPROF, PVER).transpose(0, 2, 1)          # (n, 30, 33): 12 basic, 15 aer, 3 dg, 3 qw
    o3 = X[:, NPROF * PVER:NPROF * PVER + PVER][:, :, None]                          # levels 1..30 of the 31
    lev = np.concatenate([prof[:, :, :12], o3, prof[:, :, 12:]], axis=2)             # 12 + 1 + 15 + 3 + 3 = 34
    return lev.astype(np.float32), X[:, NPROF * PVER + PVER + 1:].astype(np.float32)

def wrapper_source():
    args = ", ".join(TABLE_NAMES); ann = ", ".join(["Tensor"] * len(TABLE_NAMES)); ret = ", ".join(["Tensor"] * len(TABLE_OUTPUTS))
    return f'''
import torch
from typing import Tuple
class RadiationTransformer(torch.nn.Module):
    """The slot's 46 inputs as tensors (Fortran index order) -> the 12 outputs; the network inside is float32."""
    def __init__(self, net, x_mean, x_std, s_mean, s_std, y_mean_lev, y_std_lev, y_mean_col, y_std_col, y_lo_lev, y_hi_lev, y_lo_col, y_hi_col):
        super().__init__()
        self.net = net
        for k, v in dict(x_mean=x_mean, x_std=x_std, s_mean=s_mean, s_std=s_std, y_mean_lev=y_mean_lev, y_std_lev=y_std_lev,
                         y_mean_col=y_mean_col, y_std_col=y_std_col, y_lo_lev=y_lo_lev, y_hi_lev=y_hi_lev, y_lo_col=y_lo_col, y_hi_col=y_hi_col).items():
            self.register_buffer(k, v)
    def forward(self, {args}):
        # type: ({ann}) -> Tuple[{ret}]
        floor = 1e-30
        lev = torch.stack([state_t, torch.log10(state_pmid.clamp_min(0.0) + floor), torch.log10(state_q[:, :, 0].clamp_min(0.0) + floor),
                           cld, torch.log10(iclwp.clamp_min(0.0) + floor), torch.log10(iciwp.clamp_min(0.0) + floor), dei, mu,
                           torch.log10(lambdac.clamp_min(0.0) + floor), cldfsnow, torch.log10(icswp.clamp_min(0.0) + floor), des,
                           torch.log10(rstate_o3vmr[:, :{PVER}].clamp_min(0.0) + floor)], dim=2)
        aer = torch.log10(state_q[:, :, 42:57].clamp_min(0.0) + floor)
        dg = torch.log10(dgnumwet.clamp_min(0.0) + floor); qw = torch.log10(qaerwat.clamp_min(0.0) + floor)
        x = torch.cat([lev, aer, dg, qw], dim=2)
        x = torch.nan_to_num((x - self.x_mean) / self.x_std, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
        n = state_t.shape[0]
        s = torch.stack([coszrs, cam_in_asdir, cam_in_asdif, cam_in_aldir, cam_in_aldif, cam_in_lwup, clat, calday.expand(n)], dim=1)
        s = torch.nan_to_num((s - self.s_mean) / self.s_std, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
        tok = torch.cat([x, s.unsqueeze(1).expand(n, {PVER}, {NSCAL})], dim=2).to(torch.float32)
        y_lev, y_col = self.net(tok)
        y_lev = torch.minimum(torch.maximum(y_lev.to(torch.float64) * self.y_std_lev + self.y_mean_lev, self.y_lo_lev), self.y_hi_lev)
        y_col = torch.minimum(torch.maximum(y_col.to(torch.float64) * self.y_std_col + self.y_mean_col, self.y_lo_col), self.y_hi_col)
        lit = (coszrs > 0.0).to(torch.float64)
        qrs = y_lev[:, :, 0] * lit.unsqueeze(1); qrl = y_lev[:, :, 1]
        fsns = y_col[:, 0] * lit; fsnt = y_col[:, 1] * lit; flns = y_col[:, 2]; flnt = y_col[:, 3]; fsds = y_col[:, 4] * lit
        sols = y_col[:, 5] * lit; soll = y_col[:, 6] * lit; solsd = y_col[:, 7] * lit; solld = y_col[:, 8] * lit; flwds = y_col[:, 9]
        return (qrs, qrl, fsns, fsnt, flns, flnt, fsds, sols, soll, solsd, solld, flwds)
'''

def build_net(torch, d, layers):
    class LevelTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Linear(NF_TOK, d)
            self.pos = torch.nn.Parameter(torch.zeros(1, PVER, d))
            layer = torch.nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.enc = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.head_lev = torch.nn.Linear(d, 2)
            self.head_col = torch.nn.Linear(d, 10)
        def forward(self, tok):
            h = self.enc(self.embed(tok) + self.pos)
            return self.head_lev(h), self.head_col(h.mean(1))
    return LevelTransformer()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("capture_dir"); ap.add_argument("out_prefix")
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--layers", type=int, default=2); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=64); ap.add_argument("--val-ranks", type=int, default=64); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-ranks", type=int, default=None); ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    t0 = time.time()
    files = sorted(glob.glob(f"{a.capture_dir}/radiation_tend.rank-*.npz"))[: a.max_ranks]
    with ProcessPoolExecutor(a.workers) as pool: parts = list(pool.map(T.load_rank, files))
    ntrain = len(files) - a.val_ranks
    X = np.concatenate([p[0] for p in parts[:ntrain]]); Y = np.concatenate([p[1] for p in parts[:ntrain]])
    Xv = np.concatenate([p[0] for p in parts[ntrain:]]); Yv = np.concatenate([p[1] for p in parts[ntrain:]]); del parts
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1); X, Y = X[ok], Y[ok]
    okv = np.isfinite(Xv).all(1) & np.isfinite(Yv).all(1); Xv, Yv = Xv[okv], Yv[okv]
    L, Sc = tokens(X); Lv, Scv = tokens(Xv); del X, Xv
    print(f"loaded train {L.shape[0]:,} columns ({ntrain} ranks), validation {Lv.shape[0]:,} ({a.val_ranks}); tokens {L.shape[1:]} + scalars {Sc.shape[1]} in {time.time()-t0:.0f}s", flush=True)
    cosz_i = T.SCALARS.index("coszrs"); lit = Sc[:, cosz_i] > 0; litv = Scv[:, cosz_i] > 0
    # feature statistics per level feature (over columns and levels) and per scalar
    x_mean = L.reshape(-1, NLEV_FEAT).mean(0, dtype=np.float64); x_std = L.reshape(-1, NLEV_FEAT).std(0, dtype=np.float64)
    x_std = np.maximum(np.maximum(x_std, 1e-4 * np.abs(L.reshape(-1, NLEV_FEAT)).max(0)), 1e-20)
    s_mean = Sc.mean(0, dtype=np.float64); s_std = np.maximum(np.maximum(Sc.std(0, dtype=np.float64), 1e-4 * np.abs(Sc).max(0)), 1e-20)
    # targets: (n, 30, 2) levels and (n, 10) column, statistics per level and target (SW on lit columns)
    Ylev = np.stack([Y[:, 0:PVER], Y[:, PVER:2 * PVER]], axis=2); Ycol = Y[:, 2 * PVER:]
    Ylevv = np.stack([Yv[:, 0:PVER], Yv[:, PVER:2 * PVER]], axis=2); Ycolv = Yv[:, 2 * PVER:]
    col_names = T.OUT_SCALARS; sw_col = np.array([n in T.SW_OUTPUTS for n in col_names])
    ym_lev = np.stack([Ylev[lit, :, 0].mean(0), Ylev[:, :, 1].mean(0)], 1); ys_lev = np.stack([Ylev[lit, :, 0].std(0), Ylev[:, :, 1].std(0)], 1)
    ys_lev = np.maximum(ys_lev, 1e-3 * np.abs(Ylev).reshape(-1, 2).max(0)); ys_lev = np.where(np.abs(Ylev).reshape(-1, 2).max(0) > 1e-30, ys_lev, 1.0)
    ym_col = np.where(sw_col, Ycol[lit].mean(0), Ycol.mean(0)); ys_col = np.where(sw_col, Ycol[lit].std(0), Ycol.std(0))
    ys_col = np.maximum(ys_col, 1e-3 * np.abs(Ycol).max(0)); ys_col = np.where(np.abs(Ycol).max(0) > 1e-30, ys_col, 1.0)
    def bounds(v):
        lo, hi = v.min(0), v.max(0); span = hi - lo; return lo - 0.05 * span * (lo < 0), hi + 0.05 * span
    ylo_lev, yhi_lev = bounds(Ylev); ylo_col, yhi_col = bounds(Ycol)
    f32 = lambda v: torch.tensor(np.asarray(v, np.float32)); f64 = lambda v: torch.tensor(np.asarray(v, np.float64))
    Lt = ((torch.tensor(L) - f32(x_mean)) / f32(x_std)).clamp_(-20, 20); St = ((torch.tensor(Sc) - f32(s_mean)) / f32(s_std)).clamp_(-20, 20)
    Lvt = ((torch.tensor(Lv) - f32(x_mean)) / f32(x_std)).clamp_(-20, 20); Svt = ((torch.tensor(Scv) - f32(s_mean)) / f32(s_std)).clamp_(-20, 20)
    Ylt = (torch.tensor(Ylev, dtype=torch.float32) - f32(ym_lev)) / f32(ys_lev); Yct = (torch.tensor(Ycol, dtype=torch.float32) - f32(ym_col)) / f32(ys_col)
    Ylvt = (torch.tensor(Ylevv, dtype=torch.float32) - f32(ym_lev)) / f32(ys_lev); Ycvt = (torch.tensor(Ycolv, dtype=torch.float32) - f32(ym_col)) / f32(ys_col)
    litt = torch.tensor(lit, dtype=torch.float32); litvt = torch.tensor(litv, dtype=torch.float32)
    def masks(litm, n):
        ml = torch.ones((n, PVER, 2), dtype=torch.float32); ml[:, :, 0] = litm[:, None]
        mc = torch.ones((n, 10), dtype=torch.float32); mc[:, sw_col] = litm[:, None]
        return ml, mc
    ML, MC = masks(litt, L.shape[0]); MLv, MCv = masks(litvt, Lv.shape[0])
    net = build_net(torch, a.d, a.layers); nparam = sum(p.numel() for p in net.parameters())
    print(f"model d={a.d} layers={a.layers}: {nparam:,} parameters ({4*nparam/1e6:.2f} MB)", flush=True)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-5)
    n = Lt.shape[0]; steps = (n + a.batch - 1) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * steps, pct_start=0.1)
    def batch_loss(idx, Lx, Sx, Yl, Yc, Ml, Mc):
        tok = torch.cat([Lx[idx], Sx[idx].unsqueeze(1).expand(-1, PVER, -1)], dim=2)
        pl, pc = net(tok)
        return ((Ml[idx] * (pl - Yl[idx]) ** 2).sum() + (Mc[idx] * (pc - Yc[idx]) ** 2).sum()) / (Ml[idx].sum() + Mc[idx].sum())
    for epoch in range(a.epochs):
        net.train(); perm = torch.randperm(n); total = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]; opt.zero_grad()
            loss = batch_loss(idx, Lt, St, Ylt, Yct, ML, MC); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); total += float(loss.detach()) * len(idx)
        net.eval()
        with torch.no_grad():
            vl = sum(float(batch_loss(torch.arange(i, min(i + 4096, Lvt.shape[0])), Lvt, Svt, Ylvt, Ycvt, MLv, MCv)) * min(4096, Lvt.shape[0] - i) for i in range(0, Lvt.shape[0], 4096)) / Lvt.shape[0]
        print(f"epoch {epoch+1:3d}: train {total/n:.4f}  val {vl:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    # validation in physical units
    with torch.no_grad():
        pls, pcs = [], []
        for i in range(0, Lvt.shape[0], 4096):
            tok = torch.cat([Lvt[i:i + 4096], Svt[i:i + 4096].unsqueeze(1).expand(-1, PVER, -1)], dim=2); pl, pc = net(tok); pls.append(pl); pcs.append(pc)
        pl = torch.cat(pls).double().numpy() * ys_lev + ym_lev; pc = torch.cat(pcs).double().numpy() * ys_col + ym_col
    pl = np.clip(pl, ylo_lev, yhi_lev); pc = np.clip(pc, ylo_col, yhi_col); pl[:, :, 0] *= litv[:, None]; pc[:, sw_col] *= litv[:, None]
    report = {"d": a.d, "layers": a.layers, "parameters": int(nparam), "train_columns": int(n), "val_columns": int(Lv.shape[0]), "epochs": a.epochs, "r2": {}, "rmse": {}}
    for name, pred, true, rows in (("qrs", pl[:, :, 0], Ylevv[:, :, 0], litv), ("qrl", pl[:, :, 1], Ylevv[:, :, 1], np.ones(len(litv), bool))):
        report["r2"][name] = float(T.r2(pred[rows], true[rows])); report["rmse"][name] = float(np.sqrt(((pred[rows] - true[rows]) ** 2).mean()))
    for j, name in enumerate(col_names):
        rows = litv if sw_col[j] else np.ones(len(litv), bool)
        report["r2"][name] = float(T.r2(pc[rows, j:j + 1], Ycolv[rows, j:j + 1])); report["rmse"][name] = float(np.sqrt(((pc[rows, j] - Ycolv[rows, j]) ** 2).mean()))
    print("validation R2:", {k: round(v, 3) for k, v in report["r2"].items()}, flush=True)
    print("validation RMSE:", {k: f"{v:.3g}" for k, v in report["rmse"].items()}, flush=True)
    torch.save({"net": net.state_dict(), "d": a.d, "layers": a.layers}, f"{a.out_prefix}.ckpt.pt")
    # export: the slot's signature around the network
    src_path = Path(f"{a.out_prefix}.src.py"); src_path.write_text(wrapper_source())
    spec_ = importlib.util.spec_from_file_location("rad_tf_src", src_path); mod = importlib.util.module_from_spec(spec_); sys.modules["rad_tf_src"] = mod; spec_.loader.exec_module(mod)
    wrapper = mod.RadiationTransformer(net.eval(), f64(x_mean), f64(x_std), f64(s_mean), f64(s_std), f64(ym_lev), f64(ys_lev), f64(ym_col), f64(ys_col),
                                       f64(ylo_lev), f64(yhi_lev), f64(ylo_col), f64(yhi_col)).eval()
    scripted = torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.script(wrapper)))
    scripted.save(f"{a.out_prefix}.pt")
    # smoke: one captured chunk through the saved module versus the eager network on the same features
    z = np.load(files[-1], allow_pickle=True); meta = json.loads(str(z["meta"])); i0 = 0; ncol = int(meta[i0]["ncol"])
    ins = []
    for name, rank in TABLE_INPUTS:
        if rank == 0: ins.append(torch.tensor([float(meta[i0][name])], dtype=torch.float64))
        else:
            v = np.asarray(z[f"in/{i0}/{name}"], np.float64); v = np.nan_to_num(v)
            if v.shape[0] < 16: v = np.concatenate([v, np.zeros((16 - v.shape[0],) + v.shape[1:])])
            ins.append(torch.tensor(v))
    loaded = torch.jit.load(f"{a.out_prefix}.pt"); torch.set_num_threads(1)
    with torch.no_grad():
        outs = loaded(*ins)
        for _ in range(5): loaded(*ins)
        t1 = time.perf_counter(); N = 100
        for _ in range(N): loaded(*ins)
        dt = (time.perf_counter() - t1) / N
        Xc, _ = T.load_rank(files[-1]); Lc, Scc = tokens(Xc[:ncol])
        tok = torch.cat([((torch.tensor(Lc) - f32(x_mean)) / f32(x_std)).clamp(-20, 20), ((torch.tensor(Scc) - f32(s_mean)) / f32(s_std)).clamp(-20, 20).unsqueeze(1).expand(-1, PVER, -1)], dim=2)
        el, ec = net(tok); el = np.clip(el.double().numpy() * ys_lev + ym_lev, ylo_lev, yhi_lev); ec = np.clip(ec.double().numpy() * ys_col + ym_col, ylo_col, yhi_col)
        lit0 = Scc[:, cosz_i] > 0; el[:, :, 0] *= lit0[:, None]; ec[:, sw_col] *= lit0[:, None]
    diff = max(float(np.abs(outs[0].numpy()[:ncol] - el[:, :, 0]).max()), float(np.abs(outs[1].numpy()[:ncol] - el[:, :, 1]).max()),
               max(float(np.abs(outs[2 + j].numpy()[:ncol] - ec[:, j]).max()) for j in range(10)))
    report["scripted_vs_eager_max_abs"] = diff; report["scripted_forward_ms_16_columns_1_thread"] = dt * 1e3
    Path(f"{a.out_prefix}.report.json").write_text(json.dumps(report, indent=1))
    print(f"saved {a.out_prefix}.pt; scripted vs eager on a captured chunk: max |diff| {diff:.3g}; forward {dt*1e3:.2f} ms per 16 columns single-thread", flush=True)
if __name__ == "__main__":
    main()
