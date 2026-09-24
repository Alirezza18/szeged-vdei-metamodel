# ===== CELL P6-FC R5c-v3 + FIGURES 7/8 (CPU) - FULL-DOMAIN 3-panel maps (FIXED + CACHED) =====
# PRECONDITION: run sealed CELL 18b (VDEIDatasetV2) and CELL 19 (build) first in THIS session.
# v3 FIX: VDEIDatasetV2.__getitem__ reads the GLOBAL AIR_KEYS and expects TUPLE form
#         [('T',(28.0,4.0)),...]; CELL 27 v4 overwrites it with plain strings ->
#         "not enough values to unpack". We restore tuple-form keys BEFORE any dataset use.
# v2 FIX: dataset is indexed by SAMPLE = point*n_time + t -> Subset uses ix*n_time + t_noon.
# Predictions cached to /kaggle/working/cache (re-runs skip inference).
import json, time, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
FIG = W / '03_Results' / '03_Figures'; FIG.mkdir(parents=True, exist_ok=True)
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
CACHE = W / 'cache'; CACHE.mkdir(exist_ok=True)
SITES = ('canyon', 'plaza'); SEEDS = (0, 1, 2, 3, 4); K_PED = 4
AIR = {'T': (28.0, 4.0), 'TMRT': (45.0, 20.0)}
CMAP = {'T': 'viridis', 'TMRT': 'inferno'}; FIGN = {'T': 'fig07', 'TMRT': 'fig08'}
UNITS = {'T': r'$T_a$ (°C)', 'TMRT': r'$T_{mrt}$ (°C)'}
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'mathtext.fontset': 'stix'})

def find(pat):
    for d in sorted(IN.glob('*')):
        h = sorted(d.rglob(pat))
        if h: return h[0]
    return None

def load_npz(pat):
    p = find(pat)
    if p is None: print(f'  !! missing: {pat}'); return None
    print(f'  [{pat}] <- {p.parent.parent.name}/{p.parent.name}/{p.name}')
    return np.load(p)

if 'VDEIDatasetV2' not in globals() or 'build' not in globals():
    raise SystemExit('!! Run sealed CELL 18b + CELL 19 first in THIS session, then re-run.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device} | k={K_PED} full pedestrian slice, ensemble of 5, cached\n')

# --- RESTORE tuple-form keys (VDEIDatasetV2.__getitem__ reads the GLOBAL AIR_KEYS;
#     CELL 27 v4 overwrites it with plain strings -> crash). Sealed canonical form: ---
AIR_KEYS = [('T', (28.0, 4.0)), ('RelHum', (40.0, 15.0)), ('WindSpd', (2.0, 1.5)),
            ('TKE', (50.0, 100.0)), ('TMRT', (45.0, 20.0))]
FAC_KEYS = ['Twall', 'Qsens', 'SWabs', 'LWbal']

summary = {}
for SITE in SITES:
    t_site = time.time()
    tf = load_npz(f'targets_forcing_{SITE}.npz'); split = load_npz(f'split_{SITE}.npz')
    geom = load_npz(f'geometry_{SITE}.npz'); bbox = load_npz(f'bbox_{SITE}.npz')
    if tf is None or split is None: print(f'[{SITE}] missing inputs - skip'); continue
    i0 = np.asarray(tf['i0']); j0 = np.asarray(tf['j0']); k0 = np.asarray(tf['k0'])
    spl = np.asarray(split['split']).astype(str)
    n_points, n_time = tf['target_T'].shape
    zc = np.asarray(geom['z_center']) if geom is not None else None
    z_ped = float(zc[K_PED]) if zc is not None else float('nan')
    t_noon = int(np.argmax(np.asarray(tf['target_TMRT']).mean(axis=0)))
    print(f'  [{SITE}] n_points={n_points:,} n_time={n_time} | k={K_PED} -> z={z_ped:.2f} m | noon=t{t_noon}')

    # ---- predictions: cache or CPU inference ----
    cpath = CACHE / f'{SITE}_k4noon_preds.npz'
    pred_T = pred_TMRT = None
    if cpath.exists():
        z = np.load(cpath)
        if int(z['t_noon']) == t_noon:
            pred_T = z['pred_T']; pred_TMRT = z['pred_TMRT']
            print(f'  [{SITE}] CACHE hit - inference skipped')
        z.close()
    if pred_T is None:
        ped = (k0 == K_PED)
        local = {s: np.where(ped & (spl == s))[0] for s in ('train', 'val', 'test')}
        for s, ix in local.items(): print(f'  [{SITE}] k=4 {s}: {len(ix):,} points')
        pred_T = np.full(n_points, np.nan, dtype=np.float32)
        pred_TMRT = np.full(n_points, np.nan, dtype=np.float32)
        ckpts = []
        for S in SEEDS:
            p = find(f'{SITE}_dualhead_optuna_best_seed{S}_best.pt') \
                or find(f'{SITE}_dualhead_p5a_best_seed{S}_best.pt')
            ckpts.append(p)
        if any(p is None for p in ckpts): print(f'  !! [{SITE}] checkpoints missing - skip'); continue
        loaders = {}
        for s, ix in local.items():
            if len(ix) == 0: continue
            ds = VDEIDatasetV2(SITE, s)
            sub = (ix * n_time + t_noon).tolist()          # sample index = point*n_time + t
            loaders[s] = (ix, DataLoader(Subset(ds, sub), batch_size=256, shuffle=False,
                                         num_workers=2))
        for S, ck_path in zip(SEEDS, ckpts):
            t0 = time.time()
            ck = torch.load(ck_path, map_location='cpu', weights_only=False)
            model, _ = build(ck.get('config', 'optuna_best'))
            model.load_state_dict(ck['model_state']); model.to(device).eval()
            with torch.no_grad():
                for s, (ix, loader) in loaders.items():
                    pos = 0
                    for b in loader:
                        pa, _ = model(b['vdei'].to(device), b['forcing'].to(device))
                        pa = pa.float().cpu().numpy()
                        pred_T[ix[pos:pos+len(pa)]] = pa[:, 0]
                        pred_TMRT[ix[pos:pos+len(pa)]] = pa[:, 4]
                        pos += len(pa)
            print(f'  [{SITE}] seed {S} inference {time.time()-t0:.0f}s'); del model
        np.savez(cpath, pred_T=pred_T, pred_TMRT=pred_TMRT, t_noon=t_noon)
        print(f'  [{SITE}] [CACHED] {cpath.name}')

    # ---- consistency check vs sealed ensemble5 npz (test points, noon) ----
    ens = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    if ens is not None:
        ped = (k0 == K_PED); ix_te = np.where(ped & (spl == 'test'))[0]
        rows = ix_te * n_time + t_noon
        for vi, key in ((0, 'T'), (4, 'TMRT')):
            sealed = ens['pred_air'][rows, vi].astype(np.float32)
            ours = (pred_T if key == 'T' else pred_TMRT)[ix_te]
            dmax = float(np.nanmax(np.abs(sealed - ours)))
            print(f'  [{SITE}] ensemble check ({key}): max|d|={dmax:.4f} '
                  f'{"OK" if dmax < 0.05 else "!! MISMATCH - paste output back"}')
        ens.close()

    # ---- 2D canvases + figures ----
    pred_T = pred_T * AIR['T'][1] + AIR['T'][0]
    pred_TMRT = pred_TMRT * AIR['TMRT'][1] + AIR['TMRT'][0]
    i_min, i_max = int(bbox['i_min']), int(bbox['i_max'])
    j_min, j_max = int(bbox['j_min']), int(bbox['j_max'])
    ni, nj = i_max - i_min + 1, j_max - j_min + 1
    ii = i0 - i_min; jj = j0 - j_min; m = (k0 == K_PED)
    def canvas(vals):
        c = np.full((ni, nj), np.nan, dtype=np.float64)
        c[ii[m], jj[m]] = vals[m].astype(np.float64)
        return np.ma.masked_invalid(c)
    truth = {'T': np.asarray(tf['target_T'])[:, t_noon], 'TMRT': np.asarray(tf['target_TMRT'])[:, t_noon]}
    preds = {'T': pred_T, 'TMRT': pred_TMRT}
    bmask = None
    if geom is not None:
        try:
            ib = np.asarray(geom['is_building']); nk = len(zc)
            if ib.ndim == 3:
                ax_k = int(np.argwhere(np.array(ib.shape) == nk).ravel()[0])
                cand = np.take(ib, K_PED, axis=ax_k)
                if cand.shape == (ni, nj): bmask = cand
        except Exception: bmask = None

    stats_site = {'z_m': z_ped, 't_noon': t_noon, 'vars': {}}
    for key in ('T', 'TMRT'):
        R = canvas(truth[key]); P = canvas(preds[key]); E = canvas(preds[key] - truth[key])
        r = R.compressed(); p = P.compressed(); e = E.compressed()
        gr = np.hypot(np.gradient(R.filled(np.nan), axis=0), np.gradient(R.filled(np.nan), axis=1))
        gp = np.hypot(np.gradient(P.filled(np.nan), axis=0), np.gradient(P.filled(np.nan), axis=1))
        ok = np.isfinite(gr) & np.isfinite(gp)
        rgrad = float(np.corrcoef(gr[ok], gp[ok])[0, 1]) if ok.sum() > 10 else float('nan')
        print(f'  [{SITE}] {key} FULL k=4 slice (n={r.size:,}):')
        print(f'    truth range=[{r.min():.2f},{r.max():.2f}] mean={r.mean():.2f} sd={r.std():.2f}')
        print(f'    pred  range=[{p.min():.2f},{p.max():.2f}] mean={p.mean():.2f} sd={p.std():.2f}')
        print(f'    MAE={np.abs(e).mean():.3f} RMSE={np.sqrt((e**2).mean()):.3f} '
              f'p95={np.percentile(np.abs(e), 95):.3f} grad-r={rgrad:.3f}')
        vmin = float(min(r.min(), p.min())); vmax = float(max(r.max(), p.max()))
        elim = float(np.percentile(np.abs(e), 98))
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
        ext = [j_min*2, (j_max+1)*2, i_min*2, (i_max+1)*2]
        for ax, (C, cm, v0, v1, ttl) in zip(axes, [
                (R, CMAP[key], vmin, vmax, 'ENVI-met (ground truth)'),
                (P, CMAP[key], vmin, vmax, 'V-DEI 3D-CNN (ensemble of 5)'),
                (E, 'RdBu_r', -elim, elim, 'Prediction − Truth')]):
            im = ax.imshow(C, origin='lower', extent=ext, cmap=cm, vmin=v0, vmax=v1,
                           interpolation='nearest')
            if bmask is not None:
                ax.imshow(np.ma.masked_invalid(np.where(bmask > 0, 1.0, np.nan)),
                          origin='lower', extent=ext, cmap='gray_r', vmin=0, vmax=1,
                          interpolation='nearest')
            ax.set_title(ttl); ax.set_xlabel('x (m)'); ax.set_facecolor('0.92')
        axes[0].set_ylabel('y (m)')
        cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015); cb.set_label(UNITS[key])
        fig.suptitle(f'{SITE.capitalize()} – {UNITS[key]} at pedestrian level (z ≈ {z_ped:.1f} m), '
                     f'solar noon (t = {t_noon})', fontsize=11, y=1.02)
        out = FIG / f'{FIGN[key]}_spatial_{key}_{SITE}.png'
        fig.savefig(out, bbox_inches='tight'); plt.close(fig)
        print(f'    [SAVED] {out.name}')
        stats_site['vars'][key] = {'truth_mean': float(r.mean()), 'truth_sd': float(r.std()),
                                   'pred_mean': float(p.mean()), 'pred_sd': float(p.std()),
                                   'spatial_MAE': float(np.abs(e).mean()),
                                   'p95_err': float(np.percentile(np.abs(e), 95)), 'grad_r': rgrad}
    summary[SITE] = stats_site
    print(f'  [{SITE}] done in {(time.time()-t_site)/60:.1f} min\n')

(MET / 'fig7_8_spatial_stats.json').write_text(json.dumps(summary, indent=2))
print('===== DONE - paste this entire output back =====')
