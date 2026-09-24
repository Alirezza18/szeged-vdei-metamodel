# ===== CELL R5c-v4 + FIGURES 7/8 FINAL (GPU, ~15 min) =====
# v4 FIXES vs R5c-v3 (both confirmed by R16b):
#   (1) TRUE ENSEMBLE AVERAGING - v3 wrote pred per seed sequentially, so the LAST SEED
#       overwrote the rest while the title said 'ensemble of 5' (R16b cache check
#       max|d|=6.6/3.7 C vs sealed mean = single-seed spread). Now accumulates across
#       seeds and divides; ensemble-consistency check vs the sealed pool is RE-RUN and
#       must pass (max|d| < 0.05 normalized).
#   (2) SEPARATE COLORBARS - v3 colorbarred the last image (error) but labeled it
#       'T_mrt (C)'. Now: one colorbar for the shared truth/pred scale, one for error.
#   (3) canvas() returns a masked array (v4a crash fix: .compressed() needs it).
# Decoding for display: manuscript constants (verified == anchored decoder to 0.1%).
# PRECONDITION: sealed CELL 18b (VDEIDatasetV2) + CELL 19 (build) in THIS session.
import json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
FIG = W / '03_Results' / '03_Figures'; FIG.mkdir(parents=True, exist_ok=True)
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza'); SEEDS = (0, 1, 2, 3, 4); K_PED = 4
DEC = {'T': (28.0, 4.0), 'TMRT': (45.0, 20.0)}       # manuscript constants (verified)
CMAP = {'T': 'viridis', 'TMRT': 'inferno'}
FIGN = {'T': 'fig07', 'TMRT': 'fig08'}
UNITS = {'T': r'$T_a$ ($^\circ$C)', 'TMRT': r'$T_{mrt}$ ($^\circ$C)'}
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'axes.linewidth': 0.7, 'mathtext.fontset': 'stix'})

def find(pat):
    for d_ in sorted(IN.glob('*')):
        h = sorted(d_.rglob(pat))
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
print(f'device: {device} | R5c-v4: TRUE 5-seed ensemble mean, separate colorbars\n')
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
    print(f'  [{SITE}] n_points={n_points:,} n_time={n_time} | z={z_ped:.2f} m | noon=t{t_noon}')

    # ---- TRUE ensemble inference (accumulate over seeds, divide once) ----
    ped = (k0 == K_PED)
    local = {s: np.where(ped & (spl == s))[0] for s in ('train', 'val', 'test')}
    for s, ix in local.items(): print(f'  [{SITE}] k=4 {s}: {len(ix):,} points')
    acc_T = np.zeros(n_points, np.float64); acc_M = np.zeros(n_points, np.float64)
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
        sub = (ix * n_time + t_noon).tolist()
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
                    pa = pa.double().cpu().numpy()
                    acc_T[ix[pos:pos+len(pa)]] += pa[:, 0]
                    acc_M[ix[pos:pos+len(pa)]] += pa[:, 4]
                    pos += len(pa)
        print(f'  [{SITE}] seed {S} inference {time.time()-t0:.0f}s'); del model
    pred_T = (acc_T / len(SEEDS)).astype(np.float64)     # <-- true ensemble MEAN
    pred_TMRT = (acc_M / len(SEEDS)).astype(np.float64)

    # ---- consistency check vs sealed ensemble5 npz (normalized units; MUST pass) ----
    ens = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    if ens is not None:
        ix_te = np.where(ped & (spl == 'test'))[0]
        rws = ix_te * n_time + t_noon
        for vi, key in ((0, 'T'), (4, 'TMRT')):
            sealed = ens['pred_air'][rws, vi].astype(np.float64)
            ours = pred_T[ix_te] if key == 'T' else pred_TMRT[ix_te]
            dmax = float(np.abs(sealed - ours).max())
            print(f'  [{SITE}] ensemble check ({key}): max|d|={dmax:.4f} '
                  f'{"OK" if dmax < 0.05 else "!! MISMATCH - paste back"}')
        ens.close()

    # ---- decode for display (manuscript constants, verified) ----
    mu_T, sd_T = DEC['T']; mu_M, sd_M = DEC['TMRT']
    pred_T = pred_T * sd_T + mu_T
    pred_TMRT = pred_TMRT * sd_M + mu_M
    i_min, i_max = int(bbox['i_min']), int(bbox['i_max'])
    j_min, j_max = int(bbox['j_min']), int(bbox['j_max'])
    ni, nj = i_max - i_min + 1, j_max - j_min + 1
    ii = i0 - i_min; jj = j0 - j_min; m = ped
    def canvas(vals):
        c = np.full((ni, nj), np.nan)
        c[ii[m], jj[m]] = vals[m].astype(np.float64)
        return np.ma.masked_invalid(c)                   # <-- v4a fix (masked array)
    truth = {'T': np.asarray(tf['target_T'])[:, t_noon].astype(np.float64),
             'TMRT': np.asarray(tf['target_TMRT'])[:, t_noon].astype(np.float64)}
    preds = {'T': pred_T, 'TMRT': pred_TMRT}
    bmask = None
    if geom is not None:
        try:
            ib = np.asarray(geom['is_building']); nk = len(zc)
            if ib.ndim == 3:
                ax_k = int(np.argwhere(np.array(ib.shape) == nk).ravel()[0])
                low = np.take(ib, slice(0, min(11, nk)), axis=ax_k).any(axis=ax_k)
                cand = low if low.shape == (ni, nj) else np.take(ib, K_PED, axis=ax_k)
                if cand.shape == (ni, nj): bmask = cand
        except Exception: bmask = None

    stats_site = {'z_m': z_ped, 't_noon': t_noon, 'vars': {}}
    for key in ('T', 'TMRT'):
        R = canvas(truth[key]); P = canvas(preds[key]); E = canvas(preds[key] - truth[key])
        r = R.compressed(); p = P.compressed(); e = E.compressed()
        gr = np.hypot(*np.gradient(R.filled(np.nan)))
        gp = np.hypot(*np.gradient(P.filled(np.nan)))
        okf = np.isfinite(gr) & np.isfinite(gp)
        rgrad = float(np.corrcoef(gr[okf], gp[okf])[0, 1])
        print(f'  [{SITE}] {key} FULL ped slice (n={r.size:,}): '
              f'truth [{r.min():.1f},{r.max():.1f}] pred [{p.min():.1f},{p.max():.1f}] | '
              f'MAE={np.abs(e).mean():.2f} grad-r={rgrad:.3f}')
        vmin = float(min(r.min(), p.min())); vmax = float(max(r.max(), p.max()))
        elim = float(np.percentile(np.abs(e), 98))
        fig = plt.figure(figsize=(13.8, 4.6))
        gs = fig.add_gridspec(1, 3, wspace=0.16, left=0.05, right=0.88)
        cax1 = fig.add_axes([0.895, 0.42, 0.012, 0.5])
        cax2 = fig.add_axes([0.895, 0.08, 0.012, 0.26])
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ext = [j_min*2, (j_max+1)*2, i_min*2, (i_max+1)*2]
        panels = [(R, CMAP[key], vmin, vmax, 'ENVI-met (ground truth)'),
                  (P, CMAP[key], vmin, vmax, 'V-DEI 3D-CNN (5-seed ensemble mean)'),
                  (E, 'RdBu_r', -elim, elim, 'Prediction \u2212 Truth')]
        for ax, (C, cm, v0, v1, ttl) in zip(axes, panels):
            im = ax.imshow(C, origin='lower', extent=ext, cmap=cm, vmin=v0, vmax=v1,
                           interpolation='nearest')
            if bmask is not None:
                ax.imshow(np.ma.masked_invalid(np.where(bmask > 0, 1.0, np.nan)),
                          origin='lower', extent=ext, cmap='gray_r', vmin=0, vmax=1,
                          interpolation='nearest')
            ax.set_title(ttl, fontsize=9.5); ax.set_xlabel('x (m)'); ax.set_facecolor('0.92')
        axes[0].set_ylabel('y (m)')
        cb1 = fig.colorbar(plt.cm.ScalarMappable(            # <-- v4a fix (norm at build)
            cmap=plt.get_cmap(CMAP[key]),
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)), cax=cax1)
        cb1.set_label(UNITS[key])
        cb2 = fig.colorbar(plt.cm.ScalarMappable(
            cmap=plt.get_cmap('RdBu_r'),
            norm=TwoSlopeNorm(vmin=-elim, vcenter=0.0, vmax=elim)), cax=cax2)
        cb2.set_label(f'$\\Delta$ {UNITS[key].split()[0]}')
        fig.suptitle(f'{SITE.capitalize()} \u2013 {UNITS[key]} at pedestrian level '
                     f'(z \u2248 {z_ped:.1f} m), solar noon (t = {t_noon})',
                     fontsize=11, y=1.03)
        out = FIG / f'{FIGN[key]}_spatial_{key}_{SITE}.png'
        fig.savefig(out, bbox_inches='tight'); plt.close(fig)
        print(f'    [SAVED] {out.name}')
        stats_site['vars'][key] = {'truth_min': float(r.min()), 'truth_max': float(r.max()),
                                   'pred_min': float(p.min()), 'pred_max': float(p.max()),
                                   'spatial_MAE': float(np.abs(e).mean()),
                                   'p95_err': float(np.percentile(np.abs(e), 95)),
                                   'grad_r': rgrad}
    summary[SITE] = stats_site
    print(f'  [{SITE}] done in {(time.time()-t_site)/60:.1f} min\n')

(MET / 'fig7_8_spatial_stats_v4.json').write_text(json.dumps(summary, indent=2))
print('===== DONE - paste this entire output back (ensemble checks MUST be OK) =====')
