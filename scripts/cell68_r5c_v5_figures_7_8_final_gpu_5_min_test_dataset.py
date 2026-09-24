# ===== CELL R5c-v5 + FIGURES 7/8 FINAL (GPU, ~5 min) - TEST-DATASET-ONLY INFERENCE =====
# v5 ROOT-CAUSE FIX vs v4: v4's full-domain fields had Tmrt MAE 7.6/8.7 C and grad-r~0.02
#   even though test-point checks passed at 0.001 -> the train/val VDEIDatasetV2 splits
#   return corrupted inputs (train-mode augmentation inside __getitem__ is the prime
#   suspect; the old loop only ever verified TEST rows, so this was never visible).
# v5 STRATEGY: run ALL ped-level indices through the TEST dataset. Evidence this spans the
#   full pool: v4's train/val Subsets used sample indices up to n_points*n_time and did
#   not raise IndexError. Every predicted row is verified against the tf reference; rows
#   that fail (corrupted inputs) are dropped and counted (expect ~0% if the test dataset
#   is clean everywhere; the count is the audit trail).
# Decode: manuscript constants; sanity line prints decoded vs tf extremes (R16e
#   arbitrates the absolute-scale question independently on CPU).
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
DEC = {'T': (28.0, 4.0), 'TMRT': (45.0, 20.0)}
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
print(f'device: {device} | R5c-v5: ALL ped indices via TEST dataset + row verification\n')
AIR_KEYS = [('T', (28.0, 4.0)), ('RelHum', (40.0, 15.0)), ('WindSpd', (2.0, 1.5)),
            ('TKE', (50.0, 100.0)), ('TMRT', (45.0, 20.0))]
FAC_KEYS = ['Twall', 'Qsens', 'SWabs', 'LWbal']

summary = {}
for SITE in SITES:
    t_site = time.time()
    tf = load_npz(f'targets_forcing_{SITE}.npz'); split = load_npz(f'split_{SITE}.npz')
    geom = load_npz(f'geometry_{SITE}.npz'); bbox = load_npz(f'bbox_{SITE}.npz')
    if any(x is None for x in (tf, split)):
        print(f'[{SITE}] missing inputs - skip'); continue
    i0 = np.asarray(tf['i0']); j0 = np.asarray(tf['j0']); k0 = np.asarray(tf['k0'])
    spl = np.asarray(split['split']).astype(str)
    n_points, n_time = tf['target_T'].shape
    zc = np.asarray(geom['z_center']) if geom is not None else None
    z_ped = float(zc[K_PED]) if zc is not None else float('nan')
    t_noon = int(np.argmax(np.asarray(tf['target_TMRT']).mean(axis=0)))
    print(f'  [{SITE}] n_points={n_points:,} n_time={n_time} | z={z_ped:.2f} m | noon=t{t_noon}')

    # ---- ALL ped-level indices through the TEST dataset ----
    ped = (k0 == K_PED)
    all_ped = np.where(ped)[0]
    ds = VDEIDatasetV2(SITE, 'test')
    sub = (all_ped * n_time + t_noon).tolist()
    loader = DataLoader(Subset(ds, sub), batch_size=256, shuffle=False, num_workers=2)
    print(f'  [{SITE}] all-ped indices via TEST dataset: {len(sub):,} samples')

    sum_T = np.zeros(n_points); sum_M = np.zeros(n_points); cnt = 0
    for S in SEEDS:
        p = find(f'{SITE}_dualhead_optuna_best_seed{S}_best.pt') \
            or find(f'{SITE}_dualhead_p5a_best_seed{S}_best.pt')
        if p is None: print(f'  !! seed {S} checkpoint missing - skip'); continue
        ck = torch.load(p, map_location='cpu', weights_only=False)
        model, _ = build(ck.get('config', 'optuna_best'))
        model.load_state_dict(ck['model_state']); model.to(device).eval()
        t0 = time.time()
        with torch.no_grad():
            pos = 0
            for b in loader:
                pa, _ = model(b['vdei'].to(device), b['forcing'].to(device))
                pa = pa.double().cpu().numpy()
                sum_T[all_ped[pos:pos+len(pa)]] += pa[:, 0]
                sum_M[all_ped[pos:pos+len(pa)]] += pa[:, 4]
                pos += len(pa)
        cnt += 1
        print(f'  [{SITE}] seed {S} inference {time.time()-t0:.0f}s'); del model
    pred_T = sum_T / cnt
    pred_TMRT = sum_M / cnt

    # ---- row verification vs tf reference (drops corrupted-input rows) ----
    mu_T, sd_T = DEC['T']; mu_M, sd_M = DEC['TMRT']
    dec_T = pred_T * sd_T + mu_T
    dec_M = pred_TMRT * sd_M + mu_M
    tv_T = np.asarray(tf['target_T'][all_ped, t_noon], np.float64)
    tv_M = np.asarray(tf['target_TMRT'][all_ped, t_noon], np.float64)
    bad = (np.abs(dec_T[all_ped] - tv_T) > 10.0) | (np.abs(dec_M[all_ped] - tv_M) > 30.0)
    n_bad = int(bad.sum())
    print(f'  [{SITE}] row verification vs tf: dropped {n_bad:,}/{len(all_ped):,} '
          f'({100*n_bad/len(all_ped):.1f}%) '
          f'{"- CLEAN" if n_bad == 0 else "- inspect drop pattern"}')
    dec_T[all_ped[bad]] = np.nan; dec_M[all_ped[bad]] = np.nan
    print(f'  [{SITE}] decode sanity: pred Tmrt max={np.nanmax(dec_M):.1f} vs tf '
          f'{tv_M.max():.1f} | pred Ta max={np.nanmax(dec_T):.1f} vs tf {tv_T.max():.1f} '
          f'(R16e arbitrates absolute scale)')

    # ---- canvases + figures ----
    i_min, i_max = int(bbox['i_min']), int(bbox['i_max'])
    j_min, j_max = int(bbox['j_min']), int(bbox['j_max'])
    ni, nj = i_max - i_min + 1, j_max - j_min + 1
    ii = i0 - i_min; jj = j0 - j_min; m = ped
    def canvas(vals):
        c = np.full((ni, nj), np.nan)
        c[ii[m], jj[m]] = vals[m].astype(np.float64)
        return np.ma.masked_invalid(c)
    truth = {'T': np.asarray(tf['target_T'])[:, t_noon].astype(np.float64),
             'TMRT': np.asarray(tf['target_TMRT'])[:, t_noon].astype(np.float64)}
    preds = {'T': dec_T, 'TMRT': dec_M}
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

    stats_site = {'z_m': z_ped, 't_noon': t_noon, 'dropped_rows': n_bad, 'vars': {}}
    for key in ('T', 'TMRT'):
        R = canvas(truth[key]); Pd = canvas(preds[key]); Ee = canvas(preds[key] - truth[key])
        r = R.compressed(); p = Pd.compressed(); e = Ee.compressed()
        gr = np.hypot(*np.gradient(R.filled(np.nan)))
        gp = np.hypot(*np.gradient(Pd.filled(np.nan)))
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
                  (Pd, CMAP[key], vmin, vmax, 'V-DEI 3D-CNN (5-seed ensemble mean)'),
                  (Ee, 'RdBu_r', -elim, elim, 'Prediction \u2212 Truth')]
        for ax, (C, cm, v0, v1, ttl) in zip(axes, panels):
            ax.imshow(C, origin='lower', extent=ext, cmap=cm, vmin=v0, vmax=v1,
                      interpolation='nearest')
            if bmask is not None:
                ax.imshow(np.ma.masked_invalid(np.where(bmask > 0, 1.0, np.nan)),
                          origin='lower', extent=ext, cmap='gray_r', vmin=0, vmax=1,
                          interpolation='nearest')
            ax.set_title(ttl, fontsize=9.5); ax.set_xlabel('x (m)'); ax.set_facecolor('0.92')
        axes[0].set_ylabel('y (m)')
        cb1 = fig.colorbar(plt.cm.ScalarMappable(
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

(MET / 'fig7_8_spatial_stats_v5.json').write_text(json.dumps(summary, indent=2))
print('===== DONE - paste this entire output back =====')
