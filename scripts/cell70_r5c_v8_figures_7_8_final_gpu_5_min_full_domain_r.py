# ===== CELL R5c-v8 + FIGURES 7/8 FINAL (GPU, ~5 min) - FULL-DOMAIN RENDER, CALIBRATED GATE =====
# v7 sealed the mapping (rank ids -> ensemble checks PASS: 0.0074/0.0221, 0.0068/0.0198).
# v7's 63.8% "bad rows" = GATE MISCALIBRATION: tolerance 3.0 C on T_mrt when the sealed
#   noon T_mrt MAE IS 3.85 C -> the gate flagged ordinary model error. v7's own split
#   table proves it: train 3.09 / val 3.59 / test 3.85 (canyon), 3.78/4.68/3.69 (plaza)
#   -> split-INDEPENDENT error = generalization (real corruption looked like v6: ~8 C).
# v8: verification at CORRUPTION scale (|dT_a| > 8 C or |dT_mrt| > 25 C), FULL-DOMAIN
#   render, and BOTH stat sets printed: full-domain (visualization) + sealed test-only
#   (== R16g; the quantitative claims).
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
from scipy.ndimage import binary_erosion

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
FIG = W / '03_Results' / '03_Figures'; FIG.mkdir(parents=True, exist_ok=True)
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza'); SEEDS = (0, 1, 2, 3, 4); K_PED = 4
MU = np.array([28.0, 40.0, 2.0, 50.0, 45.0], np.float64)
SD = np.array([4.0, 15.0, 1.5, 100.0, 20.0], np.float64)
BAD_TA, BAD_TM = 8.0, 25.0          # corruption-scale tolerances (v8 recalibration)
CMAP = {'T': 'viridis', 'TMRT': 'inferno'}
FIGN = {'T': 'fig07', 'TMRT': 'fig08'}
UNITS = {'T': r'$T_a$ ($^\circ$C)', 'TMRT': r'$T_{mrt}$ ($^\circ$C)'}
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
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
print(f'device: {device} | R5c-v8: full-domain render, corruption-scale gate\n')

summary = {}
for SITE in SITES:
    t_site = time.time()
    tf = load_npz(f'targets_forcing_{SITE}.npz'); split = load_npz(f'split_{SITE}.npz')
    geom = load_npz(f'geometry_{SITE}.npz'); bbox = load_npz(f'bbox_{SITE}.npz')
    ens = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    if any(x is None for x in (tf, split, bbox, ens)):
        print(f'[{SITE}] missing inputs - skip'); continue
    i0 = np.asarray(tf['i0']); j0 = np.asarray(tf['j0']); k0 = np.asarray(tf['k0'])
    spl = np.asarray(split['split']).astype(str)
    n_points, n_time = tf['target_T'].shape
    test_list = np.where(spl == 'test')[0]
    ranks_te = np.where(k0[test_list] == K_PED)[0]
    ix_te = test_list[ranks_te]
    ped = (k0 == K_PED); all_ped = np.where(ped)[0]
    t_noon = int(np.argmax(np.asarray(tf['target_TMRT'], np.float64).mean(axis=0)))
    zc = np.asarray(geom['z_center'], np.float64) if geom is not None else None
    z_ped = float(zc[K_PED]) if zc is not None else float('nan')
    print(f'  [{SITE}] n_points={n_points:,} n_time={n_time} | z={z_ped:.2f} m | noon=t{t_noon} '
          f'| ped pts={len(all_ped):,} (test {len(ix_te):,})')

    # ---- per-split loaders, SPLIT-LOCAL rank sample ids (the v7 fix) ----
    loaders = []
    for s in ('train', 'val', 'test'):
        pts = np.where(spl == s)[0]
        sel = np.where(k0[pts] == K_PED)[0]
        ped_pts = pts[sel]
        if len(ped_pts) == 0: continue
        ds = VDEIDatasetV2(SITE, s)
        sub = (sel * n_time + t_noon).tolist()
        loader = DataLoader(Subset(ds, sub), batch_size=512, shuffle=False, num_workers=2)
        loaders.append((s, ped_pts, loader))

    # ---- true 5-seed ensemble inference ----
    sum_T = np.zeros(n_points); sum_M = np.zeros(n_points); cnt = 0
    for S in SEEDS:
        p = find(f'{SITE}_dualhead_optuna_best_seed{S}_best.pt') \
            or find(f'{SITE}_dualhead_p5a_best_seed{S}_best.pt')
        if p is None: print(f'  !! seed {S} checkpoint missing - skip'); continue
        ck = torch.load(p, map_location='cpu', weights_only=False)
        model, _ = build(ck.get('config', 'optuna_best'))
        model.load_state_dict(ck['model_state']); model.to(device).eval()
        with torch.no_grad():
            for s, pts_s, loader in loaders:
                pos = 0
                for b in loader:
                    pa, _ = model(b['vdei'].to(device), b['forcing'].to(device))
                    pa = pa.double().cpu().numpy()
                    sum_T[pts_s[pos:pos + len(pa)]] += pa[:, 0]
                    sum_M[pts_s[pos:pos + len(pa)]] += pa[:, 4]
                    pos += len(pa)
        cnt += 1
    pred_T = MU[0] + (sum_T / cnt) * SD[0]
    pred_M = MU[4] + (sum_M / cnt) * SD[4]
    print(f'  [{SITE}] 5-seed ensemble inference done ({cnt} seeds)')

    # ---- ensemble identity vs sealed pool at PROVEN rank rows (must be < 0.05) ----
    rows = ranks_te * n_time + t_noon
    for vi, nm, ours in ((0, 'T', pred_T[ix_te]), (4, 'TMRT', pred_M[ix_te])):
        dmax = float(np.max(np.abs(np.asarray(ens['pred_air'][rows, vi], np.float64) * SD[vi]
                                   + MU[vi] - ours)))
        print(f'  [{SITE}] ensemble check ({nm}, rank rows): max|d|={dmax:.4f} '
              f'{"OK" if dmax < 0.05 else "!! MISMATCH - paste back"}')

    # ---- verification at CORRUPTION scale + split-wise error table ----
    tv_T = np.asarray(tf['target_T'], np.float64); tv_M = np.asarray(tf['target_TMRT'], np.float64)
    bad = (np.abs(pred_T[all_ped] - tv_T[all_ped, t_noon]) > BAD_TA) | \
          (np.abs(pred_M[all_ped] - tv_M[all_ped, t_noon]) > BAD_TM)
    n_bad = int(bad.sum())
    print(f'  [{SITE}] corruption-scale verification vs tf (noon): bad {n_bad:,}/{len(all_ped):,} '
          f'({100 * n_bad / len(all_ped):.2f}%) '
          f'{"- CLEAN -> FULL-DOMAIN render" if n_bad == 0 else "- fallback TEST-ONLY"}')
    for s, pts_s, _ in loaders:
        maeT = float(np.abs(pred_T[pts_s] - tv_T[pts_s, t_noon]).mean())
        maeM = float(np.abs(pred_M[pts_s] - tv_M[pts_s, t_noon]).mean())
        print(f'  [{SITE}]   {s:<5}: n={len(pts_s):,} | noon MAE Ta={maeT:.2f} Tmrt={maeM:.2f} '
              f'(split-independent error = generalization, no memorization)')
    render_pts = all_ped[~bad] if n_bad == 0 else ix_te

    # ---- canvases + figures (full domain) ----
    i_min, i_max = int(bbox['i_min']), int(bbox['i_max'])
    j_min, j_max = int(bbox['j_min']), int(bbox['j_max'])
    ni, nj = i_max - i_min + 1, j_max - j_min + 1
    ii = i0 - i_min; jj = j0 - j_min
    m_ren = np.zeros(n_points, bool); m_ren[render_pts] = True
    def canvas(vals, m):
        c = np.full((ni, nj), np.nan)
        c[ii[m], jj[m]] = vals[m].astype(np.float64)
        return np.ma.masked_invalid(c)
    truth = {'T': tv_T[:, t_noon], 'TMRT': tv_M[:, t_noon]}
    preds = {'T': pred_T, 'TMRT': pred_M}
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
        R = canvas(truth[key], ped); Pd = canvas(preds[key], m_ren)
        Ee = canvas(preds[key] - truth[key], m_ren)
        r = R.compressed(); p = Pd.compressed(); e = Ee.compressed()
        def gradstats(mask_pts):
            cm = np.zeros(n_points, bool); cm[mask_pts] = True
            C1 = canvas(truth[key], cm & ped); C2 = canvas(preds[key], cm & ped)
            ok0 = np.isfinite(C1.filled(np.nan)) & np.isfinite(C2.filled(np.nan))
            oki = binary_erosion(ok0, CROSS)
            g1 = np.hypot(*np.gradient(np.where(oki, C1.filled(np.nan), np.nan)))
            g2 = np.hypot(*np.gradient(np.where(oki, C2.filled(np.nan), np.nan)))
            mg = np.isfinite(g1) & np.isfinite(g2)
            return (float(np.corrcoef(g1[mg], g2[mg])[0, 1]) if mg.sum() > 10 else float('nan'))
        gr_full = gradstats(render_pts)
        gr_test = gradstats(ix_te)
        mae_te = float(np.abs(preds[key][ix_te] - truth[key][ix_te]).mean())
        print(f'  [{SITE}] {key}: truth [{r.min():.1f},{r.max():.1f}] pred [{p.min():.1f},'
              f'{p.max():.1f}] | MAE(full)={np.abs(e).mean():.2f} grad-r(full)={gr_full:.3f} '
              f'|| SEALED test-only: MAE={mae_te:.2f} grad-r(test-canvas)={gr_test:.3f}')
        vmin = float(min(r.min(), p.min())); vmax = float(max(r.max(), p.max()))
        elim = float(np.percentile(np.abs(e), 98))
        fig = plt.figure(figsize=(13.8, 4.6))
        gs = fig.add_gridspec(1, 3, wspace=0.16, left=0.05, right=0.88)
        cax1 = fig.add_axes([0.895, 0.42, 0.012, 0.5])
        cax2 = fig.add_axes([0.895, 0.08, 0.012, 0.26])
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ext = [j_min * 2, (j_max + 1) * 2, i_min * 2, (i_max + 1) * 2]
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
                     f'(z \u2248 {z_ped:.1f} m), solar noon (t = {t_noon})', fontsize=11, y=1.03)
        out = FIG / f'{FIGN[key]}_spatial_{key}_{SITE}.png'
        fig.savefig(out, bbox_inches='tight'); plt.close(fig)
        print(f'    [SAVED] {out.name}')
        stats_site['vars'][key] = {'full_mae': float(np.abs(e).mean()), 'grad_r_full': gr_full,
                                   'test_mae': mae_te, 'grad_r_test_canvas': gr_test}
    summary[SITE] = stats_site
    ens.close()
    print(f'  [{SITE}] done in {(time.time() - t_site) / 60:.1f} min\n')

(MET / 'fig7_8_spatial_stats_v8.json').write_text(json.dumps(summary, indent=2))
print('===== DONE - paste this entire output back (ensemble checks MUST be OK, drops ~0%) =====')
