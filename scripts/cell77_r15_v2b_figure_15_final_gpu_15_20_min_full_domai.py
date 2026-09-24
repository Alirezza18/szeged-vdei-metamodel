# ===== CELL R15-v2b + FIGURE 15 FINAL (GPU, ~15-20 min) - FULL-DOMAIN TIME-MEAN MAE, VERIFIED SCALE =====
# v2b FIXES vs v2a (both gate-side; the MAE fields were already correct in v2a's run):
#   FIX 1 (GATE 1 design flaw): v2a averaged predictions across ALL timesteps and compared
#     against the NOON pool -> daily-mean vs noon (~7 C by design, exactly what printed).
#     GATE 1 now uses a dedicated NOON-only accumulation.
#   FIX 2 (GATE 2 crash): variable selection [:, [0, 4]] added before the SD multiply.
# Machinery unchanged: split-local rank ids (rank -> global point via the FULL split list),
# true 5-seed ensemble at all timesteps, sealed-anchor + physical-range gates.
# PRECONDITION: sealed CELL 18b (VDEIDatasetV2) + CELL 19 (build) in THIS session.
import json, time
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
SITES = ('canyon', 'plaza'); SEEDS = (0, 1, 2, 3, 4); K_PED = 4
MU = np.array([28.0, 40.0, 2.0, 50.0, 45.0], np.float64)
SD = np.array([4.0, 15.0, 1.5, 100.0, 20.0], np.float64)
SEALED_38 = {'canyon': (0.465, 2.143), 'plaza': (0.392, 2.284)}
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
print(f'device: {device} | R15-v2b: full-domain time-mean MAE, verified scale\n')

summary, store = {}, {}
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
    all_ped = np.where(k0 == K_PED)[0]
    t_noon = int(np.argmax(np.asarray(tf['target_TMRT'], np.float64).mean(axis=0)))
    zc = np.asarray(geom['z_center'], np.float64) if geom is not None else None
    print(f'  [{SITE}] n_points={n_points:,} n_time={n_time} | z={zc[K_PED]:.2f} m | '
          f'ped pts={len(all_ped):,} (test {len(ix_te):,}) | noon=t{t_noon}')
    tv_T = np.asarray(tf['target_T']); tv_M = np.asarray(tf['target_TMRT'])

    # ---- per-split loaders with SPLIT-LOCAL rank ids at ALL timesteps ----
    loaders = []
    for s in ('train', 'val', 'test'):
        pts = np.where(spl == s)[0]
        sel = np.where(k0[pts] == K_PED)[0]
        if len(sel) == 0: continue
        ds = VDEIDatasetV2(SITE, s)
        sub = np.concatenate([sel * n_time + t for t in range(n_time)])
        loader = DataLoader(Subset(ds, sub.tolist()), batch_size=512,
                            shuffle=False, num_workers=2)
        loaders.append((s, pts[sel], pts, sub, loader))  # pts = FULL split list (rank target)
        print(f'  [{SITE}] {s:<5}: {len(sel):,} ped pts x {n_time} t = {len(sub):,} samples')

    # ---- true 5-seed ensemble; accumulate |error| per point over (seed x t) ----
    # v2b FIX 1: GATE 1 gets a dedicated NOON-only accumulation (v2a compared the
    #   all-timestep daily mean against the noon pool -> ~7 C by design).
    sumET = np.zeros(n_points); sumEM = np.zeros(n_points)
    cnts = np.zeros(n_points, np.int64)
    no_T = np.zeros(n_points); no_M = np.zeros(n_points); ncnt = np.zeros(n_points, np.int64)
    for S in SEEDS:
        p = find(f'{SITE}_dualhead_optuna_best_seed{S}_best.pt') \
            or find(f'{SITE}_dualhead_p5a_best_seed{S}_best.pt')
        if p is None: print(f'  !! seed {S} checkpoint missing - skip'); continue
        ck = torch.load(p, map_location='cpu', weights_only=False)
        model, _ = build(ck.get('config', 'optuna_best'))
        model.load_state_dict(ck['model_state']); model.to(device).eval()
        with torch.no_grad():
            for s, pts_s, pts_full, sub, loader in loaders:
                pos = 0
                for b in loader:
                    pa, _ = model(b['vdei'].to(device), b['forcing'].to(device))
                    pa = pa.double().cpu().numpy()
                    ids = sub[pos:pos + len(pa)]
                    rk = ids // n_time; tv = ids % n_time
                    gp = pts_full[rk]  # rank within FULL split point list -> global point id
                    np.add.at(cnts, gp, 1)
                    np.add.at(sumET, gp, np.abs(MU[0] + pa[:, 0] * SD[0] - tv_T[gp, tv]))
                    np.add.at(sumEM, gp, np.abs(MU[4] + pa[:, 4] * SD[4] - tv_M[gp, tv]))
                    mnoon = (tv == t_noon)
                    if mnoon.any():
                        g2 = gp[mnoon]
                        np.add.at(no_T, g2, pa[mnoon, 0]); np.add.at(no_M, g2, pa[mnoon, 4])
                        np.add.at(ncnt, g2, 1)
                    pos += len(pa)
    safe = np.maximum(cnts, 1)
    mae_T = np.where(cnts > 0, sumET / safe, np.nan)
    mae_M = np.where(cnts > 0, sumEM / safe, np.nan)
    safeN = np.maximum(ncnt, 1)
    pred_noon_T = MU[0] + (no_T / safeN) * SD[0]
    pred_noon_M = MU[4] + (no_M / safeN) * SD[4]
    print(f'  [{SITE}] 5-seed ensemble inference done ({int((cnts[all_ped] > 0).sum()):,}'
          f'/{len(all_ped):,} ped pts covered)')

    # ---- GATE 1: noon ensemble check vs sealed pool at PROVEN rank rows ----
    rows = ranks_te * n_time + t_noon
    for vi, nm, ours in ((0, 'T', pred_noon_T[ix_te]), (4, 'TMRT', pred_noon_M[ix_te])):
        dmax = float(np.max(np.abs(np.asarray(ens['pred_air'][rows, vi], np.float64)
                                   * SD[vi] + MU[vi] - ours)))
        print(f'  [{SITE}] ensemble check ({nm}, noon, rank rows): max|d|={dmax:.4f} '
              f'{"OK" if dmax < 0.05 else "!! MISMATCH - paste back"}')

    # ---- GATE 2: all-t test-only MAE from the sealed pool == sealed Section 3.8 ----
    tt = np.arange(n_time)
    rows_all = (ranks_te[:, None] * n_time + tt[None, :]).ravel()
    Pa = np.asarray(ens['pred_air'][rows_all], np.float64)[:, [0, 4]]   # v2b FIX 2: select vars
    Ta_ = np.asarray(ens['target_air'][rows_all], np.float64)[:, [0, 4]]  # before SD multiply
    mae_pool = (SD[[0, 4]] * np.abs(Pa - Ta_)).mean(axis=0)
    sealed = SEALED_38[SITE]
    print(f'  [{SITE}] GATE2 pool all-t test MAE: Ta={mae_pool[0]:.3f} (sealed {sealed[0]:.3f}) '
          f'Tmrt={mae_pool[1]:.3f} (sealed {sealed[1]:.3f}) -> '
          f'{"OK" if np.max(np.abs(mae_pool - np.array(sealed))) < 0.02 else "!! MISMATCH"}')
    ours_te = (float(mae_T[ix_te].mean()), float(mae_M[ix_te].mean()))
    print(f'  [{SITE}] GATE2b OUR all-t test MAE: Ta={ours_te[0]:.3f}/{sealed[0]:.3f} '
          f'Tmrt={ours_te[1]:.3f}/{sealed[1]:.3f} -> '
          f'{"OK" if max(abs(ours_te[0] - sealed[0]), abs(ours_te[1] - sealed[1])) < 0.05 else "!! MISMATCH"}')

    # ---- GATE 3: full-domain physical sanity ----
    gate3 = {}
    for nm, arr, lo, hi in (('Ta', mae_T, 0.2, 1.5), ('Tmrt', mae_M, 0.8, 6.0)):
        v = arr[all_ped]; med, mx = float(np.median(v)), float(v.max())
        ok = (lo < med < hi) and (mx < 4 * hi)
        gate3[nm] = {'median': med, 'p95': float(np.percentile(v, 95)), 'max': mx}
        print(f'  [{SITE}] GATE3 full-domain {nm} time-mean MAE: median={med:.3f} '
              f'p95={gate3[nm]["p95"]:.3f} max={mx:.3f} '
              f'{"OK" if ok else "!! OUT OF PHYSICAL RANGE - paste back"}')

    # ---- canvas + building mask ----
    i_min, i_max = int(bbox['i_min']), int(bbox['i_max'])
    j_min, j_max = int(bbox['j_min']), int(bbox['j_max'])
    ni, nj = i_max - i_min + 1, j_max - j_min + 1
    ii = i0 - i_min; jj = j0 - j_min
    def canvas(vals):
        c = np.full((ni, nj), np.nan)
        c[ii[all_ped], jj[all_ped]] = vals[all_ped]
        return np.ma.masked_invalid(c)
    bmask = None
    if geom is not None:
        try:
            ib = np.asarray(geom['is_building']); nk = len(zc)
            if ib.ndim == 3:
                ax_k = int(np.argwhere(np.array(ib.shape) == nk).ravel()[0])
                low = np.take(ib, slice(0, min(11, nk)), axis=ax_k).any(axis=ax_k)
                cand = low if low.shape == (ni, nj) else np.take(ib, K_PED, axis=ax_k)
                if cand.shape == (ni, nj): bmask = cand
        except Exception:
            bmask = None
    store[SITE] = {'Ta': canvas(mae_T), 'Tmrt': canvas(mae_M),
                   'ext': [j_min * 2, (j_max + 1) * 2, i_min * 2, (i_max + 1) * 2],
                   'bmask': bmask}
    summary[SITE] = {'full_domain': gate3, 'test_mae': ours_te,
                     'pool_mae': [float(mae_pool[0]), float(mae_pool[1])]}
    ens.close()
    print(f'  [{SITE}] done in {(time.time() - t_site) / 60:.1f} min\n')

# ---- render 2x2, shared vmax per variable across sites (98th pct) ----
if len(store) == 2:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8))
    for r, (key, vlab) in enumerate((('Ta', r'$T_a$'), ('Tmrt', r'$T_{mrt}$'))):
        vmax = float(np.nanpercentile(
            np.concatenate([store['canyon'][key].compressed(),
                            store['plaza'][key].compressed()]), 98))
        print(f'  [FIGURE 15] {vlab} colorbar: 0 - {vmax:.2f} degC (98th pct, shared across sites)')
        for cc, site in enumerate(('canyon', 'plaza')):
            ax = axes[r, cc]
            im = ax.imshow(store[site][key], origin='lower', extent=store[site]['ext'],
                           cmap='magma', vmin=0, vmax=vmax, interpolation='nearest')
            bm = store[site]['bmask']
            if bm is not None:
                ax.imshow(np.ma.masked_invalid(np.where(bm > 0, 1.0, np.nan)),
                          origin='lower', extent=store[site]['ext'], cmap='gray_r',
                          vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f'({"ab"[cc]}) {site.capitalize()} {vlab}', fontsize=9.5)
            ax.set_facecolor('0.92'); ax.set_xlabel('x (m)')
            if cc == 0: ax.set_ylabel('y (m)')
        cb = fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.028, pad=0.015)
        cb.set_label(f'{vlab} time-mean MAE ($^\\circ$C)')
    fig.suptitle('Spatial MAE fields - pedestrian level, full domain, all timesteps '
                 '(5-seed ensemble-mean prediction)', fontsize=10.5, y=0.995)
    out = FIG / 'fig15_spatial_MAE_full.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f'  [SAVED] {out.name}')

(MET / 'spatial_mae_fulldomain_summary.json').write_text(json.dumps(summary, indent=2))
print('\n===== DONE - paste this entire output back (ALL GATES must be OK; '
      'expect T_a colorbar ~1 C, T_mrt ~4-6 C) =====')
