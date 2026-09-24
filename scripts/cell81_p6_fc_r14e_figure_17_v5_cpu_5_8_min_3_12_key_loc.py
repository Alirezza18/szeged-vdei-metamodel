# ===== CELL P6-FC R14e + FIGURE 17 v5 (CPU, ~5-8 min) - §3.12 KEY-LOCATION DIURNAL VALIDATION =====
# v5 PROFESSIONAL FIGURE (stats/JSON/Table A14 identical to R14d - analysis untouched):
#   * journal palette (tab10), ref dashed 1.1 / ensemble solid 1.8, bands alpha .10
#   * daylight shading subtler (alpha .08, behind), noon dotted line kept
#   * error strips: nice 1-2-5 y-limits (no more 12.7/18.58), x labels only on bottom row
#     (fixes plaza-title/tick collision), Delta label only left column
#   * maps: ROBUST building footprint (union of levels <= 10 m, bbox preferred / i0-j0
#     fallback) + status line, domain border, 50 m scale bar, stars + letters
#   * panel letters (a)-(f) kept, embedded suptitle dropped (caption carries it)
# v4 analysis (kept exactly): anchored decoder EXACT by linearity (float64), de-duplicated
#   key locations, 5-seed order-verified min-max band, per-location stats.
# Key locations: HOTSPOT / COOLSPOT / CONFINED (min ENVI-met SVF) / MEDIAN (median error).
# Outputs: fig17_key_locations.png + key_location_summary.json. Paste ENTIRE output back.
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
FIG = W / '03_Results' / '03_Figures'; FIG.mkdir(parents=True, exist_ok=True)
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza'); K_PED = 4
AL = (r'$T_a$', r'$RH$', r'$V$', r'$TKE$', r'$T_{mrt}$')
MU = np.array([28.0, 40.0, 2.0, 50.0, 45.0], np.float32)
SD0 = np.array([4.0, 15.0, 1.5, 100.0, 20.0], np.float32)
SEALED = {'canyon': (0.383, 1.222, 0.273, 3.470, 1.111),
          'plaza': (0.365, 1.285, 0.351, 2.109, 1.472)}
GATE = 0.005
LOCS = ('HOTSPOT', 'COOLSPOT', 'CONFINED', 'MEDIAN')
LET = {'HOTSPOT': 'H', 'COOLSPOT': 'C', 'CONFINED': 'F', 'MEDIAN': 'M'}
LC = {'HOTSPOT': '#D62728', 'COOLSPOT': '#1F77B4',
      'CONFINED': '#9467BD', 'MEDIAN': '#2CA02C'}
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'axes.linewidth': 0.7, 'mathtext.fontset': 'stix',
                     'font.family': 'DejaVu Sans'})

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

def seed_npz(SITE, S):
    """Per-seed fullpool npz (for the min-max band); None if absent."""
    for pat in (f'{SITE}_dualhead_optuna_best_seed{S}_test_fullpool.npz',
                f'{SITE}_dualhead_p5a_best_seed{S}_test_fullpool.npz',
                f'{SITE}_dualhead_optuna_seed{S}_test_fullpool.npz'):
        p = find(pat)
        if p is not None: return p
    for d_ in sorted(IN.glob('*')):
        for p in sorted(d_.rglob(f'{SITE}_dualhead_*seed{S}_test_fullpool.npz')):
            n = p.name
            if 'ensemble5' in n or 'zs_' in n or 'ft_' in n or 'from' in n: continue
            return p
    return None

def runs_of(idxs):
    out = []
    if len(idxs) == 0: return out
    s = p_ = int(idxs[0])
    for t in idxs[1:]:
        t = int(t)
        if t == p_ + 1: p_ = t
        else: out.append((s, p_)); s = p_ = t
    out.append((s, p_))
    return out

def nice_sym(v):
    """Round a symmetric limit up to the nearest 1/2/2.5/5 x 10^k (clean strip ticks)."""
    if v is None or v <= 0 or v != v: return 1.0
    e = 10.0 ** np.floor(np.log10(v)); m = v / e
    for c in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m <= c + 1e-9: return c * e
    return 10.0 * e

SERIES = {}; SUMMARY = {'decoder': {}, 'sites': {}}
for SITE in SITES:
    print(f'\n########## [{SITE}] ##########')
    tf = load_npz(f'targets_forcing_{SITE}.npz')
    split = load_npz(f'split_{SITE}.npz')
    d = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    svf = load_npz(f'svf_{SITE}.npz')
    geom = load_npz(f'geometry_{SITE}.npz')
    bbox = load_npz(f'bbox_{SITE}.npz')
    if any(x is None for x in (tf, split, d)): continue
    n_points, n_time = tf['target_T'].shape
    spl = np.asarray(split['split']).astype(str)
    ix_te = np.where(spl == 'test')[0]; n_test = len(ix_te)
    i0_all = np.asarray(tf['i0']); j0_all = np.asarray(tf['j0']); k0_all = np.asarray(tf['k0'])
    zc = np.asarray(geom['z_center']) if (geom is not None and 'z_center' in geom.files) else None
    z_ped = float(zc[K_PED]) if zc is not None else 2.0

    P0 = np.asarray(d['pred_air'], np.float32).reshape(n_test, n_time, 5)
    T0 = np.asarray(d['target_air'], np.float32).reshape(n_test, n_time, 5)
    assert P0.size == n_test * n_time * 5, 'reshape mismatch - paste back'

    # ---- anchored decoder: EXACT by linearity (MAE_phys = SD_eff * MAE_norm) ----
    sealed = np.asarray(SEALED[SITE], np.float32)
    mae_norm = np.abs(P0 - T0).mean((0, 1), dtype=np.float64)
    SD_eff = (sealed / mae_norm).astype(np.float32)
    P = P0 * SD_eff + MU; T = T0 * SD_eff + MU
    mae = np.abs(P - T).mean((0, 1), dtype=np.float64)
    gate = float(np.abs(mae - sealed).max())
    print('  anchored decoder: SD_eff=(' + ', '.join(f'{s:.4f}' for s in SD_eff) + ') [exact]')
    print('  cross-check MAE (must match §3.1.1): ' +
          '  '.join(f'{AL[v]}={mae[v]:.3f}' for v in range(5)))
    print(f'  anchored gate: max|dMAE|={gate:.5f} ' +
          ('OK - physical scale reproduces sealed' if gate < GATE else '!! MISMATCH - paste back'))
    SUMMARY['decoder'][SITE] = {'SD_eff': [round(float(s), 4) for s in SD_eff],
                                'gate': round(gate, 6)}

    # ---- time axis + daytime ----
    if 'forcing_IsDaytime' in tf.files:
        day = np.asarray(tf['forcing_IsDaytime']).astype(bool).ravel()
    else:
        day = np.asarray(tf['forcing_SunHeight']).ravel() > 0
    if day.size != n_time: day = np.ones(n_time, bool)
    day_i = np.where(day)[0] if day.sum() >= 3 else np.arange(n_time)
    t_noon = int(np.argmax(np.asarray(tf['target_TMRT']).mean(0)))
    dt = 24.0 / (n_time - 1)
    print(f'  x-mapping: dt={dt:.2f} h/step, solar noon t={t_noon}, daytime steps={int(day.sum())}')

    # ---- key-location selection (ped-level test points, de-duplicated) ----
    ped = np.where(k0_all[ix_te] == K_PED)[0]
    Tm = T[:, :, 4]
    daymean = Tm[:, day_i].mean(1)
    mae_pt = np.abs(P0[:, :, 4] - T0[:, :, 4]).mean(1)   # normalized; scale-invariant
    def pick_free(order, taken):
        """First candidate index in priority order not already taken."""
        for c in order:
            if int(c) not in taken: return int(c)
        return int(order[0])

    hot = int(ped[int(np.argmax(daymean[ped]))])
    cool = int(ped[int(np.argmin(daymean[ped]))])
    svf_em = None; conf = cool
    if svf is not None and 'svf_envimet' in svf.files:
        s = np.asarray(svf['svf_envimet'], np.float64)[ix_te]
        if np.isfinite(s[ped]).any():
            svf_em = s
            sv = s[ped].copy(); sv[~np.isfinite(sv)] = np.inf
            conf = pick_free(ped[np.argsort(sv)], {hot, cool})
    med = float(np.median(mae_pt[ped]))
    medp = pick_free(ped[np.argsort(np.abs(mae_pt[ped] - med))], {hot, cool, conf})
    locs = {'HOTSPOT': hot, 'COOLSPOT': cool, 'CONFINED': conf, 'MEDIAN': medp}
    print(f'  key locations: {len(set(locs.values()))}/4 distinct grid points')

    # ---- V-DEI context ----
    vf = load_npz(f'vdei_features_{SITE}.npz')
    cls = np.asarray(vf['cls'])[ix_te] if (vf is not None and 'cls' in vf.files) else None

    # ---- 5-seed min-max band (each seed order-verified vs ensemble) ----
    loc_ix = np.array([locs[L] for L in LOCS])
    Gmin = Gmax = None; ngot = 0
    for S in range(5):
        pth = seed_npz(SITE, S)
        if pth is None: continue
        try:
            dz = np.load(pth)
            if 'pred_air' not in dz.files: dz.close(); continue
            Pps = np.asarray(dz['pred_air'], np.float32).reshape(n_test, n_time, 5)
            cpp = float(np.corrcoef(np.abs(Pps[:, :, 4] - T0[:, :, 4]).mean(1), mae_pt)[0, 1])
            if cpp < 0.8:
                print(f'  seed {S}: per-point err corr={cpp:.3f} < 0.8 - DROPPED (ordering?)')
            else:
                Gs = Pps[loc_ix] * SD_eff + MU
                Gmin = Gs if Gmin is None else np.minimum(Gmin, Gs)
                Gmax = Gs if Gmax is None else np.maximum(Gmax, Gs)
                ngot += 1
                print(f'  seed {S}: per-point err corr={cpp:.3f} OK')
            del Pps
            dz.close()
        except Exception as ex:
            print(f'  !! seed {S} npz unreadable ({ex}) - skipped')
    if ngot < 2:
        Gmin = Gmax = None
        print('  seed band: not available (<2 verified seeds) - mean line only')
    else:
        print(f'  seed band: {ngot}/5 per-seed npz verified')

    # ---- per-location stats ----
    print('\n  ===== KEY-LOCATION DIURNAL STATS (anchored scale) =====')
    stats = {}
    for L in LOCS:
        p = locs[L]; g = int(ix_te[p]); pr = P[p]; tr = T[p]; e = pr - tr
        rec = {'grid': {'i0': int(i0_all[g]), 'j0': int(j0_all[g]), 'k0': int(k0_all[g]),
                        'x_m': int(j0_all[g]) * 2, 'y_m': int(i0_all[g]) * 2,
                        'z_m': round(z_ped, 2)}}
        if svf_em is not None and np.isfinite(svf_em[p]):
            rec['grid']['svf_em'] = round(float(svf_em[p]), 3)
        if cls is not None:
            cc = np.asarray(cls[p])
            rec['grid'].update({'sky_frac': round(float((cc == 0).mean()), 3),
                                'bld_frac': round(float((cc == 1).mean()), 3),
                                'veg_frac': round(float((cc == 2).mean()), 3)})
        line = (f"  [{L}] grid(i={rec['grid']['i0']},j={rec['grid']['j0']}) "
                f"x={rec['grid']['x_m']}m y={rec['grid']['y_m']}m z={rec['grid']['z_m']}m")
        if 'svf_em' in rec['grid']: line += f" | SVF={rec['grid']['svf_em']:.2f}"
        if 'sky_frac' in rec['grid']:
            line += (f" sky={rec['grid']['sky_frac']:.2f}"
                     f" bld={rec['grid']['bld_frac']:.2f} veg={rec['grid']['veg_frac']:.2f}")
        print(line)
        for v, nm in ((4, 'T_mrt'), (0, 'T_a')):
            maeL = float(np.abs(e[:, v]).mean()); biasL = float(e[:, v].mean())
            trd = tr[day_i, v]; prd = pr[day_i, v]
            tref = int(day_i[int(np.argmax(trd))]); tpred = int(day_i[int(np.argmax(prd))])
            rL = float(np.corrcoef(tr[:, v], pr[:, v])[0, 1])
            rec[nm] = {'MAE': round(maeL, 3), 'bias': round(biasL, 3),
                       'peak_ref_C': round(float(trd.max()), 2),
                       'peak_pred_C': round(float(prd.max()), 2),
                       'peak_t_ref': tref, 'peak_t_pred': tpred,
                       'shift_h': round((tpred - tref) * dt, 2), 'r': round(rL, 4)}
            print(f"    {nm:5s}: MAE={maeL:.2f} bias={biasL:+.2f} | "
                  f"peak ref {trd.max():.1f} @ {(tref - t_noon) * dt:+.1f}h vs "
                  f"pred {prd.max():.1f} @ {(tpred - t_noon) * dt:+.1f}h "
                  f"(shift {(tpred - tref) * dt:+.1f}h, dAmp {prd.max() - trd.max():+.1f}) "
                  f"| r={rL:.3f}")
        stats[L] = rec
    SUMMARY['sites'][SITE] = {'t_noon': t_noon, 'dt_h': dt, 'z_m': round(z_ped, 2),
                              'locations': stats}

    # ---- building footprint + extent for the map panel (ROBUST) ----
    bmask = None; ext = None
    try:
        if bbox is not None and all(k in bbox.files for k in
                                    ('i_min', 'i_max', 'j_min', 'j_max')):
            i_min, i_max = int(bbox['i_min']), int(bbox['i_max'])
            j_min, j_max = int(bbox['j_min']), int(bbox['j_max'])
        else:                      # fallback: air-point domain from tf
            i_min, i_max = int(i0_all.min()), int(i0_all.max())
            j_min, j_max = int(j0_all.min()), int(j0_all.max())
        ni, nj = i_max - i_min + 1, j_max - j_min + 1
        ext = [j_min * 2, (j_max + 1) * 2, i_min * 2, (i_max + 1) * 2]
        if geom is not None and 'is_building' in geom.files:
            ib = np.asarray(geom['is_building'])
            if ib.ndim == 2 and ib.shape == (ni, nj):
                bmask = (ib > 0)
            elif ib.ndim == 3:
                nk = len(zc) if zc is not None else 0
                hits = np.argwhere(np.array(ib.shape) == nk).ravel()
                ax_k = int(hits[0]) if (nk and hits.size) else 2
                ibm = np.moveaxis(ib, ax_k, 2) if ax_k != 2 else ib
                if ibm.shape[:2] == (ni, nj):
                    zlim = 10.0
                    kmax = (int(np.searchsorted(np.asarray(zc), zlim, side='right'))
                            if zc is not None else ibm.shape[2])
                    kmax = max(1, min(kmax, ibm.shape[2]))
                    bmask = ibm[..., :kmax].max(axis=2) > 0   # footprint: union <= 10 m
        print(f'  map: footprint={"OK" if bmask is not None else "unavailable"}'
              + (f' (cells={int(bmask.sum())})' if bmask is not None else '')
              + f' | ext={[round(v) for v in ext]}')
    except Exception as ex:
        print(f'  map: fallback failed ({ex}) - plain map'); bmask = None; ext = None

    # ---- store everything the figure needs ----
    x = (np.arange(n_time) - t_noon) * dt
    SERIES[SITE] = {
        'x': x, 'dt': dt, 'stats': stats, 'day_i': np.asarray(day_i, int),
        'series': {L: (T[locs[L]][:, [4, 0]].astype(np.float64),
                       P[locs[L]][:, [4, 0]].astype(np.float64)) for L in LOCS},
        'band': ({L: (Gmin[i][:, [4, 0]].astype(np.float64),
                      Gmax[i][:, [4, 0]].astype(np.float64))
                  for i, L in enumerate(LOCS)} if Gmin is not None else None),
        'day_spans': [(int(s), int(e)) for s, e in runs_of(day_i)],
        'map': {'bmask': bmask, 'ext': ext,
                'xy': {L: (float(j0_all[int(ix_te[locs[L]])] * 2 + 1),
                           float(i0_all[int(ix_te[locs[L]])] * 2 + 1)) for L in LOCS}}}
    del P0, T0, P, T, d, svf, vf, cls
    if Gmin is not None: del Gmin, Gmax

# ---- FIGURE 17 v5 (professional) ----
print('\n===== FIGURE 17 (v5) =====')
fig = plt.figure(figsize=(12.6, 9.0))
gs = fig.add_gridspec(4, 3, height_ratios=[3.0, 0.95, 3.0, 0.95],
                      width_ratios=[1.18, 1.18, 0.92], hspace=0.10, wspace=0.26)
PANEL = iter('abcdef')
for r_, SITE in enumerate(SITES):
    if SITE not in SERIES: continue
    D = SERIES[SITE]; x = D['x']; dt = D['dt']; stats = D['stats']; day_i = D['day_i']
    rows = (0, 1) if r_ == 0 else (2, 3)
    for c_, (vi, ylab, ttl, vkey) in enumerate(
            ((0, r'$T_{mrt}$ (°C)', r'$T_{mrt}$', 'T_mrt'),
             (1, r'$T_a$ (°C)', r'$T_a$', 'T_a'))):
        ax = fig.add_subplot(gs[rows[0], c_])
        axe = fig.add_subplot(gs[rows[1], c_], sharex=ax)
        for s_, e_ in D['day_spans']:
            for a_ in (ax, axe):
                a_.axvspan(x[s_] - dt / 2, x[e_] + dt / 2, color='#F5D76E',
                           alpha=0.08, lw=0, zorder=0)
        errs = []
        for L in LOCS:
            ref, pred = D['series'][L]
            if D['band'] is not None:
                lo, hi = D['band'][L]
                ax.fill_between(x, lo[:, vi], hi[:, vi], color=LC[L],
                                alpha=0.10, lw=0, zorder=1)
            ax.plot(x, ref[:, vi], '--', color=LC[L], lw=1.1, alpha=0.9, zorder=3)
            ax.plot(x, pred[:, vi], '-', color=LC[L], lw=1.8, zorder=4)
            errs.append(pred[:, vi] - ref[:, vi])
            ti = int(np.argmax(ref[day_i, vi])); tj = int(np.argmax(pred[day_i, vi]))
            ax.plot(x[day_i[ti]], ref[day_i[ti], vi], 'o', color=LC[L], ms=3.4,
                    mec='k', mew=0.4, zorder=6)
            ax.plot(x[day_i[tj]], pred[day_i[tj], vi], 's', color=LC[L], ms=3.4,
                    mec='k', mew=0.4, zorder=6)
        # hotspot peak-shift arrow (T_mrt panels, only if shift >= half a step)
        if vkey == 'T_mrt':
            rec = stats['HOTSPOT'][vkey]
            x1 = (rec['peak_t_ref'] - SUMMARY['sites'][SITE]['t_noon']) * dt
            x2 = (rec['peak_t_pred'] - SUMMARY['sites'][SITE]['t_noon']) * dt
            y = rec['peak_ref_C']
            if abs(x2 - x1) >= dt / 2:
                ax.annotate('', xy=(x2, y), xytext=(x1, y),
                            arrowprops=dict(arrowstyle='<->', color='0.15', lw=0.8))
                ax.text((x1 + x2) / 2, y, f"{rec['shift_h']:+.1f} h", fontsize=6,
                        ha='center', va='bottom', color='0.15')
        ax.axvline(0, color='k', lw=0.7, ls=':', alpha=0.55, zorder=2)
        ax.set_title(f'{SITE.capitalize()} – {ttl}', pad=3)
        if c_ == 0: ax.set_ylabel(ylab)
        ax.grid(alpha=0.22, lw=0.4)
        ax.tick_params(labelbottom=False)
        ax.text(0.012, 0.97, f'({next(PANEL)})', transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top')
        mae_txt = 'MAE  ' + '  '.join(f"{LET[L]} {stats[L][vkey]['MAE']:.2f}"
                                      for L in LOCS)
        ax.text(0.985, 0.03, mae_txt, transform=ax.transAxes, ha='right',
                va='bottom', fontsize=6.4,
                bbox=dict(fc='white', ec='0.8', alpha=0.8, pad=1.6))
        # error strip (nice limits, x labels only on the bottom site)
        E = np.vstack(errs)
        sym = nice_sym(float(np.percentile(np.abs(E), 98)))
        for i, L in enumerate(LOCS):
            axe.plot(x, E[i], '-', color=LC[L], lw=1.0)
        axe.axhline(0, color='k', lw=0.6, ls=':', alpha=0.7)
        axe.axvline(0, color='k', lw=0.7, ls=':', alpha=0.55)
        axe.set_ylim(-sym, sym)
        axe.set_xlim(x.min() - dt / 2, x.max() + dt / 2)
        axe.grid(alpha=0.18, lw=0.3)
        axe.yaxis.set_major_locator(MultipleLocator(sym))
        axe.xaxis.set_major_locator(MultipleLocator(3))
        axe.tick_params(labelsize=7, labelbottom=(r_ == 1))
        if c_ == 0:
            axe.set_ylabel(r'$\Delta$ (pred$-$ref)', fontsize=7)
        if r_ == 1:
            axe.set_xlabel('hours relative to solar noon', fontsize=8)
    # map panel spanning both rows of this site
    axm = fig.add_subplot(gs[rows[0]:rows[1] + 1, 2])
    M = D['map']
    axm.set_facecolor('0.94')
    if M['bmask'] is not None and M['ext'] is not None:
        axm.imshow(np.ma.masked_invalid(np.where(M['bmask'], 1.0, np.nan)),
                   origin='lower', extent=M['ext'], cmap='gray_r', vmin=0, vmax=1,
                   interpolation='nearest')
        axm.set_xlim(M['ext'][0], M['ext'][1]); axm.set_ylim(M['ext'][2], M['ext'][3])
        axm.add_patch(Rectangle((M['ext'][0], M['ext'][2]),
                                M['ext'][1] - M['ext'][0], M['ext'][3] - M['ext'][2],
                                fill=False, ec='0.4', lw=0.7))
        # 50 m scale bar, bottom-left
        x0 = M['ext'][0] + 0.06 * (M['ext'][1] - M['ext'][0])
        y0 = M['ext'][2] + 0.05 * (M['ext'][3] - M['ext'][2])
        axm.plot([x0, x0 + 50], [y0, y0], color='k', lw=1.6, solid_capstyle='butt')
        axm.text(x0 + 25, y0, '50 m', fontsize=6, ha='center', va='bottom',
                 bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.8))
    for L in LOCS:
        xx, yy = M['xy'][L]
        axm.scatter(xx, yy, marker='*', s=120, color=LC[L], edgecolor='k',
                    linewidth=0.6, zorder=6)
        axm.text(xx + 2.5, yy + 2.5, LET[L], fontsize=7, fontweight='bold',
                 ha='left', va='bottom', zorder=7,
                 bbox=dict(fc='white', ec='none', alpha=0.65, pad=0.9))
    axm.set_title(f'{SITE.capitalize()} – key locations', fontsize=9, pad=3)
    axm.set_xlabel('x (m)', fontsize=7); axm.set_ylabel('y (m)', fontsize=7)
    axm.set_aspect('equal'); axm.tick_params(labelsize=6.5)
    axm.set_xticks(axm.get_xticks()[::2]); axm.set_yticks(axm.get_yticks()[::2])
    axm.text(0.012, 0.97, f'({next(PANEL)})', transform=axm.transAxes,
             fontsize=9, fontweight='bold', va='top')

handles = [Line2D([], [], color='k', ls='--', lw=1.1, label='ENVI-met reference'),
           Line2D([], [], color='k', ls='-', lw=1.8, label='V-DEI ensemble (5-seed mean)'),
           Patch(facecolor='0.55', alpha=0.25, label='5-seed range'),
           Patch(facecolor='#F5D76E', alpha=0.35, label='daylight'),
           Line2D([], [], color='k', marker='o', ls='none', ms=4, label='ref peak'),
           Line2D([], [], color='k', marker='s', ls='none', ms=4, label='pred peak')] + \
          [Line2D([], [], color=LC[L], lw=2, marker='*', ms=9, mec='k', mew=0.5,
                  label=f'{LET[L]} = {L}') for L in LOCS]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=7.2,
           frameon=False, bbox_to_anchor=(0.5, -0.045))
fig.savefig(FIG / 'fig17_key_locations.png', bbox_inches='tight')
plt.close(fig)
print('  [SAVED] fig17_key_locations.png')

# ---- TABLE A14 rows (markdown, paste-ready) ----
print('\n===== TABLE A14 ROWS (markdown) =====')
for SITE in SITES:
    if SITE not in SUMMARY['sites']: continue
    st = SUMMARY['sites'][SITE]
    print(f"\n**{SITE.capitalize()}** (z = {st['z_m']} m; dt = {st['dt_h']} h; "
          f"solar noon t = {st['t_noon']})")
    print('| Location | Var | MAE | Bias | Peak ref | Peak pred | Shift (h) | r |')
    print('|---|---|---|---|---|---|---|---|')
    for L in LOCS:
        rec = st['locations'][L]
        for nm, key in ((r'$T_{mrt}$', 'T_mrt'), (r'$T_a$', 'T_a')):
            rr = rec[key]
            print(f"| {L} | {nm} | {rr['MAE']:.2f} | {rr['bias']:+.2f} | "
                  f"{rr['peak_ref_C']:.1f} | {rr['peak_pred_C']:.1f} | "
                  f"{rr['shift_h']:+.1f} | {rr['r']:.3f} |")

(MET / 'key_location_summary.json').write_text(json.dumps(SUMMARY, indent=2))
print('\n===== DONE - paste this entire output back =====')
