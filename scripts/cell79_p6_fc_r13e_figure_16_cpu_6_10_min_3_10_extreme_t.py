# ===== CELL P6-FC R13e + FIGURE 16 (CPU, ~6-10 min) - §3.10 EXTREME TAILS + §3.11 FAILURE ATTRIBUTION (STAT-RESOLVED FINAL) =====
# R13d verdict baked in:
#   * single npz copy (no shadowing); pred_air = 5-seed ENSEMBLE-MEAN in NORMALIZED units;
#   * per-seed fingerprint (canyon Ta seed CV = 0.43) matches sealed §3.5 -> artifacts consistent;
#   * MISMATCH cause: ensemble5 npz re-encoded with the PER-SITE TRAINING normalization stats,
#     while earlier cells used rounded manuscript constants (gap TKE -4.8% = SD 100 vs ~104.8).
# THIS CELL: (0) derives candidate decoders from targets_forcing + split (C0 rounded /
#     C1 train / C2 train+val / C3 all points) and AUTO-SELECTS the one reproducing sealed
#     §3.1.1 pool MAEs (gate: max|dMAE| < 0.003); (A) §3.10 5% reference tails (threshold,
#     n, MAE, amplification, bias, recovery%, flagged-ratio); (B) comfort subsets (Tmrt>=60,
#     Ta>=35); (C) §3.11 worst-5% failure lifts (SVF terciles NaN-safe, distance terciles,
#     vegetation, pedestrian band, solar regime, sun-exposed/shaded from tf sun_hit);
#     (D) Figure 16 re-render + extremes_failures_summary.json.
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
FIG = W / '03_Results' / '03_Figures'; FIG.mkdir(parents=True, exist_ok=True)
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza'); K_PED = 4
AK = ('T', 'RelHum', 'WindSpd', 'TKE', 'TMRT')
AL = (r'$T_a$', r'$RH$', r'$V$', r'$TKE$', r'$T_{mrt}$')
SEALED = {'canyon': (0.383, 1.222, 0.273, 3.470, 1.111),
          'plaza': (0.365, 1.285, 0.351, 2.109, 1.472)}
TAIL_VARS = (0, 2, 3, 4)          # Ta, V, TKE, Tmrt (RH omitted per section scope)
GATE = 0.003
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'mathtext.fontset': 'stix'})

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

def ter3(x):
    """NaN-safe tercile labels (-1 = invalid); None if any finite bin < 1%."""
    if x is None: return None, 0.0
    fin = np.isfinite(x)
    t = np.full(x.shape, -1, dtype=np.int8)
    if fin.sum() < 1000: return None, 0.0
    e = np.nanquantile(x[fin], [1 / 3, 2 / 3])
    t[fin] = np.digitize(x[fin], e)
    frac = np.bincount(t[fin], minlength=3) / fin.sum()
    if (frac < 0.01).any(): return None, float(fin.mean())
    return t, float(fin.mean())

DECODERS = {}
summary = {}
for SITE in SITES:
    print(f'\n########## [{SITE}] ##########')
    tf = load_npz(f'targets_forcing_{SITE}.npz')
    split = load_npz(f'split_{SITE}.npz')
    d = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    svf = load_npz(f'svf_{SITE}.npz')
    vf = load_npz(f'vdei_features_{SITE}.npz')
    if any(x is None for x in (tf, split, d, vf)): continue
    n_points, n_time = tf['target_T'].shape
    spl = np.asarray(split['split']).astype(str)
    ix_te = np.where(spl == 'test')[0]; n_test = len(ix_te)
    k0 = np.asarray(tf['k0'])[ix_te]

    # ---- (0) decoder auto-selection ----
    m_masks = {'C1_train': spl == 'train',
               'C2_train+val': np.isin(spl, ('train', 'val')),
               'C3_all': np.ones(spl.shape, bool)}
    cands = {'C0_rounded': (np.array([28., 40., 2., 50., 45.], np.float32),
                            np.array([4., 15., 1.5, 100., 20.], np.float32))}
    for nm, m in m_masks.items():
        mus, sds = [], []
        for k in AK:
            v = np.asarray(tf[f'target_{k}'], np.float64)[m]
            mus.append(v.mean()); sds.append(v.std()); del v
        cands[nm] = (np.array(mus, np.float32), np.array(sds, np.float32))
    Pn = np.asarray(d['pred_air'], np.float32).reshape(n_test, n_time, 5)
    Tn = np.asarray(d['target_air'], np.float32).reshape(n_test, n_time, 5)
    mae_norm = np.abs(Pn - Tn).reshape(-1, 5).mean(0)      # mu cancels; MAE_phys = MAE_norm*sd
    print('  decoder candidates (max|dMAE| vs sealed §3.1.1):')
    best_nm, best_err = None, np.inf
    for nm, (mu, sd) in cands.items():
        err = float(np.max(np.abs(mae_norm * sd - np.array(SEALED[SITE]))))
        print(f'    {nm:12} sd=({", ".join(f"{s:7.3f}" for s in sd)})  max|d|={err:.4f}')
        if err < best_err: best_nm, best_err = nm, err
    mu, sd = cands[best_nm]
    ok = best_err < GATE
    print(f'  SELECTED decoder = {best_nm} (max|d|={best_err:.4f}) '
          f'{"OK - physical scale resolved" if ok else "!! GATE FAILED - inspect, do not write text"}')
    DECODERS[SITE] = {'selected': best_nm, 'max_dMAE': best_err,
                      'mu': mu.tolist(), 'sd': sd.tolist()}

    P = Pn * sd + mu; T = Tn * sd + mu
    E = P - T; ae = np.abs(E)
    mae_pool = ae.reshape(-1, 5).mean(0)
    print('  cross-check MAE (must match §3.1.1): ' +
          '  '.join(f'{AL[v]}={mae_pool[v]:.3f}' for v in range(5)))
    ok2 = max(abs(float(mae_pool[v]) - SEALED[SITE][v]) for v in range(5))
    print(f'  cross-check max|d| vs sealed = {ok2:.4f} {"OK" if ok2 < GATE else "!! MISMATCH - paste back"}')

    # solar regimes (sealed convention: ref domain-mean Tmrt terciles)
    domref = T[:, :, 4].mean(0)
    q1, q2 = np.quantile(domref, [1 / 3, 2 / 3])
    reg = np.where(domref <= q1, 0, np.where(domref <= q2, 1, 2))

    # ---- descriptors (test points) ----
    svf_em = np.asarray(svf['svf_envimet'], np.float32)[ix_te] if 'svf_envimet' in svf.files else None
    svf_ray = np.asarray(svf['svf_rays'], np.float32)[ix_te] if 'svf_rays' in svf.files else None
    ck = 'cls' if 'cls' in vf.files else next(k for k in vf.files if 'class' in k.lower())
    cls = np.asarray(vf[ck])[ix_te]
    print(f'  cls {cls.shape} code freq: ' +
          ', '.join(f'{c}:{(cls == c).mean():.3f}' for c in range(4)))
    f_veg = (cls == 2).mean(1)
    dk = 'dist' if 'dist' in vf.files else None
    dmean = np.asarray(vf[dk], np.float32)[ix_te].mean(1) if dk else None
    sun = None
    if 'sun_hit' in tf.files:
        s = np.asarray(tf['sun_hit'])
        if s.shape[0] == n_points: s = s[ix_te]
        if s.ndim == 2 and s.shape[0] == n_test:
            sun = s
            u, c_ = np.unique(sun, return_counts=True)
            print(f'  sun flags: tf.sun_hit shape={sun.shape} values={u} '
                  f'freq={np.round(c_ / sun.size, 3)}')
    if sun is None: print('  sun flags: NOT FOUND - exposure strata skipped')
    veg_pt = f_veg > 0; ped = k0 == K_PED
    svf_ter, vfrac = ter3(svf_em)
    svf_src = 'ENVI-met'
    if svf_ter is None:
        svf_ter, vfrac = ter3(svf_ray); svf_src = 'ray'
    d_ter, _ = ter3(dmean)
    if svf_ter is not None:
        print(f'  SVF stratifier={svf_src} valid={vfrac:.3f} '
              f'bins={np.round(np.bincount(svf_ter[svf_ter >= 0]) / max((svf_ter >= 0).sum(), 1), 3)}')
    if d_ter is not None:
        print(f'  dist tercile bins={np.round(np.bincount(d_ter[d_ter >= 0]) / max((d_ter >= 0).sum(), 1), 3)}')
    print(f'  base rates (test pts): veg_any={veg_pt.mean():.3f} ped={ped.mean():.3f}')

    # ---- (A) §3.10 extreme tails ----
    print('\n  ===== §3.10 EXTREME-TAIL STATISTICS (reference-defined 5% tails) =====')
    print(f'  {"var":>8} {"tail":>5} {"thresh":>8} {"n":>9} {"MAE_tail":>9} {"MAE_pool":>9} '
          f'{"amplif":>8} {"bias":>8} {"ref_mean":>8} {"pred_mean":>9} {"recov%":>7} {"flagR":>6}')
    site = {'decoder': best_nm, 'mae_pool': {AL[v]: float(mae_pool[v]) for v in range(5)},
            'tails': {}, 'comfort': {}, 'failure': {}}
    for v in TAIL_VARS:
        r = T[:, :, v].ravel().astype(np.float64)
        p = P[:, :, v].ravel().astype(np.float64)
        e = E[:, :, v].ravel().astype(np.float64)
        q95, q05 = np.quantile(r, [0.95, 0.05])
        for tag in ('upper', 'lower'):
            thr = q95 if tag == 'upper' else q05
            mask = (r >= thr) if tag == 'upper' else (r <= thr)
            clear = (p >= q95) if tag == 'upper' else (p <= q05)
            n = int(mask.sum())
            mae_t = float(np.abs(e[mask]).mean()); bias = float(e[mask].mean())
            rec = float(clear[mask].mean()) * 100
            flagR = float(clear.mean()) / 0.05
            print(f'  {AL[v]:>8} {tag:>5} {thr:>8.2f} {n:>9,} {mae_t:>9.3f} {mae_pool[v]:>9.3f} '
                  f'{mae_t / float(mae_pool[v]):>7.2f}x {bias:>+8.3f} {r[mask].mean():>8.2f} '
                  f'{p[mask].mean():>9.2f} {rec:>6.1f}% {flagR:>6.2f}')
            site['tails'][f'{AK[v]}_{tag}'] = dict(
                thr=float(thr), n=n, mae_tail=mae_t,
                amplif=mae_t / float(mae_pool[v]), bias=bias,
                ref_mean=float(r[mask].mean()), pred_mean=float(p[mask].mean()),
                recovery_pct=rec, flagged_ratio=flagR)
        del r, p, e

    # ---- (B) comfort-relevant subsets ----
    print('\n  ===== COMFORT-RELEVANT SUBSETS =====')
    for nm, v, thr in ((r'$T_{mrt}$>=60C', 4, 60.0), (r'$T_a$>=35C', 0, 35.0)):
        m = T[:, :, v] >= thr
        if m.sum() == 0:
            print(f'  {nm}: no samples'); continue
        e_m = E[:, :, v][m]
        amp = float(np.abs(e_m).mean()) / float(mae_pool[v])
        print(f'  {nm}: n={int(m.sum()):,} ({100 * float(m.mean()):.2f}% of pool)  '
              f'MAE={np.abs(e_m).mean():.3f} ({amp:.2f}x pool)  bias={e_m.mean():+.3f}')
        site['comfort'][nm] = dict(n=int(m.sum()), share=float(m.mean()),
                                   mae=float(np.abs(e_m).mean()), amplif=amp,
                                   bias=float(e_m.mean()))

    # ---- (C) §3.11 failure-mode attribution ----
    print('\n  ===== §3.11 FAILURE-MODE ATTRIBUTION (worst 5% |err|, lift vs base) =====')
    for v in (0, 4, 2, 3):
        A = ae[:, :, v]; worst = A >= np.quantile(A, 0.95)
        def lf(mask_pt=None, mask_samp=None):
            m = mask_pt if mask_pt is not None else mask_samp
            base = float(m.mean())
            if base <= 0: return float('nan')
            hit = float(worst[m].mean()) if mask_pt is not None else float(worst[:, m].mean())
            return hit / base
        row = {}
        if svf_ter is not None:
            for i, t_ in enumerate(('SVF_low', 'SVF_mid', 'SVF_high')):
                row[t_] = lf(mask_pt=svf_ter == i)
        if d_ter is not None:
            row['dist_near'] = lf(mask_pt=d_ter == 0)
            row['dist_far'] = lf(mask_pt=d_ter == 2)
        row['veg_any'] = lf(mask_pt=veg_pt)
        row['ped_band'] = lf(mask_pt=ped)
        for i, t_ in enumerate(('night', 'trans', 'peak')):
            row[f'reg_{t_}'] = lf(mask_samp=reg == i)
        if sun is not None:
            mcol = sun.shape[1]; w = worst[:, :mcol]
            exp = sun == 2; shd = sun == 1
            if exp.any(): row['sun_exposed'] = float(w[exp].mean() / exp.mean())
            if shd.any(): row['shaded'] = float(w[shd].mean() / shd.mean())
        print(f'  [{AL[v]}] worst5% n={int(worst.sum()):,}: ' +
              '  '.join(f'{k}={val:.2f}' for k, val in row.items()))
        site['failure'][AK[v]] = row
        if v == 4:
            if svf_ter is not None:
                print('    Tmrt pool MAE by SVF tercile: ' +
                      '  '.join(f'{t}={A[svf_ter == i].mean():.2f}'
                                for i, t in enumerate(('low', 'mid', 'high'))))
            if sun is not None:
                mcol = sun.shape[1]
                print(f'    Tmrt pool MAE daylight: exposed={np.abs(E[:, :mcol, 4])[sun == 2].mean():.2f} '
                      f'shaded={np.abs(E[:, :mcol, 4])[sun == 1].mean():.2f} (daylight n={sun.size:,})')
        del A, worst
    summary[SITE] = site
    del P, T, E, ae, Pn, Tn, d

# ---- (D) FIGURE 16 ----
if len(summary) == 2:
    TV = [0, 2, 3, 4]; TVL = [AL[v] for v in TV]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    ax = axes[0]; wb = 0.2
    for si, (S, c_) in enumerate(zip(SITES, ('#4C72B0', '#C44E52'))):
        up = [summary[S]['tails'][f'{AK[v]}_upper']['amplif'] for v in TV]
        lo = [summary[S]['tails'][f'{AK[v]}_lower']['amplif'] for v in TV]
        xs = np.arange(len(TV))
        ax.bar(xs + (2 * si - 1.5) * wb, up, wb, color=c_, label=f'{S} upper 5%')
        ax.bar(xs + (2 * si - 0.5) * wb, lo, wb, color=c_, alpha=0.45, label=f'{S} lower 5%')
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xticks(np.arange(len(TV))); ax.set_xticklabels(TVL)
    ax.set_ylabel('MAE amplification (tail / pool)')
    ax.set_title('(a) Extreme-tail error amplification (5% reference tails)')
    ax.legend(fontsize=7)
    ax = axes[1]
    STR = ['SVF_low', 'SVF_mid', 'SVF_high', 'dist_near', 'dist_far', 'veg_any',
           'ped_band', 'reg_night', 'reg_trans', 'reg_peak', 'sun_exposed', 'shaded']
    xs = np.arange(len(STR))
    for si, (S, c_) in enumerate(zip(SITES, ('#4C72B0', '#C44E52'))):
        vals = [summary[S]['failure']['TMRT'].get(k, np.nan) for k in STR]
        ax.bar(xs + (si - 0.5) * 0.38, vals, 0.38, color=c_, label=S)
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xticks(xs); ax.set_xticklabels(STR, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('lift (share among worst 5% / base)')
    ax.set_title(r'(b) Failure-mode lift - worst 5% $T_{mrt}$ errors')
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    out = FIG / 'fig16_extremes_failures.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f'\n  [SAVED] {out.name}')

(MET / 'extremes_failures_summary.json').write_text(json.dumps(
    {'decoders': DECODERS, 'sites': summary}, indent=2))
print('\n===== VERDICT HINTS =====')
print(' * decoder line MUST print "OK - physical scale resolved" at both sites; else paste back')
print(' * expect: hot tails amplified 2-3x with negative bias (compression); TKE upper 8-11x')
print(' * failure lifts: ped_band ~5-11x; low-SVF/dist-near/sun-exposed > 1 for Tmrt; night ~0')
print('===== DONE - paste this entire output back =====')
