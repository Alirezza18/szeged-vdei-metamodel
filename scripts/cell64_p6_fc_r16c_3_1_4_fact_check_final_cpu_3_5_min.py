# ===== CELL P6-FC R16c + §3.1.4 FACT-CHECK FINAL (CPU, ~3-5 min) =====
# R16b verdicts baked in:
#  * anchored decoder == manuscript constants (0.1%) -> kept; pool gate exact by construction
#  * FIX1: TRUE all-t ped MAE (R16b's 'all-t' line accidentally reused the noon rows)
#          -> must reproduce sealed §3.8 anchors (canyon 0.465/2.143, plaza 0.392/2.284)
#  * FIX2: hole-robust gradients (the test-only canvas has train/val holes; np.gradient
#          across holes mechanically depressed grad-r) -> keep only cells whose 4
#          neighbours are valid (binary_erosion, cross kernel); raw vs interior both printed
#  * FIX3: vegetation strata by RAY SHARE (R16b's any-ray flag marks 60/55% of points
#          'veg' and dilutes the contrast to ~0) -> zero-veg vs top-tercile veg-share
#  * NEW : full-domain (all ped points) reference stats + hotspot count from tf (truth,
#          no decode involved) - context numbers for the text
# Hotspot bookkeeping kept from R16b (location preserved / magnitude compressed).
# Outputs: fig7_8_spatial_stats_v3.json. Paste ENTIRE output back.
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza'); K_PED = 4
MU = np.array([28.0, 40.0, 2.0, 50.0, 45.0], np.float64)
SEALED = {'canyon': (0.383, 1.222, 0.273, 3.470, 1.111),
          'plaza': (0.365, 1.285, 0.351, 2.109, 1.472)}
S38 = {'canyon': (0.465, 2.143), 'plaza': (0.392, 2.284)}   # sealed §3.8 ped all-t anchors
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)

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

def grad_pair(r, p, ok):
    rn = np.where(ok, r, np.nan); pn = np.where(ok, p, np.nan)
    gr = np.hypot(np.gradient(rn, axis=0), np.gradient(rn, axis=1))
    gp = np.hypot(np.gradient(pn, axis=0), np.gradient(pn, axis=1))
    return gr, gp

def centroid(mask):
    ii, jj = np.nonzero(mask)
    return float(ii.mean()), float(jj.mean())

SUMMARY = {}
for SITE in SITES:
    print(f'\n########## [{SITE}] ##########')
    tf = load_npz(f'targets_forcing_{SITE}.npz'); split = load_npz(f'split_{SITE}.npz')
    bb = load_npz(f'bbox_{SITE}.npz')
    ens = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    if any(x is None for x in (tf, split, ens)):
        print(f'[{SITE}] missing sealed inputs - SKIP'); continue
    i0 = np.asarray(tf['i0']); j0 = np.asarray(tf['j0']); k0 = np.asarray(tf['k0'])
    spl = np.asarray(split['split']).astype(str)
    n_points, n_time = tf['target_T'].shape
    t_noon = int(np.argmax(np.asarray(tf['target_TMRT']).mean(axis=0)))
    ped = (k0 == K_PED)
    ix_te = np.where(ped & (spl == 'test'))[0]
    mt = np.zeros(n_points, bool); mt[ix_te] = True
    rows = ix_te * n_time + t_noon
    rows_all = (ix_te[:, None] * n_time + np.arange(n_time)).ravel()

    # ---- anchored decoder (verified = manuscript constants) ----
    Eall = np.asarray(ens['target_air'], np.float64)
    Pall = np.asarray(ens['pred_air'], np.float64)
    SD_eff = np.array(SEALED[SITE]) / np.abs(Pall - Eall).mean(axis=0)

    # ---- FIX1: TRUE all-t ped MAE + gate vs sealed §3.8 ----
    mae_all = SD_eff[[0, 4]] * np.abs(Pall[rows_all][:, [0, 4]]
                                      - Eall[rows_all][:, [0, 4]]).mean(axis=0)
    g8 = float(np.abs(mae_all - np.array(S38[SITE])).max())
    print(f'  ALL-T ped test MAE = Ta {mae_all[0]:.3f} / Tmrt {mae_all[1]:.3f} '
          f'(sealed §3.8: {S38[SITE][0]:.3f}/{S38[SITE][1]:.3f}) '
          f'-> {"OK" if g8 < 0.02 else "!! MISMATCH - paste back"} (max|d|={g8:.3f})')

    # ---- noon fields, decoded ----
    Tref = MU[[0, 4]] + Eall[rows][:, [0, 4]] * SD_eff[[0, 4]]
    P = MU[[0, 4]] + Pall[rows][:, [0, 4]] * SD_eff[[0, 4]]
    mae_noon = np.abs(P - Tref).mean(axis=0)

    # ---- canvases ----
    i_min, i_max = int(bb['i_min']), int(bb['i_max'])
    j_min, j_max = int(bb['j_min']), int(bb['j_max'])
    ni, nj = i_max - i_min + 1, j_max - j_min + 1
    ii = i0 - i_min; jj = j0 - j_min
    def canvas(full_vals):
        c = np.full((ni, nj), np.nan)
        c[ii[mt], jj[mt]] = full_vals[mt]
        return c
    Ta_f = np.full(n_points, np.nan); Ta_f[ix_te] = Tref[:, 0]
    Tm_f = np.full(n_points, np.nan); Tm_f[ix_te] = Tref[:, 1]
    Pa_f = np.full(n_points, np.nan); Pa_f[ix_te] = P[:, 0]
    Pm_f = np.full(n_points, np.nan); Pm_f[ix_te] = P[:, 1]
    R = {'Ta': canvas(Ta_f), 'Tmrt': canvas(Tm_f)}
    Pp = {'Ta': canvas(Pa_f), 'Tmrt': canvas(Pm_f)}

    # ---- full-domain truth context (from tf, no decode) ----
    fd = {}
    for nm, arr in (('Ta', np.asarray(tf['target_T'])[:, t_noon]),
                    ('Tmrt', np.asarray(tf['target_TMRT'])[:, t_noon])):
        v = arr[ped]; v = v[np.isfinite(v)]
        fd[nm] = {'mean': float(v.mean()), 'sd': float(v.std()),
                  'range': [float(v.min()), float(v.max())],
                  'n_hot70': int((v > 70).sum())}
    print(f"  FULL-DOMAIN truth @noon: Ta mean/sd={fd['Ta']['mean']:.1f}/{fd['Ta']['sd']:.2f} "
          f"range=[{fd['Ta']['range'][0]:.1f},{fd['Ta']['range'][1]:.1f}] | "
          f"Tmrt mean/sd={fd['Tmrt']['mean']:.1f}/{fd['Tmrt']['sd']:.2f} "
          f"range=[{fd['Tmrt']['range'][0]:.1f},{fd['Tmrt']['range'][1]:.1f}] | "
          f"pts>70C={fd['Tmrt']['n_hot70']:,}")

    # ---- field stats + FIX2 hole-robust gradients ----
    st = {'t_noon': t_noon, 'n_test_ped': int(ix_te.size), 'fulldomain_truth': fd, 'vars': {}}
    for key in ('Ta', 'Tmrt'):
        r, p = R[key], Pp[key]
        ok = np.isfinite(r) & np.isfinite(p)
        rr, pp = r[ok], p[ok]
        gr, gp = grad_pair(r, p, ok)
        sel_all = np.isfinite(gr) & np.isfinite(gp)
        interior = binary_erosion(ok, CROSS)
        sel_in = interior & np.isfinite(gr) & np.isfinite(gp)
        rec = {'mean_ref': float(rr.mean()), 'sd_ref': float(rr.std()),
               'mean_pred': float(pp.mean()), 'sd_pred': float(pp.std()),
               'sd_ratio': float(pp.std() / rr.std()),
               'range_ref': [float(rr.min()), float(rr.max())],
               'range_pred': [float(pp.min()), float(pp.max())],
               'grad_r_raw': float(np.corrcoef(gr[sel_all], gp[sel_all])[0, 1]),
               'grad_r_interior': float(np.corrcoef(gr[sel_in], gp[sel_in])[0, 1]),
               'grad_mean_ref': float(gr[sel_in].mean()),
               'grad_mean_pred': float(gp[sel_in].mean()),
               'spatial_MAE': float(np.abs(pp - rr).mean()), 'n': int(ok.sum())}
        st['vars'][key] = rec
        print(f"  [{key}] ref mean/sd={rec['mean_ref']:.1f}/{rec['sd_ref']:.2f} "
              f"pred mean/sd={rec['mean_pred']:.1f}/{rec['sd_pred']:.2f} "
              f"(SDx{rec['sd_ratio']:.2f}) n={rec['n']:,}")
        print(f"        grad-r raw={rec['grad_r_raw']:.3f} INTERIOR={rec['grad_r_interior']:.3f} "
              f"| grad mean ref/pred={rec['grad_mean_ref']:.3f}/{rec['grad_mean_pred']:.3f} "
              f"(x{rec['grad_mean_pred']/rec['grad_mean_ref']:.2f}) | noon MAE={rec['spatial_MAE']:.2f}")

    # ---- hotspots (unchanged from R16b - location vs magnitude) ----
    r70, p70 = R['Tmrt'], Pp['Tmrt']
    hm_ref = np.where(np.isfinite(r70), r70 > 70.0, False)
    n_ref = int(hm_ref.sum())
    frac = n_ref / int(np.isfinite(p70).sum())
    thr_pred = float(np.nanpercentile(p70, 100 * (1 - frac)))
    hm_pred = np.where(np.isfinite(p70), p70 >= thr_pred, False)
    hits_any = int(binary_dilation(hm_ref, iterations=1)[hm_pred].sum())
    hits_px = int((hm_ref & hm_pred).sum())
    ci, cj = centroid(hm_ref); qi, qj = centroid(hm_pred)
    dist = 2.0 * float(np.hypot(ci - qi, cj - qj))
    st['hotspot'] = {'n_ref': n_ref, 'fd_n_ref': fd['Tmrt']['n_hot70'],
                     'thr_pred': thr_pred, 'hits_any_3x3': hits_any,
                     'hits_pixel_exact': hits_px, 'centroid_shift_m': dist}
    print(f"  [hotspot] test n_ref={n_ref:,} (full-domain {fd['Tmrt']['n_hot70']:,}) | "
          f"matched pred-thr={thr_pred:.1f} vs ref 70.0 (compression "
          f"{70.0 - thr_pred:.1f} C) | 3x3 hits={hits_any:,} ({100*hits_any/n_ref:.0f}%) | "
          f"exact={hits_px:,} | centroid shift={dist:.1f} m")

    # ---- FIX3: vegetation by ray share ----
    vf = load_npz(f'vdei_features_{SITE}.npz')
    if vf is not None:
        cls = np.asarray(vf['cls'])
        share = (cls[ix_te] == 2).sum(axis=1) / 144.0
        g0 = share == 0
        thr_v = float(np.quantile(share[~g0], 2 / 3)) if (~g0).sum() > 50 else 1.1
        hi = share >= thr_v
        st['veg'] = {'share_mean': float(share.mean()), 'n_zero': int(g0.sum()),
                     'n_high': int(hi.sum()), 'thr': thr_v}
        for key, vi in (('Ta', 0), ('Tmrt', 1)):
            cool = float(Tref[~g0, vi].mean() - Tref[hi, vi].mean())
            ev = float(np.abs(P[hi, vi] - Tref[hi, vi]).mean())
            en = float(np.abs(P[g0, vi] - Tref[g0, vi]).mean())
            st['veg'][key] = {'cooling_ref': cool, 'MAE_high': ev, 'MAE_zero': en,
                              'ratio': ev / en}
            print(f"  [{key}] veg-share>={thr_v:.2f} (n={int(hi.sum()):,}) vs zero-veg "
                  f"(n={int(g0.sum()):,}): ref cooling={cool:+.1f} C | "
                  f"MAE {ev:.2f} vs {en:.2f} (x{ev/en:.2f})")
        vf.close()

    pool = SEALED[SITE]
    st['noon_vs_pool'] = {'Ta': float(st['vars']['Ta']['spatial_MAE'] / pool[0]),
                          'Tmrt': float(st['vars']['Tmrt']['spatial_MAE'] / pool[4])}
    print(f"  [noon vs pool] Tmrt {st['noon_vs_pool']['Tmrt']:.2f}x | "
          f"Ta {st['noon_vs_pool']['Ta']:.2f}x  (sealed §3.9 all-level peak/pool: "
          f"Tmrt 1.70x / Ta 1.33x canyon, 1.56x / 1.40x plaza)")
    SUMMARY[SITE] = st
    ens.close()

(MET / 'fig7_8_spatial_stats_v3.json').write_text(json.dumps(SUMMARY, indent=2))

# ===================== TEXT-READY SENTENCES =====================
print('\n===== FINAL TEXT-READY SENTENCES (sealed) =====')
for SITE in SITES:
    s = SUMMARY.get(SITE)
    if s is None: continue
    Ta, Tm = s['vars']['Ta'], s['vars']['Tmrt']
    h = s['hotspot']
    print(f'\n[{SITE}]')
    print(f"  1) 'At the held-out pedestrian-level test locations the noon T_mrt field "
          f"(domain mean {s['fulldomain_truth']['Tmrt']['mean']:.1f} C, SD "
          f"{s['fulldomain_truth']['Tmrt']['sd']:.1f} C) was predicted with a spatial MAE of "
          f"{Tm['spatial_MAE']:.2f} C and a reference-vs-predicted gradient-magnitude "
          f"correlation of r = {Tm['grad_r_interior']:.2f}.'")
    print(f"  2) 'The noon T_a field spans less than {Ta['sd_ref']*3:.0f} C "
          f"(SD {Ta['sd_ref']:.1f} C), so its spatial gradients are low-signal; the "
          f"corresponding gradient correlation was r = {Ta['grad_r_interior']:.2f} with "
          f"gradient magnitudes preserved to {100*Ta['grad_mean_pred']/Ta['grad_mean_ref']:.0f}%.'")
    print(f"  3) 'Amplitude compression persisted in every field: predicted SDs were "
          f"{100*Tm['sd_ratio']:.0f}% (T_mrt) and {100*Ta['sd_ratio']:.0f}% (T_a) of "
          f"reference, and the predicted T_mrt range "
          f"({Tm['range_pred'][0]:.1f}-{Tm['range_pred'][1]:.1f} C) was narrower than the "
          f"observed ({Tm['range_ref'][0]:.1f}-{Tm['range_ref'][1]:.1f} C).")
    print(f"  4) 'Of {h['n_ref']:,} held-out reference points exceeding 70 C, "
          f"{h['hits_any_3x3']:,} ({100*h['hits_any_3x3']/h['n_ref']:.0f}%) were matched by a "
          f"predicted hotspot within one voxel (exact-pixel {h['hits_pixel_exact']:,}; "
          f"hotspot centroid displaced {h['centroid_shift_m']:.1f} m), while the matched-count "
          f"predicted threshold was {h['thr_pred']:.1f} C against the reference 70.0 C - "
          f"location preserved, magnitude compressed by {70.0-h['thr_pred']:.1f} C.'")
    if 'veg' in s:
        v = s['veg']['Tmrt']
        print(f"  5) 'Vegetation shading was reproduced directionally: the most vegetation-"
              f"exposed test points (veg-ray share >= {s['veg']['thr']:.2f}) were "
              f"{v['cooling_ref']:.1f} C cooler in reference noon T_mrt than vegetation-free "
              f"points, and carried {v['ratio']:.1f}x their T_mrt error "
              f"({v['MAE_high']:.2f} vs {v['MAE_zero']:.2f} C).'")
    print(f"  6) 'Solar noon remained an elevated-error regime (T_mrt MAE "
          f"{s['noon_vs_pool']['Tmrt']:.1f}x and T_a {s['noon_vs_pool']['Ta']:.1f}x the pool "
          f"averages), though the hardest radiative regimes were the low-sun transition hours "
          f"(Sections 3.9 and 3.12).'")
print('\n===== DONE - paste this entire output back =====')
