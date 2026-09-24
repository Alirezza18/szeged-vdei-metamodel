# ===== CELL P6-FC R16b + §3.1.4 FACT-CHECK (GPU, ~10 min) - SPATIAL FIDELITY vs SEALED =====
# v2 FIXES vs R16: (1) targets are decoded to physical units with the SAME anchored SD_eff
#   (R16 compared denormalized preds against NORMALIZED targets -> 32/66 C nonsense);
#   (2) canvases are filled via a test-point mask (R16 indexed with the all-ped mask ->
#   shape mismatch); (3) the pool gate is EXACT BY CONSTRUCTION (SD_eff = sealed/MAE_norm,
#   float64) and is reported, while noon MAE is a reported statistic (NOT a gate - sealed
#   values are full-pool numbers). Ped-level all-t sanity anchors from sealed §3.8:
#   canyon Ta 0.465 / Tmrt 2.143; plaza 0.392 / 2.284.
# Recomputes EVERY §3.1.4 draft claim from the SEALED ensemble5 pool:
#   (A) ped-level (k=4) solar-noon fields -> mean, SD, gradient-magnitude r, grad mean, MAE
#   (B) amplitude compression: SD ratio, grad ratio, range compression
#   (C) hotspots: n_ref(>70C) / reproduced (3x3 any-overlap + exact pixel) + centroid shift
#   (D) vegetation contrast: ref cooling + MAE amplification (veg vs non-veg)
#   (E) noon-vs-pool MAE amplification factor
#   (F) consistency vs the freshly rendered Figures 7/8 cache + EXACT text-ready sentences
# Outputs: fig7_8_spatial_stats_v2.json. Paste ENTIRE output back.
import json
from pathlib import Path
import numpy as np

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza'); K_PED = 4
MU = np.array([28.0, 40.0, 2.0, 50.0, 45.0], np.float64)
SEALED = {'canyon': (0.383, 1.222, 0.273, 3.470, 1.111),
          'plaza': (0.365, 1.285, 0.351, 2.109, 1.472)}
CACHE = W / 'cache'

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

def gradmag(C):
    g = np.hypot(np.gradient(C, axis=0), np.gradient(C, axis=1))
    return g, np.isfinite(g)

def centroid(mask):
    ii, jj = np.nonzero(mask)
    return float(ii.mean()), float(jj.mean())

SUMMARY = {}
for SITE in SITES:
    print(f'\n########## [{SITE}] ##########')
    tf = load_npz(f'targets_forcing_{SITE}.npz'); split = load_npz(f'split_{SITE}.npz')
    geom = load_npz(f'geometry_{SITE}.npz'); bb = load_npz(f'bbox_{SITE}.npz')
    ens = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    if any(x is None for x in (tf, split, ens)):
        print(f'[{SITE}] missing sealed inputs - SKIP'); continue
    i0 = np.asarray(tf['i0']); j0 = np.asarray(tf['j0']); k0 = np.asarray(tf['k0'])
    spl = np.asarray(split['split']).astype(str)
    n_points, n_time = tf['target_T'].shape
    t_noon = int(np.argmax(np.asarray(tf['target_TMRT']).mean(axis=0)))
    dom_noon = float(np.asarray(tf['target_TMRT'])[:, t_noon].mean())
    print(f'  solar noon t={t_noon} | domain-mean ref T_mrt (all ped pts) = {dom_noon:.1f} C')

    ped = (k0 == K_PED)
    ix_te = np.where(ped & (spl == 'test'))[0]
    mt = np.zeros(n_points, bool); mt[ix_te] = True          # test-ped mask (canvas fill)
    rows = ix_te * n_time + t_noon

    # ---- anchored decoder (R14d verdict): pool MAE == sealed EXACTLY by construction ----
    Eall = np.asarray(ens['target_air'], np.float64)         # normalized, (n_pool,5)
    Pall = np.asarray(ens['pred_air'], np.float64)
    mae_norm = np.abs(Pall - Eall).mean(axis=0)              # float64 mean over full pool
    SD_eff = np.array(SEALED[SITE]) / mae_norm
    gate = float(np.abs(SD_eff * mae_norm - np.array(SEALED[SITE])).max())
    print(f'  anchored SD_eff = ({", ".join(f"{v:.4f}" for v in SD_eff)}) | '
          f'pool gate max|d| = {gate:.2e} (exact by construction)')

    # ---- noon slices, decoded to physical units (BOTH pred and target) ----
    Tn = Eall[rows][:, [0, 4]]; Pn = Pall[rows][:, [0, 4]]
    Tref = MU[[0, 4]] + Tn * SD_eff[[0, 4]]
    P = MU[[0, 4]] + Pn * SD_eff[[0, 4]]
    mae_noon = np.abs(P - Tref).mean(axis=0)
    mae_all_t = SD_eff[[0, 4]] * np.abs(Pall[rows][:, [0, 4]] - Eall[rows][:, [0, 4]]).mean(axis=0)
    print(f'  sanity: ped test MAE all-t = Ta {mae_all_t[0]:.3f} / Tmrt {mae_all_t[1]:.3f} '
          f'(sealed §3.8: {0.465 if SITE=="canyon" else 0.392:.3f} / '
          f'{2.143 if SITE=="canyon" else 2.284:.3f})')
    print(f'  NOON spatial MAE (reported, not a gate) = Ta {mae_noon[0]:.3f} / '
          f'Tmrt {mae_noon[1]:.3f}')

    # ---- 2D canvases (test points only) ----
    i_min, i_max = int(bb['i_min']), int(bb['i_max'])
    j_min, j_max = int(bb['j_min']), int(bb['j_max'])
    ni, nj = i_max - i_min + 1, j_max - j_min + 1
    ii = i0 - i_min; jj = j0 - j_min
    def canvas(full_vals):
        c = np.full((ni, nj), np.nan)
        c[ii[mt], jj[mt]] = full_vals[mt]
        return c
    Ta_all = np.full(n_points, np.nan); Ta_all[ix_te] = Tref[:, 0]
    Tm_all = np.full(n_points, np.nan); Tm_all[ix_te] = Tref[:, 1]
    Pa_all = np.full(n_points, np.nan); Pa_all[ix_te] = P[:, 0]
    Pm_all = np.full(n_points, np.nan); Pm_all[ix_te] = P[:, 1]
    R = {'Ta': canvas(Ta_all), 'Tmrt': canvas(Tm_all)}
    Pp = {'Ta': canvas(Pa_all), 'Tmrt': canvas(Pm_all)}

    # ---- (A) field statistics + gradient preservation ----
    st = {'t_noon': t_noon, 'n_test_ped': int(ix_te.size), 'vars': {}}
    for key in ('Ta', 'Tmrt'):
        r, p = R[key], Pp[key]
        ok = np.isfinite(r) & np.isfinite(p)
        rr, pp = r[ok], p[ok]
        gr, gok_r = gradmag(r); gp, gok_p = gradmag(p)
        nok = gok_r & gok_p
        rgcorr = float(np.corrcoef(gr[nok], gp[nok])[0, 1])
        rec = {'mean_ref': float(rr.mean()), 'sd_ref': float(rr.std()),
               'mean_pred': float(pp.mean()), 'sd_pred': float(pp.std()),
               'sd_ratio': float(pp.std() / rr.std()),
               'range_ref': [float(rr.min()), float(rr.max())],
               'range_pred': [float(pp.min()), float(pp.max())],
               'grad_r': rgcorr,
               'grad_mean_ref': float(gr[nok].mean()), 'grad_mean_pred': float(gp[nok].mean()),
               'spatial_MAE': float(np.abs(pp - rr).mean()), 'n': int(ok.sum())}
        st['vars'][key] = rec
        print(f"  [{key}] ref mean/sd = {rec['mean_ref']:.1f}/{rec['sd_ref']:.1f} | "
              f"pred mean/sd = {rec['mean_pred']:.1f}/{rec['sd_pred']:.1f} "
              f"(SD ratio {rec['sd_ratio']:.2f}) | n={rec['n']:,}")
        print(f"        grad-r = {rgcorr:.3f} | grad mean ref/pred = "
              f"{rec['grad_mean_ref']:.3f}/{rec['grad_mean_pred']:.3f} "
              f"(ratio {rec['grad_mean_pred']/rec['grad_mean_ref']:.2f})")

    # ---- (C) hotspots (reference > 70 C, ped level) ----
    r70, p70 = R['Tmrt'], Pp['Tmrt']
    hm_ref = np.where(np.isfinite(r70), r70 > 70.0, False)
    n_ref = int(hm_ref.sum())
    if n_ref > 0:
        frac = n_ref / int(np.isfinite(p70).sum())
        thr_pred = float(np.nanpercentile(p70, 100 * (1 - frac)))
        hm_pred = np.where(np.isfinite(p70), p70 >= thr_pred, False)
        from scipy.ndimage import binary_dilation
        hits_any = int(binary_dilation(hm_ref, iterations=1)[hm_pred].sum())
        hits_px = int((hm_ref & hm_pred).sum())
        ci, cj = centroid(hm_ref); qi, qj = centroid(hm_pred)
        dist = 2.0 * float(np.hypot(ci - qi, cj - qj))       # 2 m voxels -> metres
        st['hotspot'] = {'n_ref': n_ref, 'thr_pred': thr_pred, 'hits_any_3x3': hits_any,
                         'hits_pixel_exact': hits_px, 'centroid_shift_m': dist}
        print(f"  [hotspot >70C] n_ref={n_ref:,} | matched pred-thr={thr_pred:.1f}C | "
              f"hits(3x3 tol)={hits_any:,} ({100*hits_any/n_ref:.0f}%) | "
              f"exact-pixel={hits_px:,} | centroid shift={dist:.1f} m")
    else:
        print('  [hotspot >70C] no reference points above 70 C at ped level')

    # ---- (D) vegetation contrast (vdei cls code 2 = vegetation, any ray) ----
    vf = load_npz(f'vdei_features_{SITE}.npz')
    if vf is not None:
        cls = np.asarray(vf['cls'])
        veg_te = (cls[ix_te] == 2).any(axis=1)               # test points only
        for key, vi in (('Ta', 0), ('Tmrt', 1)):
            rv, rn = Tref[veg_te, vi], Tref[~veg_te, vi]
            ev = np.abs(P[veg_te, vi] - rv); en = np.abs(P[~veg_te, vi] - rn)
            rec = st['vars'][key]
            rec['veg_cooling_ref'] = float(rn.mean() - rv.mean())
            rec['veg_MAE'] = float(ev.mean()); rec['nonveg_MAE'] = float(en.mean())
            rec['veg_MAE_ratio'] = float(ev.mean() / en.mean())
            rec['n_veg'] = int(veg_te.sum())
            print(f"  [{key}] veg cooling (ref) = {rec['veg_cooling_ref']:.1f} C | "
                  f"MAE veg/non = {ev.mean():.2f}/{en.mean():.2f} "
                  f"(x{rec['veg_MAE_ratio']:.2f}, n_veg={rec['n_veg']:,})")
        vf.close()

    # ---- (E) noon-vs-pool amplification (pool = sealed) ----
    pool = SEALED[SITE]
    st['noon_vs_pool'] = {'Ta': float(st['vars']['Ta']['spatial_MAE'] / pool[0]),
                          'Tmrt': float(st['vars']['Tmrt']['spatial_MAE'] / pool[4])}
    print(f"  [noon amplification] Ta {st['vars']['Ta']['spatial_MAE']:.2f}/{pool[0]:.2f} = "
          f"{st['noon_vs_pool']['Ta']:.2f}x | Tmrt {st['vars']['Tmrt']['spatial_MAE']:.2f}/"
          f"{pool[4]:.2f} = {st['noon_vs_pool']['Tmrt']:.2f}x")

    # ---- (F) consistency vs freshly rendered Figures 7/8 (R5c cache) ----
    cpath = CACHE / f'{SITE}_k4noon_preds.npz'
    if cpath.exists():
        z = np.load(cpath)
        ours = {'Ta': z['pred_T'][ix_te].astype(np.float64) * 4.0 + 28.0,
                'Tmrt': z['pred_TMRT'][ix_te].astype(np.float64) * 20.0 + 45.0}
        dmax = max(float(np.abs(ours['Ta'] - P[:, 0]).max()),
                   float(np.abs(ours['Tmrt'] - P[:, 1]).max()))
        print(f'  [fig7/8 cache check] max|d| vs anchored sealed decode = {dmax:.3f} '
              f'({"OK - same model, rounded-constant display decode" if dmax < 2.5 else "!! different decode - note in Methods"})')
        z.close()
    SUMMARY[SITE] = st
    ens.close()

(MET / 'fig7_8_spatial_stats_v2.json').write_text(json.dumps(SUMMARY, indent=2))

# ===================== EXACT TEXT-READY SENTENCES =====================
print('\n===== DRAFT-CLAIM VERDICTS + TEXT-READY SENTENCES =====')
for SITE in SITES:
    s = SUMMARY.get(SITE)
    if s is None: continue
    Ta, Tm = s['vars']['Ta'], s['vars']['Tmrt']
    print(f'\n[{SITE}]')
    print(f"  grad-r: 'The Pearson correlation between reference and predicted "
          f"gradient-magnitude fields was {Tm['grad_r']:.3f} for T_mrt and {Ta['grad_r']:.3f} "
          f"for T_a.'  (draft: Tmrt/Ta listed first - keep var order consistent)")
    print(f"  domain stats: 'the reference T_mrt field (mean {Tm['mean_ref']:.1f} C, "
          f"SD {Tm['sd_ref']:.1f} C) was predicted with mean {Tm['mean_pred']:.1f} C and "
          f"SD {Tm['sd_pred']:.1f} C' | spatial MAE: Ta {Ta['spatial_MAE']:.2f} / "
          f"Tmrt {Tm['spatial_MAE']:.2f}")
    print(f"  compression: 'predicted SDs were {100*Tm['sd_ratio']:.0f}% (T_mrt) and "
          f"{100*Ta['sd_ratio']:.0f}% (T_a) of reference; gradient means "
          f"{Tm['grad_mean_pred']:.3f} vs {Tm['grad_mean_ref']:.3f} (T_mrt), "
          f"{Ta['grad_mean_pred']:.3f} vs {Ta['grad_mean_ref']:.3f} (T_a) C/voxel; "
          f"predicted range {Tm['range_pred'][0]:.1f}-{Tm['range_pred'][1]:.1f} vs observed "
          f"{Tm['range_ref'][0]:.1f}-{Tm['range_ref'][1]:.1f} C'")
    print(f"  noon amplification: '{s['noon_vs_pool']['Tmrt']:.1f}x (T_mrt), "
          f"{s['noon_vs_pool']['Ta']:.1f}x (T_a)'  (draft: ~3.5x / ~2.5x)")
    if 'hotspot' in s:
        h = s['hotspot']
        print(f"  hotspots: 'of {h['n_ref']:,} reference points above 70 C, "
              f"{h['hits_any_3x3']:,} were reproduced within one voxel "
              f"({100*h['hits_any_3x3']/h['n_ref']:.0f}%; exact-pixel {h['hits_pixel_exact']:,}), "
              f"with the matched predicted hotspot centroid displaced by "
              f"{h['centroid_shift_m']:.1f} m'")
    if 'veg_MAE_ratio' in Tm:
        print(f"  vegetation: 'reference T_mrt in vegetated pedestrian points was "
              f"{Tm['veg_cooling_ref']:.1f} C cooler; absolute error at vegetated points was "
              f"{Tm['veg_MAE_ratio']:.1f}x the non-vegetated error ({Tm['veg_MAE']:.2f} vs "
              f"{Tm['nonveg_MAE']:.2f} C)'")
print('\n===== DONE - paste this entire output back =====')
