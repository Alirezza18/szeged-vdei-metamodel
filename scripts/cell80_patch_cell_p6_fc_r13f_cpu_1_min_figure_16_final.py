# ===== PATCH CELL P6-FC R13f (CPU, <1 min) - Figure 16 FINAL + sun-flag semantic audit =====
# Reads extremes_failures_summary.json written by R13e (same /kaggle/working session).
# Re-renders BOTH panels:
#   (a) tail amplification (data unchanged - scale-invariant, values as printed by R13e)
#   (b) TRUE ENRICHMENT  e = P(worst-5% | stratum) / 0.05  per Tmrt stratum.
#       R13e printed lf = P(worst|s)/P(s) (lift vs stratum base); enrichment vs the uniform
#       5% worst-rate is the manuscript metric. Bases embedded from the R13e audit.
#   * sun_exposed/shaded EXCLUDED from panel (b): R13e's exposed=code2 mapping (1.4% share)
#     conflicts with SS3.9's sealed exposed=code>0 (9.2%). The audit block recomputes the
#     sun strata per code from targets_forcing.sun_hit + the fullpool npz and prints both
#     interpretations so the mapping can be settled before SS3.11 cites sun strata.
# Outputs: fig16_extremes_failures.png (overwrites) + fig16_panel_b_enrichment.json
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
FIG = W / '03_Results' / '03_Figures'; FIG.mkdir(parents=True, exist_ok=True)
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza')
SDc = np.array([4.0, 15.0, 1.5, 100.0, 20.0], np.float32)   # decode used for R13e ratios
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'mathtext.fontset': 'stix'})

def find(pat):
    for d_ in sorted(IN.glob('*')):
        h = sorted(d_.rglob(pat))
        if h: return h[0]
    return None

S = json.loads((MET / 'extremes_failures_summary.json').read_text())['sites']

# ---------- panel (a): tail amplification (unchanged) ----------
TV = ('T', 'WindSpd', 'TKE', 'TMRT'); TVL = (r'$T_a$', r'$V$', r'$TKE$', r'$T_{mrt}$')
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
ax = axes[0]; wb = 0.2
for si, (site, c_) in enumerate(zip(SITES, ('#4C72B0', '#C44E52'))):
    up = [S[site]['tails'][f'{v}_upper']['amplif'] for v in TV]
    lo = [S[site]['tails'][f'{v}_lower']['amplif'] for v in TV]
    xs = np.arange(len(TV))
    ax.bar(xs + (2 * si - 1.5) * wb, up, wb, color=c_, label=f'{site} upper 5%')
    ax.bar(xs + (2 * si - 0.5) * wb, lo, wb, color=c_, alpha=0.45, label=f'{site} lower 5%')
ax.axhline(1.0, color='k', lw=0.8, ls='--')
ax.set_xticks(np.arange(len(TV))); ax.set_xticklabels(TVL)
ax.set_ylabel('MAE amplification (tail / pool)')
ax.set_title('(a) Extreme-tail error amplification (5% reference tails)')
ax.legend(fontsize=7)

# ---------- panel (b): TRUE enrichment for Tmrt worst-5% ----------
STR = ['SVF_low', 'SVF_mid', 'SVF_high', 'dist_near', 'dist_far', 'veg_any',
       'ped_band', 'reg_night', 'reg_trans', 'reg_peak']
BASE = {  # sample-level stratum bases (R13e audited prints; every test point has all timesteps,
#         so sample base = point base; SVF bins = equal thirds of valid points)
 'canyon': dict(SVF_low=0.268, SVF_mid=0.268, SVF_high=0.268, dist_near=0.333,
                dist_far=0.334, veg_any=0.597, ped_band=0.015,
                reg_night=0.320, reg_trans=0.320, reg_peak=0.360),
 'plaza': dict(SVF_low=0.292, SVF_mid=0.292, SVF_high=0.292, dist_near=0.333,
               dist_far=0.333, veg_any=0.546, ped_band=0.016,
               reg_night=0.327, reg_trans=0.327, reg_peak=0.347)}
ax = axes[1]; xs = np.arange(len(STR))
enr_out = {}
for si, (site, c_) in enumerate(zip(SITES, ('#4C72B0', '#C44E52'))):
    row = S[site]['failure']['TMRT']
    vals = [row.get(k, np.nan) * BASE[site][k] / 0.05 for k in STR]
    ax.bar(xs + (si - 0.5) * 0.38, vals, 0.38, color=c_, label=site)
    enr_out[site] = {k: float(v) for k, v in zip(STR, vals)}
ax.axhline(1.0, color='k', lw=0.8, ls='--')
ax.set_xticks(xs); ax.set_xticklabels(STR, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('enrichment  e = P(worst 5% | stratum) / 0.05')
ax.set_title(r'(b) Failure-mode enrichment - worst 5% $T_{mrt}$ errors')
ax.legend(fontsize=7.5)
fig.tight_layout()
out = FIG / 'fig16_extremes_failures.png'
fig.savefig(out, bbox_inches='tight'); plt.close(fig)
print(f'  [SAVED] {out.name}  (panel b = enrichment)')
(MET / 'fig16_panel_b_enrichment.json').write_text(json.dumps(enr_out, indent=2))

# ---------- sun-flag semantic audit (why sun strata are excluded) ----------
print('\n===== SUN-FLAG SEMANTIC AUDIT (tf.sun_hit) =====')
for SITE in SITES:
    tf = np.load(find(f'targets_forcing_{SITE}.npz'))
    d = np.load(find(f'{SITE}_dualhead_ensemble5_test_fullpool.npz'))
    split = np.load(find(f'split_{SITE}.npz'))
    spl = np.asarray(split['split']).astype(str)
    ix_te = np.where(spl == 'test')[0]
    n_test = len(ix_te); n_time = tf['target_T'].shape[1]
    sun = np.asarray(tf['sun_hit'])[ix_te]
    P = np.asarray(d['pred_air'], np.float32).reshape(n_test, n_time, 5) * SDc
    T = np.asarray(d['target_air'], np.float32).reshape(n_test, n_time, 5) * SDc
    E = np.abs(P - T)
    A = E[:, :, 4]; worst = A >= np.quantile(A, 0.95)
    mcol = sun.shape[1]
    for code, nm in ((0, 'code0'), (1, 'code1'), (2, 'code2'), ('>0', 'exposed>0')):
        m = (sun > 0) if code == '>0' else (sun == code)
        if not m.any(): continue
        print(f'  [{SITE}] {nm}: share={m.mean():.4f}  '
              f'P(worst5%|{nm})={float(worst[:, :mcol][m].mean()):.4f}  '
              f'enrichment={float(worst[:, :mcol][m].mean()) / 0.05:.2f}  '
              f'Tmrt MAE={A[:, :mcol][m].mean():.2f}')
    k0 = np.asarray(tf['k0'])[ix_te]
    ped = k0 == 4
    wped = worst[:, :mcol][ped]
    sped = sun[ped]
    m2 = sped > 0
    if wped.any() and m2.any():
        print(f'  [{SITE}] ped pts: n={int(ped.sum()):,} | ped-worst5% that are exposed(>0)='
              f'{float(wped[m2].mean()):.3f} (exposed share among ped={float(m2.mean()):.3f})')
    del tf, d, split, P, T, E, A, worst, sun
print('\nDECISION RULE (sun strata): SS3.9 sealed convention is exposed=code>0 (9.2%).')
print('If the >0 enrichment/MAE ordering matches SS3.8/SS3.9 (exposed error > shaded,')
print('gap grows toward midday), adopt >0 for SS3.11; otherwise retire sun strata with a note.')
print('===== DONE - paste this entire output back =====')
