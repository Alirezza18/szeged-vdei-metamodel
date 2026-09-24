# ===== CELL P6-FC R1 + FIGURE 6 (CPU, zero GPU quota) - §3.1.1 fact-check + performance figure =====
# Verifies EVERY number in the §3.1.1 draft and renders Figure 6 (R2 / MAE / RMSE,
# air + facade heads, both sites, 5-seed mean +- SD, physical units).
# Inputs (attached): szeged-backup (optuna_best seed0-4 + ensemble5 npz),
#                    szeged-vdei-processed (facade_targets_{site}.npz for per-site stats).
# Outputs: 03_Results/02_Metrics/fig6_perf_summary.json
#          03_Results/03_Figures/fig06_performance_summary.png
# Runtime: ~3-5 min CPU (loads 10 fullpool npz). Paste the whole output back.
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN = Path('/kaggle/input')
W = Path('/kaggle/working')
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
FIG = W / '03_Results' / '03_Figures'; FIG.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza')
SEEDS = (0, 1, 2, 3, 4)
# (key, (mu, sd), manuscript label, unit)  -- shared air constants (declared, sealed)
AIR = [('T', (28.0, 4.0), r'$T_a$', 'degC'),
       ('RelHum', (40.0, 15.0), r'$RH$', '%'),
       ('WindSpd', (2.0, 1.5), r'$V$', 'm/s'),
       ('TKE', (50.0, 100.0), r'$TKE$', 'm2/s2'),
       ('TMRT', (45.0, 20.0), r'$T_{mrt}$', 'degC')]
FAC = [('Twall', r'$T_{wall}$', 'degC'), ('Qsens', r'$Q_{sens}$', 'W/m2'),
       ('SWabs', r'$SW_{abs}$', 'W/m2'), ('LWbal', r'$LW_{bal}$', 'W/m2')]
AIR_KEYS = [a[0] for a in AIR]
FAC_KEYS = [f[0] for f in FAC]
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'mathtext.fontset': 'stix'})
COLORS = {'canyon': '#4C72B0', 'plaza': '#D55E00'}


def find(pat):
    for d in sorted(IN.glob('*')):
        h = sorted(d.rglob(pat))
        if h:
            return h[0]
    return None


def style(ax):
    """grid network in the background of every panel (locked figure style)"""
    ax.set_axisbelow(True)
    ax.grid(True, which='major', ls=':', lw=0.5, color='0.55', alpha=0.7)
    ax.grid(True, which='minor', ls=':', lw=0.3, color='0.8', alpha=0.5)
    return ax


def load(pat):
    p = find(pat)
    if p is None:
        print(f'  !! missing: {pat}')
        return None
    print(f'  [{pat}] <- {p.parent.parent.name}/{p.parent.name}/{p.name}')
    return np.load(p)


def facade_stats(site):
    fc = load(f'facade_targets_{site}.npz')
    if fc is None:
        return None, None
    mu = np.asarray(fc['facade_norm_mean'], dtype=np.float64)
    sd = np.asarray(fc['facade_norm_std'], dtype=np.float64)
    fc.close()
    return mu, sd


def metrics(p, t):
    """MAE, RMSE, R2 in physical units (float64). p,t already physical."""
    p = p.astype(np.float64); t = t.astype(np.float64)
    d = p - t
    ss_res = float((d * d).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(np.abs(d).mean()), float(np.sqrt((d * d).mean())), r2


summary = {'meta': {'config': 'optuna_best', 'seeds': list(SEEDS),
                    'pool': 'full test pool (physical units)',
                    'air_norm': 'shared declared constants',
                    'facade_norm': 'per-site sealed stats',
                    'note': 'R2 computed here (metrics log stores MAE/RMSE only)'},
           'sites': {}}
# per-seed metric arrays for the figure: seedstore[(site, head, var)] -> (5, 3) [MAE, RMSE, R2]
seedstore = {}
print('device: CPU (zero GPU quota) | inputs: szeged-backup + szeged-vdei-processed\n')
print('NOTE: npz targets are NORMALIZED -> all metrics denormalized to physical units.')

for site in SITES:
    fac_mu, fac_sd = facade_stats(site)
    if fac_sd is None:
        print(f'[{site}] facade stats missing - abort site'); continue
    n_air = n_fac = None
    per_seed = {k: [] for k in AIR_KEYS + FAC_KEYS}
    ens = None
    n_loaded = 0
    for s in SEEDS:
        d = load(f'{site}_dualhead_optuna_best_seed{s}_test_fullpool.npz')
        if d is None:
            continue
        n_loaded += 1
        pa = d['pred_air'].astype(np.float32); ta = d['target_air'].astype(np.float32)
        pf = d['pred_fac'].astype(np.float32); tf = d['target_fac'].astype(np.float32)
        valid = d['valid'].astype(bool)
        n_air = pa.shape[0]; n_fac = int(valid.sum())
        if ens is None:  # accumulate ensemble prediction sums (float32 is exact enough for 5 terms)
            ens = {'pa': np.zeros_like(pa), 'pf': np.zeros_like(pf),
                   'ta': ta, 'tf': tf, 'valid': valid}
        else:
            assert np.array_equal(ta, ens['ta']) and np.array_equal(valid, ens['valid'])
        ens['pa'] += pa; ens['pf'] += pf
        for vi, k in enumerate(AIR_KEYS):
            mu, sd = AIR[vi][1]
            p = pa[:, vi].astype(np.float64) * sd + mu
            t = ta[:, vi].astype(np.float64) * sd + mu
            per_seed[k].append(metrics(p, t))
        for vi, k in enumerate(FAC_KEYS):
            p = pf[valid, vi].astype(np.float64) * fac_sd[vi] + fac_mu[vi]
            t = tf[valid, vi].astype(np.float64) * fac_sd[vi] + fac_mu[vi]
            per_seed[k].append(metrics(p, t))
        d.close()
    ens['pa'] /= n_loaded; ens['pf'] /= n_loaded

    res = {'n_air': n_air, 'n_facade_valid': n_fac,
           'facade_frac_pct': round(100 * n_fac / n_air, 3), 'air': {}, 'facade': {}}
    for k in AIR_KEYS + FAC_KEYS:
        seedstore[(site, 'air' if k in AIR_KEYS else 'facade', k)] = \
            np.array(per_seed[k], dtype=np.float64)
    print(f'\n===== [{site}] PER-SEED MAE / RMSE / R2 (physical, n_air={n_air:,}, n_fac={n_fac:,}) =====')
    print(f'{"var":8}{"metric":6}' + ''.join(f'{"s"+str(s):>9}' for s in SEEDS) + f'{"mean":>9}{"std":>8}')
    for k in AIR_KEYS + FAC_KEYS:
        arr = np.array(per_seed[k], dtype=np.float64)  # (5, 3) -> mae, rmse, r2
        print(f'--- {k}')
        for mi, mname in enumerate(('MAE', 'RMSE', 'R2')):
            vals = arr[:, mi]
            print(f'  {mname:5}' + ''.join(f'{v:>9.4f}' for v in vals)
                  + f'{vals.mean():>9.4f}{vals.std(ddof=1):>8.4f}')

    # ---- ensemble metrics (5-seed mean of predictions, then scored) ----
    print(f'\n----- [{site}] ENSEMBLE5 metrics (physical) -----')
    for vi, k in enumerate(AIR_KEYS):
        mu, sd = AIR[vi][1]
        p = ens['pa'][:, vi].astype(np.float64) * sd + mu
        t = ens['ta'][:, vi].astype(np.float64) * sd + mu
        mae, rmse, r2 = metrics(p, t)
        res['air'][k] = {'mae': round(mae, 4), 'rmse': round(rmse, 4), 'r2': round(r2, 4)}
        print(f'  {k:8} MAE={mae:9.4f} RMSE={rmse:9.4f} R2={r2:8.4f}')
    for vi, k in enumerate(FAC_KEYS):
        p = ens['pf'][ens['valid'], vi].astype(np.float64) * fac_sd[vi] + fac_mu[vi]
        t = ens['tf'][ens['valid'], vi].astype(np.float64) * fac_sd[vi] + fac_mu[vi]
        mae, rmse, r2 = metrics(p, t)
        res['facade'][k] = {'mae': round(mae, 4), 'rmse': round(rmse, 4), 'r2': round(r2, 4)}
        print(f'  {k:8} MAE={mae:9.4f} RMSE={rmse:9.4f} R2={r2:8.4f}')

    # ---- facade head stability: CV of seed MAE ----
    cv = {}
    for k in FAC_KEYS:
        arr = np.array(per_seed[k], dtype=np.float64)
        cv[k] = round(float(arr[:, 0].std(ddof=1) / arr[:, 0].mean()), 4)
    res['facade_cv_mae'] = cv
    print(f'  facade CV(MAE) across seeds: {cv}')
    summary['sites'][site] = res

# ================= DRAFT-CLAIM VERDICTS =================
print('\n================= DRAFT-CLAIM VERDICTS (§3.1.1) =================')
c, p = summary['sites']['canyon'], summary['sites']['plaza']
air_keys = AIR_KEYS

def chk(label, cond, detail):
    print(f'  [{"OK " if cond else "XX "}] {label}: {detail}')

r2_air = {s: [summary["sites"][s]["air"][k]["r2"] for k in air_keys] for s in SITES}
r2_fac = {s: [summary["sites"][s]["facade"][k]["r2"] for k, _, _ in FAC] for s in SITES}
mae_fac = {s: {k: summary["sites"][s]["facade"][k]["mae"] for k, _, _ in FAC} for s in SITES}
chk('Air R2 >= 0.92 at both sites',
    min(min(r2_air[s]) for s in SITES) >= 0.92,
    f'canyon min={min(r2_air["canyon"]):.4f}, plaza min={min(r2_air["plaza"]):.4f}')
chk('T_a, RH, T_mrt all R2 > 0.96 (both sites)',
    all(r2_air[s][i] > 0.96 for s in SITES for i in (0, 1, 4)),
    f'canyon T={r2_air["canyon"][0]:.4f} RH={r2_air["canyon"][1]:.4f} TMRT={r2_air["canyon"][4]:.4f} | '
    f'plaza T={r2_air["plaza"][0]:.4f} RH={r2_air["plaza"][1]:.4f} TMRT={r2_air["plaza"][4]:.4f}')
chk('TKE lowest air R2 (draft: 0.916 / 0.941)',
    abs(r2_air['canyon'][3] - 0.916) < 0.01 and abs(r2_air['plaza'][3] - 0.941) < 0.01,
    f'canyon TKE R2={r2_air["canyon"][3]:.4f} (draft 0.916), plaza={r2_air["plaza"][3]:.4f} (draft 0.941)')
chk('Facade R2 within [0.847, 0.969]',
    min(min(v) for v in r2_fac.values()) >= 0.847 - 0.005 and max(max(v) for v in r2_fac.values()) <= 0.969 + 0.005,
    f'range [{min(min(v) for v in r2_fac.values()):.4f}, {max(max(v) for v in r2_fac.values()):.4f}] (draft 0.847-0.969)')
chk('SWabs = largest facade MAE at both sites',
    all(mae_fac[s]['SWabs'] == max(mae_fac[s].values()) for s in SITES),
    f'canyon SWabs={mae_fac["canyon"]["SWabs"]:.2f}, plaza={mae_fac["plaza"]["SWabs"]:.2f} (draft 23.5 / 32.9)')
chk('LWbal = lowest facade R2 at plaza (draft 0.847)',
    r2_fac['plaza'][3] == min(r2_fac['plaza']) and abs(r2_fac['plaza'][3] - 0.847) < 0.01,
    f'plaza LWbal R2={r2_fac["plaza"][3]:.4f} (draft 0.847), min-facade-plaza={min(r2_fac["plaza"]):.4f}')

# ================= FIGURE 6 =================
print('\n===== rendering Figure 6 =====')
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
panel_titles = [('R$^2$', ''), ('MAE', '(physical units)'), ('RMSE', '(physical units)')]
groups = [('Air-microclimate head', AIR, 'air'), ('Fa\u00e7ade-thermal head', FAC, 'facade')]
for row, (gtitle, vars_, head) in enumerate(groups):
    for col, (mname, _) in enumerate(panel_titles):
        ax = style(axes[row, col])
        mi = {'R$^2$': 2, 'MAE': 0, 'RMSE': 1}[mname]
        labels, units = [], []
        for vi, entry in enumerate(vars_):
            k = entry[0]
            if head == 'air':
                _, _, lab, unit = entry
            else:
                lab, unit = entry[1], entry[2]
            labels.append(lab); units.append(unit)
            for si, site in enumerate(SITES):
                arr = seedstore[(site, head, k)]          # (5, 3)
                mean = float(arr[:, mi].mean())
                std = float(arr[:, mi].std(ddof=1))
                x = vi + (si - 0.5) * 0.72
                ax.bar(x, mean, 0.62, color=COLORS[site], alpha=0.85,
                       yerr=std, capsize=2.5, error_kw={'lw': 0.8},
                       label=site.capitalize() if vi == 0 else None)
        ax.set_xticks(range(len(vars_)))
        ax.set_xticklabels([f'{l}\n({u})' for l, u in zip(labels, units)], fontsize=8)
        ax.set_ylabel(mname + (f'  {panel_titles[col][1]}' if col else ''))
        ax.set_title(f'({"abcdef"[row * 3 + col]}) {gtitle} \u2013 {mname}')
        if mname == 'R$^2$':
            ax.set_ylim(0.80, 1.005)
        if row == 0 and col == 0:
            ax.legend(fontsize=8, frameon=False)
fig.suptitle('Figure 6 \u2013 Test-set predictive performance of the dual-head 3D-CNN '
             '(5-seed mean \u00b1 SD, full test pool, physical units)', fontsize=11, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = FIG / 'fig06_performance_summary.png'
fig.savefig(out, bbox_inches='tight'); plt.close(fig)
print(f'  [SAVED] {out}')

jpath = MET / 'fig6_perf_summary.json'
jpath.write_text(json.dumps(summary, indent=2))
print(f'[SAVED] {jpath.name}')
print('\n===== DONE - paste this entire output back =====')
print('Then I rewrite §3.1.1 with artifact-true numbers + the refined Figure 6 caption.')
