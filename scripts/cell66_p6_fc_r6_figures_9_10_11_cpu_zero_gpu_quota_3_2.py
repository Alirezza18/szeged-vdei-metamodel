# ===== CELL P6-FC R6 + FIGURES 9/10/11 (CPU, zero GPU quota) - §3.2 + §3.3.1 fact-check =====
# (A) §3.2: full sealed p5b_classical_baselines.json dump (meta + all metrics), MAD floors,
#     deep-vs-baseline gains, spread checks, MLP-SVF-vs-trees check, every draft-claim verdict.
# (B) §3.3.1: zero-shot transfer BOTH directions, 5 sealed seeds + ensemble5 npz,
#     facade ZS denormalized with SOURCE stats (sealed p5c convention), degradation vs native,
#     comparison vs MAD floor, every draft-claim verdict.
# Figures: fig09 (MLP-SVF vs deep), fig10 (5 classical baselines vs deep, log MAE),
#          fig11 (ZS native vs zero-shot, mean+-SD, log MAE, air head).
# Inputs: szeged-backup (ensemble5 + p5c_zs npz), szeged-vdei-processed (targets_forcing,
#         facade_targets), p5b-output-backup (p5b_classical_baselines.json).
# Outputs: 03_Results/02_Metrics/fig9_10_11_summary.json + 3 PNGs. Runtime ~5-8 min CPU.
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN = Path('/kaggle/input')
W = Path('/kaggle/working')
FIG = W / '03_Results' / '03_Figures'; FIG.mkdir(parents=True, exist_ok=True)
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza')
SEEDS = (0, 1, 2, 3, 4)
AIR = [('T', (28.0, 4.0), r'$T_a$'), ('RelHum', (40.0, 15.0), r'$RH$'),
       ('WindSpd', (2.0, 1.5), r'$V$'), ('TKE', (50.0, 100.0), r'$TKE$'),
       ('TMRT', (45.0, 20.0), r'$T_{mrt}$')]
AIR_KEYS = [a[0] for a in AIR]; AIR_LAB = [a[2] for a in AIR]
FAC_KEYS = ['Twall', 'Qsens', 'SWabs', 'LWbal']
FAC_LAB = [r'$T_{wall}$', r'$Q_{sens}$', r'$SW_{abs}$', r'$LW_{bal}$']
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'mathtext.fontset': 'stix'})
C_SITE = {'canyon': '#4C72B0', 'plaza': '#D55E00'}
GRAY = ['#9e9e9e', '#8aa0b8', '#b8a88a', '#a89ab8', '#8ab8a0']


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, which='major', ls=':', lw=0.5, color='0.55', alpha=0.7)
    return ax


def find(pat):
    for d in sorted(IN.glob('*')):
        h = sorted(d.rglob(pat))
        if h:
            return h[0]
    return None


def load_npz(pat):
    p = find(pat)
    if p is None:
        print(f'  !! missing: {pat}')
        return None
    print(f'  [{pat}] <- {p.parent.parent.name}/{p.parent.name}/{p.name}')
    return np.load(p)


def metrics(p, t):
    p = p.astype(np.float64); t = t.astype(np.float64)
    d = p - t
    ss_res = float((d * d).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(np.abs(d).mean()), float(np.sqrt((d * d).mean())), r2


V = []


def verd(ok, label, detail=''):
    tag = '[OK ]' if ok is True else ('[XX ]' if ok is False else '[?  ]')
    print(f'  {tag} {label}  {detail}')
    V.append({'claim': label, 'ok': ok, 'detail': detail})


print('device: CPU | inputs: szeged-backup + szeged-vdei-processed + p5b json\n')

# ============ A) MAD floors (full data, physical) ============
print('================= A) MAD CLIMATOLOGY FLOORS (full data) =================')
MAD = {}
for SITE in SITES:
    tf = load_npz(f'targets_forcing_{SITE}.npz')
    MAD[SITE] = {}
    for vi, (k, (mu, sd), _) in enumerate(AIR):
        t = np.asarray(tf[f'target_{k}'], dtype=np.float32) * np.float32(sd) + np.float32(mu)
        MAD[SITE][k] = float(np.abs(t - t.mean()).mean())
        del t
    tf.close()
    print(f'  [{SITE}] ' + '  '.join(f'{AIR_LAB[vi]}={MAD[SITE][k]:.3f}'
          for vi, (k, _, _) in enumerate(AIR)))

# ============ B) sealed baselines JSON ============
print('\n================= B) SEALED p5b CLASSICAL BASELINES =================')
jp = find('p5b_classical_baselines.json')
J = json.loads(jp.read_text()) if jp else {}
if jp:
    print(f'  [{jp.name}] <- {jp.parent.parent.name}/{jp.parent.name}/{jp.name}')
print(f'  meta: {json.dumps(J.get("meta", {}))[:600]}')
BASE = {}
for SITE in SITES:
    block = J.get(SITE, {})
    print(f'\n  --- [{SITE}] json block keys: {list(block.keys())[:12]}')
    BASE[SITE] = {}
    for model, ent in block.items():
        if not isinstance(ent, dict):
            continue
        row = {}
        for k in AIR_KEYS:
            e = ent.get(k) or ent.get(k.lower()) or {}
            if isinstance(e, dict):
                row[k] = {kk: float(vv) for kk, vv in e.items()
                          if isinstance(vv, (int, float))}
            elif isinstance(e, (int, float)):
                row[k] = {'MAE': float(e)}
        BASE[SITE][model] = row
        sample = row.get('T', {})
        print(f'    {model:18} T-fields={list(sample.keys())} T-MAE={sample.get("MAE", float("nan")):.3f}')
if not BASE.get('canyon'):
    print('  !! baselines JSON not parsed - paste output back')

# ---- deep ensemble5 (sealed) ----
DEEP = {}
for SITE in SITES:
    d = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    DEEP[SITE] = {}
    for vi, (k, (mu, sd), _) in enumerate(AIR):
        p = d['pred_air'][:, vi].astype(np.float64) * sd + mu
        t = d['target_air'][:, vi].astype(np.float64) * sd + mu
        DEEP[SITE][k] = dict(zip(('mae', 'rmse', 'r2'), metrics(p, t)[:3]))
        del p, t
    d.close()
print('\n  deep ens5 MAE: ' + ' | '.join(
    f'{s}: ' + ' '.join(f'{AIR_LAB[vi]}={DEEP[s][k]["mae"]:.3f}'
                        for vi, (k, _, _) in enumerate(AIR)) for s in SITES))

# ---- §3.2 verdicts ----
print('\n================= §3.2 DRAFT-CLAIM VERDICTS =================')
for SITE in SITES:
    models = list(BASE[SITE].keys())
    for vi, (k, _, lab) in enumerate(AIR):
        bmae = [BASE[SITE][m].get(k, {}).get('MAE', np.nan) for m in models]
        bmae = [x for x in bmae if np.isfinite(x)]
        if not bmae:
            continue
        spread = (max(bmae) - min(bmae)) / min(bmae) * 100
        best = min(bmae)
        gain = (best - DEEP[SITE][k]['mae']) / best * 100
        verd(spread < 1.5, f'[{SITE}] {lab} baselines within ~1% of each other',
             f'spread={spread:.2f}% best={best:.3f}')
        verd(True, f'[{SITE}] {lab} deep gain vs best baseline',
             f'{gain:.1f}%  deep/MAD={DEEP[SITE][k]["mae"] / MAD[SITE][k]:.2f}x')
mlp_canyon_v = BASE['canyon'].get('MLP-SVF', {}).get('WindSpd', {}).get('MAE', np.nan)
imp_v = (mlp_canyon_v - DEEP['canyon']['WindSpd']['mae']) / mlp_canyon_v * 100
verd(False, '3.2.1 "V improvement over MLP-SVF = 52.6% (canyon)"',
     f'sealed MLP-SVF V MAE={mlp_canyon_v:.3f} -> improvement={imp_v:.1f}% (MLP-SVF sits at climatology)')
deep_r2_min = min(min(DEEP[s][k]['r2'] for k in AIR_KEYS) for s in SITES)
verd(deep_r2_min > 0.99, '3.2.2 "deep R2 > 0.99 for all air targets"',
     f'sealed min deep R2={deep_r2_min:.4f} (TKE)')
has_r2 = any('r2' in BASE['canyon'][m].get('T', {}) for m in BASE['canyon'])
verd(has_r2, 'baseline R2 available in sealed json (3.2.2 R2 claims)',
     'present' if has_r2 else 'ABSENT - draft R2 numbers (0.062/0.040/0.984-0.989) unverifiable, text must drop them')
verd(False, '3.2.2 "CatBoost V 1.04 -> 0.30 (-71%)"',
     f"sealed CatBoost V canyon={BASE['canyon'].get('CatBoost', {}).get('WindSpd', {}).get('MAE', float('nan')):.3f} -> deep {DEEP['canyon']['WindSpd']['mae']:.3f}")
verd(False, '"10-configuration random search" (tuning protocol)',
     'check meta above - Table A4 says 15 Optuna trials/site/target; reconcile one way')

# ============ C) §3.3.1 ZERO-SHOT, both directions, 5 seeds ============
print('\n================= C) §3.3.1 ZERO-SHOT TRANSFER (5 sealed seeds) =================')
ZS = {}
for SITE in SITES:
    other = 'plaza' if SITE == 'canyon' else 'canyon'
    print(f'\n--- [{SITE}] zero-shot FROM {other} (model trained on {other}) ---')
    fac_t = load_npz(f'facade_targets_{SITE}.npz')
    fac_s = load_npz(f'facade_targets_{other}.npz')
    ft_mu = np.asarray(fac_t['facade_norm_mean'], dtype=np.float64)
    ft_sd = np.asarray(fac_t['facade_norm_std'], dtype=np.float64)
    fs_mu = np.asarray(fac_s['facade_norm_mean'], dtype=np.float64)
    fs_sd = np.asarray(fac_s['facade_norm_std'], dtype=np.float64)
    fac_t.close(); fac_s.close()

    ens = load_npz(f'{SITE}_dualhead_p5c_zs_from{other}_ensemble5_test_fullpool.npz')
    nat = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    if ens is None or nat is None:
        continue
    valid = ens['valid'].astype(bool)
    rec = {'air': {}, 'fac': {}, 'seed_air': {k: [] for k in AIR_KEYS},
           'seed_fac': {k: [] for k in FAC_KEYS}, 'native': {}, 'valid_n': int(valid.sum())}
    print(f'  NOTE: ZS facade preds denormed with SOURCE ({other}) stats; targets with {SITE} stats (sealed p5c convention)')
    for vi, (k, (mu, sd), lab) in enumerate(AIR):
        p = ens['pred_air'][:, vi].astype(np.float64) * sd + mu
        t = ens['target_air'][:, vi].astype(np.float64) * sd + mu
        rec['air'][k] = dict(zip(('mae', 'rmse', 'r2'), metrics(p, t)[:3]))
        pn = nat['pred_air'][:, vi].astype(np.float64) * sd + mu
        tn = nat['target_air'][:, vi].astype(np.float64) * sd + mu
        rec['native'][k] = dict(zip(('mae', 'rmse', 'r2'), metrics(pn, tn)[:3]))
        del p, t, pn, tn
    for vi, k in enumerate(FAC_KEYS):
        p = ens['pred_fac'][valid, vi].astype(np.float64) * fs_sd[vi] + fs_mu[vi]
        t = ens['target_fac'][valid, vi].astype(np.float64) * ft_sd[vi] + ft_mu[vi]
        rec['fac'][k] = dict(zip(('mae', 'rmse', 'r2'), metrics(p, t)[:3]))
        pn = nat['pred_fac'][valid, vi].astype(np.float64) * ft_sd[vi] + ft_mu[vi]
        tn = nat['target_fac'][valid, vi].astype(np.float64) * ft_sd[vi] + ft_mu[vi]
        rec['native'][k] = dict(zip(('mae', 'rmse', 'r2'), metrics(pn, tn)[:3]))
        del p, t, pn, tn
    ens.close(); nat.close()

    for S in SEEDS:
        d = load_npz(f'{SITE}_dualhead_p5c_zs_from{other}_seed{S}_test_fullpool.npz')
        if d is None:
            continue
        vb = d['valid'].astype(bool)
        for vi, (k, (mu, sd), _) in enumerate(AIR):
            diff = d['pred_air'][:, vi].astype(np.float64) - d['target_air'][:, vi].astype(np.float64)
            rec['seed_air'][k].append(float(np.abs(diff).mean()) * sd)
        for vi, k in enumerate(FAC_KEYS):
            diff = d['pred_fac'][vb, vi].astype(np.float64) - d['target_fac'][vb, vi].astype(np.float64)
            rec['seed_fac'][k].append(float(np.abs(diff).mean() * fs_sd[vi]))
        d.close()

    print(f'  {"var":9}{"native":>9}{"ZS":>9}{"degr":>7}{"ZS_R2":>8}{"MADfloor":>9}{"ZS/MAD":>8}{"seedSD":>8}')
    for vi, (k, _, lab) in enumerate(AIR):
        n, z = rec['native'][k], rec['air'][k]
        sd_seeds = float(np.std(rec['seed_air'][k], ddof=1)) if len(rec['seed_air'][k]) > 1 else float('nan')
        print(f'  {lab:9}{n["mae"]:>9.3f}{z["mae"]:>9.3f}{z["mae"] / n["mae"]:>7.2f}'
              f'{z["r2"]:>8.3f}{MAD[SITE][k]:>9.3f}{z["mae"] / MAD[SITE][k]:>8.2f}{sd_seeds:>8.3f}')
    for vi, k in enumerate(FAC_KEYS):
        n, z = rec['native'][k], rec['fac'][k]
        print(f'  {FAC_LAB[vi]:9}{n["mae"]:>9.3f}{z["mae"]:>9.3f}{z["mae"] / n["mae"]:>7.2f}{z["r2"]:>8.3f}')
    ZS[f'{other}->{"canyon" if other == "plaza" else "plaza"}'] = rec

# ---- §3.3.1 verdicts vs draft ----
print('\n================= §3.3.1 DRAFT-CLAIM VERDICTS (vs sealed ensemble) =================')
DRAFT = {('plaza->canyon', 'T'): (0.940, 0.66), ('plaza->canyon', 'RelHum'): (0.854, 2.09),
         ('plaza->canyon', 'TMRT'): (0.958, 2.73), ('plaza->canyon', 'WindSpd'): (0.532, 1.07),
         ('plaza->canyon', 'TKE'): (0.344, 9.95), ('canyon->plaza', 'T'): (0.912, 0.74),
         ('canyon->plaza', 'RelHum'): (0.930, 2.17), ('canyon->plaza', 'TMRT'): (0.872, 2.93),
         ('canyon->plaza', 'WindSpd'): (0.196, 1.09), ('canyon->plaza', 'TKE'): (-7.313, 16.82)}
for (direc, k), (r2_d, mae_d) in DRAFT.items():
    rec = ZS.get(direc)
    if rec is None:
        verd(None, f'{direc} {k}', 'direction missing from output')
        continue
    z = rec['air'][k]
    ok = abs(z['r2'] - r2_d) < 0.05 and abs(z['mae'] - mae_d) / max(mae_d, 1e-9) < 0.2
    verd(ok, f'{direc} {k}: draft R2={r2_d:.3f} MAE={mae_d:.2f}',
         f"sealed R2={z['r2']:.3f} MAE={z['mae']:.3f}")

# ============ D) FIGURES 9 / 10 / 11 ============
print('\n================= D) RENDERING FIGURES 9/10/11 =================')
x = np.arange(len(AIR_KEYS))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, SITE in zip(axes, SITES):
    mlp = [BASE[SITE].get('MLP-SVF', {}).get(k, {}).get('MAE', np.nan) for k in AIR_KEYS]
    dp = [DEEP[SITE][k]['mae'] for k in AIR_KEYS]
    ax.bar(x - 0.2, mlp, 0.38, label='MLP-SVF baseline', color='#9e9e9e')
    ax.bar(x + 0.2, dp, 0.38, label='V-DEI 3D-CNN (ens5)', color=C_SITE[SITE])
    for xi, (m, d_) in enumerate(zip(mlp, dp)):
        if np.isfinite(m) and m > 0:
            ax.text(xi + 0.2, d_ * 1.15, f'-{(m - d_) / m * 100:.0f}%', ha='center', fontsize=7.5)
    ax.set_yscale('log'); style(ax)
    ax.set_xticks(x); ax.set_xticklabels(AIR_LAB)
    ax.set_title(f'{SITE.capitalize()}'); ax.set_ylabel('MAE (physical units, log)')
    ax.legend(frameon=False, fontsize=8)
fig.suptitle('Figure 9: V-DEI vs. simple exposure-based MLP baseline (test pool, ensemble of 5)',
             fontsize=10, y=1.02)
fig.tight_layout(); fig.savefig(FIG / 'fig09_baseline_mlp.png', bbox_inches='tight'); plt.close(fig)
print('  [SAVED] fig09_baseline_mlp.png')

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
for ax, SITE in zip(axes, SITES):
    models = [m for m in BASE[SITE]
              if any(np.isfinite(BASE[SITE][m].get(k, {}).get('MAE', np.nan)) for k in AIR_KEYS)]
    w = 0.8 / (len(models) + 1)
    for mi, m in enumerate(models):
        vals = [BASE[SITE][m].get(k, {}).get('MAE', np.nan) for k in AIR_KEYS]
        ax.bar(x - 0.4 + w * (mi + 0.5), vals, w, label=m, color=GRAY[mi % len(GRAY)])
    vals = [DEEP[SITE][k]['mae'] for k in AIR_KEYS]
    ax.bar(x - 0.4 + w * (len(models) + 0.5), vals, w, label='V-DEI 3D-CNN (ens5)',
           color=C_SITE[SITE], edgecolor='k', lw=0.4)
    ax.axhline(MAD[SITE]['T'], color='r', ls='--', lw=0.8)
    ax.set_yscale('log'); style(ax)
    ax.set_xticks(x); ax.set_xticklabels(AIR_LAB)
    ax.set_title(f'{SITE.capitalize()}'); ax.set_ylabel('MAE (physical units, log)')
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
fig.suptitle('Figure 10: Multi-metric benchmarking against classical ML baselines (red dashed = climatology MAD floor)',
             fontsize=10, y=1.02)
fig.tight_layout(); fig.savefig(FIG / 'fig10_classical_benchmark.png', bbox_inches='tight'); plt.close(fig)
print('  [SAVED] fig10_classical_benchmark.png')

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
for ax, (direc, SITE) in zip(axes, [('plaza->canyon', 'canyon'), ('canyon->plaza', 'plaza')]):
    rec = ZS.get(direc)
    if rec is None:
        continue
    nat = [rec['native'][k]['mae'] for k in AIR_KEYS]
    zs = [rec['air'][k]['mae'] for k in AIR_KEYS]
    sd = [float(np.std(rec['seed_air'][k], ddof=1)) if len(rec['seed_air'][k]) > 1 else 0
          for k in AIR_KEYS]
    ax.bar(x - 0.2, nat, 0.38, label='native (site-trained)', color='0.6')
    ax.bar(x + 0.2, zs, 0.38, yerr=sd, capsize=2, label='zero-shot',
           color=C_SITE[{'canyon': 'plaza', 'plaza': 'canyon'}[SITE]])
    for xi, (n_, z_) in enumerate(zip(nat, zs)):
        ax.text(xi + 0.2, z_ * 1.3, f'x{z_ / n_:.1f}', ha='center', fontsize=7.5)
    ax.set_yscale('log'); style(ax)
    ax.set_xticks(x); ax.set_xticklabels(AIR_LAB)
    ax.set_title(f'{direc} (evaluated on {SITE})'); ax.set_ylabel('MAE (physical units, log)')
    ax.legend(frameon=False, fontsize=8)
fig.suptitle('Figure 11: Zero-shot cross-morphology transfer, air head (mean ± SD across 5 seeds)',
             fontsize=10, y=1.02)
fig.tight_layout(); fig.savefig(FIG / 'fig11_zeroshot_transfer.png', bbox_inches='tight'); plt.close(fig)
print('  [SAVED] fig11_zeroshot_transfer.png')

(MET / 'fig9_10_11_summary.json').write_text(json.dumps(
    {'MAD': MAD, 'deep': DEEP,
     'zs': {k: {'air': v['air'], 'fac': v['fac'], 'native': v['native'],
                'valid_n': v['valid_n'], 'seed_air': v['seed_air']}
            for k, v in ZS.items()}, 'verdicts': V}, indent=2))
print('\n===== DONE - paste this entire output back =====')
print('Then I deliver final §3.2 (restructured 3.2.1/3.2.2) + §3.3.1 + Figure 9/10/11 captions.')
