# ===== CELL P6-FC R6b (CPU, ~2 min) - FIX: correct MAD floors + re-render fig09/fig10 =====
# Baselines hardcoded from the SEALED R4 output (p5b_classical_baselines.json, verified run);
# MAD recomputed from RAW physical targets (no re-scaling); deep recomputed from ensemble5 npz.
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
AIR = [('T', r'$T_a$'), ('RelHum', r'$RH$'), ('WindSpd', r'$V$'),
       ('TKE', r'$TKE$'), ('TMRT', r'$T_{mrt}$')]
AK = [a[0] for a in AIR]; AL = [a[1] for a in AIR]
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'mathtext.fontset': 'stix'})
C_SITE = {'canyon': '#4C72B0', 'plaza': '#D55E00'}
GRAY = ['#9e9e9e', '#8aa0b8', '#b8a88a', '#a89ab8', '#8ab8a0']

def find(pat):
    for d in sorted(IN.glob('*')):
        h = sorted(d.rglob(pat))
        if h: return h[0]
    return None

# ---- sealed baseline MAE (from sealed R4 cell output, physical units) ----
BASE = {'canyon': {
    'LinearRegression': [3.583, 10.778, 1.965, 12.746, 18.456],
    'RandomForest':     [3.583, 10.777, 1.921, 12.235, 18.443],
    'XGBoost':          [3.583, 10.777, 1.922, 12.406, 18.450],
    'CatBoost':         [3.583, 10.776, 1.921, 12.289, 18.447],
    'MLP-SVF':          [3.583, 10.795, 1.931, 12.348, 18.443]},
    'plaza': {
    'LinearRegression': [3.732, 11.417, 1.944, 9.122, 18.767],
    'RandomForest':     [3.732, 11.416, 1.924, 9.050, 18.780],
    'XGBoost':          [3.732, 11.416, 1.926, 9.125, 18.784],
    'CatBoost':         [3.732, 11.417, 1.924, 9.040, 18.778],
    'MLP-SVF':          [3.732, 11.407, 1.930, 9.053, 18.776]}}
# ---- sealed zero-shot ensemble MAE (from R6 output just pasted) ----
ZS = {'plaza->canyon': [0.628, 2.098, 1.540, 11.368, 3.422],
      'canyon->plaza': [0.549, 1.497, 0.998, 23.043, 3.802]}

# ---- CORRECT MAD floors: raw physical targets, no transform ----
MAD = {}
for SITE in SITES:
    tf = np.load(find(f'targets_forcing_{SITE}.npz'))
    MAD[SITE] = [float(np.abs(np.asarray(tf[f'target_{k}'], dtype=np.float64)
                              - np.asarray(tf[f'target_{k}'], dtype=np.float64).mean()).mean())
                 for k in AK]
    tf.close()
    print(f'[{SITE}] MAD floors: ' + '  '.join(f'{l}={v:.3f}' for l, v in zip(AL, MAD[SITE])))

# ---- deep ensemble5 MAE ----
DEEP = {}
for SITE in SITES:
    d = np.load(find(f'{SITE}_dualhead_ensemble5_test_fullpool.npz'))
    sd = {'T': 4.0, 'RelHum': 15.0, 'WindSpd': 1.5, 'TKE': 100.0, 'TMRT': 20.0}
    DEEP[SITE] = [float(np.abs(d['pred_air'][:, vi].astype(np.float64)
                               - d['target_air'][:, vi].astype(np.float64)).mean()) * sd[k]
                  for vi, k in enumerate(AK)]
    d.close()
    print(f'[{SITE}] deep ens5 MAE: ' + '  '.join(f'{l}={v:.3f}' for l, v in zip(AL, DEEP[SITE])))

# ---- corrected gain + floor table ----
print('\n===== CORRECTED §3.2 GAINS + §3.3.1 ZS-vs-FLOOR =====')
for SITE in SITES:
    best = np.min(np.array(list(BASE[SITE].values())), axis=0)
    print(f'[{SITE}] gain vs best baseline: ' +
          '  '.join(f'{l}={(b - d) / b * 100:.1f}%' for l, b, d in zip(AL, best, DEEP[SITE])) +
          '  |  deep/MAD: ' + '  '.join(f'{l}={d / m:.2f}x' for l, d, m in zip(AL, DEEP[SITE], MAD[SITE])))
for direc, site in (('plaza->canyon', 'canyon'), ('canyon->plaza', 'plaza')):
    print(f'[{direc}] ZS/MAD floor: ' +
          '  '.join(f'{l}={z / m:.2f}x' for l, z, m in zip(AL, ZS[direc], MAD[site])))

def style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, which='major', ls=':', lw=0.5, color='0.55', alpha=0.7)

x = np.arange(len(AK))
# ---- fig09 ----
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, SITE in zip(axes, SITES):
    mlp = BASE[SITE]['MLP-SVF']
    ax.bar(x - 0.2, mlp, 0.38, label='MLP-SVF (static scalars)', color='#9e9e9e')
    ax.bar(x + 0.2, DEEP[SITE], 0.38, label='V-DEI 3D-CNN (ens5)', color=C_SITE[SITE])
    for xi, (m, d_) in enumerate(zip(mlp, DEEP[SITE])):
        ax.text(xi + 0.2, d_ * 1.15, f'-{(m - d_) / m * 100:.0f}%', ha='center', fontsize=7.5)
    ax.set_yscale('log'); style(ax); ax.set_xticks(x); ax.set_xticklabels(AL)
    ax.set_title(SITE.capitalize()); ax.set_ylabel('MAE (physical units, log)')
    ax.legend(frameon=False, fontsize=8)
fig.suptitle('V-DEI vs. simple exposure-based MLP baseline (test pool, ensemble of 5)', fontsize=10, y=1.02)
fig.tight_layout(); fig.savefig(FIG / 'fig09_baseline_mlp.png', bbox_inches='tight'); plt.close(fig)
print('  [SAVED] fig09_baseline_mlp.png')
# ---- fig10 ----
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
for ax, SITE in zip(axes, SITES):
    models = list(BASE[SITE].keys()); w = 0.8 / (len(models) + 1)
    for mi, m in enumerate(models):
        ax.bar(x - 0.4 + w * (mi + 0.5), BASE[SITE][m], w, label=m, color=GRAY[mi % len(GRAY)])
    ax.bar(x - 0.4 + w * (len(models) + 0.5), DEEP[SITE], w, label='V-DEI 3D-CNN (ens5)',
           color=C_SITE[SITE], edgecolor='k', lw=0.4)
    ax.axhline(MAD[SITE][0], color='r', ls='--', lw=0.8)
    ax.set_yscale('log'); style(ax); ax.set_xticks(x); ax.set_xticklabels(AL)
    ax.set_title(SITE.capitalize()); ax.set_ylabel('MAE (physical units, log)')
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
fig.suptitle('Benchmarking against classical ML baselines (red dashed = T_a climatology MAD floor)',
             fontsize=10, y=1.02)
fig.tight_layout(); fig.savefig(FIG / 'fig10_classical_benchmark.png', bbox_inches='tight'); plt.close(fig)
print('  [SAVED] fig10_classical_benchmark.png')
(MET / 'fig9_10_summary_fixed.json').write_text(json.dumps(
    {'MAD': dict(zip(SITES, MAD)), 'deep': dict(zip(SITES, DEEP)),
     'baselines_sealed': BASE, 'zs_sealed': ZS}, indent=2))
print('===== DONE - paste this short output back =====')
