# ===== CELL P6-FC R7 + FIGURE 12 (CPU, ~5 min) - §3.3.2 FINE-TUNE TRANSFER fact-check =====
# Computes the never-computed metrics of the sealed FT artifacts, both directions, 5 seeds:
#   FT ensemble5 air + facade metrics (physical), per-seed MAE mean+-SD,
#   ZS vs FT vs native comparison + GAP-CLOSED %, then renders Figure 12.
# Sealed conventions (p5c_transfer.py): air = shared manuscript stats; FT facade preds
#   denormed with TARGET stats (ZS used SOURCE stats); targets always TARGET stats.
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
FIG = W / '03_Results' / '03_Figures'; FIG.mkdir(parents=True, exist_ok=True)
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza'); SEEDS = (0, 1, 2, 3, 4)
AIR = [('T', (28.0, 4.0), r'$T_a$'), ('RelHum', (40.0, 15.0), r'$RH$'),
       ('WindSpd', (2.0, 1.5), r'$V$'), ('TKE', (50.0, 100.0), r'$TKE$'),
       ('TMRT', (45.0, 20.0), r'$T_{mrt}$')]
AK = [a[0] for a in AIR]; AL = [a[2] for a in AIR]
FAC = [('Twall', r'$T_{wall}$'), ('Qsens', r'$Q_{sens}$'),
       ('SWabs', r'$SW_{abs}$'), ('LWbal', r'$LW_{bal}$')]
FK = [f[0] for f in FAC]; FL = [f[1] for f in FAC]
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'mathtext.fontset': 'stix'})
C = {'native': '0.6', 'zs': '#D55E00', 'ft': '#4C72B0'}

def find(pat):
    for d in sorted(IN.glob('*')):
        h = sorted(d.rglob(pat))
        if h: return h[0]
    return None

def load_npz(pat):
    p = find(pat)
    if p is None: print(f'  !! missing: {pat}'); return None
    print(f'  [{pat}] <- {p.parent.parent.name}/{p.parent.name}/{p.name}')
    return np.load(p)

def met(p, t):
    p = p.astype(np.float64); t = t.astype(np.float64)
    d = p - t
    r2 = 1.0 - float((d*d).sum()) / float(((t - t.mean())**2).sum())
    return float(np.abs(d).mean()), float(np.sqrt((d*d).mean())), r2

out_sum = {}
for SITE in SITES:
    other = 'plaza' if SITE == 'canyon' else 'canyon'
    print(f'\n########## [{SITE}] FINE-TUNED FROM {other} ##########')
    ft = load_npz(f'{SITE}_dualhead_p5c_ft_from{other}_ensemble5_test_fullpool.npz')
    zs = load_npz(f'{SITE}_dualhead_p5c_zs_from{other}_ensemble5_test_fullpool.npz')
    nat = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    fc = load_npz(f'facade_targets_{SITE}.npz')
    if any(x is None for x in (ft, zs, nat, fc)): continue
    fmu = np.asarray(fc['facade_norm_mean'], np.float64)
    fsd = np.asarray(fc['facade_norm_std'], np.float64); fc.close()
    valid = ft['valid'].astype(bool)
    rec = {'air': {}, 'fac': {}, 'seed_air': {k: [] for k in AK}, 'seed_fac': {k: [] for k in FK}}

    print(f'--- FT ensemble5 AIR (physical, n={ft["pred_air"].shape[0]:,}) ---')
    print(f'  {"var":9}{"native":>9}{"ZS":>9}{"FT":>9}{"FT_R2":>8}{"gap_closed":>11}')
    for vi, (k, (mu, sd), lab) in enumerate(AIR):
        t = nat['target_air'][:, vi].astype(np.float64) * sd + mu
        pn = nat['pred_air'][:, vi].astype(np.float64) * sd + mu
        pz = zs['pred_air'][:, vi].astype(np.float64) * sd + mu
        pf = ft['pred_air'][:, vi].astype(np.float64) * sd + mu
        mn, _, _ = met(pn, t); mz, _, _ = met(pz, t); mf, _, r2f = met(pf, t)
        gc = (mz - mf) / (mz - mn) * 100 if mz > mn else float('nan')
        rec['air'][k] = {'native': mn, 'zs': mz, 'ft': mf, 'ft_r2': r2f, 'gap_closed': gc}
        print(f'  {lab:9}{mn:>9.3f}{mz:>9.3f}{mf:>9.3f}{r2f:>8.3f}{gc:>10.1f}%')
        del t, pn, pz, pf
    print(f'--- FT ensemble5 FACADE (physical, TARGET stats, n_valid={int(valid.sum()):,}) ---')
    print(f'  {"var":9}{"native":>9}{"ZS":>9}{"FT":>9}{"FT_R2":>8}{"gap_closed":>11}')
    for vi, (k, lab) in enumerate(FAC):
        t = nat['target_fac'][valid, vi].astype(np.float64) * fsd[vi] + fmu[vi]
        pn = nat['pred_fac'][valid, vi].astype(np.float64) * fsd[vi] + fmu[vi]
        pz = zs['pred_fac'][valid, vi].astype(np.float64) * np.asarray(
            np.load(find(f'facade_targets_{other}.npz'))['facade_norm_std'], np.float64)[vi] \
            + np.asarray(np.load(find(f'facade_targets_{other}.npz'))['facade_norm_mean'], np.float64)[vi]
        pf = ft['pred_fac'][valid, vi].astype(np.float64) * fsd[vi] + fmu[vi]
        mn, _, _ = met(pn, t); mz, _, _ = met(pz, t); mf, _, r2f = met(pf, t)
        gc = (mz - mf) / (mz - mn) * 100 if mz > mn else float('nan')
        rec['fac'][k] = {'native': mn, 'zs': mz, 'ft': mf, 'ft_r2': r2f, 'gap_closed': gc}
        print(f'  {lab:9}{mn:>9.3f}{mz:>9.3f}{mf:>9.3f}{r2f:>8.3f}{gc:>10.1f}%')
        del t, pn, pz, pf

    # per-seed FT MAE (mean +- SD)
    for S in SEEDS:
        d = load_npz(f'{SITE}_dualhead_p5c_ft_from{other}_seed{S}_test_fullpool.npz')
        if d is None: continue
        vb = d['valid'].astype(bool)
        for vi, (k, (mu, sd), _) in enumerate(AIR):
            diff = d['pred_air'][:, vi].astype(np.float64) - d['target_air'][:, vi].astype(np.float64)
            rec['seed_air'][k].append(float(np.abs(diff).mean()) * sd)
        for vi, k in enumerate(FK):
            diff = d['pred_fac'][vb, vi].astype(np.float64) - d['target_fac'][vb, vi].astype(np.float64)
            rec['seed_fac'][k].append(float(np.abs(diff).mean() * fsd[vi]))
        d.close()
    print('--- per-seed FT AIR MAE (mean +- SD across 5 seeds) ---')
    for vi, k in enumerate(AK):
        v = rec['seed_air'][k]
        print(f'  {AL[vi]:9}{np.mean(v):>9.3f} +- {np.std(v, ddof=1):.3f}' if len(v) > 1 else f'  {AL[vi]:9}{v[0]:>9.3f}')
    for x in (ft, zs, nat): x.close()
    out_sum[f'{other}->{SITE}'] = rec

# ---- Figure 12: native / ZS / FT per direction (air, log MAE) ----
print('\n===== D) RENDERING FIGURE 12 =====')
x = np.arange(len(AK))
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
for ax, (direc, SITE) in zip(axes, [('plaza->canyon', 'canyon'), ('canyon->plaza', 'plaza')]):
    r = out_sum.get(direc)
    if r is None: continue
    nat = [r['air'][k]['native'] for k in AK]
    zs = [r['air'][k]['zs'] for k in AK]
    ftt = [r['air'][k]['ft'] for k in AK]
    sds = [np.std(r['seed_air'][k], ddof=1) if len(r['seed_air'][k]) > 1 else 0 for k in AK]
    ax.bar(x - 0.25, nat, 0.24, label='native (site-trained)', color=C['native'])
    ax.bar(x, zs, 0.24, label='zero-shot', color=C['zs'])
    ax.bar(x + 0.25, ftt, 0.24, yerr=sds, capsize=2, label='fine-tuned', color=C['ft'])
    for xi, k in enumerate(AK):
        gc = r['air'][k]['gap_closed']
        ax.text(xi + 0.25, ftt[xi] * 1.35, f'{gc:.0f}%', ha='center', fontsize=7.5)
    ax.set_yscale('log')
    ax.set_axisbelow(True); ax.grid(True, ls=':', lw=0.5, color='0.55', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(AL)
    ax.set_title(f'{direc} (evaluated on {SITE})'); ax.set_ylabel('MAE (physical units, log)')
    ax.legend(frameon=False, fontsize=8)
fig.suptitle('Fine-tune transfer: native vs zero-shot vs fine-tuned MAE (% = share of ZS gap recovered)',
             fontsize=10, y=1.02)
fig.tight_layout(); fig.savefig(FIG / 'fig12_finetune_transfer.png', bbox_inches='tight')
plt.close(fig)
print('  [SAVED] fig12_finetune_transfer.png')
(MET / 'fig12_finetune_summary.json').write_text(json.dumps(out_sum, indent=2))
print('\n===== DONE - paste this entire output back =====')
print('Then I deliver final §3.3.2 + Figure 12 caption.')
