# ===== CELL P6-FIX (CPU) - regenerate F8 + F9 + F10 (v3 float64-safe) ============
# Self-contained. F10 stays in skip mode until p5c-transfer-backup is attached;
# after attaching, just re-run this same cell -> F10 renders too.
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN = Path('/kaggle/input')
FIG = Path('/kaggle/working/03_Results/03_Figures'); FIG.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza')
AIR_KEYS = [('T', (28., 4.)), ('RelHum', (40., 15.)), ('WindSpd', (2., 1.5)),
            ('TKE', (50., 100.)), ('TMRT', (45., 20.))]
AIR_NAMES = [k for k, _ in AIR_KEYS]
AIR_LABELS = [r'$T_a$', r'$RH$', r'$V$', r'$TKE$', r'$T_{mrt}$']
FAC_LABELS = [r'$T_{wall}$', r'$Q_{sens}$', r'$SW_{abs}$', r'$LW_{bal}$']
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'mathtext.fontset': 'stix'})
done = []

def style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, which='major', ls=':', lw=0.5, color='0.55', alpha=0.7)
    ax.grid(True, which='minor', ls=':', lw=0.3, color='0.8', alpha=0.5)
    return ax

def find(pat):
    for d in sorted(IN.glob('*')):
        h = sorted(d.rglob(pat))
        if h:
            return h[0]
    return None

def facade_stats(site):
    p = find(f'facade_targets_{site}.npz')
    if p is None:
        return None, None
    fc = np.load(p)
    mu = np.asarray(fc['facade_norm_mean'], dtype=np.float64)
    sd = np.asarray(fc['facade_norm_std'], dtype=np.float64)
    fc.close()
    return mu, sd

def load_npz(pat):
    p = find(pat)
    if p is None:
        return None
    print(f'  [{pat}] <- {p.parent.parent.name}/{p.parent.name}/{p.name}')
    return np.load(p)

def mae_air(d, vi):
    sd = float(AIR_KEYS[vi][1][1])
    diff = d['pred_air'][:, vi].astype(np.float64) - d['target_air'][:, vi].astype(np.float64)
    return float(np.mean(np.abs(diff))) * sd

def mae_fac(d, vi, fac_sd):
    valid = d['valid'].astype(bool)
    diff = (d['pred_fac'][valid, vi].astype(np.float64)
            - d['target_fac'][valid, vi].astype(np.float64))
    return float(np.mean(np.abs(diff))) * float(fac_sd[vi])

FALLBACK_P5B = {   # provenance: P5b v2 final tables (2026-09-05)
    'canyon': {'LinearRegression': [3.583, 10.778, 1.965, 12.746, 18.456],
               'RandomForest':     [3.583, 10.777, 1.921, 12.235, 18.443],
               'XGBoost':          [3.583, 10.777, 1.922, 12.406, 18.450],
               'CatBoost':         [3.583, 10.776, 1.921, 12.289, 18.447],
               'MLP-SVF':          [3.583, 10.795, 1.931, 12.348, 18.443]},
    'plaza':  {'LinearRegression': [3.732, 11.417, 1.944, 9.122, 18.767],
               'RandomForest':     [3.732, 11.416, 1.924, 9.050, 18.780],
               'XGBoost':          [3.732, 11.416, 1.926, 9.125, 18.784],
               'CatBoost':         [3.732, 11.417, 1.924, 9.040, 18.778],
               'MLP-SVF':          [3.732, 11.407, 1.930, 9.053, 18.776]},
}

# ---- F8: per-seed spread vs ensemble --------------------------------------------
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for col, site in enumerate(SITES):
        seeds = []
        for s in range(5):
            d = load_npz(f'{site}_dualhead_optuna_best_seed{s}_test_fullpool.npz')
            if d is None:
                break
            seeds.append([mae_air(d, vi) for vi in range(5)])
            d.close()
        if not seeds:
            axes[col].set_title(f'{site}: seed npz missing'); continue
        seeds = np.array(seeds, dtype=np.float64)
        ens = load_npz(f'{site}_dualhead_ensemble5_test_fullpool.npz')
        ax = style(axes[col])
        for vi in range(5):
            ax.scatter([vi] * seeds.shape[0], seeds[:, vi], s=14, color='steelblue', zorder=3)
            ax.plot([vi - .25, vi + .25], [float(seeds[:, vi].mean())] * 2, 'b-', lw=2)
            if ens is not None:
                ax.plot([vi - .3, vi + .3], [mae_air(ens, vi)] * 2, 'r--', lw=1.4, zorder=4)
        if ens is not None:
            ens.close()
        ax.set_xticks(range(5)); ax.set_xticklabels(AIR_LABELS)
        ax.set_title(f'{site}: blue = seeds, line = mean, red = ensemble5')
        ax.set_ylabel('Test MAE')
    fig.suptitle('F8 \u2013 Multi-seed robustness (optuna_best)')
    fig.tight_layout(); fig.savefig(FIG / 'fig08_seed_robustness.png', bbox_inches='tight'); plt.close(fig)
    done.append('F8')
except Exception as e:
    print(f'SKIP F8: {e}')

# ---- F9: baselines vs dual-head ---------------------------------------------------
try:
    jpath = find('p5b_classical_baselines.json')
    base = {}
    if jpath is not None:
        j = json.loads(jpath.read_text())
        res = j.get('results', {})
        for site in SITES:
            try:
                base[site] = {m: [float(res[site][var][m]['mae']) for var in AIR_NAMES]
                              for m in res[site][AIR_NAMES[0]].keys()}
            except Exception:
                pass
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for col, site in enumerate(SITES):
        b = base.get(site, FALLBACK_P5B[site])
        if base.get(site) is None:
            print(f'  F9 {site}: json incomplete -> frozen P5b table constants')
        rows = [[float(x) for x in b[m]] for m in b.keys()]
        models = list(b.keys()) + ['Dual-Head 3D-CNN (ens5)']
        ens = load_npz(f'{site}_dualhead_ensemble5_test_fullpool.npz')
        rows.append([mae_air(ens, vi) for vi in range(5)] if ens is not None else [np.nan] * 5)
        if ens is not None:
            ens.close()
        ax = style(axes[col])
        x = np.arange(5); w = 0.15
        for mi, m in enumerate(models):
            ax.bar(x + (mi - (len(models) - 1) / 2) * w, rows[mi], w, label=m)
        ax.set_xticks(x); ax.set_xticklabels(AIR_LABELS)
        ax.set_title(site); ax.set_ylabel('Test MAE')
        if col == 0:
            ax.legend(fontsize=6.5)
    fig.suptitle('F9 \u2013 Classical baselines vs dual-head 3D-CNN (air head, full test pool)')
    fig.tight_layout(); fig.savefig(FIG / 'fig09_baselines.png', bbox_inches='tight'); plt.close(fig)
    done.append('F9')
except Exception as e:
    print(f'SKIP F9: {e}')

# ---- F10: transfer triangle -------------------------------------------------------
try:
    for site in SITES:
        other = 'plaza' if site == 'canyon' else 'canyon'
        dsets = [load_npz(p) for p in
                 (f'{site}_dualhead_ensemble5_test_fullpool.npz',
                  f'{site}_dualhead_p5c_zs_from{other}_ensemble5_test_fullpool.npz',
                  f'{site}_dualhead_p5c_ft_from{other}_ensemble5_test_fullpool.npz')]
        if any(d is None for d in dsets):
            print(f'  F10 {site}: needs scratch + p5c ZS/FT ensembles (attach p5c-transfer-backup)')
            continue
        fac_mu, fac_sd = facade_stats(site)
        if fac_sd is None:
            print(f'  F10 {site}: facade_targets missing'); continue
        systems = ['scratch5 (native)', 'zero-shot from ' + other, 'fine-tune from ' + other]
        air_vals = [[mae_air(d, vi) for vi in range(5)] for d in dsets]
        fac_vals = [[mae_fac(d, vi, fac_sd) for vi in range(4)] for d in dsets]
        for d in dsets:
            d.close()
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        ax0, ax1 = style(axes[0]), style(axes[1])
        for si, (s, av) in enumerate(zip(systems, air_vals)):
            ax0.bar(np.arange(5) + (si - 1) * .27, av, .27, label=s)
        ax0.set_xticks(range(5)); ax0.set_xticklabels(AIR_LABELS)
        ax0.set_title(f'{site} \u2013 climate factors'); ax0.set_ylabel('Test MAE')
        for si, (s, fv) in enumerate(zip(systems, fac_vals)):
            ax1.bar(np.arange(4) + (si - 1) * .27, fv, .27, label=s)
        ax1.set_xticks(range(4)); ax1.set_xticklabels(FAC_LABELS)
        ax1.set_title(f'{site} \u2013 fa\u00e7ade factors')
        ax0.legend(fontsize=6.5)
        fig.suptitle(f'F10 \u2013 Cross-site transfer: scratch vs zero-shot vs fine-tune ({site})')
        fig.tight_layout(); fig.savefig(FIG / f'fig10_transfer_{site}.png', bbox_inches='tight'); plt.close(fig)
        done.append(f'F10-{site}')
except Exception as e:
    print(f'SKIP F10: {e}')

print(f'\n===== P6-FIX DONE - generated: {done} =====')
