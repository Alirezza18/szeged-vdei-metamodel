# ===== CELL P6 (CPU, zero GPU quota) - paper figures v2 =========================
# v2: grid background on every panel + manuscript abbreviations + STIX mathtext.
# Missing input -> that figure SKIPS with a note; nothing crashes.
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
FAC_KEYS = ['Twall', 'Qsens', 'SWabs', 'LWbal']
AIR_LABELS = [r'$T_a$', r'$RH$', r'$V$', r'$TKE$', r'$T_{mrt}$']          # manuscript abbr.
FAC_LABELS = [r'$T_{wall}$', r'$Q_{sens}$', r'$SW_{abs}$', r'$LW_{bal}$']  # manuscript abbr.
N_TIME = {'canyon': 25, 'plaza': 49}
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'mathtext.fontset': 'stix'})
rng = np.random.RandomState(0)
done = []

def style(ax):
    """grid network in the background of every panel"""
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
    mu, sd = fc['facade_norm_mean'].astype(np.float32), fc['facade_norm_std'].astype(np.float32)
    fc.close()
    return mu, sd

def load_npz(pat):
    p = find(pat)
    if p is None:
        return None
    print(f'  [{pat}] <- {p.parent.parent.name}/{p.parent.name}/{p.name}')
    return np.load(p)

def phys_air(arr, vi):
    mu, sd = AIR_KEYS[vi][1]
    return arr[:, vi].astype(np.float32) * sd + mu

FALLBACK_P5B = {   # air MAE only; provenance: P5b v2 final tables (2026-09-05)
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

# ---- F5: predicted-vs-ENVI-met density (T_a + T_mrt), both sites ---------------
try:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.6))
    for col, site in enumerate(SITES):
        d = load_npz(f'{site}_dualhead_ensemble5_test_fullpool.npz')
        if d is None:
            axes[0, col].set_title(f'{site}: input missing'); continue
        for row, var in enumerate(('T', 'TMRT')):
            vi = AIR_NAMES.index(var)
            p, t = phys_air(d['pred_air'], vi), phys_air(d['target_air'], vi)
            if len(p) > 300_000:
                sel = rng.choice(len(p), 300_000, replace=False); p, t = p[sel], t[sel]
            ax = style(axes[row, col])
            ax.hexbin(t, p, gridsize=90, mincnt=1, bins='log', cmap='viridis')
            lim = [min(t.min(), p.min()), max(t.max(), p.max())]
            ax.plot(lim, lim, 'r--', lw=1)
            mae = float(np.abs(p - t).mean())
            ax.set_title(f'{site} \u2013 {AIR_LABELS[vi]}   MAE = {mae:.3f}')
            ax.set_xlabel('ENVI-met truth'); ax.set_ylabel('V-DEI dual-head prediction')
        d.close()
    fig.suptitle('F5 \u2013 Predicted vs ENVI-met (5-seed ensemble, full test pool)')
    fig.tight_layout(); fig.savefig(FIG / 'fig05_pred_vs_truth.png', bbox_inches='tight'); plt.close(fig)
    done.append('F5')
except Exception as e:
    print(f'SKIP F5: {e}')

# ---- F7: diurnal profiles (domain mean), both sites ---------------------------
try:
    for site in SITES:
        d = load_npz(f'{site}_dualhead_ensemble5_test_fullpool.npz')
        if d is None:
            continue
        nt = N_TIME[site]
        fig, axes = plt.subplots(1, 5, figsize=(16, 2.9), sharey=False)
        for vi, (var, _) in enumerate(AIR_KEYS):
            p = d['pred_air'][:, vi].astype(np.float32).reshape(-1, nt).mean(axis=0)
            t = d['target_air'][:, vi].astype(np.float32).reshape(-1, nt).mean(axis=0)
            mu, sd = AIR_KEYS[vi][1]
            p, t = p * sd + mu, t * sd + mu
            ax = style(axes[vi])
            hrs = np.arange(nt) * (24 / nt)
            ax.plot(hrs, t, 'k-', lw=1.6, label='ENVI-met')
            ax.plot(hrs, p, 'r--', lw=1.6, label='V-DEI ens5')
            ax.set_title(AIR_LABELS[vi]); ax.set_xlabel('Hour of day')
            if vi == 0:
                ax.legend(fontsize=7)
        fig.suptitle(f'F7 \u2013 Diurnal domain-mean profiles ({site}, full test pool)')
        fig.tight_layout(); fig.savefig(FIG / f'fig07_diurnal_{site}.png', bbox_inches='tight'); plt.close(fig)
        d.close(); done.append(f'F7-{site}')
except Exception as e:
    print(f'SKIP F7: {e}')

# ---- F8: per-seed spread vs ensemble, air vars, both sites ---------------------
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for col, site in enumerate(SITES):
        seeds = []
        for s in range(5):
            d = load_npz(f'{site}_dualhead_optuna_best_seed{s}_test_fullpool.npz')
            if d is None:
                break
            maes = [np.mean(np.abs(d['pred_air'][:, vi].astype(np.float32)
                                    - d['target_air'][:, vi].astype(np.float32))) * sd
                    for vi, (_, sd) in enumerate(AIR_KEYS)]
            seeds.append(maes); d.close()
        if not seeds:
            axes[col].set_title(f'{site}: seed npz missing'); continue
        seeds = np.array(seeds)
        ens = load_npz(f'{site}_dualhead_ensemble5_test_fullpool.npz')
        ax = style(axes[col])
        for vi, var in enumerate(AIR_NAMES):
            ax.scatter([vi] * 5, seeds[:, vi], s=14, color='steelblue', zorder=3)
            ax.plot([vi - .25, vi + .25], [seeds[:, vi].mean()] * 2, 'b-', lw=2)
            if ens is not None:
                emae = np.mean(np.abs(ens['pred_air'][:, vi].astype(np.float32)
                                      - ens['target_air'][:, vi].astype(np.float32))) * AIR_KEYS[vi][1][1]
                ax.plot([vi - .3, vi + .3], [emae] * 2, 'r--', lw=1.4, zorder=4)
        if ens is not None:
            ens.close()
        ax.set_xticks(range(5)); ax.set_xticklabels(AIR_LABELS)
        ax.set_title(f'{site}: blue = 5 seeds, line = mean, red = ensemble5')
        ax.set_ylabel('Test MAE')
    fig.suptitle('F8 \u2013 Multi-seed robustness (optuna_best)')
    fig.tight_layout(); fig.savefig(FIG / 'fig08_seed_robustness.png', bbox_inches='tight'); plt.close(fig)
    done.append('F8')
except Exception as e:
    print(f'SKIP F8: {e}')

# ---- F9: baselines vs dual-head (air), both sites ------------------------------
try:
    jpath = find('p5b_classical_baselines.json')
    base = {}
    if jpath is not None:
        j = json.loads(jpath.read_text())
        res = j.get('results', {})
        for site in SITES:
            try:
                base[site] = {m: [res[site][var][m]['mae'] for var in AIR_NAMES]
                              for m in res[site][AIR_NAMES[0]].keys()}
            except Exception:
                pass
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for col, site in enumerate(SITES):
        b = base.get(site, FALLBACK_P5B[site])
        if base.get(site) is None:
            print(f'  F9 {site}: json incomplete -> frozen P5b table constants')
        ens = load_npz(f'{site}_dualhead_ensemble5_test_fullpool.npz')
        models = list(b.keys()) + ['Dual-Head 3D-CNN (ens5)']
        vals = np.array([b[m] for m in b.keys()]
                        + ([[np.mean(np.abs(ens['pred_air'][:, vi].astype(np.float32)
                                            - ens['target_air'][:, vi].astype(np.float32))) * sd
                              for vi, (_, sd) in enumerate(AIR_KEYS)]
                            if ens is not None else [np.nan] * 5]))
        if ens is not None:
            ens.close()
        ax = style(axes[col])
        x = np.arange(5); w = 0.15
        for mi, m in enumerate(models):
            ax.bar(x + (mi - len(models) / 2) * w, vals[mi], w, label=m)
        ax.set_xticks(x); ax.set_xticklabels(AIR_LABELS)
        ax.set_title(site); ax.set_ylabel('Test MAE')
        if col == 0:
            ax.legend(fontsize=6.5)
    fig.suptitle('F9 \u2013 Classical baselines vs dual-head 3D-CNN (air head, full test pool)')
    fig.tight_layout(); fig.savefig(FIG / 'fig09_baselines.png', bbox_inches='tight'); plt.close(fig)
    done.append('F9')
except Exception as e:
    print(f'SKIP F9: {e}')

# ---- F10: transfer triangle, air from npz + facade per-site stats --------------
try:
    for site in SITES:
        other = 'plaza' if site == 'canyon' else 'canyon'
        d_sc = load_npz(f'{site}_dualhead_ensemble5_test_fullpool.npz')
        d_zs = load_npz(f'{site}_dualhead_p5c_zs_from{other}_ensemble5_test_fullpool.npz')
        d_ft = load_npz(f'{site}_dualhead_p5c_ft_from{other}_ensemble5_test_fullpool.npz')
        if d_sc is None or d_zs is None or d_ft is None:
            print(f'  F10 {site}: needs scratch + p5c ZS/FT ensembles (upload p5c-transfer-backup!)')
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        systems = ['scratch5 (native)', 'zero-shot from ' + other, 'fine-tune from ' + other]
        dsets = [d_sc, d_zs, d_ft]
        air_vals = [[np.mean(np.abs(d['pred_air'][:, vi].astype(np.float32)
                                     - d['target_air'][:, vi].astype(np.float32))) * sd
                     for vi, (_, sd) in enumerate(AIR_KEYS)] for d in dsets]
        fac_mu, fac_sd = facade_stats(site)
        fac_vals = [[np.mean(np.abs(d['pred_fac'][d['valid'].astype(bool), vi].astype(np.float32)
                                     - d['target_fac'][d['valid'].astype(bool), vi].astype(np.float32))) * fac_sd[vi]
                     for vi in range(4)] for d in dsets]
        for d in dsets:
            d.close()
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

# ---- F12: SVF validation scatter ------------------------------------------------
try:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for col, site in enumerate(SITES):
        p = find(f'svf_{site}.npz')
        if p is None:
            axes[col].set_title(f'{site}: svf npz missing'); continue
        d = np.load(p)
        ev, ra, r = d['svf_envimet'].astype(np.float32), d['svf_rays'].astype(np.float32), float(d['pearson_r'])
        d.close()
        ok = np.isfinite(ev) & np.isfinite(ra)
        ev, ra = ev[ok], ra[ok]
        if len(ev) > 100_000:
            sel = rng.choice(len(ev), 100_000, replace=False); ev, ra = ev[sel], ra[sel]
        ax = style(axes[col])
        ax.hexbin(ev, ra, gridsize=70, mincnt=1, bins='log', cmap='viridis')
        ax.plot([0, 1], [0, 1], 'r--', lw=1)
        ax.set_title(f'{site}  Pearson $r$ = {r:.3f} (all-K, n = {ok.sum():,})')
        ax.set_xlabel('ENVI-met Sky View Factor'); ax.set_ylabel('V-DEI ray-based SVF')
    fig.suptitle('F12 \u2013 SVF geometric cross-validation')
    fig.tight_layout(); fig.savefig(FIG / 'fig12_svf_validation.png', bbox_inches='tight'); plt.close(fig)
    done.append('F12')
except Exception as e:
    print(f'SKIP F12: {e}')

print(f'\n===== P6 v2 DONE - figures generated: {done} =====')
print(f'[DIR] {FIG}')
print('NEXT: zip figs -> download -> tomorrow: GPU P5e joint + F6/F11/F13 figures.')
