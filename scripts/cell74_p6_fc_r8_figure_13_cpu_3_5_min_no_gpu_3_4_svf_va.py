# ===== CELL P6-FC R8 + FIGURE 13 (CPU, ~3-5 min, no GPU) - §3.4 SVF VALIDATION fact-check =====
# (A) sealed svf_{site}.npz (svf_envimet, svf_rays, saved pearson_r)
# (B) recompute: Pearson r, Spearman rho, MAD, bias, RMSE, coverage |d|<0.1/0.2,
#     bias by ENVI-met SVF decile -> everything §3.4 needs
# (C) DEFINITION CHECK: recompute ray-based sky fraction from vdei_features cls
#     (144 dirs = 16 az x 9 el, az-major) for el>=4 (80 dirs, phi>=0),
#     el>=5 (64 dirs, phi>0), all 144 -> identify which subset svf_rays equals exactly
# (D) Figure 13: hexbin V-DEI rays vs ENVI-met SVF per site (1:1 line, stats annotated)
# Inputs: szeged-vdei-processed (svf_*.npz, vdei_features_*.npz).
# Outputs: 03_Results/03_Figures/fig13_svf_validation_{site}.png + 02_Metrics/fig13_svf_summary.json
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
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'mathtext.fontset': 'stix'})

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

def spearman(a, b, cap=500_000, seed=0):
    if a.size > cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(a.size, cap, replace=False)
        a, b = a[idx], b[idx]
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])

summary = {}
for SITE in SITES:
    print(f'\n########## [{SITE}] ##########')
    sv = load_npz(f'svf_{SITE}.npz')
    if sv is None: continue
    em = np.asarray(sv['svf_envimet'], dtype=np.float64).ravel()
    ry = np.asarray(sv['svf_rays'], dtype=np.float64).ravel()
    print(f'  saved pearson_r = {float(sv["pearson_r"]):.4f} | len(em)={em.size:,} len(ry)={ry.size:,}')
    ok = np.isfinite(em) & np.isfinite(ry)
    e, r = em[ok], ry[ok]
    n = e.size
    d = r - e
    pear = float(np.corrcoef(e, r)[0, 1])
    rho = spearman(e, r)
    mad = float(np.abs(d).mean()); bias = float(d.mean())
    rmse = float(np.sqrt((d * d).mean()))
    cov10 = float((np.abs(d) < 0.1).mean()); cov20 = float((np.abs(d) < 0.2).mean())
    print(f'  matched n = {n:,} / {em.size:,} ({100*n/em.size:.1f}%)')
    print(f'  ENVI-met SVF: mean={e.mean():.3f} sd={e.std():.3f} '
          f'q25/50/75={np.percentile(e, [25, 50, 75]).round(3)}')
    print(f'  V-DEI  rays : mean={r.mean():.3f} sd={r.std():.3f} '
          f'q25/50/75={np.percentile(r, [25, 50, 75]).round(3)}')
    print(f'  Pearson r={pear:.4f} (saved {float(sv["pearson_r"]):.4f}) | Spearman rho={rho:.4f}')
    print(f'  MAD={mad:.4f}  bias(rays-envimet)={bias:+.4f}  RMSE={rmse:.4f}')
    print(f'  coverage: |d|<0.1 -> {100*cov10:.1f}%  |d|<0.2 -> {100*cov20:.1f}%')
    # bias by ENVI-met SVF decile
    qb = np.quantile(e, np.linspace(0, 1, 11)); qb[0], qb[-1] = -np.inf, np.inf
    print('  bias by ENVI-met SVF decile (bin-center, n, bias, MAE):')
    for k in range(10):
        m = (e >= qb[k]) & (e < qb[k + 1])
        if m.sum() == 0: continue
        c = float(e[m].mean())
        print(f'    SVF~{c:.2f}  n={int(m.sum()):>9,}  bias={d[m].mean():+.3f}  MAE={np.abs(d[m]).mean():.3f}')

    # ---- (C) definition check: which ray subset == svf_rays? ----
    vf = load_npz(f'vdei_features_{SITE}.npz')
    if vf is not None:
        cls = np.asarray(vf['cls'])
        P = cls.shape[0]
        print(f'  vdei cls shape={cls.shape} dtype={cls.dtype} (144 = 16 az x 9 el, az-major)')
        cols80 = np.array([a * 9 + t for a in range(16) for t in range(9) if t >= 4])
        cols64 = np.array([a * 9 + t for a in range(16) for t in range(9) if t >= 5])
        rng = np.random.default_rng(0)
        sub = np.arange(P) if P <= 1_000_000 else rng.choice(P, 1_000_000, replace=False)
        sky = (cls[sub] == 0)
        for name, cols in (('el>=4 (80 dirs, phi>=0)', cols80),
                           ('el>=5 (64 dirs, phi>0)', cols64),
                           ('all 144 dirs', np.arange(144))):
            sf = sky[:, cols].mean(1)
            dd = np.abs(sf - ry[sub])
            rr = float(np.corrcoef(sf, ry[sub])[0, 1]) if sf.std() > 0 else float('nan')
            print(f'  svf_rays vs {name:<24} r={rr:.4f} max|d|={dd.max():.4f} mean={sf.mean():.3f}')
        del cls

    # ---- (D) figure ----
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    hb = ax.hexbin(e, r, gridsize=60, mincnt=1, cmap='viridis', bins='log')
    ax.plot([0, 1], [0, 1], 'r--', lw=1, label='1:1')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('ENVI-met SVF (native SVFUpSky)')
    ax.set_ylabel('V-DEI ray-cast sky fraction')
    ax.set_title(f'{SITE.capitalize()}:  r={pear:.2f}, MAD={mad:.2f}, bias={bias:+.2f}, n={n:,}')
    ax.legend(loc='upper left')
    cb = fig.colorbar(hb, ax=ax); cb.set_label('log10 count')
    out = FIG / f'fig13_svf_validation_{SITE}.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f'  [SAVED] {out.name}')
    summary[SITE] = {'saved_pearson_r': float(sv['pearson_r']), 'pearson_r': pear,
                     'spearman': rho, 'n_matched': int(n), 'MAD': mad, 'bias': bias,
                     'RMSE': rmse, 'cov_|d|<0.1': cov10, 'cov_|d|<0.2': cov20,
                     'envimet_mean': float(e.mean()), 'rays_mean': float(r.mean())}

(MET / 'fig13_svf_summary.json').write_text(json.dumps(summary, indent=2))
print('\n===== DONE - paste this entire output back =====')
print('Then I deliver final §3.4 + Figure 13 caption.')
