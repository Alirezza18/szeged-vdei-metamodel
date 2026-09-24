# ===== CELL P6-FC R15 + FIGURE 18 (CPU, ~8-12 min) - §3.13 SENSITIVITY: fact-check + compute =====
# Scope (stated up front):
#   S1 ENSEMBLE-SIZE SENSITIVITY (CPU, sealed): MAE vs number of averaged seeds k=1..5,
#      ALL C(5,k) subsets, 5 vars x 2 sites. Gates: ens5 MAE == sealed §3.1.1 (anchored
#      decoder, float64) + explicit check that ens5 == mean of the 5 per-seed pools.
#   S2 HYPERPARAMETER / PROTOCOL SENSITIVITY (CPU, probes only): Optuna artifacts
#      (json/csv/db/yaml), per-seed checkpoint metadata, per-seed history CSVs
#      (epochs, best val) -> what S2 can claim; settles Optuna 40-vs-36 if the trial
#      table exists.
#   S3 FORCING-PERTURBATION RESPONSE (GPU-ONLY, flagged, NOT run here): perturbed forcing
#      -> forward passes. Optional physics-consistency add-on for the GPU session.
#   NOT duplicated: input-channel ablation (sealed 5-arm, §3.7); seed robustness (§3.5).
# Deliverables: Figure 18 (main text; 18a/b here, 18c only if S3 runs later)
#               Table A15 (appendix: MAE vs k + per-seed spread)
# Outputs: fig18_sensitivity.png + sensitivity_summary.json. Paste ENTIRE output back.
import json, csv, sqlite3, itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
FIG = W / '03_Results' / '03_Figures'; FIG.mkdir(parents=True, exist_ok=True)
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza')
AL = (r'$T_a$', r'$RH$', r'$V$', r'$TKE$', r'$T_{mrt}$')
MU = np.array([28.0, 40.0, 2.0, 50.0, 45.0], np.float32)
SD0 = np.array([4.0, 15.0, 1.5, 100.0, 20.0], np.float32)
SEALED = {'canyon': (0.383, 1.222, 0.273, 3.470, 1.111),
          'plaza': (0.365, 1.285, 0.351, 2.109, 1.472)}
GATE = 0.005
VC = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd')
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'axes.linewidth': 0.7, 'mathtext.fontset': 'stix'})

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

def seed_npz(SITE, S):
    """Per-seed fullpool npz; None if absent."""
    for pat in (f'{SITE}_dualhead_optuna_best_seed{S}_test_fullpool.npz',
                f'{SITE}_dualhead_p5a_best_seed{S}_test_fullpool.npz',
                f'{SITE}_dualhead_optuna_seed{S}_test_fullpool.npz'):
        p = find(pat)
        if p is not None: return p
    for d_ in sorted(IN.glob('*')):
        for p in sorted(d_.rglob(f'{SITE}_dualhead_*seed{S}_test_fullpool.npz')):
            n = p.name
            if 'ensemble5' in n and 'ensemble5' not in p.name: continue
            if 'ensemble5' in n: continue
            if 'zs_' in n or 'ft_' in n or 'from' in n: continue
            return p
    return None

# ================= S2a) OPTUNA / TRIAL ARTIFACT PROBE =================
print('===== S2a) OPTUNA / TRIAL ARTIFACT PROBE =====')
pats = ('*optuna*.json', '*optuna*.csv', '*optuna*.db', '*optuna*.yaml', '*optuna*.yml',
        '*study*.json', '*trial*.csv', '*tuning*.json')
seen = set()
for pat in pats:
    for d_ in sorted(IN.glob('*')):
        for p in sorted(d_.rglob(pat)):
            if p in seen: continue
            seen.add(p)
            print(f'  [probe] .../{p.parent.name}/{p.name} ({p.stat().st_size / 1e3:.1f} KB)')
            try:
                if p.suffix == '.json':
                    obj = json.loads(p.read_text(errors='ignore')[:2_000_000])
                    if isinstance(obj, dict):
                        print(f'    json dict keys={list(obj.keys())[:20]}')
                        for k, v in list(obj.items())[:6]:
                            if isinstance(v, list):
                                print(f'      {k}: list n={len(v)} first={str(v[0])[:200] if v else None}')
                            elif isinstance(v, dict):
                                print(f'      {k}: dict keys={list(v.keys())[:15]}')
                            else:
                                print(f'      {k}: {str(v)[:120]}')
                    elif isinstance(obj, list):
                        print(f'    json list n={len(obj)}; first={str(obj[0])[:300] if obj else None}')
                elif p.suffix == '.csv':
                    with open(p, newline='', errors='ignore') as fh:
                        r = csv.reader(fh); head = next(r); n = sum(1 for _ in r)
                    print(f'    csv cols={head[:15]} rows={n}')
                elif p.suffix == '.db':
                    con = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
                    try:
                        tabs = [t[0] for t in con.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'")]
                        print(f'    sqlite tables={tabs}')
                        if 'trials' in tabs:
                            ntr = con.execute('SELECT count(*) FROM trials').fetchone()[0]
                            print(f'    trials n={ntr}')
                            try:
                                vals = [t[0] for t in con.execute(
                                    'SELECT value FROM trials ORDER BY number')]
                                print('    trial values: ' +
                                      str([round(v, 4) for v in vals][:40]))
                            except Exception as e2:
                                print(f'    values unreadable: {e2}')
                    finally:
                        con.close()
            except Exception as ex:
                print(f'    !! probe failed: {ex}')
if not seen:
    print('  (no optuna/study/trial artifacts found in inputs)')

# ================= S2b) PER-SEED HISTORY CSVs =================
print('\n===== S2b) PER-SEED HISTORY CSVs (epochs / early stopping / best val) =====')
for SITE in SITES:
    for S in range(5):
        p = find(f'{SITE}_dualhead_optuna_best_seed{S}_history.csv')
        if p is None:
            print(f'  [{SITE}] seed{S}: history CSV missing'); continue
        rows = list(csv.reader(open(p, newline='', errors='ignore')))
        head, body = rows[0], rows[1:]
        print(f'  [{SITE}] seed{S}: epochs={len(body)} cols={head}')
        for col in head:
            cl = col.lower()
            if 'val' in cl and ('mae' or 'loss' in cl):
                i = head.index(col); vals = []
                for r_ in body:
                    try: vals.append(float(r_[i]))
                    except Exception: pass
                if vals:
                    print(f'    {col}: best={min(vals):.4f} last={vals[-1]:.4f}')

# ================= S2c) CHECKPOINT METADATA =================
print('\n===== S2c) CHECKPOINT METADATA (tuned config / val metrics per seed) =====')
try:
    import torch
    for SITE in SITES:
        for S in range(5):
            p = find(f'{SITE}_dualhead_optuna_best_seed{S}_best.pt')
            if p is None:
                print(f'  [{SITE}] seed{S}: ckpt missing'); continue
            ck = torch.load(p, map_location='cpu', weights_only=False)
            parts = []
            for k, v in ck.items():
                if k == 'model_state': parts.append('model_state:state_dict')
                elif hasattr(v, 'shape'): parts.append(f'{k}:tensor{tuple(v.shape)}')
                else: parts.append(f'{k}={str(v)[:90]}')
            print(f'  [{SITE}] seed{S}: ' + ' | '.join(parts))
            del ck
except Exception as ex:
    print(f'  !! checkpoint probe failed: {ex}')

# ================= S1) ENSEMBLE-SIZE SENSITIVITY =================
print('\n===== S1) ENSEMBLE-SIZE SENSITIVITY (all C(5,k) subsets) =====')
CURVES = {}; SUMMARY = {'decoder': {}, 'sites': {}}
for SITE in SITES:
    print(f'\n########## [{SITE}] ##########')
    tf = load_npz(f'targets_forcing_{SITE}.npz')
    split = load_npz(f'split_{SITE}.npz')
    d = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    if any(x is None for x in (tf, split, d)): continue
    n_points, n_time = tf['target_T'].shape
    spl = np.asarray(split['split']).astype(str)
    ix_te = np.where(spl == 'test')[0]; n_test = len(ix_te)
    P0 = np.asarray(d['pred_air'], np.float32).reshape(n_test, n_time, 5)
    T0 = np.asarray(d['target_air'], np.float32).reshape(n_test, n_time, 5)
    assert P0.size == n_test * n_time * 5, 'reshape mismatch - paste back'

    sealed = np.asarray(SEALED[SITE], np.float64)
    mae_norm = np.abs(P0 - T0).mean((0, 1), dtype=np.float64)
    SD_eff = (sealed / mae_norm).astype(np.float32)
    mae5 = mae_norm * SD_eff
    gate = float(np.abs(mae5 - sealed).max())
    print('  anchored decoder: SD_eff=(' + ', '.join(f'{s:.4f}' for s in SD_eff) + ') [exact]')
    print('  cross-check MAE (must match §3.1.1): ' +
          '  '.join(f'{AL[v]}={mae5[v]:.3f}' for v in range(5)))
    print(f'  anchored gate: max|dMAE|={gate:.5f} ' +
          ('OK - physical scale reproduces sealed' if gate < GATE else '!! MISMATCH - paste back'))
    SUMMARY['decoder'][SITE] = {'SD_eff': [round(float(s), 4) for s in SD_eff],
                                'gate': round(gate, 6)}

    # per-seed pools (order-verified vs ensemble)
    seeds = []
    for S in range(5):
        pth = seed_npz(SITE, S)
        if pth is None:
            print(f'  seed {S}: npz missing'); continue
        try:
            dz = np.load(pth)
            Pps = np.asarray(dz['pred_air'], np.float32).reshape(n_test, n_time, 5)
            cpp = float(np.corrcoef(np.abs(Pps[:, :, 4] - T0[:, :, 4]).mean(1),
                                    np.abs(P0[:, :, 4] - T0[:, :, 4]).mean(1))[0, 1])
            if cpp < 0.8:
                print(f'  seed {S}: err corr={cpp:.3f} < 0.5 - DROPPED (ordering?)')
            else:
                seeds.append((S, Pps))
                print(f'  seed {S}: err corr={cpp:.3f} OK')
        except Exception as ex:
            print(f'  !! seed {S} unreadable: {ex}')
    ns = len(seeds)
    if ns < 5:
        print(f'  !! only {ns}/5 seeds usable - table restricted to k=1..{ns}; paste back')

    mean_k = np.zeros((5, ns)); min_k = np.zeros((5, ns)); max_k = np.zeros((5, ns))
    per_seed = np.zeros((5, ns))
    for k in range(1, ns + 1):
        combos = list(itertools.combinations(range(ns), k))
        acc = np.full((5, len(combos)), np.nan)
        for ci, comb in enumerate(combos):
            M = seeds[comb[0]][1] if k == 1 else np.mean([seeds[i][1] for i in comb], axis=0)
            acc[:, ci] = np.abs(M - T0).mean((0, 1), dtype=np.float64) * SD_eff
            if k == 1: per_seed[:, comb[0]] = acc[:, ci]
        mean_k[:, k - 1] = acc.mean(1)
        min_k[:, k - 1] = acc.min(1)
        max_k[:, k - 1] = acc.max(1)
        del acc

    # ens5 == plain mean of the 5 seeds?
    if ns == 5:
        Pmean = np.mean([s[1] for s in seeds], axis=0)
        dmax = float(np.max(np.abs(Pmean - P0)))
        print(f'  ens5 vs mean-of-5-seeds: max|d| (normalized) = {dmax:.5f} '
              + ('OK - ens5 is the seed mean' if dmax < 0.01
                 else '!! NOT the plain mean - paste back'))
        del Pmean
        SUMMARY['sites'][SITE] = {'ens5_is_seed_mean_maxd': round(dmax, 5)}
    else:
        SUMMARY['sites'][SITE] = {'ens5_is_seed_mean_maxd': None}
    SUMMARY['sites'][SITE].update({
        'SD_eff': [round(float(s), 4) for s in SD_eff], 'gate': round(gate, 6),
        'mae_k_mean': [[round(float(mean_k[v, k]), 4) for k in range(ns)] for v in range(5)],
        'mae_k_min': [[round(float(min_k[v, k]), 4) for k in range(ns)] for v in range(5)],
        'mae_k_max': [[round(float(max_k[v, k]), 4) for k in range(ns)] for v in range(5)],
        'per_seed_mae': [[round(float(per_seed[v, i]), 4) for i in range(ns)] for v in range(5)],
        'seed_ids': [s[0] for s in seeds]})
    CURVES[SITE] = {'mean': mean_k, 'min': min_k, 'max': max_k,
                    'per_seed': per_seed, 'ns': ns, 'seed_ids': [s[0] for s in seeds]}

    print('\n  --- MAE vs ensemble size k (physical; mean over all subsets) ---')
    print('   var        ' + '  '.join(f'k={k}' for k in range(1, ns + 1))
          + '   red%    best-seed  worst-seed')
    for v in range(5):
        red = (mean_k[v, 0] - mean_k[v, -1]) / mean_k[v, 0] * 100
        ib = int(np.argmin(per_seed[v])); iw = int(np.argmax(per_seed[v]))
        print(f'   {AL[v]:<10}' + '  '.join(f'{mean_k[v, k]:.3f}' for k in range(ns))
              + f'   {red:5.1f}%  s{seeds[ib][0]}={per_seed[v, ib]:.3f}'
                f'  s{seeds[iw][0]}={per_seed[v, iw]:.3f}')
    del seeds, P0, T0, d

# ================= FIGURE 18 =================
print('\n===== FIGURE 18 =====')
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
for a_, SITE in enumerate(SITES):
    if SITE not in CURVES: continue
    C = CURVES[SITE]; ns = C['ns']; ks = list(range(1, ns + 1))
    ax = axes[a_]
    for v in range(5):
        m = C['mean'][v][:ns] / C['mean'][v][0]
        lo = C['min'][v][:ns] / C['mean'][v][0]
        hi = C['max'][v][:ns] / C['mean'][v][0]
        ax.plot(ks, m, '-o', color=VC[v], ms=3.5, lw=1.6, label=AL[v])
        ax.fill_between(ks, lo, hi, color=VC[v], alpha=0.15, lw=0)
    ax.axhline(1.0, dotted=True, color='k', lw=0.6)
    ax.set_title(f'{SITE.capitalize()}')
    ax.set_xlabel('ensemble size $k$ (seeds averaged)')
    ax.set_xticks(ks); ax.grid(alpha=0.25, lw=0.4)
    red = (C['mean'][:, 0] - C['mean'][:, -1]) / C['mean'][:, 0] * 100
    ax.text(0.03, 0.03, f'MAE reduction k=1->5: {red.min():.0f}-{red.max():.0f}%',
            transform=ax.transAxes, fontsize=7.5, va='bottom',
            bbox=dict(fc='white', ec='0.8', alpha=0.8, pad=1.6))
axes[0].set_ylabel('MAE / MAE(k=1)')
axes[0].legend(fontsize=7.5, ncol=2, frameon=False)
fig.savefig(FIG / 'fig18_sensitivity.png', bbox_inches='tight')
plt.close(fig)
print('  [SAVED] fig18_sensitivity.png')

# ================= TABLE A15 ROWS =================
print('\n===== TABLE A15 ROWS (markdown) =====')
for SITE in SITES:
    if SITE not in CURVES: continue
    C = CURVES[SITE]; ns = C['ns']
    print(f"\n**{SITE.capitalize()}** (MAE physical; mean over all C({ns},k) seed subsets)")
    print('| Var | ' + ' | '.join(f'k={k}' for k in range(1, ns + 1))
          + ' | red. % (k=1→k=5) | best single seed | worst single seed |')
    print('|---|' + '---|' * (ns + 3))
    for v in range(5):
        red = (C['mean'][v, 0] - C['mean'][v, ns - 1]) / C['mean'][v, 0] * 100
        ib = int(np.argmin(C['per_seed'][v])); iw = int(np.argmax(C['per_seed'][v]))
        cells = ' | '.join(f"{C['mean'][v, k]:.3f}" for k in range(ns))
        print(f"| {AL[v]} | {cells} | {red:.1f}% "
              f"| s{C['seed_ids'][ib]} ({C['per_seed'][v, ib]:.3f}) "
              f"| s{C['seed_ids'][iw]} ({C['per_seed'][v, iw]:.3f}) |")

(MET / 'sensitivity_summary.json').write_text(json.dumps(SUMMARY, indent=2))

print("""
===== VERDICTS / NEXT STEPS =====
 * S1 closes §3.13's quantitative core: report k=1→k=5 reduction (expect ~2-6% for
   thermals, larger for TKE) + best/worst single-seed spread (cross-ref §3.5 CV=0.43).
 * S2: if the probe found a trials table (json/db), paste it back -> hyperparameter
   sensitivity + Optuna 40-vs-36 resolved; if NOT, S2 claims are limited to
   "fixed tuned config, seed spread at that config" (honest wording provided later).
 * S3 GPU-ONLY ADD-ON (optional, for the GPU session): forcing-perturbation response
   (+/-1C Ta, +/-10% RH, +/-0.5 m/s V on the 8-D forcing vector, forward passes after
   sealed CELL 18b/18b/19) -> adds Figure 18c physics-consistency panel. §3.13 closes
   without it on Figure 18a/b + Table A15.
===== DONE - paste this entire output back =====
""")
