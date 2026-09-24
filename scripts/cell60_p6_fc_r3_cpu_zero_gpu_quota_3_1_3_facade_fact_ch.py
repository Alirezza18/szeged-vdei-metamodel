# ===== CELL P6-FC R3 (CPU, zero GPU quota) - §3.1.3 facade fact-check =====
# Computes EVERY number a facade section needs, from sealed artifacts:
#   (1) ensemble5 facade MAE / RMSE / R2 per variable, both sites (physical units)
#   (2) per-seed facade MAE + coefficient of variation across seeds (stability)
#   (3) facade sample size: unique facade points, fraction of air points,
#       test-pool valid rows + fraction  (the "sample-size constraint" claims)
#   (4) per-site facade normalization stats (confirms the site-specific claim)
# Inputs: szeged-backup (optuna_best seed0-4 + ensemble5 npz),
#         szeged-vdei-processed (facade_targets_{site}.npz, targets_forcing_{site}.npz)
# Outputs: printed tables only. Runtime ~3-5 min CPU.
import json
from pathlib import Path
import numpy as np

IN = Path('/kaggle/input')
SITES = ('canyon', 'plaza')
SEEDS = (0, 1, 2, 3, 4)
FAC = [('Twall', r'$T_{wall}$', 'degC'), ('Qsens', r'$Q_{sens}$', 'W/m2'),
       ('SWabs', r'$SW_{abs}$', 'W/m2'), ('LWbal', r'$LW_{bal}$', 'W/m2')]
FAC_KEYS = [f[0] for f in FAC]


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


def facade_stats(site):
    fc = load_npz(f'facade_targets_{site}.npz')
    if fc is None:
        return None, None, None
    mu = np.asarray(fc['facade_norm_mean'], dtype=np.float64)
    sd = np.asarray(fc['facade_norm_std'], dtype=np.float64)
    idx = np.asarray(fc['facade_point_idx'])
    fc.close()
    return mu, sd, idx


def metrics(p, t):
    p = p.astype(np.float64); t = t.astype(np.float64)
    d = p - t
    ss_res = float((d * d).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(np.abs(d).mean()), float(np.sqrt((d * d).mean())), r2


print('device: CPU | inputs: szeged-backup (seed0-4 + ensemble5) + szeged-vdei-processed\n')

for site in SITES:
    fac_mu, fac_sd, fac_idx = facade_stats(site)
    if fac_sd is None:
        print(f'[{site}] facade stats missing'); continue

    print(f'\n========== [{site}] ==========')
    print(f'--- (4) FACADE NORMALIZATION STATS (per-site, sealed) ---')
    for vi, k in enumerate(FAC_KEYS):
        print(f'  {k:7} mu={fac_mu[vi]:>9.3f}  sd={fac_sd[vi]:>9.3f}')

    # ---- (3) facade sample size ----
    tf = load_npz(f'targets_forcing_{site}.npz')
    n_points = tf['target_T'].shape[0] if tf is not None else None
    if tf is not None:
        tf.close()
    n_fac_pts = int(len(fac_idx))
    print(f'--- (3) FACADE SAMPLE SIZE ---')
    print(f'  unique facade points      = {n_fac_pts:,}  (facade_point_idx)')
    print(f'  total air points          = {n_points:,}')
    if n_points:
        print(f'  facade fraction of points = {n_fac_pts / n_points * 100:.3f}%')

    # ---- (1)+(2) per-seed + ensemble facade metrics ----
    per_seed = {k: [] for k in FAC_KEYS}
    ens = None
    n_air = n_valid = 0
    for s in SEEDS:
        d = load_npz(f'{site}_dualhead_optuna_best_seed{s}_test_fullpool.npz')
        if d is None:
            continue
        pf = d['pred_fac'].astype(np.float32); tf2 = d['target_fac'].astype(np.float32)
        valid = d['valid'].astype(bool)
        n_air = d['pred_air'].shape[0]; n_valid = int(valid.sum())
        if ens is None:
            ens = {'pf': np.zeros_like(pf), 'tf': tf2, 'valid': valid}
        else:
            assert np.array_equal(tf2, ens['tf']) and np.array_equal(valid, ens['valid'])
        ens['pf'] += pf
        for vi, k in enumerate(FAC_KEYS):
            p = pf[valid, vi].astype(np.float64) * fac_sd[vi] + fac_mu[vi]
            t = tf2[valid, vi].astype(np.float64) * fac_sd[vi] + fac_mu[vi]
            per_seed[k].append(metrics(p, t))
        d.close()
    ens['pf'] /= len(SEEDS)

    print(f'--- (2) PER-SEED FACADE MAE (physical, n_valid={n_valid:,}) ---')
    hdr = f'{"var":7}' + ''.join(f'{"s"+str(s):>10}' for s in SEEDS) + f'{"mean":>10}{"std":>9}{"CV":>7}'
    print(hdr)
    for k in FAC_KEYS:
        arr = np.array(per_seed[k], dtype=np.float64)          # (5, 3) -> MAE, RMSE, R2
        mae = arr[:, 0]
        cv = float(mae.std(ddof=1) / mae.mean())
        print(f'{k:7}' + ''.join(f'{v:>10.3f}' for v in mae)
              + f'{mae.mean():>10.3f}{mae.std(ddof=1):>9.3f}{cv:>7.3f}')

    print(f'--- (1) ENSEMBLE5 FACADE METRICS (physical) ---')
    for vi, k in enumerate(FAC_KEYS):
        p = ens['pf'][ens['valid'], vi].astype(np.float64) * fac_sd[vi] + fac_mu[vi]
        t = ens['tf'][ens['valid'], vi].astype(np.float64) * fac_sd[vi] + fac_mu[vi]
        mae, rmse, r2 = metrics(p, t)
        print(f'  {k:7} MAE={mae:9.4f} RMSE={rmse:9.4f} R2={r2:8.4f}')

    # ---- test-pool valid row fraction ----
    print(f'--- test-pool facade-valid rows = {n_valid:,} / {n_air:,} '
          f'= {n_valid / n_air * 100:.3f}% of test samples ---')

print('\n===== DONE - paste this entire output back =====')
print('Then I rewrite §3.1.3 with artifact-true facade numbers.')
