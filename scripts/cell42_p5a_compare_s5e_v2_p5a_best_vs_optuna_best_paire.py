# ===== CELL P5a-COMPARE (S5e v2) - p5a_best vs optuna_best + paired bootstrap 95% CIs =====
# CPU-only. Reads the 2 ensemble npz per site from Inputs - nothing retrained, no globals touched.
import json, csv, time
from datetime import datetime
from pathlib import Path
import numpy as np

W = Path('/kaggle/working')
OUT_DIR = W / '03_Results' / '02_Metrics'; OUT_DIR.mkdir(parents=True, exist_ok=True)
IN = Path('/kaggle/input')
B_BOOT, RNG_SEED, CHUNK = 1000, 0, 25
N_TIME = {'canyon': 25, 'plaza': 49}
AIR_KEYS = [('T', (28.0, 4.0)), ('RelHum', (40.0, 15.0)), ('WindSpd', (2.0, 1.5)),
            ('TKE', (50.0, 100.0)), ('TMRT', (45.0, 20.0))]
FAC_KEYS = ['Twall', 'Qsens', 'SWabs', 'LWbal']
AIR_UNITS = {'T': 'degC', 'RelHum': '%', 'WindSpd': 'm/s', 'TKE': 'm2/s2', 'TMRT': 'degC'}
FAC_UNITS = {'Twall': 'degC', 'Qsens': 'W/m2', 'SWabs': 'W/m2', 'LWbal': 'W/m2'}
SITES = ('canyon', 'plaza')

def _find(pattern):
    hit = sorted(OUT_DIR.glob(pattern))
    if hit: return hit[0]
    for d in sorted(IN.glob('*')):
        hits = sorted(d.rglob(pattern))
        if hits: return hits[0]
    raise FileNotFoundError(f'{pattern} not found - check attached Inputs!')

def point_abs_sums(pred, targ, scale, n_pts, n_time):
    err = np.abs(pred.astype(np.float32) - targ.astype(np.float32))
    return err.reshape(n_pts, n_time, -1).sum(axis=1) * scale[None, :]

def paired_bootstrap(S_new, S_old, n_rows, B=B_BOOT, seed=RNG_SEED, chunk=CHUNK):
    n_pts, V = S_new.shape
    rng = np.random.default_rng(seed)
    out = {k: np.empty((B, V)) for k in ('new', 'old', 'delta')}
    done = 0
    while done < B:
        c = min(chunk, B - done)
        idx = rng.integers(0, n_pts, size=(c, n_pts))
        sn = S_new[idx].sum(axis=1, dtype=np.float64) / n_rows
        so = S_old[idx].sum(axis=1, dtype=np.float64) / n_rows
        out['new'][done:done+c] = sn; out['old'][done:done+c] = so
        out['delta'][done:done+c] = sn - so
        done += c
    return out

def verdict(d_lo, d_hi):
    if d_hi < 0: return 'new better (sig.)'
    if d_lo > 0: return 'old better (sig.)'
    return 'tie (n.s.)'

r4 = lambda x: round(float(x), 4)
results = {'meta': {'B': B_BOOT, 'seed': RNG_SEED, 'bootstrap': 'point-cluster (paired)',
                    'delta_note': 'delta = p5a_best - optuna_best (negative = new better)',
                    'computed_at': datetime.now().strftime('%Y-%m-%d %H:%M')}}
csv_rows = []; t00 = time.time()

for site in SITES:
    n_time = N_TIME[site]
    old_p = _find(f'{site}_dualhead_ensemble5_test_fullpool.npz')
    new_p = _find(f'{site}_dualhead_p5a_best_ensemble5_test_fullpool.npz')
    print(f'\n[{site}] old (optuna_best): {old_p}')
    print(f'[{site}] new (p5a_best)   : {new_p}')
    d_old, d_new = np.load(old_p), np.load(new_p)
    assert np.array_equal(d_old['target_air'], d_new['target_air']), f'{site}: target_air mismatch!'
    assert np.array_equal(d_old['valid'], d_new['valid']), f'{site}: valid mask mismatch!'
    n_tot = d_new['target_air'].shape[0]; n_pts = n_tot // n_time
    assert n_pts * n_time == n_tot, f'{site}: n_tot {n_tot} not divisible by n_time {n_time}'

    sd_air = np.array([sd for _, (mu, sd) in AIR_KEYS], np.float32)
    S_new = point_abs_sums(d_new['pred_air'], d_new['target_air'], sd_air, n_pts, n_time)
    S_old = point_abs_sums(d_old['pred_air'], d_old['target_air'], sd_air, n_pts, n_time)
    exact = {'air': (S_old.sum(0) / n_tot, S_new.sum(0) / n_tot)}
    boot_air = paired_bootstrap(S_new, S_old, n_tot)
    del S_new, S_old

    vmap = d_new['valid'].reshape(n_pts, n_time).all(axis=1)
    fc = np.load(_find(f'facade_targets_{site}.npz'))
    fs_fac = fc['facade_norm_std'].astype(np.float32)
    Sf_new = point_abs_sums(d_new['pred_fac'], d_new['target_fac'], fs_fac, n_pts, n_time)[vmap]
    Sf_old = point_abs_sums(d_old['pred_fac'], d_old['target_fac'], fs_fac, n_pts, n_time)[vmap]
    n_fac = int(vmap.sum()) * n_time
    exact['facade'] = (Sf_old.sum(0) / n_fac, Sf_new.sum(0) / n_fac)
    boot_fac = paired_bootstrap(Sf_new, Sf_old, n_fac)
    del Sf_new, Sf_old, d_old, d_new

    site_res = {'n_tot': int(n_tot), 'n_pts': int(n_pts), 'n_time': n_time, 'n_fac_rows': n_fac}
    for head, boot, keys, units in (('air', boot_air, AIR_KEYS, AIR_UNITS),
                                    ('facade', boot_fac, FAC_KEYS, FAC_UNITS)):
        eo, en = exact[head]
        print(f'\n[{site}] {head.upper()} - ensemble5: optuna_best (old) vs p5a_best (new), physical units')
        print(f'{"Variable":8} {"old_MAE":>9} {"new_MAE":>9} {"delta%":>8} '
              f'{"new 95% CI":>17} {"old 95% CI":>17} {"delta 95% CI":>17} {"P(new<old)":>10}  verdict')
        block = {}
        for vi, key in enumerate(keys):
            var = key if isinstance(key, str) else key[0]
            n_lo, n_hi = np.percentile(boot['new'][:, vi], [2.5, 97.5])
            o_lo, o_hi = np.percentile(boot['old'][:, vi], [2.5, 97.5])
            dd_lo, dd_hi = np.percentile(boot['delta'][:, vi], [2.5, 97.5])
            o_mae, n_mae = float(eo[vi]), float(en[vi])
            dpct = (n_mae - o_mae) / o_mae * 100.0
            p_better = float((boot['delta'][:, vi] < 0).mean())
            v = verdict(dd_lo, dd_hi)
            print(f'{var:8} {o_mae:>9.3f} {n_mae:>9.3f} {dpct:>+7.1f}% '
                  f'[{n_lo:>7.3f},{n_hi:>7.3f}] [{o_lo:>7.3f},{o_hi:>7.3f}] '
                  f'[{dd_lo:>+7.3f},{dd_hi:>+7.3f}] {p_better:>10.3f}  {v}  {units[var]}')
            block[var] = {'old_mae': r4(o_mae), 'new_mae': r4(n_mae), 'delta_pct': r4(dpct),
                          'unit': units[var], 'new_ci95': [r4(n_lo), r4(n_hi)],
                          'old_ci95': [r4(o_lo), r4(o_hi)], 'delta_ci95': [r4(dd_lo), r4(dd_hi)],
                          'p_new_better': r4(p_better), 'verdict': v}
            csv_rows.append([results['meta']['computed_at'], site, head, var,
                             f'{o_mae:.4f}', f'{n_mae:.4f}', f'{dpct:.2f}',
                             f'{n_lo:.4f}', f'{n_hi:.4f}', f'{o_lo:.4f}', f'{o_hi:.4f}',
                             f'{dd_lo:.4f}', f'{dd_hi:.4f}', f'{p_better:.3f}', v])
        site_res[head] = block
    results[site] = site_res

jpath = OUT_DIR / 'p5a_vs_optuna_compare.json'
jpath.write_text(json.dumps(results, indent=2))
cpath = OUT_DIR / 'p5a_vs_optuna_compare.csv'
with open(cpath, 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['ts', 'site', 'head', 'var', 'old_mae', 'new_mae', 'delta_pct',
                'new_ci_lo', 'new_ci_hi', 'old_ci_lo', 'old_ci_hi',
                'delta_ci_lo', 'delta_ci_hi', 'p_new_better', 'verdict'])
    w.writerows(csv_rows)
print(f'\n[SAVED] {jpath.name}')
print(f'[SAVED] {cpath.name}')
print(f'\n===== P5a-COMPARE DONE in {(time.time() - t00) / 60:.1f} min =====')
print('NEXT: tiny backup cell -> download zip -> paste me the table.')
