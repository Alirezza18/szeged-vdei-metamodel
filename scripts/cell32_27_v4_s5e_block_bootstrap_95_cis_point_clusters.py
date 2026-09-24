# ===== CELL 27 v4 (S5e) - Block-bootstrap 95% CIs (point clusters, physical units) =====
# Fixes vs v3: (1) n_tot defined locally (v3 leaked plaza's n_tot -> canyon MAE wrong),
# (2) MAE/CI denormalized to physical units, (3) facade errors masked by valid mask.
import csv
from datetime import datetime
import numpy as np

SITES  = ('canyon', 'plaza')
N_BOOT = 1000

AIR_KEYS = ['T', 'RelHum', 'WindSpd', 'TKE', 'TMRT']
AIR_SD   = {'T': 4.0, 'RelHum': 15.0, 'WindSpd': 1.5, 'TKE': 100.0, 'TMRT': 20.0}   # D006
FAC_KEYS = ['Twall', 'Qsens', 'SWabs', 'LWbal']
AIR_UNITS = {'T': 'degC', 'RelHum': '%', 'WindSpd': 'm/s', 'TKE': 'm2/s2', 'TMRT': 'degC'}
FAC_UNITS = {'Twall': 'degC', 'Qsens': 'W/m2', 'SWabs': 'W/m2', 'LWbal': 'W/m2'}

mlog = output('03_Results/02_Metrics/test_metrics_log.csv')
ts = datetime.now().strftime('%Y-%m-%d %H:%M')

for SITE in SITES:
    npz_path = output(f'03_Results/02_Metrics/{SITE}_dualhead_ensemble5_test_fullpool.npz')
    assert npz_path.exists(), f'ensemble npz missing for {SITE} - run CELL 26 (recovery) first!'
    d = np.load(npz_path)
    pred_air = d['pred_air'].astype(np.float32); targ_air = d['target_air'].astype(np.float32)
    pred_fac = d['pred_fac'].astype(np.float32); targ_fac = d['target_fac'].astype(np.float32)
    valid    = d['valid'].astype(bool)
    d.close()

    n_time   = {'canyon': 25, 'plaza': 49}[SITE]
    n_tot    = pred_air.shape[0]                 # LOCAL - never reuse kernel globals
    n_points = n_tot // n_time
    assert n_tot == n_points * n_time

    fs     = np.asarray(VDEIDatasetV2(SITE, 'test').d['fac_std'], dtype=np.float32)
    sd_air = np.array([AIR_SD[v] for v in AIR_KEYS], dtype=np.float32)
    fac_point_mask = valid.reshape(n_points, n_time).any(axis=1)
    n_facade_pts = int(fac_point_mask.sum())
    n_valid = int(valid.sum())
    print(f'[{SITE}] {n_points:,} points ({n_facade_pts:,} facade pts) x {n_time} t -> B={N_BOOT}')

    # Per-point error sums in PHYSICAL units, facade masked to valid samples only
    abs_err_air = np.abs((pred_air - targ_air) * sd_air)                      # (n, 5)
    abs_err_fac = np.abs((pred_fac - targ_fac) * fs) * valid[..., None]       # masked
    point_air = abs_err_air.reshape(n_points, n_time, len(AIR_KEYS)).sum(axis=1)
    point_fac = abs_err_fac.reshape(n_points, n_time, len(FAC_KEYS)).sum(axis=1)[fac_point_mask]

    rng = np.random.RandomState(0)
    boot_air = np.zeros((N_BOOT, len(AIR_KEYS)))
    boot_fac = np.zeros((N_BOOT, len(FAC_KEYS)))
    for b in range(N_BOOT):
        idx  = rng.randint(0, n_points, n_points)
        fidx = rng.randint(0, n_facade_pts, n_facade_pts)
        boot_air[b] = point_air[idx].sum(axis=0) / (n_points * n_time)
        boot_fac[b] = point_fac[fidx].sum(axis=0) / (n_facade_pts * n_time)
        if (b + 1) % 200 == 0:
            print(f'  boot {b+1}/{N_BOOT}', flush=True)

    lo_a, hi_a = np.percentile(boot_air, [2.5, 97.5], axis=0)
    lo_f, hi_f = np.percentile(boot_fac, [2.5, 97.5], axis=0)
    mae_air = point_air.sum(axis=0) / n_tot
    mae_fac = point_fac.sum(axis=0) / max(1, n_facade_pts * n_time)

    print(f'\n===== [{SITE.upper()}] ENSEMBLE 95% CI (point-cluster bootstrap, B={N_BOOT}) =====')
    print(f'{"Variable":10} {"MAE":>10} {"95% CI":>24} {"Unit":>8}')
    for vi, var in enumerate(AIR_KEYS):
        print(f'{var:10} {mae_air[vi]:>10.3f}   [{lo_a[vi]:.3f}, {hi_a[vi]:.3f}] {AIR_UNITS[var]:>8}')
    for vi, var in enumerate(FAC_KEYS):
        print(f'{var:10} {mae_fac[vi]:>10.3f}   [{lo_f[vi]:.3f}, {hi_f[vi]:.3f}] {FAC_UNITS[var]:>8}')

    with open(mlog, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        for vi, var in enumerate(AIR_KEYS):
            w.writerow([ts, SITE, 'ensemble5-ci', '-', 'air', var, n_tot,
                        f'{mae_air[vi]:.4f}', f'CI[{lo_a[vi]:.4f},{hi_a[vi]:.4f}]'])
        for vi, var in enumerate(FAC_KEYS):
            w.writerow([ts, SITE, 'ensemble5-ci', '-', 'facade', var, n_valid,
                        f'{mae_fac[vi]:.4f}', f'CI[{lo_f[vi]:.4f},{hi_f[vi]:.4f}]'])

print('\nS5e DONE (v4) - paste this output to Buffy.')
