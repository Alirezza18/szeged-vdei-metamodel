# ===== CELL P6-FC R4 (CPU, zero GPU quota) - §3.2 baselines + §3.3.1 transfer fact-check =====
# (A) §3.2: full baseline table from p5b_classical_baselines.json (air + facade if present)
#     vs the sealed dual-head ensemble5 (frozen from the §3.1.1 cell output), plus a
#     CLIMATOLOGY CHECK: baseline MAE vs the target's mean-absolute-deviation (MAD).
#     Static-input baselines should sit at ~MAD (they cannot track the diurnal cycle) -
#     this quantifies WHERE the deep model's advantage comes from.
# (B) §3.3.1: zero-shot cross-site transfer from the sealed p5c ZS ensemble5 npz,
#     degradation factors vs native scratch, per variable, both directions.
# Inputs: szeged-backup (ensemble5 npz), szeged-vdei-processed (targets_forcing for MAD),
#         p5b_output_backup (p5b_classical_baselines.json), p5c-transfer-backup (ZS npz).
# Outputs: printed tables + draft-claim verdicts. Runtime ~3-5 min CPU.
import json
from pathlib import Path
import numpy as np

IN = Path('/kaggle/input')
SITES = ('canyon', 'plaza')
AIR = [('T', (28.0, 4.0), r'$T_a$', 'degC'),
       ('RelHum', (40.0, 15.0), r'$RH$', '%'),
       ('WindSpd', (2.0, 1.5), r'$V$', 'm/s'),
       ('TKE', (50.0, 100.0), r'$TKE$', 'm2/s2'),
       ('TMRT', (45.0, 20.0), r'$T_{mrt}$', 'degC')]
AIR_KEYS = [a[0] for a in AIR]
AIR_LABELS = [a[2] for a in AIR]
FAC = [('Twall', r'$T_{wall}$', 'degC'), ('Qsens', r'$Q_{sens}$', 'W/m2'),
       ('SWabs', r'$SW_{abs}$', 'W/m2'), ('LWbal', r'$LW_{bal}$', 'W/m2')]
FAC_KEYS = [f[0] for f in FAC]
# sealed dual-head ensemble5 MAE (from §3.1.1 cell output - frozen constants)
ENS5 = {'canyon': {'T': 0.3834, 'RelHum': 1.2220, 'WindSpd': 0.2729, 'TKE': 3.4703, 'TMRT': 1.1111,
                   'Twall': 1.2795, 'Qsens': 10.4167, 'SWabs': 21.2542, 'LWbal': 6.8061},
        'plaza':  {'T': 0.3649, 'RelHum': 1.2849, 'WindSpd': 0.3509, 'TKE': 2.1090, 'TMRT': 1.4716,
                   'Twall': 1.2132, 'Qsens': 7.4736, 'SWabs': 22.4633, 'LWbal': 5.6052}}


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
        return None, None
    mu = np.asarray(fc['facade_norm_mean'], dtype=np.float64)
    sd = np.asarray(fc['facade_norm_std'], dtype=np.float64)
    fc.close()
    return mu, sd


def mae_air(d, vi):
    sd = float(AIR[vi][1][1])
    diff = d['pred_air'][:, vi].astype(np.float64) - d['target_air'][:, vi].astype(np.float64)
    return float(np.abs(diff).mean()) * sd


def mae_fac(d, vi, fac_sd):
    valid = d['valid'].astype(bool)
    diff = (d['pred_fac'][valid, vi].astype(np.float64)
            - d['target_fac'][valid, vi].astype(np.float64))
    return float(np.abs(diff).mean()) * float(fac_sd[vi])


print('device: CPU | inputs: szeged-backup + szeged-vdei-processed + p5b json + p5c ZS npz\n')

# ================= A) §3.2 BASELINES =================
print('================= A) §3.2 BASELINE TABLE (air head, physical MAE) =================')
jpath = find('p5b_classical_baselines.json')
base = {}
j = None
if jpath is not None:
    print(f'  [p5b_classical_baselines.json] <- {jpath.parent.parent.name}/{jpath.parent.name}/{jpath.name}')
    j = json.loads(jpath.read_text())
    res = j.get('results', {})
    for site in SITES:
        try:
            base[site] = {m: [float(res[site][var][m]['mae']) for var in AIR_KEYS]
                          for m in res[site][AIR_KEYS[0]].keys()}
        except Exception as e:
            print(f'  [{site}] json parse issue: {e}')
else:
    print('  !! p5b json not found')

# target MAD (mean absolute deviation from the mean) per air var - climatology floor
print('\n--- climatology floor (target MAD, physical) ---')
mad = {}
for site in SITES:
    d = load_npz(f'targets_forcing_{site}.npz')
    if d is None:
        continue
    mad[site] = {}
    for vi, k in enumerate(AIR_KEYS):
        t = d[f'target_{k}'].astype(np.float64)
        mad[site][k] = float(np.abs(t - t.mean()).mean())
        print(f'  [{site}] {AIR_LABELS[vi]:8} MAD = {mad[site][k]:9.3f}')
    d.close()

if base:
    for site in SITES:
        print(f'\n--- [{site}] air MAE by model ---')
        print(f'{"model":<28}' + ''.join(f'{l:>10}' for l in AIR_LABELS))
        models = sorted(base[site].keys())
        for m in models:
            print(f'{m:<28}' + ''.join(f'{v:>10.3f}' for v in base[site][m]))
        print(f'{"Dual-Head 3D-CNN (ens5)":<28}'
              + ''.join(f'{ENS5[site][k]:>10.3f}' for k in AIR_KEYS))
        # improvement of the deep model over the best baseline, per var
        print(f'  deep vs BEST baseline per var:')
        for vi, k in enumerate(AIR_KEYS):
            best = min(base[site][m][vi] for m in models)
            deep = ENS5[site][k]
            floor = mad.get(site, {}).get(k, float('nan'))
            print(f'  {AIR_LABELS[vi]:8} best={best:9.3f}  deep={deep:9.3f}  '
                  f'gain={100 * (1 - deep / best):>6.1f}%  MAD_floor={floor:9.3f}  '
                  f'deep/MAD={deep / floor:5.2f}x')

# facade baselines (if the json carries them)
if j is not None:
    print('\n--- facade baselines in json? ---')
    try:
        fac_res = j.get('results', {}).get('canyon', {}).get('Twall', None)
        print('  Twall entry keys:', list(fac_res.keys()) if fac_res else 'none')
    except Exception:
        print('  none')

# ================= B) §3.3.1 ZERO-SHOT TRANSFER =================
print('\n================= B) §3.3.1 ZERO-SHOT TRANSFER (ensemble5, physical MAE) =================')
for site in SITES:
    other = 'plaza' if site == 'canyon' == False else 'canyon'
    fac_mu, fac_sd = facade_stats(site)
    d_native = load_npz(f'{site}_dualhead_ensemble5_test_fullpool.npz')
    d_zs = load_npz(f'{site}_dualhead_p5c_zs_from{other}_ensemble5_test_fullpool.npz')
    if d_native is None or d_zs is None or fac_sd is None:
        print(f'  [{site}] missing inputs - skip'); continue
    print(f'\n--- [{site}] native scratch vs zero-shot from {other} ---')
    print(f'{"var":8}{"native":>10}{"zero-shot":>11}{"degr.x":>8}{"degr.%":>8}')
    for vi, k in enumerate(AIR_KEYS):
        n = mae_air(d_native, vi); z = mae_air(d_zs, vi)
        print(f'{AIR_LABELS[vi]:8}{n:>10.3f}{z:>11.3f}{z / n:>8.2f}{100 * (z / n - 1):>+7.1f}%')
    for vi, k in enumerate(FAC_KEYS):
        n = mae_fac(d_native, vi, fac_sd); z = mae_fac(d_zs, vi, fac_sd)
        print(f'{k:8}{n:>10.3f}{z:>11.3f}{z / n:>8.2f}{100 * (z / n - 1):>+7.1f}%')
    d_native.close(); d_zs.close()

print('\n===== DONE - paste this entire output back =====')
print('Then I write §3.2 + §3.3.1 with artifact-true numbers.')
