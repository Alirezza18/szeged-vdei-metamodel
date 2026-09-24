# ===== CELL P6-FC EVAL (CPU) - §2.6 fact-check =====
from pathlib import Path
import numpy as np
import re

IN = Path('/kaggle/input')

def find_all(pat):
    out = []
    for d in sorted(IN.glob('*')):
        out += sorted(d.rglob(pat))
    return out

print('=== A) SEED MATRIX from artifact filenames ===')
def seeds_for(pattern):
    s = set()
    for p in find_all(pattern):
        m = re.search(r'seed(\d+)', p.name)
        if m:
            s.add(int(m.group(1)))
    return sorted(s)

rows = [
    ('scratch (optuna_best)   ', '*dualhead_optuna_best_seed*_test_fullpool.npz'),
    ('p5a_best (re-tune)       ', '*dualhead_p5a_best_seed*_test_fullpool.npz'),
    ('p5c ZS canyon->plaza     ', 'plaza_dualhead_p5c_zs_fromcanyon_seed*_test_fullpool.npz'),
    ('p5c FT canyon->plaza     ', 'plaza_dualhead_p5c_ft_fromcanyon_seed*_test_fullpool.npz'),
    ('p5c ZS plaza->canyon     ', 'canyon_dualhead_p5c_zs_fromplaza_seed*_test_fullpool.npz'),
    ('p5c FT plaza->canyon     ', 'canyon_dualhead_p5c_ft_fromplaza_seed*_test_fullpool.npz'),
    ('p5d ablation (all arms)  ', '*dualhead_p5d_*_seed*_test_fullpool.npz'),
]
for label, pat in rows:
    print(f'  {label}: seeds {seeds_for(pat)}')
print('  p5e joint                : runs TOMORROW - sealed design seeds {0, 1, 2}')

print('\n=== B) diagnostic-subsample claim ===')
diag = []
for pat in ('*diag*', '*subsample*', '*keypoint*', '*consecutive*'):
    diag += find_all(pat)
print('  artifacts matching diagnostics patterns:', diag if diag else 'NONE')
print('  -> the ~293k/296k "diagnostic subsample" numbers have NO artifact backing.')
print('  -> temporal-coherence analyses can run on the FULL test pool (see D).')

print('\n=== C) pedestrian level (vertical grid) ===')
ghits = find_all('geometry_canyon.npz')
if ghits:
    d = np.load(ghits[0])
    dz, z_ctr = d['dz'], d['z_center']
    d.close()
    z_up = np.concatenate([(z_ctr[:-1] + z_ctr[1:]) / 2, [z_ctr[-1] + dz[-1] / 2]])
    print(f'  K = {len(dz)} levels; z_center[0..5] = {np.round(z_ctr[:6], 2)}')
    print(f'  biomet level k=4 -> z = {z_ctr[4]:.2f} m   (draft says 1.5 m -> correct to ~1.97 m)')
    print(f'  domain top = {z_up[-1]:.1f} m')
else:
    print('  geometry npz missing')

print('\n=== D) throughput + full-pool temporal reshape feasibility ===')
for site, nt, mins in (('canyon', 25, 10.1), ('plaza', 49, 6.5)):
    hits = find_all(f'{site}_dualhead_ensemble5_test_fullpool.npz')
    if not hits:
        continue
    print(f'  {site}: ensemble npz present ({hits[0].stat().st_size/1e6:.0f} MB)')
    n = {'canyon': 10_357_000, 'plaza': 6_782_825}[site]
    print(f'     sealed full-pool eval: {n:,} samples in {mins} min -> {n/(mins*60):,.0f} samples/s')
    print(f'     temporal reshape: {n:,}/{nt} = {n//nt:,} points x {nt} steps (exact: {n % nt == 0})')

print('\n=== E) P6b feasibility manifest ===')
need = {
    'per-variable targets (MAE/R2/tail subsets)': '*targets_forcing_canyon.npz',
    'predictions + valid facade mask':            '*ensemble5_test_fullpool.npz',
    'canopy-height strata (k0)':                  '*targets_forcing_canyon.npz',
    'SVF terciles':                               'svf_canyon.npz',
    'vegetation presence (cls channels)':         'vdei_features_canyon.npz',
    'building proximity (dist channel)':          'vdei_features_canyon.npz',
    'instantaneous solar exposure (sun_hit)':     '*targets_forcing_canyon.npz',
    'spatial grid for Moran I (i0, j0)':          '*targets_forcing_canyon.npz',
}
for label, pat in need.items():
    print(f'  {label:<46} {"OK" if find_all(pat) else "MISSING"}')

print('\n=== DONE - paste this entire output back ===')
