# ===== CHECK (5 s) - are all needed files findable? =====
from pathlib import Path
IN = Path('/kaggle/input')
def find(pat):
    for d in sorted(IN.glob('*')):
        h = sorted(d.rglob(pat))
        if h: return str(h[0])
    return None
ok = True
for site in ('canyon', 'plaza'):
    for pat in (f'{site}_dualhead_ensemble5_test_fullpool.npz',
                f'{site}_dualhead_p5a_best_ensemble5_test_fullpool.npz',
                f'facade_targets_{site}.npz'):
        p = find(pat); ok &= (p is not None)
        print(f'{"OK" if p else "MISSING":8} {pat}')
print('\nALL GOOD -> run the P5a-COMPARE cell' if ok else '\nMISSING FILES -> paste me this output')
