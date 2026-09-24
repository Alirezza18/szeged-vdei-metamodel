# ===== CELL 7 — V-DEI raw sanity check (run AFTER CELL 6 finishes) =====
import numpy as np

CLASS_NAMES = {0: 'sky/air', 1: 'building', 2: 'vegetation', 3: 'ground'}

for site in SITES:
    f = output(f'01_Data/02_Processed/vdei_raw_{site}.npz')
    d = np.load(f)
    cls, dist = d['cls'], d['dist']
    print(f'\n=== [{site.upper()}] V-DEI sanity ===')
    print(f'file: {f.name} ({f.stat().st_size/1e6:.0f} MB) | cls={cls.shape} dirs={int(d["n_dirs"])}')
    vals, cnts = np.unique(cls, return_counts=True)
    for v, c in zip(vals, cnts):
        m = cls == v; dd = dist[m]
        print(f'  {CLASS_NAMES.get(int(v), v):<11} {c:>12,} ({100*c/cls.size:5.1f}%)  '
              f'mean={dd.mean():6.2f}m  p50={np.percentile(dd,50):6.2f}  p95={np.percentile(dd,95):6.2f}')
    g = np.load(output(f'01_Data/02_Processed/geometry_{site}.npz'))
    exp = int(g['is_air'][:, ::2, ::2].sum())
    ok = (exp == cls.shape[0])
    print(f'  consistency: expected points={exp:,}  got={cls.shape[0]:,}  -> {"OK ✅" if ok else "MISMATCH ❌"}')
