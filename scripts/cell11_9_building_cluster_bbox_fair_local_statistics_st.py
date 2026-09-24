# ===== CELL 9 — building-cluster bbox + fair LOCAL statistics (stride-aware) =====
import numpy as np, xarray as xr

MARGIN_CELLS = 15   # cells (2 m each) added around the building cluster

for site in SITES:
    g = np.load(output(f'01_Data/02_Processed/geometry_{site}.npz'))
    is_b_2d = g['is_building'].any(axis=0)                  # (J,I) columns with buildings
    jj_b, ii_b = np.where(is_b_2d)
    nJ, nI = is_b_2d.shape
    j_min, j_max = max(0, jj_b.min()-MARGIN_CELLS), min(nJ, jj_b.max()+1+MARGIN_CELLS)
    i_min, i_max = max(0, ii_b.min()-MARGIN_CELLS), min(nI, ii_b.max()+1+MARGIN_CELLS)
    print(f'\n[{site}] bbox: J[{j_min}:{j_max}] I[{i_min}:{i_max}]  '
          f'cropped {(j_max-j_min)}x{(i_max-i_min)} (original {nJ}x{nI})')

    d = np.load(output(f'01_Data/02_Processed/vdei_raw_{site}.npz'))
    cls, i0, j0, k0 = d['cls'], d['i0'], d['j0'], d['k0']

    ds = xr.open_dataset(raw_file(site), decode_times=False)
    zb = ds['ZNodeBiomet'].values
    if zb.ndim == 3: zb = np.nanmax(zb, axis=0)
    ds.close()
    zbi = (zb - 1).astype(float)

    kb = zbi[j0, i0]
    m_bio = (k0 == kb) & np.isfinite(kb)
    m_reg = (i0 >= i_min) & (i0 < i_max) & (j0 >= j_min) & (j0 < j_max)

    names = {0: 'sky/air', 1: 'building', 2: 'vegetation', 3: 'ground'}
    for label, m in (('GLOBAL', m_bio), ('LOCAL(bbox)', m_bio & m_reg)):
        sub = cls[m]
        vals, cnts = np.unique(sub, return_counts=True)
        line = '  '.join(f'{names.get(int(v), v)}:{100*c/sub.size:5.1f}%' for v, c in zip(vals, cnts))
        print(f'  [{site}] {label:<12} n={sub.shape[0]:>8,}  {line}')

    np.savez(output(f'01_Data/02_Processed/bbox_{site}.npz'),
             j_min=j_min, j_max=j_max, i_min=i_min, i_max=i_max, margin=MARGIN_CELLS)
    print(f'  [SAVED] bbox_{site}.npz  (CELL 10 uses this)')
