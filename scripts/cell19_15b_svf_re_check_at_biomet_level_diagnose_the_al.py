# ===== CELL 15b - SVF re-check at BIOMET level (diagnose the all-K dilution) =====
# 2D ENVI-met SVF is a SURFACE-level quantity; apples-to-apples = per-cell biomet level.
import numpy as np, xarray as xr

for site in SITES:
    svf = np.load(output(f'01_Data/02_Processed/svf_{site}.npz'))
    envimet, rays = svf['svf_envimet'], svf['svf_rays']
    d = np.load(output(f'01_Data/02_Processed/vdei_features_{site}.npz'))
    i0, j0, k0 = d['i0'], d['j0'], d['k0']

    ds = xr.open_dataset(raw_file(site), decode_times=False)
    zb = ds['ZNodeBiomet'].values
    if zb.ndim == 3: zb = np.nanmax(zb, axis=0)
    ds.close()
    kb = (zb - 1).astype(float)[j0, i0]

    m = (k0 == kb) & np.isfinite(kb) & np.isfinite(envimet)
    r_all = float(svf['pearson_r'])
    r_bio = np.corrcoef(envimet[m], rays[m])[0, 1]
    mad_bio = np.mean(np.abs(envimet[m] - rays[m]))
    print(f'[{site}] ALL-K  : r={r_all:.3f}  (n={int(np.isfinite(envimet).sum()):,})')
    print(f'[{site}] BIOMET : r={r_bio:.3f}  mean|diff|={mad_bio:.3f}  (n={int(m.sum()):,})')
    print(f'[{site}] biomet means: envimet={np.nanmean(envimet[m]):.3f}  '
          f'rays={float(np.nanmean(rays[m])):.3f}')

print('\nINTERPRET: BIOMET r >= 0.8 -> dilution explained, geometry VALIDATED, move on.'
      '  Still low (<0.7) -> deeper geometry audit before S3.')
