# ===== CELL 8 — building-exposure map at TRUE per-cell biomet level =====
import numpy as np, xarray as xr
import matplotlib.pyplot as plt

for site in SITES:
    d = np.load(output(f'01_Data/02_Processed/vdei_raw_{site}.npz'))
    cls, i0, j0, k0 = d['cls'], d['i0'], d['j0'], d['k0']

    ds = xr.open_dataset(raw_file(site), decode_times=False)
    zb = ds['ZNodeBiomet'].values
    if zb.ndim == 3: zb = np.nanmax(zb, axis=0)     # (J,I)
    ds.close()
    zbi = (zb - 1).astype(float)                     # 0-based biomet k per (j,i)

    kb = zbi[j0, i0]
    m = (k0 == kb) & np.isfinite(kb)
    bfrac = (cls[m] == 1).mean(axis=1)

    nJ, nI = zbi.shape
    hm = np.full((nJ, nI), np.nan)
    hm[j0[m], i0[m]] = bfrac

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(hm, origin='lower', cmap='inferno', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label='building-hit fraction (enclosure proxy)')
    ax.set_title(f'[{site}] building exposure at biomet level (per-cell ZNodeBiomet)')
    out_png = output(f'03_Results/diagnostics/{site}_building_exposure_biomet.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight'); plt.show()
    print(f'[{site}] biomet-level points: {m.sum():,} | mean={100*bfrac.mean():.1f}% '
          f'max={100*bfrac.max():.1f}% | [SAVED] {out_png}')
