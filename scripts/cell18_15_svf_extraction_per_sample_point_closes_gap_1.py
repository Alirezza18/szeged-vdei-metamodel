# ===== CELL 15 - SVF extraction per sample point (closes GAP-1: svf_validation.npz) =====
# svf_envimet = ENVI-met SkyViewFactor (2D) at each sample column (old 'svf_real').
# svf_rays    = model-side sky fraction from V-DEI rays (upper hemisphere, el>0: 64 dirs).
# Pearson r between the two = independent validation of the locked V-DEI geometry (D021).
import numpy as np, xarray as xr

EL_IDX_UP = slice(5, 9)          # elevations linspace(-80,80,9): idx 5..8 = +20..+80 deg

for site in SITES:
    print(f"\n===== [{site.upper()}] SVF extraction =====")
    d = np.load(output(f'01_Data/02_Processed/vdei_features_{site}.npz'))
    cls, i0, j0 = d['cls'], d['i0'], d['j0']
    n = cls.shape[0]

    cls3 = cls.reshape(n, 16, 9)
    svf_rays = (cls3[:, :, EL_IDX_UP] == 0).mean(axis=(1, 2)).astype(np.float32)

    ds = xr.open_dataset(raw_file(site), decode_times=False)
    svf2d = ds['SkyViewFactor']
    svf2d = svf2d.isel(Time=0).values if svf2d.ndim == 3 else svf2d.values   # (J, I) static
    ds.close()
    svf_envimet = svf2d[j0, i0].astype(np.float32)

    ok = np.isfinite(svf_envimet)
    r = np.corrcoef(svf_envimet[ok], svf_rays[ok])[0, 1]
    mad = np.mean(np.abs(svf_envimet[ok] - svf_rays[ok]))
    print(f'[{site}] n={n:,} | valid ENVI-met SVF: {ok.mean()*100:.1f}% | '
          f'Pearson r(svf_envimet, svf_rays) = {r:.3f} | mean|diff| = {mad:.3f}')
    print(f'[{site}] svf_envimet mean={np.nanmean(svf_envimet):.3f}  '
          f'svf_rays mean={svf_rays.mean():.3f}')

    out = output(f'01_Data/02_Processed/svf_{site}.npz')
    np.savez_compressed(out, svf_envimet=svf_envimet, svf_rays=svf_rays,
                        i0=i0, j0=j0, pearson_r=float(r))
    print(f'[SAVED] {out.name} ({out.stat().st_size/1e6:.0f} MB)')

print('\nSVF extraction DONE - feeds baselines (C20) and the P5c SVF check.')
