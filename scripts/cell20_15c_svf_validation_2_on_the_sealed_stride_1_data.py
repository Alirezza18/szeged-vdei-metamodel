# ===== CELL 15c - SVF validation #2 on the SEALED stride-1 data: all-K, per-level (D040) =====
import numpy as np, xarray as xr

for site in SITES:
    print(f'\n===== [{site.upper()}] SVF 3D check (all-K, per-point level) =====')
    svf = np.load(processed_dir(site) / f'svf_{site}.npz')
    rays = svf['svf_rays'].astype(np.float32); svf.close()
    vf = np.load(processed_dir(site) / f'vdei_features_{site}.npz')
    k0 = vf['k0'].astype(np.int64); j0 = vf['j0'].astype(np.int64); i0 = vf['i0'].astype(np.int64)
    vf.close()
    print(f'[{site}] points={len(k0):,}')

    ds = xr.open_dataset(raw_file(site), decode_times=False)
    v = ds['SVFUpSky']
    svf3d = v.isel(Time=0).values if v.ndim == 4 else v.values
    ds.close()
    print(f'[{site}] SVFUpSky shape={svf3d.shape}')

    ref = svf3d[k0, j0, i0]
    ok = np.isfinite(ref) & np.isfinite(rays)
    r = float(np.corrcoef(ref[ok], rays[ok])[0, 1])
    mad = float(np.mean(np.abs(ref[ok] - rays[ok])))
    print(f'[{site}] n={int(ok.sum()):,} | Pearson r(svf_rays, SVFUpSky@own K) = {r:.3f} | mean|diff| = {mad:.3f}')
    print(f'[{site}] means: envimet={np.nanmean(ref[ok]):.3f}  rays={np.nanmean(rays[ok]):.3f}')

    out = output(f'03_Results/diagnostics/svf3d_check_{site}.npz')
    np.savez_compressed(out, r=r, mad=mad, n=int(ok.sum()),
                        mean_envimet=float(np.nanmean(ref[ok])),
                        mean_rays=float(np.nanmean(rays[ok])))
    print(f'[SAVED] {out.name}  |  gate: {"PASS (>=0.90) ✅" if r >= 0.90 else "CHECK (<0.90) ⚠️"}')

print('\nCELL 15c DONE - per-level geometry validation on sealed data (D040).')
