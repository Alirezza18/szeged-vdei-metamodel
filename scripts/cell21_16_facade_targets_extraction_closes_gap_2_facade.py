# ===== CELL 16 - facade targets extraction (closes GAP-2: facade_targets.npz) =====
# 4 facade targets from the X facade system (D032; Y/Z counts recorded as QC):
#   Twall = XFac_WallTempNode1Outside        Qsens = XFac_WallSystemSensHeatFlux
#   SWabs = XFac_WallSystemSWReceived        LWbal = XFac_WallSystemLWEnergyBalance
# Facade points = sample-set rows with valid X-system vars at t0.
# Normalization stats (mean/std over ALL times x facade points) computed ONCE, FROZEN (D030).
import numpy as np, xarray as xr, gc

FACADE_MAP = {
    'Twall': 'XFac_WallTempNode1Outside',
    'Qsens': 'XFac_WallSystemSensHeatFlux',
    'SWabs': 'XFac_WallSystemSWReceived',
    'LWbal': 'XFac_WallSystemLWEnergyBalance',
}
SYS_PREFIXES = ('XFac_', 'YFac_', 'ZFac_')

for site in SITES:
    print(f"\n===== [{site.upper()}] facade targets (X system) =====")
    d = np.load(output(f'01_Data/02_Processed/vdei_features_{site}.npz'))
    i0, j0, k0 = d['i0'], d['j0'], d['k0']

    ds = xr.open_dataset(raw_file(site), decode_times=False)
    n_time = ds.sizes['Time']

    for pref in SYS_PREFIXES:
        v = f'{pref}WallTempNode1Outside'
        if v in ds.variables:
            a = ds[v].isel(Time=0).values
            nv = int(np.isfinite(a[k0, j0, i0]).sum())
            print(f'  [{site}] {pref} valid facade points at t0: {nv:,}')
            del a
        else:
            print(f'  [{site}] {v}: NOT in file')

    tw0 = ds[FACADE_MAP['Twall']].isel(Time=0).values
    valid = np.isfinite(tw0[k0, j0, i0])
    n_f = int(valid.sum())
    print(f'[{site}] facade-adjacent sample points (X): {n_f:,} / {len(i0):,} '
          f'({100*n_f/len(i0):.2f}%)')
    del tw0; gc.collect()

    facade = {}
    stats_mean, stats_std = {}, {}
    for name, var in FACADE_MAP.items():
        arr = ds[var].values                          # (Time, K, J, I)
        extracted = arr[:, k0[valid], j0[valid], i0[valid]].astype(np.float32).T  # (n_f, T)
        facade[name] = extracted
        mu, sd = float(np.nanmean(extracted)), float(np.nanstd(extracted) + 1e-6)
        stats_mean[name], stats_std[name] = mu, sd
        print(f'  [{site}] {name:6} shape={extracted.shape}  '
              f'NaN%={100*np.isnan(extracted).mean():.2f}%  '
              f'range=[{np.nanmin(extracted):.2f}, {np.nanmax(extracted):.2f}]  '
              f'frozen stats: mean={mu:.3f} std={sd:.3f}')
        del arr; gc.collect()
    ds.close()

    out = output(f'01_Data/02_Processed/facade_targets_{site}.npz')
    np.savez_compressed(out,
        facade_point_idx=valid.nonzero()[0].astype(np.int64),
        **{f'facade_{k}': v for k, v in facade.items()},
        facade_norm_mean=np.array([stats_mean[k] for k in FACADE_MAP], np.float32),
        facade_norm_std=np.array([stats_std[k] for k in FACADE_MAP], np.float32),
        facade_vars=np.array(list(FACADE_MAP.keys())),
        system='X', n_time=n_time)
    print(f'[SAVED] {out.name} ({out.stat().st_size/1e6:.0f} MB)  |  D030 stats frozen')
    del facade; gc.collect()

print('\nFacade targets DONE - S2 is now 100% complete (D032: X system, paper-phase check).')
