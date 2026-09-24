# ===== CELL 2 — vertical grid + geometry foundation (P2->P3 bridge) =====
# MERGED: old Colab cell A (static QC, now BOTH sites) + old cell B (grid/ZBiomet/masks)
# Uses raw_file()/output() from CELL 0.
import numpy as np
import pandas as pd
import xarray as xr

STATIC_QC_VARS = ['BuildingNumber', 'BuildingHeight', 'ZTopo', 'Objects', 'GridIndex']
geom, audit_rows = {}, []

for site in SITES:
    ds = xr.open_dataset(raw_file(site), decode_times=False)
    print(f'\n===== [{site.upper()}] {raw_file(site).name} =====')

    # --- 1) static-variable QC (both sites; merged old cell A) ---
    print('\n[STATIC QC]')
    for v in STATIC_QC_VARS:
        if v not in ds.variables:
            print(f'  {v:<18} NOT FOUND'); continue
        da = ds[v]
        data = da.values if 'Time' not in da.dims else da.isel(Time=0).values
        fin = np.isfinite(data) if np.issubdtype(data.dtype, np.floating) else np.ones(data.shape, bool)
        vals = data[fin]
        print(f'  {v:<18} shape={str(data.shape):<16} dtype={data.dtype} '
              f'min={vals.min():.2f} max={vals.max():.2f} unique={len(np.unique(vals))}')
        if v in ('BuildingNumber', 'Objects') and np.issubdtype(data.dtype, np.floating):
            print(f'  {"":<18} sentinel -999 count: {int((data == -999).sum()):,}')

    # --- 2) vertical grid: REAL heights from SizeDZ (per site — grids are NOT identical) ---
    dz = np.asarray(ds.attrs['SizeDZ'], dtype=float)
    z_top = np.cumsum(dz); z_ctr = z_top - dz / 2
    gridsk = ds['GridsK'].values
    print(f'\n[VERTICAL GRID] K={len(dz)}  DZ {dz[0]:.2f}->{dz[-1]:.2f} m  TOTAL HEIGHT={z_top[-1]:.1f} m')
    print(f'  GridsK stored: first={gridsk[0]:.2f} last={gridsk[-1]:.2f} '
          f'(computed centers: {z_ctr[0]:.2f} ... {z_ctr[-1]:.2f})')

    # --- 3) ZNodeBiomet -> physical height (1-based index) ---
    zn = ds['ZNodeBiomet'].values
    for i in np.unique(zn[np.isfinite(zn)]):
        i0 = int(i) - 1
        print(f'  ZNodeBiomet={int(i)} (1-based) -> height {z_ctr[i0]:.3f} m')

    # --- 4) geometry masks at t0 (06:00, daytime) ---
    bn = ds['BuildingNumber'].isel(Time=0).values   # (K,J,I)
    ob = ds['Objects'].isel(Time=0).values
    bhv = ds['BuildingHeight']
    bh = bhv.values if 'Time' not in bhv.dims else bhv.isel(Time=0).values
    is_b = np.isfinite(bn) & (bn > -900)
    print(f'\n[GEOMETRY t0] total={bn.size:,}  building={is_b.sum():,} ({100*is_b.mean():.2f}%)  '
          f'air={(~is_b).sum():,} ({100*(~is_b).mean():.2f}%)')
    print(f'  BuildingHeight max={np.nanmax(bh):.1f} m  |  building ids={len(np.unique(bn[is_b]))}')
    print(f'  Objects unique={np.unique(ob[np.isfinite(ob) & (ob > -900)])}')

    # --- 5) facade-cell fraction audit (manuscript expects ~1.95% canyon / 2.61% plaza) ---
    tw = ds['XFac_WallTempNode1Outside'].isel(Time=0).values
    f_ok = np.isfinite(tw) & (tw > -900)
    frac = 100 * f_ok.mean()
    print(f'\n[FACADE FRACTION t0] valid Twall cells={f_ok.sum():,} ({frac:.2f}%) '
          f'  <- expect ~1.95% (canyon) / ~2.61% (plaza)')

    audit_rows.append({'site': site, 'J': ds.sizes['GridsJ'], 'I': ds.sizes['GridsI'],
                       'K': len(dz), 'total_height_m': round(float(z_top[-1]), 1),
                       'n_time': ds.sizes['Time'],
                       'model_rotation_deg': float(ds.attrs['ModelRotation']),
                       'wind_inflow_deg': float(np.asarray(ds.attrs['WindInflow'])[0]),
                       'building_pct': round(100*is_b.mean(), 2),
                       'facade_pct': round(frac, 2),
                       'max_building_height_m': round(float(np.nanmax(bh)), 1)})

    geom[site] = {'building_number': bn.astype(np.float32), 'objects': ob.astype(np.float32),
                  'building_height': bh.astype(np.float32), 'is_building': is_b, 'is_air': ~is_b,
                  'z_center': z_ctr.astype(np.float32), 'dz': dz.astype(np.float32)}
    ds.close()

    out_npz = output(f'01_Data/02_Processed/geometry_{site}.npz')
    np.savez_compressed(out_npz, **geom[site])
    print(f'[SAVED] {out_npz}')

# --- cross-site audit table ---
audit = pd.DataFrame(audit_rows)
print('\n===== CROSS-SITE AUDIT =====')
print(audit.to_string(index=False))
csv = output('01_Data/03_Metadata/grid_geometry_audit.csv')
audit.to_csv(csv, index=False)
print(f'[SAVED] {csv}')
print('\n⚠️ Vertical grids DIFFER per site — all height-based processing uses per-site z_center.')
