# ===== CELL 3 — object codes identification + facade-% definition reconciliation =====
# MERGED: old Colab cell 2 (objects x building per K) + old cell 3 (LAD confirmation)
# Masks come from CELL 2 npz (fast); LAD and facade vars read once from the raw nc.
import numpy as np
import xarray as xr

for site in SITES:
    g = np.load(output(f'01_Data/02_Processed/geometry_{site}.npz'))
    objects, is_building, z_ctr = g['objects'], g['is_building'], g['z_center']

    print(f"\n{'='*70}\n[{site.upper()}] Objects code vs is_building mask\n{'='*70}")
    codes = np.unique(objects[np.isfinite(objects)])
    for code in codes:
        mc = (objects == code); nt = int(mc.sum()); nb = int((mc & is_building).sum())
        print(f"Objects=={code:5.1f}: total={nt:>9,}  building_overlap={nb:>9,} "
              f"({100*nb/nt:5.1f}%)  air={100*(nt-nb)/nt:5.1f}%")

    print(f"\n[{site}] Objects distribution by K-level (near ground):")
    for k in range(8):
        vals, cnts = np.unique(objects[k][np.isfinite(objects[k])], return_counts=True)
        print(f"  K={k} (h={z_ctr[k]:.2f}m): {dict(zip(vals.tolist(), cnts.tolist()))}")

    ds = xr.open_dataset(raw_file(site), decode_times=False)
    lad = ds['LAD'].isel(Time=0).values   # (K,J,I)

    print(f"\n{'='*60}\n[{site}] LAD overlap per Objects code (vegetation confirmation)\n{'='*60}")
    for code in codes:
        mc = (objects == code); lad_in = lad[mc]
        nz = int(np.sum((lad_in > 0) & np.isfinite(lad_in)))
        mean_lad = float(np.nanmean(lad_in[lad_in > 0])) if nz else 0.0
        print(f"Objects=={code:5.1f}: total={int(mc.sum()):>9,}  LAD>0={nz:>9,} "
              f"({100*nz/mc.sum():5.1f}%)  mean_LAD={mean_lad:.4f}")
    print(f"  [sanity] total LAD>0 cells in domain: "
          f"{int(np.sum((lad > 0) & np.isfinite(lad))):,}")

    # --- facade-% reconciliation: X-only vs SUM(X+Y+Z) vs UNION (manuscript def?) ---
    print(f"\n{'='*60}\n[{site}] FACADE-% DEFINITION RECONCILIATION (t0)\n{'='*60}")
    ok = lambda a: np.isfinite(a) & (a > -900)
    n, pcts, shapes = 0, [], []
    for d in ('X', 'Y', 'Z'):
        v = ds[f'{d}Fac_WallTempNode1Outside'].isel(Time=0).values
        nv = int(ok(v).sum())
        print(f"  {d}: shape={v.shape}  valid={nv:,}  ({100*nv/v.size:.2f}% of {v.size:,})")
        pcts.append(100*nv/v.size)
        if v.ndim == 3:  # only 3D grids enter the union/sum comparison
            n = v.size if n == 0 else n
            shapes.append(v.shape)
    if len(set(shapes)) == 1 and n > 0:
        fx, fy, fz = [ds[f'{d}Fac_WallTempNode1Outside'].isel(Time=0).values for d in 'XYZ']
        uni = int((ok(fx) | ok(fy) | ok(fz)).sum())
        print(f"  --> SUM(X+Y+Z) = {sum(pcts):.2f}%   UNION(any) = {100*uni/n:.2f}%")
        print(f"      manuscript expects ~1.95% (canyon) / ~2.61% (plaza)")
    else:
        print("  --> facade grids have different shapes (staggered) — SUM over per-grid % above")
    ds.close()

print('\nDONE — vegetation codes identified + facade-% definition resolved.')
