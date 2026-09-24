# ===== CELL 1 — raw ENVI-met NetCDF inspection (Phase 2 audit) =====
from pathlib import Path
import pandas as pd
import xarray as xr

GROUPS = {
    'atmospheric':      ['Ta', 'AirTemperature', 'RH', 'RelativeHumidity', 'Wind', 'WS',
                         'TKE', 'TKE_', 'Tmrt', 'MeanRadiant'],
    'facade':           ['Twall', 'WallTemperature', 'Qsens', 'Sensible', 'SWabs', 'Shortwave',
                         'LWbal', 'Longwave', 'Facade'],
    'static_morphology':['Building', 'LAD', 'LeafArea', 'SVF', 'Topography',
                         'Obstacle', 'Z0', 'Height'],
}

def group_of(var):
    vl = var.lower()
    for g, keys in GROUPS.items():
        if any(k.lower() in vl for k in keys):
            return g
    return 'other'

def open_nc(path):
    """Open robustly; ENVI-met files often have nonstandard time encoding."""
    last = None
    for eng in ('netcdf4', 'h5netcdf', 'scipy'):
        try:
            return xr.open_dataset(path, decode_times=False, engine=eng)
        except Exception as e:
            last = e
    raise RuntimeError(f'Could not open {path} with any engine: {last}')

def inspect_site(site, path, out_dir):
    print('\n' + '=' * 78)
    print(f'RESEARCH INSPECTION: {site.upper()}  ({path.name}, {path.stat().st_size/1e9:.2f} GB)')
    print('=' * 78)
    ds = open_nc(path)

    print('\n[GLOBAL ATTRS]')
    for k, v in ds.attrs.items():
        print(f'  {k:<28}: {v}')

    print('\n[DIMENSIONS]')
    for d, n in ds.sizes.items():
        print(f'  {d:<12}: {n}')

    print('\n[TIME]')
    tvar = ds.get('Time') if 'Time' in ds.variables else (ds['time'] if 'time' in ds.variables else None)
    if tvar is not None:
        tv = tvar.values
        print(f'  steps       : {len(tv)}')
        print(f'  start / end : {tv[0]}  /  {tv[-1]}')
        for k, v in tvar.attrs.items():
            print(f'  {k:<12}: {v}')
    else:
        print('  no Time variable found - check raw encoding')

    rows = []
    for var in ds.data_vars:
        da = ds[var]
        rows.append({'site': site, 'variable': var,
                     'units': da.attrs.get('units', 'N/A'),
                     'long_name': da.attrs.get('long_name', ''),
                     'dims': 'x'.join(da.dims),
                     'group': group_of(var)})
    inv = pd.DataFrame(rows)

    print(f'\n[VARIABLES] total={len(inv)}')
    for g in ['atmospheric', 'facade', 'static_morphology', 'other']:
        sub = inv[inv.group == g]
        if len(sub):
            print(f'  --- {g} ({len(sub)}) ---')
            for _, r in sub.iterrows():
                print(f'    {r.variable:<32} {str(r.units):<12} {r.long_name}')

    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f'variable_inventory_{site}.csv'
    inv.to_csv(csv, index=False, encoding='utf-8')
    print(f'\n[SAVED] {csv}')
    ds.close()
    print(f'DONE {site}')
    return inv

OUT = output('01_Data/03_Metadata')
inv_canyon = inspect_site('canyon', raw_file('canyon'), OUT)
inv_plaza  = inspect_site('plaza',  raw_file('plaza'),  OUT)
print('\nDONE — both variable inventories saved.')
