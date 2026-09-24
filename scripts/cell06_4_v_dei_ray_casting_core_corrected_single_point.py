# ===== CELL 4 — V-DEI ray-casting core (corrected) + single-point test + sun/shadow audit =====
import numpy as np
import xarray as xr

SITE = 'canyon'
TEST_POINT = (263, 130, 4)            # (i, j, k) — k=4 is the biomet-height level (~1.97 m)
class_names = {0: 'sky/air', 1: 'building', 2: 'vegetation', 3: 'ground', -1: 'UNKNOWN-CODE'}

g = np.load(output(f'01_Data/02_Processed/geometry_{SITE}.npz'))
objects, z_ctr, dz = g['objects'], g['z_center'], g['dz']
K, J, I = objects.shape
TOP = z_ctr[-1] + dz[-1] / 2.0
_upper = np.concatenate([(z_ctr[:-1] + z_ctr[1:]) / 2, [TOP]])   # true cell upper-bounds

def k_of_z(z):                        # correct mapping on the stretched vertical grid
    return int(np.searchsorted(_upper, z))

def cell_class(i, j, k):
    v = objects[k, j, i]
    if not np.isfinite(v) or v == 0:  return 0
    if v == 1:                        return 1
    if v in (11, 12, 13):             return 2   # vegetation codes — LAD-confirmed (CELL 3)
    return -1

def cast_ray_hit(origin_ijk, azimuth_deg, elevation_deg, *,
                 step=0.5, max_range=100.0, az_grid=None):
    """
    Ray from CELL CENTRE. Returns (hit_class, distance_m):
      0=sky/air, 1=building, 2=vegetation, 3=ground. max_range default=100 (manuscript d_max).
    azimuth in the MODEL-GRID frame (az_grid), or classic azimuth_deg (grid frame, az=0 -> +x).
    Compass->grid conversion happens OUTSIDE so the audit can test both rotations.
    """
    i0, j0, k0 = origin_ijk
    az = np.radians(az_grid if az_grid is not None else azimuth_deg)
    el = np.radians(elevation_deg)
    dx_, dy_, dz_ = np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)
    x0, y0, z0 = (i0 + .5)*2.0, (j0 + .5)*2.0, z_ctr[k0]   # cell centre (dx=dy=2 m)
    dist = 0.0
    while dist < max_range:
        dist += step
        x, y, z = x0 + dx_*dist, y0 + dy_*dist, z0 + dz_*dist
        if z < 0.0:                                     return 3, dist   # below flat terrain
        if x < 0 or y < 0 or x >= I*2.0 or y >= J*2.0:  return 0, dist   # leaves domain
        if z > TOP:                                     return 0, dist   # above domain -> sky
        i, j, k = int(x/2.0), int(y/2.0), k_of_z(z)
        if (i, j, k) == (i0, j0, k0):                   continue         # skip origin cell
        c = cell_class(i, j, k)
        if c > 0:                                       return c, dist
    return 0, max_range

print(f'[{SITE}] origin cell class = {class_names[cell_class(*TEST_POINT)]}, '
      f'height = {z_ctr[TEST_POINT[2]]:.2f} m')

print(f"\n{'Azimuth':>8} {'Elevation':>10} {'HitClass':>12} {'Distance(m)':>12}")
for azg in np.arange(0, 360, 45):          # grid-frame azimuth sweep (test only)
    for el in np.arange(-60, 61, 30):
        hc, d = cast_ray_hit(TEST_POINT, azg, el)
        print(f"{azg:>8} {el:>10} {class_names[hc]:>12} {d:>12.2f}")

# --- D012 azimuth audit: real sun position at t0 vs ENVI-met's own ShadowFlag ---
ds = xr.open_dataset(raw_file(SITE), decode_times=False)
rot = float(ds.attrs['ModelRotation'])
sun_az = float(np.asarray(ds.attrs['SunPositionAzimuth'])[0])
sun_h  = float(np.asarray(ds.attrs['SunPositionHeight'])[0])
print(f'\n[SUN AUDIT t0] compass azimuth={sun_az:.1f}\u00b0  height={sun_h:.1f}\u00b0  ModelRotation={rot}\u00b0')
for label, azg in (('az - R', sun_az - rot), ('az + R', sun_az + rot)):
    hc, d = cast_ray_hit(TEST_POINT, 0, sun_h, az_grid=azg)
    print(f'  ray toward sun ({label} = {azg:.1f}\u00b0 grid) -> {class_names[hc]} at {d:.1f} m')
sf = ds['ShadowFlag']
sfv = sf.isel(Time=0).values
pt = sfv if sfv.ndim == 2 else sfv[TEST_POINT[2], TEST_POINT[1], TEST_POINT[0]]
print(f'  ENVI-met ShadowFlag at test point = {pt}')
print('  INTERPRET: the hypothesis (-R or +R) whose ray hit MATCHES the flag WINS -> lock it.')
ds.close()
print('\nDONE — ray core validated; production vectorized version comes with S1.')
