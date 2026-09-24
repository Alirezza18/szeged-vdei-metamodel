# ===== CELL 5 — AZIMUTH LOCK: statistical sun/shadow agreement audit =====
import numpy as np, xarray as xr

N_SAMPLE, K_GROUND = 600, 0
TIMES = [0, 6, 12]                     # morning / near-noon / evening

def ray_hits(i, j, k, az_grid, el_deg, z_ctr, z_up, cls, nI, nJ, TOP, step=0.5, max_range=100.0):
    azr, elr = np.radians(az_grid), np.radians(el_deg)
    dx_, dy_, dz_ = np.cos(elr)*np.cos(azr), np.cos(elr)*np.sin(azr), np.sin(elr)
    x0, y0, z0 = (i+.5)*2.0, (j+.5)*2.0, z_ctr[k]
    dist = 0.0
    while dist < max_range:
        dist += step
        x, y, z = x0+dx_*dist, y0+dy_*dist, z0+dz_*dist
        if z < 0.0: return 3, dist
        if x < 0 or y < 0 or x >= nI*2.0 or y >= nJ*2.0 or z > TOP: return 0, dist
        ii, jj, kk = int(x/2.0), int(y/2.0), int(np.searchsorted(z_up, z))
        if (ii, jj, kk) == (i, j, k): continue
        v = cls[kk, jj, ii]
        if v in (1, 2): return v, dist
    return 0, max_range

for site in SITES:
    g = np.load(output(f'01_Data/02_Processed/geometry_{site}.npz'))
    objects, z_ctr, dz = g['objects'], g['z_center'], g['dz']
    nK, nJ, nI = objects.shape
    TOP = z_ctr[-1] + dz[-1]/2
    z_up = np.concatenate([(z_ctr[:-1]+z_ctr[1:])/2, [TOP]]).astype(np.float64)
    cls = np.zeros(objects.shape, np.int8)
    cls[objects == 1.0] = 1
    cls[(objects == 11.0)|(objects == 12.0)|(objects == 13.0)] = 2

    ds = xr.open_dataset(raw_file(site), decode_times=False)
    rot  = float(ds.attrs['ModelRotation'])
    sun_az = np.asarray(ds.attrs['SunPositionAzimuth'], float)
    sun_h  = np.asarray(ds.attrs['SunPositionHeight'],  float)
    rng = np.random.default_rng(42)
    jj = rng.integers(0, nJ, N_SAMPLE); ii = rng.integers(0, nI, N_SAMPLE)

    print(f'\n===== [{site.upper()}] AZIMUTH LOCK AUDIT (ModelRotation={rot}\u00b0) =====')
    best = None
    for t in TIMES:
        if t >= len(sun_h) or sun_h[t] < 3: continue
        sf = np.asarray(ds['ShadowFlag'].isel(Time=t).values, float)
        valid = np.isfinite(sf[jj, ii])
        hits = {'-R': [], '+R': []}
        for p in range(N_SAMPLE):
            if not valid[p]: continue
            i, j = int(ii[p]), int(jj[p])
            hits['-R'].append(ray_hits(i, j, K_GROUND, sun_az[t]-rot, sun_h[t], z_ctr, z_up, cls, nI, nJ, TOP)[0])
            hits['+R'].append(ray_hits(i, j, K_GROUND, sun_az[t]+rot, sun_h[t], z_ctr, z_up, cls, nI, nJ, TOP)[0])
        flag1 = sf[jj, ii][valid] > 0.5
        for lbl in ('-R', '+R'):
            pred = np.array(hits[lbl]) > 0
            acc_shaded = 100*np.mean(pred == flag1)
            acc_lit    = 100*np.mean(pred == ~flag1)
            print(f'  t={t:2d} (az={sun_az[t]:5.1f}\u00b0 h={sun_h[t]:4.1f}\u00b0) {lbl:>3}: '
                  f'acc[flag1=shaded]={acc_shaded:5.1f}%  acc[flag1=lit]={acc_lit:5.1f}%  (n={valid.sum()})')
            s = max(acc_shaded, acc_lit)
            if best is None or s > best[0]: best = (s, lbl, t)
    print(f'  --> CANDIDATE LOCK: rotation "{best[1]}" (best agreement {best[0]:.1f}% at t={best[2]})')
    ds.close()
print('\nRun CELL 5 for BOTH sites, then we lock the convention once and for all.')
