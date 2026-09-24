# ===== CELL 11 - S2a: targets (T, RelHum, WindSpd, TKE, TMRT) + forcing + per-point sun series =====
# NEW (D026): per-point sun-block series per daylight step, LOCKED convention D021
# (az_grid = compass_az - ModelRotation). Sample set = D025 (bbox-filtered points, all K).
# Indices come from CELL 10's vdei_features_{site}.npz.
import numpy as np, xarray as xr, gc, time
from numba import njit, prange

TARGET_VARS = ['T', 'RelHum', 'WindSpd', 'TKE', 'TMRT']
K_TOP = -1          # forcing slice: top grid level (as in old pipeline)

@njit(parallel=True, fastmath=True)
def sun_batch(oi, oj, ok, dxr, dyr, dzr, obj3, z_ctr, z_up, nI, nJ, step, max_range,
              out_hit, out_dst):
    n = len(oi)
    top = z_up[-1]
    for p in prange(n):
        i0, j0, k0 = oi[p], oj[p], ok[p]
        x0, y0, z0 = (i0 + 0.5) * 2.0, (j0 + 0.5) * 2.0, z_ctr[k0]
        hit, dst, dist = 0, max_range, 0.0
        while dist < max_range:
            dist += step
            x, y, z = x0 + dxr*dist, y0 + dyr*dist, z0 + dzr*dist
            if z < 0.0:
                hit, dst = 3, dist; break
            if x < 0.0 or y < 0.0 or x >= nI*2.0 or y >= nJ*2.0 or z > top:
                break                                    # clear sky -> hit stays 0
            ii, jj = int(x/2.0), int(y/2.0)
            kk = np.searchsorted(z_up, z)
            if kk >= len(z_up): kk = len(z_up) - 1
            if ii == i0 and jj == j0 and kk == k0:
                continue
            v = obj3[kk, jj, ii]
            if v == 1:   hit, dst = 1, dist; break
            elif v == 2: hit, dst = 2, dist; break
        out_hit[p] = hit
        out_dst[p] = dst

for site in SITES:
    print(f"\n{'='*60}\n[{site}] S2a: targets + forcing + per-point sun series\n{'='*60}")
    d = np.load(output(f'01_Data/02_Processed/vdei_features_{site}.npz'))
    oi = d['i0'].astype(np.int64); oj = d['j0'].astype(np.int64); ok = d['k0'].astype(np.int64)
    n_points = len(oi)

    ds = xr.open_dataset(raw_file(site), decode_times=False)
    n_time = ds.sizes['Time']
    print(f'[{site}] n_points={n_points:,}  n_timesteps={n_time}')

    # ---- targets at each point's own K level ----
    targets = {}
    for var in TARGET_VARS:
        arr = ds[var].values                             # (Time, K, J, I)
        extracted = arr[:, ok, oj, oi]
        targets[var] = extracted.astype(np.float32).T    # -> (n_points, Time)
        print(f'[{site}] {var:7} shape={targets[var].shape}  '
              f'NaN%={100*np.isnan(targets[var]).mean():.2f}%  '
              f'range=[{np.nanmin(targets[var]):.2f}, {np.nanmax(targets[var]):.2f}]')
        del arr; gc.collect()

    # ---- forcing (domain top-level means + sun position, as in old pipeline) ----
    forcing = {}
    for var in ['T', 'RelHum', 'WindSpd']:
        top_slice = ds[var].isel(GridsK=K_TOP).values
        forcing[var] = np.nanmean(top_slice, axis=(1, 2)).astype(np.float32)
    if 'WindDir' in ds.variables:
        wd = ds['WindDir'].isel(GridsK=K_TOP).values
        wdr = np.radians(wd)
        forcing['WindDir'] = np.degrees(np.arctan2(np.nanmean(np.sin(wdr), axis=(1, 2)),
                                                   np.nanmean(np.cos(wdr), axis=(1, 2)))) % 360

    sun_az = np.asarray(ds.attrs['SunPositionAzimuth'], dtype=np.float32)
    sun_h  = np.asarray(ds.attrs['SunPositionHeight'],  dtype=np.float32)
    assert len(sun_az) == n_time, f'SunPositionAzimuth length {len(sun_az)} != n_time {n_time}'
    is_day = (sun_h > -900).astype(np.float32)           # -999 = night
    az_safe = np.where(sun_az <= -900, 0.0, sun_az)
    h_safe  = np.where(sun_h  <= -900, 0.0, sun_h)
    forcing['SunAzimuthSin'] = (np.sin(np.radians(az_safe)) * is_day).astype(np.float32)
    forcing['SunAzimuthCos'] = (np.cos(np.radians(az_safe)) * is_day).astype(np.float32)
    forcing['SunHeight']     = (h_safe * is_day).astype(np.float32)
    forcing['IsDaytime']     = is_day
    print(f'[{site}] daytime steps: {int(is_day.sum())}/{n_time} | SunHeight(day) range '
          f'[{h_safe[is_day == 1].min():.2f}, {h_safe[is_day == 1].max():.2f}]')

    # ---- D026: per-point sun-block series over daylight steps (convention D021 locked) ----
    rot = float(ds.attrs['ModelRotation'])
    g = np.load(output(f'01_Data/02_Processed/geometry_{site}.npz'))
    objects, z_ctr, dz = g['objects'], g['z_center'], g['dz']
    obj3 = np.zeros(objects.shape, np.int8)
    obj3[objects == 1.0] = 1
    obj3[(objects == 11.0) | (objects == 12.0) | (objects == 13.0)] = 2
    z_ctr = z_ctr.astype(np.float64)
    z_up = np.concatenate([(z_ctr[:-1] + z_ctr[1:]) / 2.0,
                           [z_ctr[-1] + dz[-1] / 2.0]]).astype(np.float64)
    nK, nJ, nI = objects.shape

    day_t = np.where(is_day == 1)[0].astype(np.int64)
    n_day = len(day_t)
    sun_hit = np.zeros((n_points, n_day), np.int8)
    sun_dst = np.zeros((n_points, n_day), np.float32)
    t0 = time.time()
    for idx, t in enumerate(day_t):
        azg = np.radians(float(sun_az[t]) - rot)         # D021: az_grid = compass_az - R
        el  = np.radians(max(float(h_safe[t]), 0.0))
        dxr = np.cos(el) * np.cos(azg)
        dyr = np.cos(el) * np.sin(azg)
        dzr = np.sin(el)
        oh = np.zeros(n_points, np.int8)
        od = np.zeros(n_points, np.float32)
        sun_batch(oi, oj, ok, dxr, dyr, dzr, obj3, z_ctr, z_up, nI, nJ, 0.5, 100.0, oh, od)
        sun_hit[:, idx] = oh
        sun_dst[:, idx] = od
        if (idx + 1) % 5 == 0 or idx == n_day - 1:
            el_ = time.time() - t0
            print(f'  sun step {idx+1}/{n_day} | {el_:.1f}s | '
                  f'ETA {(el_/(idx+1))*(n_day-idx-1):.1f}s')
    print(f'[{site}] mean sun-blocked fraction over daylight steps: '
          f'{100*(sun_hit > 0).mean():.1f}%')

    out = output(f'01_Data/02_Processed/targets_forcing_{site}.npz')
    np.savez_compressed(out,
        **{f'target_{v}': targets[v] for v in TARGET_VARS},
        **{f'forcing_{k}': v for k, v in forcing.items()},
        sun_hit=sun_hit, sun_dst=sun_dst.astype(np.float16),
        sun_times=day_t,
        sun_az_grid_deg=(sun_az[day_t] - rot).astype(np.float32),
        sun_el_deg=h_safe[day_t].astype(np.float32),
        i0=oi.astype(np.int32), j0=oj.astype(np.int32), k0=ok.astype(np.int32))
    print(f'[SAVED] {out.name} ({out.stat().st_size/1e6:.0f} MB)')
    del targets, forcing, sun_hit, sun_dst; gc.collect()
    ds.close()

print('\nS2a DONE - next: CELL 12 (block split).')
