# ===== CELL 6 v2 — S1a production batch (stride configurable, auto-backup) =====
try:
    import numba
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numba'], check=True)
    import numba
from numba import njit, prange
import numpy as np, time
from pathlib import Path

AZS = np.arange(0.0, 360.0, 22.5)
ELS = np.linspace(-80.0, 80.0, 9)
dir_x = np.zeros(144); dir_y = np.zeros(144); dir_z = np.zeros(144)
idx = 0
for az in AZS:
    for el in ELS:
        azr, elr = np.radians(az), np.radians(el)
        dir_x[idx], dir_y[idx], dir_z[idx] = np.cos(elr)*np.cos(azr), np.cos(elr)*np.sin(azr), np.sin(elr)
        idx += 1
N_DIRS = 144

@njit(parallel=True, fastmath=True)
def vdei_batch(oi, oj, ok, dir_x, dir_y, dir_z, obj_class, z_ctr, z_up,
               nI, nJ, step, max_range, out_c, out_d):
    n, nd = len(oi), len(dir_x)
    top = z_up[-1]
    for p in prange(n):
        i0, j0, k0 = oi[p], oj[p], ok[p]
        x0, y0, z0 = (i0 + 0.5)*2.0, (j0 + 0.5)*2.0, z_ctr[k0]
        for d in range(nd):
            dxr, dyr, dzr = dir_x[d], dir_y[d], dir_z[d]
            dist = 0.0; hc = 0; hd = max_range
            while dist < max_range:
                dist += step
                x, y, z = x0 + dxr*dist, y0 + dyr*dist, z0 + dzr*dist
                if z < 0.0:
                    hc, hd = 3, dist; break
                if x < 0.0 or y < 0.0 or x >= nI*2.0 or y >= nJ*2.0 or z > top:
                    hc, hd = 0, dist; break
                ii, jj = int(x/2.0), int(y/2.0)
                kk = np.searchsorted(z_up, z)
                if kk >= len(z_up): kk = len(z_up) - 1
                if ii == i0 and jj == j0 and kk == k0: continue
                v = obj_class[kk, jj, ii]
                if v == 1:   hc, hd = 1, dist; break
                elif v == 2: hc, hd = 2, dist; break
            out_c[p, d] = hc
            out_d[p, d] = hd

OUT_DIR = output('01_Data/02_Processed')
CHK_DIR = OUT_DIR / 'vdei_chk'; CHK_DIR.mkdir(parents=True, exist_ok=True)
HORIZONTAL_STRIDE, CHUNK, STEP, MAX_RANGE = 1, 20000, 1.0, 100.0   # <-- D023: full grid

for site in SITES:
    g = np.load(output(f'01_Data/02_Processed/geometry_{site}.npz'))
    objects, z_ctr, dz = g['objects'], g['z_center'], g['dz']
    nK, nJ, nI = objects.shape
    z_up = np.concatenate([(z_ctr[:-1]+z_ctr[1:])/2, [z_ctr[-1]+dz[-1]/2]]).astype(np.float64)
    z_ctr = z_ctr.astype(np.float64)
    cls3 = np.zeros(objects.shape, np.int8)
    cls3[objects == 1.0] = 1
    cls3[(objects == 11.0)|(objects == 12.0)|(objects == 13.0)] = 2
    is_air = g['is_air']

    ks, js, is_ = np.where(is_air[:, ::HORIZONTAL_STRIDE, ::HORIZONTAL_STRIDE])
    js = js * HORIZONTAL_STRIDE; is_ = is_ * HORIZONTAL_STRIDE
    n_points = len(ks)
    print(f'\n[{site}] stride={HORIZONTAL_STRIDE}  points={n_points:,}  rays={n_points*N_DIRS:,}')

    prev = OUT_DIR / f'vdei_raw_{site}.npz'
    if prev.exists():
        prev.rename(OUT_DIR / f'vdei_raw_{site}_s2_backup.npz')
        print(f'[{site}] previous raw file backed up as _s2_backup.npz')

    all_c = np.zeros((n_points, N_DIRS), np.int8)
    all_d = np.zeros((n_points, N_DIRS), np.float32)
    done = 0
    for f in sorted(CHK_DIR.glob(f'{site}_s{HORIZONTAL_STRIDE}_chunk_*.npz')):
        z = np.load(f); s = int(z['start']); c, dd = z['cls'], z['dist']
        all_c[s:s+len(c)] = c; all_d[s:s+len(dd)] = dd
        done = s + len(c)
    if done: print(f'[{site}] resumed from {done:,}')

    t0 = time.time()
    for cs in range(done, n_points, CHUNK):
        ce = min(cs + CHUNK, n_points)
        oi = is_[cs:ce].astype(np.int64); oj = js[cs:ce].astype(np.int64); ok = ks[cs:ce].astype(np.int64)
        oc = np.zeros((ce-cs, N_DIRS), np.int8); od = np.zeros((ce-cs, N_DIRS), np.float32)
        vdei_batch(oi, oj, ok, dir_x, dir_y, dir_z, cls3, z_ctr, z_up,
                   np.int64(nI), np.int64(nJ), STEP, MAX_RANGE, oc, od)
        all_c[cs:ce] = oc; all_d[cs:ce] = od
        np.savez_compressed(CHK_DIR / f'{site}_s{HORIZONTAL_STRIDE}_chunk_{cs:09d}.npz', start=cs, cls=oc, dist=od)
        el = time.time() - t0; dn = ce - done
        eta = (el/dn)*(n_points - ce)/60 if dn else 0
        print(f'[{site}] {ce:,}/{n_points:,} | {el/60:.1f}min | ETA {eta:.1f}min', flush=True)

    final = OUT_DIR / f'vdei_raw_{site}.npz'
    np.savez_compressed(final, cls=all_c, dist=all_d,
                        i0=is_.astype(np.int32), j0=js.astype(np.int32), k0=ks.astype(np.int32),
                        z_ctr=z_ctr, z_up=z_up, n_dirs=N_DIRS, stride=HORIZONTAL_STRIDE)
    assert np.load(final)['cls'].shape == all_c.shape
    print(f'[SAVED] {final} ({final.stat().st_size/1e6:.0f} MB) — verified')
    for f in CHK_DIR.glob(f'{site}_s{HORIZONTAL_STRIDE}_chunk_*.npz'): f.unlink()
print('\nS1a v2 DONE (full grid).')
