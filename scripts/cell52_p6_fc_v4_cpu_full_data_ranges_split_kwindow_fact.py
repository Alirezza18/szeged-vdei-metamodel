# ===== CELL P6-FC v4 (CPU) - full-data ranges + split/kwindow facts + SVF definition =====
from pathlib import Path
import numpy as np

IN = Path('/kaggle/input')
SITES = ('canyon', 'plaza')

def find(pat):
    for d in sorted(IN.glob('*')):
        h = sorted(d.rglob(pat))
        if h:
            return h[0]
    return None

print('=========== P6-FC v4 ===========')

# ---------- G) FULL-DATA air + forcing ranges ----------
print('\n--- G) FULL-DATA ranges (targets_forcing files) ---')
for site in SITES:
    p = find(f'targets_forcing_{site}.npz')
    if p is None:
        print(f'  [{site}] missing'); continue
    nf = np.load(p)
    print(f'\n  [{site}] {p.name}')
    for k in nf.files:
        a = np.asarray(nf[k])
        if a.dtype.kind not in 'fiu' or a.size < 100:
            continue
        tag = 'PHYSICAL' if (abs(float(np.nanmean(a))) > 3 or float(np.nanstd(a)) > 3) else 'normalized?'
        print(f'    {k:<22} shape={str(a.shape):<16} min={np.nanmin(a):>9.2f} max={np.nanmax(a):>9.2f} '
              f'mean={np.nanmean(a):>9.2f} std={np.nanstd(a):>9.2f}  [{tag}]')
        del a
    nf.close()

# ---------- H) SPLIT facts ----------
print('\n--- H) SPLIT facts (block partitioning) ---')
for site in SITES:
    p = find(f'split_{site}.npz')
    if p is None:
        print(f'  [{site}] missing'); continue
    nf = np.load(p)
    print(f'\n  [{site}] {p.name}')
    print(f'    block_size = {nf["block_size"]}   seed = {nf["seed"]}   n_blocks = {nf["n_blocks"]}')
    split, bid = nf['split'], nf['block_id']
    vals, cnts = np.unique(split, return_counts=True)
    print(f'    split counts: ' + ', '.join(f'{v}:{c:,}' for v, c in zip(vals, cnts))
          + f'   (n_points={len(split):,})')
    print(f'    block_id: min={bid.min()} max={bid.max()} unique={len(np.unique(bid))}')
    nf.close()

# ---------- I) KWINDOW facts ----------
print('\n--- I) KWINDOW facts (vertical window + boundary handling) ---')
for site in SITES:
    p = find(f'kwindow_{site}.npz')
    if p is None:
        print(f'  [{site}] missing'); continue
    nf = np.load(p)
    print(f'\n  [{site}] {p.name}')
    print(f'    k_window = {nf["k_window"]}   half_k = {nf["half_k"]}')
    nbr, gap = nf['nbr_idx'], nf['nbr_gap']
    print(f'    nbr_idx shape={nbr.shape} min={nbr.min()} max={nbr.max()}')
    srt = np.sort(nbr, axis=1)
    dup = (np.diff(srt, axis=1) == 0).any(axis=1)
    print(f'    rows with repeated (clamped/replicated) level: {dup.sum():,} / {len(nbr):,} '
          f'({100*dup.mean():.2f}%)')
    gv, gc = np.unique(gap, return_counts=True)
    if len(gv) <= 12:
        print(f'    nbr_gap values: ' + ', '.join(f'{v}:{c:,}' for v, c in zip(gv, gc)))
    else:
        print(f'    nbr_gap: min={gap.min()} max={gap.max()} mean={gap.mean():.2f} unique={len(gv)}')
    nf.close()

# ---------- D) SVF definition check (reshape fixed) ----------
print('\n--- D) SVF definition check ---')
for site in SITES:
    vf, sv = find(f'vdei_features_{site}.npz'), find(f'svf_{site}.npz')
    if vf is None or sv is None:
        print(f'  [{site}] missing inputs'); continue
    d = np.load(vf)
    cls = d['cls']; i0f, j0f, k0f = d['i0'], d['j0'], d['k0']
    P = cls.shape[0]
    d.close()
    sky = (cls == 0).reshape(P, 16, 9)          # idx = az*9 + el
    del cls
    cands = {'phi>=0 (80 dirs)': sky[:, :, 4:].mean(axis=(1, 2)),
             'phi>0 (64 dirs)':  sky[:, :, 5:].mean(axis=(1, 2)),
             'all 144 dirs':     sky.mean(axis=(1, 2))}
    del sky
    s = np.load(sv)
    ev, ra = np.asarray(s['svf_envimet']), np.asarray(s['svf_rays'])
    i0s, j0s = s['i0'], s['j0']
    pr = float(s['pearson_r'])
    s.close()
    print(f'\n  [{site}] vdei P={P:,} | svf envimet={ev.shape} rays={ra.shape} '
          f'i0={i0s.shape} j0={j0s.shape} | saved pearson_r={pr:.4f}')
    if ra.ndim == 1 and ra.shape[0] == P:
        t_ra, t_ev, how = ra, ev, 'direct (1D, len P)'
    elif ra.ndim == 3:
        t_ra, t_ev, how = ra[k0f, j0f, i0f], ev[k0f, j0f, i0f], '3D grid via (k0,j0,i0)'
    elif ra.ndim == 2 and len(i0s) == P:
        t_ra, t_ev, how = ra[j0s, i0s], ev[j0s, i0s], '2D grid via svf (i0,j0)'
    else:
        print(f'    alignment unclear (ra {ra.shape}, P={P:,}, svf i0 len={len(i0s)}) -> paste shapes'); continue
    m = np.isfinite(t_ra) & np.isfinite(t_ev)
    print(f'    alignment: {how}   matched n={m.sum():,}')
    for lbl, c in cands.items():
        cm = c[m]
        r = float(np.corrcoef(cm, t_ra[m])[0, 1]) if m.sum() > 2 else float('nan')
        print(f'    rays vs {lbl:<20} r={r:.4f}  max|d|={np.abs(cm - t_ra[m]).max():.4f}  mean={cm.mean():.3f}')
    print(f'    svf_rays mean={t_ra[m].mean():.3f}  envimet mean={t_ev[m].mean():.3f}')
    for lbl in ('phi>=0 (80 dirs)', 'all 144 dirs'):
        c = cands[lbl][m]
        print(f'    envimet vs {lbl:<18} r={float(np.corrcoef(c, t_ev[m])[0, 1]):.4f}  '
              f'MAD={np.abs(c - t_ev[m]).mean():.4f}')

print('\n=========== DONE v4 - paste everything back ============')
