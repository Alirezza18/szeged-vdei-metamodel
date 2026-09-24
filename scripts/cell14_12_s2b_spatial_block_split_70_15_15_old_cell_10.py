# ===== CELL 12 - S2b: spatial block split 70/15/15 (old CELL 10 converted, vectorized) =====
# Leakage guard: every (i,j) column belongs to ONE split for ALL its timesteps.
# BLOCK_SIZE=20 (40 m) and SEED=42 kept EXACTLY from the old pipeline; same RandomState(42).
import numpy as np
import matplotlib.pyplot as plt

BLOCK_SIZE = 20          # cells -> 40 m x 40 m blocks (manuscript)
RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SEED = 42                # locked (old pipeline value)

for site in SITES:
    d = np.load(output(f'01_Data/02_Processed/vdei_features_{site}.npz'))
    oi, oj = d['i0'].astype(np.int64), d['j0'].astype(np.int64)
    n = len(oi)

    bid = (oi // BLOCK_SIZE).astype(np.int64) * 100000 + (oj // BLOCK_SIZE).astype(np.int64)
    ub = np.unique(bid)
    rs = np.random.RandomState(SEED)          # same RNG family as the old cell
    rs.shuffle(ub)
    n_tr = int(len(ub) * RATIOS['train'])
    n_va = int(len(ub) * RATIOS['val'])
    tr_ids, va_ids = ub[:n_tr], ub[n_tr:n_tr + n_va]
    te_ids = ub[n_tr + n_va:]

    split = np.full(n, 'test', dtype='<U5')
    split[np.isin(bid, tr_ids)] = 'train'
    split[np.isin(bid, va_ids)] = 'val'

    print(f'\n[{site}] blocks={len(ub):,}  (block = {BLOCK_SIZE} cells = {BLOCK_SIZE*2} m)')
    nbs = {'train': n_tr, 'val': n_va, 'test': len(ub) - n_tr - n_va}
    for s in ('train', 'val', 'test'):
        npt = int((split == s).sum())
        print(f'  {s:5}: {nbs[s]:>5,} blocks | {npt:>9,} points ({100*npt/n:5.1f}%)')

    # split-block map (diagnostic: visually proves spatial no-leakage; paper-grade)
    g = np.load(output(f'01_Data/02_Processed/geometry_{site}.npz'))
    nJ, nI = g['is_building'].shape[1], g['is_building'].shape[2]
    bj, bi = bid // 100000, bid % 100000
    lbl = np.full((bj.max() + 1, bi.max() + 1), np.nan)
    lbl[bj, bi] = np.where(np.isin(bid, tr_ids), 0.0,
                  np.where(np.isin(bid, va_ids), 1.0, 2.0))
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(lbl, origin='lower', cmap='coolwarm', vmin=-0.5, vmax=2.5)
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cb.ax.set_yticklabels(['train', 'val', 'test'])
    ax.set_title(f'[{site}] spatial block split - {len(ub)} blocks of {BLOCK_SIZE*2} m')
    out = output(f'03_Results/diagnostics/split_blocks_{site}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'  [SAVED] split map -> {out.name}')

    np.savez_compressed(output(f'01_Data/02_Processed/split_{site}.npz'),
                        split=split, block_id=bid, block_size=BLOCK_SIZE,
                        seed=SEED, n_blocks=len(ub))

print('\nS2b DONE - next: CELL 13 (K-window vertical index).')
