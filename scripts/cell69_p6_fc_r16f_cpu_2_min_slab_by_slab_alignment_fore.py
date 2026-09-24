# ===== CELL P6-FC R16f (CPU, ~2 min) - SLAB-BY-SLAB ALIGNMENT FORENSICS =====
# R16e verdict baked in: decode == manuscript constants (exact); pool row mapping exact
# at t in {0,5,10,15,20}; noon slab (t=8) misaligned vs tf (order-corr ~0, dec max 94.9
# vs tf ped max 81.0). This cell maps EVERY slab:
#   M[t,t'] = corr(pool-decoded target @ (ped test rows, t), tf target @ (same points, t'))
#   -> M[t,t] ~ 1 = slab aligned; argmax row t' != t = slab holds another hour;
#      flat row = slab holds wrong POINTS (then its quantiles vs tf ped/all-level tell us).
# Also: per-slab internal consistency (pool pred vs pool target corr + MAE), per-slab
# decoded-target quantiles vs tf ped and tf all-level distributions, and the same
# alignment check on the seed0 per-seed npz (is the defect pool-file-specific?).
# Outputs: slab_alignment.json. Paste ENTIRE output back.
import json
from pathlib import Path
import numpy as np

IN = Path('/kaggle/input')
MET = Path('/kaggle/working') / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
K_PED = 4
MU = np.array([28.0, 40.0, 2.0, 50.0, 45.0])
SD = np.array([4.0, 15.0, 1.5, 100.0, 20.0])

def find(pat):
    for d_ in sorted(IN.glob('*')):
        h = sorted(d_.rglob(pat))
        if h: return h[0]
    return None

def load_npz(pat):
    p = find(pat)
    if p is None: print(f'  !! missing: {pat}'); return None
    print(f'  [{pat}] <- {p.parent.parent.name}/{p.parent.name}/{p.name}')
    return np.load(p)

OUT = {}
for SITE in ('canyon', 'plaza'):
    print(f'\n########## [{SITE}] ##########')
    tf = load_npz(f'targets_forcing_{SITE}.npz'); split = load_npz(f'split_{SITE}.npz')
    ens = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    k0 = np.asarray(tf['k0']); spl = np.asarray(split['split']).astype(str)
    n_points, n_time = tf['target_T'].shape
    ix_te = np.where((k0 == K_PED) & (spl == 'test'))[0]
    tfT = np.asarray(tf['target_T'], np.float64)[ix_te]          # (n, nT) ped test truth
    tfM = np.asarray(tf['target_TMRT'], np.float64)[ix_te]
    E = np.asarray(ens['target_air'], np.float64)
    P = np.asarray(ens['pred_air'], np.float64)
    st = {}
    for v, (nm, TF) in ((0, ('T', tfT)), (4, ('TMRT', tfM))):
        dec = MU[v] + np.stack([E[ix_te * n_time + t, v] for t in range(n_time)], axis=1) * SD[v]
        A, B = dec - dec.mean(0, keepdims=True), TF - TF.mean(0, keepdims=True)
        C = (A.T @ B) / np.outer(np.sqrt((A**2).sum(0)), np.sqrt((B**2).sum(0)))
        diag = np.diag(C)
        argmax = C.argmax(axis=1)
        bad = [int(t) for t in range(n_time) if diag[t] < 0.8]
        # internal consistency + quantile fingerprints per slab
        print(f'  [{nm}] slab diag corr M[t,t]:')
        print('    ' + ' '.join(f'{d:5.2f}' for d in diag))
        print(f'    aligned (>=0.8): {n_time - len(bad)}/{n_time} | misaligned slabs: {bad}')
        if bad:
            q = np.percentile(dec[:, bad], [0, 50, 95, 100], axis=0)
            qp = np.percentile(TF, [0, 50, 95, 100])
            tfall = np.asarray(tf[f'target_{ "T" if v==0 else "TMRT"}'], np.float64)
            kte = k0[np.where(spl == "test")[0]]
            all_rows = np.where(spl == 'test')[0]
            qa = np.percentile(tfall[all_rows][:, bad[0]] if tfall.ndim == 2 else tfall[all_rows],
                               [0, 50, 95, 100])
            print(f'    misaligned slab quantiles [0/50/95/100%] vs tf-ped vs tf-allLevel:')
            for j, t in enumerate(bad):
                print(f'      t={t}: slab {np.round(q[:, j], 1)} | tf-ped {np.round(qp, 1)} | '
                      f'tf-all {np.round(qa, 1)}')
        # internal consistency: pred vs target per slab
        icorr = []
        for t in range(n_time):
            rws = ix_te * n_time + t
            icorr.append(float(np.corrcoef(P[rws, v], E[rws, v])[0, 1]))
        print(f'    internal pred-target corr per slab: ' + ' '.join(f'{c:4.2f}' for c in icorr))
        st[nm] = {'diag': [float(x) for x in diag], 'argmax': [int(x) for x in argmax],
                  'bad_slabs': bad, 'internal_corr': icorr}
    # per-seed npz: same defect?
    z = load_npz(f'{SITE}_dualhead_optuna_best_seed0_test_fullpool.npz')
    if z is not None:
        Ps = np.asarray(z['pred_air'], np.float64)
        Es = np.asarray(z['target_air'], np.float64)
        tn = int(np.argmax(np.asarray(tf['target_TMRT'])[:, :].mean(axis=0)))
        rws = ix_te * n_time + tn
        c_seed = float(np.corrcoef(MU[4] + Ps[rws, 4] * SD[4], tfM[:, tn])[0, 1])
        c_ens = float(np.corrcoef(MU[4] + E[rws, 4] * SD[4], tfM[:, tn])[0, 1])
        print(f'  seed0 vs ens at noon slab: target corr vs tf = {c_seed:.4f}/{c_ens:.4f} '
              f'-> defect {"also in per-seed files" if c_seed < 0.8 else "only in ensemble file"}')
        z.close()
    OUT[SITE] = st
    ens.close()

(MET / 'slab_alignment.json').write_text(json.dumps(OUT, indent=2))
print('\nDECISION RULES:')
print(' * All diag >= 0.8 -> pool fully aligned; R16e noon anomaly was a canvas/ordering')
print('   artifact of that cell; Figures 7/8 render straight from the pool; final text next.')
print(' * Misaligned slab set S, argmax(t) = t for all -> relabel-free; S slabs hold wrong')
print('   POINTS -> figures/per-t stats for S must come from tf-aligned fresh inference or')
print('   per-seed files (if clean).')
print(' * argmax(t) = pi(t) != t -> the pool t-axis is permuted: relabel pi and ALL per-t')
print('   sealed stats (3.9 regimes, 3.12 diurnal, A1-A3) shift by pi - I issue the')
print('   corrected table.')
print('===== DONE - paste this entire output back =====')
