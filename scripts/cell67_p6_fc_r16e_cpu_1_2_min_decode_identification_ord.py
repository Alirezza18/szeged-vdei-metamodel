# ===== CELL P6-FC R16e (CPU, ~1-2 min) - DECODE IDENTIFICATION + ORDER CHECK + §3.8 ARBITRATION =====
# Settles the two blockers before §3.1.4 ships:
#  (1) R16c contradiction: decoded test T_mrt max 94.9 C > TRUE tf ped-noon max 83.4 C.
#      Under constants (45,20) impossible -> identify the pool encoding (mu*, sd*) by
#      regression  true_tf = mu* + sd* * E_pool  over held-out rows (exact by definition),
#      AND check pool point-ordering (corr per-point decoded vs tf; MAE cannot test order).
#  (2) §3.8 'ped (k=4)' line: k=4 vs <2.5 m band vs per-seed-mean hypotheses.
# All from sealed artifacts + tf; no model, no GPU.
# Outputs: decode_identification.json. Paste ENTIRE output back.
import json, re
from pathlib import Path
import numpy as np

IN = Path('/kaggle/input')
W = Path('/kaggle/working')
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
K_PED = 4
CONST = np.array([[28.0, 4.0], [40.0, 15.0], [2.0, 1.5], [50.0, 100.0], [45.0, 20.0]])  # (mu,sd) x var
VARKEY = ('T', 'RelHum', 'WindSpd', 'TKE', 'TMRT')
SEALED_38 = {'canyon': (0.465, 2.143), 'plaza': (0.392, 2.284)}

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
    geom = load_npz(f'geometry_{SITE}.npz')
    ens = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    k0 = np.asarray(tf['k0']); spl = np.asarray(split['split']).astype(str)
    n_points, n_time = tf['target_T'].shape
    zc = np.asarray(geom['z_center'])
    TRUE = np.stack([np.asarray(tf[f'target_{k}'], np.float64) for k in VARKEY], axis=2)  # (nP,nT,5)
    E = np.asarray(ens['target_air'], np.float64)          # (nPool,5) normalized targets
    P = np.asarray(ens['pred_air'], np.float64)
    n_pool = E.shape[0]
    test_list = np.where(spl == 'test')[0]                 # ASSUMPTION: pool row r -> test_list[r//nT], t=r%nT
    print(f'  pool rows={n_pool:,} == n_test*n_time={len(test_list)*n_time:,}: '
          f'{n_pool == len(test_list)*n_time}')

    # ---- B) IDENTIFICATION: regress true on E over a uniform row sample ----
    stride = max(1, n_pool // 2_000_000)
    rs = np.arange(0, n_pool, stride)
    pidx = rs // n_time; tidx = rs % n_time
    gp = test_list[pidx]
    st = {'n_sample': int(rs.size), 'vars': {}}
    print('  var          mu*        sd*   |  const(mu,sd)   |  fitR2  order-corr  |  MAE_norm  MAE_fit  sealed  d_fit_vs_sealed')
    for v, k in enumerate(VARKEY):
        x = E[rs, v]; y = TRUE[gp, tidx, v]
        b, a = np.polyfit(x, y, 1)
        yh = a + b * x
        r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        oc = float(np.corrcoef(x, y)[0, 1])
        mae_norm = float(np.abs(P[:, v] - E[:, v]).mean())
        mae_fit = b * mae_norm
        SEALED = {'canyon': (0.383, 1.222, 0.273, 3.470, 1.111),
                  'plaza': (0.365, 1.285, 0.351, 2.109, 1.472)}[SITE][v]
        st['vars'][k] = {'mu': float(a), 'sd': float(b), 'fitR2': float(r2),
                         'order_corr': oc, 'mae_norm': mae_norm,
                         'mae_fit': float(mae_fit), 'mae_const': float(CONST[v, 1] * mae_norm),
                         'sealed': SEALED}
        print(f'  {k:<7} {a:9.3f} {b:9.3f}  |  ({CONST[v,0]:.0f},{CONST[v,1]:.0f})'
              f'       |  {r2:5.3f}    {oc:8.4f}  |  {mae_norm:7.4f} {mae_fit:7.3f} {SEALED:6.3f} '
              f'{100*(mae_fit-SEALED)/SEALED:+6.1f}%')

    # ---- A) ORDER CHECK at k=4 noon specifically ----
    ped = (k0 == K_PED)
    ix_te = np.where(ped & (spl == 'test'))[0]
    tn = int(np.argmax(np.asarray(tf['target_TMRT']).mean(axis=0)))
    rows4 = ix_te * n_time + tn
    checks = {}
    for v, k in ((0, 'T'), (4, 'TMRT')):
        dec_c = CONST[v, 0] + E[rows4, v] * CONST[v, 1]
        dec_f = st['vars'][VARKEY[v]]['mu'] + E[rows4, v] * st['vars'][VARKEY[v]]['sd']
        tru = TRUE[ix_te, tn, v]
        checks[k] = {'corr_const': float(np.corrcoef(dec_c, tru)[0, 1]),
                     'corr_fit': float(np.corrcoef(dec_f, tru)[0, 1]),
                     'max_true': float(tru.max()),
                     'max_dec_const': float(dec_c.max()), 'max_dec_fit': float(dec_f.max())}
        print(f"  [{k}] k=4 noon: order-corr const={checks[k]['corr_const']:.4f} "
              f"fit={checks[k]['corr_fit']:.4f} | max true={tru.max():.1f} "
              f"dec_const={dec_c.max():.1f} dec_fit={dec_f.max():.1f}")
    st['order_check_k4_noon'] = checks
    st['t_noon'] = tn

    # ---- C) §3.8 ped arbitration (decode with FITTED sd) ----
    SDf = np.array([st['vars'][k]['sd'] for k in VARKEY])
    SDc = CONST[:, 1]
    tt = np.arange(n_time)
    def mae(ix, SD):
        rws = (ix[:, None] * n_time + tt[None, :]).ravel()
        return SD[[0, 4]] * np.abs(P[rws][:, [0, 4]] - E[rws][:, [0, 4]]).mean(axis=0)
    k4 = test_list[k0[test_list] == K_PED]
    zt = zc[k0[test_list]]
    band = test_list[zt < 2.5]
    for nm, ix in (('k4', k4), ('band<2.5m', band)):
        for sdn, SD in (('fit', SDf), ('const', SDc)):
            e = mae(ix, SD)
            print(f'  §3.8 {nm:<10} decode={sdn:<5}: Ta={e[0]:.3f} Tmrt={e[1]:.3f}  n={len(ix):,}')
    print(f'  §3.8 sealed ped line: Ta={SEALED_38[SITE][0]:.3f} Tmrt={SEALED_38[SITE][1]:.3f}')
    seeds = set()
    for d_ in IN.glob('*'):
        for p in d_.rglob(f'{SITE}_dualhead_*seed*_test_fullpool.npz'):
            m = re.search(r'seed(\d+)_test_fullpool', p.name)
            if m: seeds.add(int(m.group(1)))
    maes = []
    for S in sorted(seeds):
        z = None
        for pat in (f'{SITE}_dualhead_optuna_best_seed{S}_test_fullpool.npz',
                    f'{SITE}_dualhead_p5a_best_seed{S}_test_fullpool.npz'):
            z = load_npz(pat)
            if z is not None: break
        if z is None: continue
        Pe = np.asarray(z['pred_air'], np.float64)
        maes.append(mae(k4, SDc))
        z.close()
    if maes:
        mm = np.mean(np.array(maes), axis=0)
        print(f'  §3.8 per-seed-mean (k4, const decode): Ta={mm[0]:.3f} Tmrt={mm[1]:.3f} '
              f'({len(maes)} seeds)')
    OUT[SITE] = st
    ens.close()

(MET / 'decode_identification.json').write_text(json.dumps(OUT, indent=2))
print('\nDECISION RULE:')
print(' * order-corr ~1.0 everywhere -> pool ordering OK; use fitted (mu*, sd*).')
print(' * fitted sd* == constants (±1%) -> sealed MAEs stand as physical; §3.1.1 safe.')
print('   fitted sd* != constants -> ALL physical MAEs rescale by sd*/const (§3.1.1 included)')
print('   - paste back and I will issue the global rescale table.')
print(' * fitted mu* != const mu -> absolute levels (means/ranges/thresholds) shift;')
print('   ratios, biases, MAEs, grad-r unchanged. Text will use fitted levels.')
print(' * §3.8 ped line: whichever row matches sealed decides the k=4-vs-band wording.')
print('===== DONE - paste this entire output back =====')
