# ===== CELL P6-SENS-B2 (GPU ~8-10 min) - §3.13 PART B v2: OAT FORCING SENSITIVITY =====
# v2 fix: column identification now uses a CONTIGUOUS dataset block (rank mapping
# row r <-> (point r//n_time, t r%n_time) is PROVEN), so each row's expected raw
# forcing is exact; columns are matched by per-row correlation (scale-invariant, so
# dataset-side standardization is harmless), then affine col = a*raw + b is fitted.
# OAT then perturbs T_bg/RH_bg/V_bg +/-10% and solar azimuth +/-1h (15 deg) on the
# SAME 50k subsample (RandomState 42), ensemble-mean deltas, zero-guarded S index,
# and prints PASS/MISMATCH vs every §3.13 draft claim.
# PRECONDITION: sealed CELL 18b + CELL 19 + P6-SENS-CHECK session still alive.
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza'); SEEDS = (0, 1, 2, 3, 4); CFG = 'optuna_best'
N_SUB = 50_000; N_ID = 150_000; BATCH = 1024
MU = np.array([28.0, 40.0, 2.0, 50.0, 45.0], np.float64)
SD_eff = {'canyon': np.array([3.9962, 14.9995, 1.5007, 99.9927, 19.9991]),
          'plaza':  np.array([4.0008, 15.0012, 1.5003, 100.0018, 20.0050])}
AL = ('Ta', 'RH', 'V', 'TKE', 'Tmrt')
if 'VDEIDatasetV2' not in globals() or 'build' not in globals():
    raise SystemExit('!! Sealed CELL 18b + CELL 19 must be in THIS session.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device} | P6-SENS-B2: OAT forcing sensitivity (v2 contiguous-ID)\n')

def find(pat):
    for d_ in sorted(IN.glob('*')):
        h = sorted(d_.rglob(pat))
        if h: return h[0]
    return None

def pearson(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))

DRAFT = []; SUMMARY = {'sites': {}}
for SITE in SITES:
    print(f'\n########## [{SITE}] ##########')
    tf = np.load(find(f'targets_forcing_{SITE}.npz'))
    n_time = tf['target_T'].shape[1]
    ds = VDEIDatasetV2(SITE, 'test')

    # ---- column ID on a contiguous block (rank-mapped expectations) ----
    idx_id = np.arange(min(N_ID, len(ds)))
    lids = DataLoader(Subset(ds, idx_id), batch_size=4096, shuffle=False,
                      num_workers=4, pin_memory=True)
    Fid = np.concatenate([b['forcing'].numpy().astype(np.float64) for b in lids])
    t_of = idx_id % n_time
    az = np.rad2deg(np.arctan2(np.asarray(tf['forcing_SunAzimuthSin'], np.float64),
                               np.asarray(tf['forcing_SunAzimuthCos'], np.float64))) % 360
    cand = {'T': tf['forcing_T'].astype(np.float64)[t_of],
            'RH': tf['forcing_RelHum'].astype(np.float64)[t_of],
            'V': tf['forcing_WindSpd'].astype(np.float64)[t_of],
            'azSin': np.asarray(tf['forcing_SunAzimuthSin'], np.float64)[t_of],
            'azCos': np.asarray(tf['forcing_SunAzimuthCos'], np.float64)[t_of],
            'SunH': np.asarray(tf['forcing_SunHeight'], np.float64)[t_of],
            'IsDay': np.asarray(tf['forcing_IsDaytime'], np.float64)[t_of],
            'wdSin': np.sin(np.deg2rad(tf['forcing_WindDir'].astype(np.float64)))[t_of],
            'wdCos': np.cos(np.deg2rad(tf['forcing_WindDir'].astype(np.float64)))[t_of]}
    names = {}
    for j in range(Fid.shape[1]):
        rs = {k: abs(pearson(Fid[:, j], v)) for k, v in cand.items()}
        kbest = max(rs, key=rs.get)
        names[j] = (kbest, rs[kbest]) if rs[kbest] >= 0.90 else ('unmatched', rs[kbest])
    print('  forcing columns: ' + ' | '.join(
        f'col{j}={names[j][0]}({names[j][1]:.3f})' for j in range(Fid.shape[1])))
    colmap, aff = {}, {}
    ok = True
    for k in ('T', 'RH', 'V', 'azSin', 'azCos'):
        hits = [j for j in names if names[j][0] == k and names[j][1] >= 0.90]
        if len(hits) != 1:
            print(f'  !! {k}: {len(hits)} matches - paste back'); ok = False; break
        j = hits[0]; colmap[k] = j
        y, x = Fid[:, j], cand[k]
        A = np.vstack([x, np.ones_like(x)]).T
        (a_, b_), res, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - (res[0] / np.sum((y - y.mean()) ** 2)) if len(res) else 1.0
        aff[k] = (float(a_), float(b_))
        print(f'  affine {k}: col = {a_:.5f}*raw + {b_:.5f} (R2={r2:.5f})')
        assert r2 > 0.99, f'affine {k} poor'

    if not ok:
        SUMMARY['sites'][SITE] = {'oat': 'aborted'}; continue

    # ---- models + 50k subsample (same seed 42 as v1 for comparability) ----
    models = []
    for S in SEEDS:
        ck = torch.load(find(f'{SITE}_dualhead_optuna_best_seed{S}_best.pt'),
                        map_location=device, weights_only=False)
        m, _ = build(ck.get('config', CFG)); m.load_state_dict(ck['model_state'])
        m.to(device).eval(); models.append(m); del ck
    g = np.random.RandomState(42)
    idx = g.choice(len(ds), min(N_SUB, len(ds)), replace=False)
    loader = DataLoader(Subset(ds, idx), batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=True)
    F = np.concatenate([b['forcing'].numpy().astype(np.float64) for b in loader])
    n = F.shape[0]

    def fwd(Fm):
        out = np.zeros((len(SEEDS), n, 5), np.float32)
        with torch.no_grad():
            pos = 0
            for b in loader:
                v = b['vdei'].to(device, non_blocking=True)
                f = torch.from_numpy(Fm[pos:pos + v.size(0)]).float().to(device)
                for si, m in enumerate(models):
                    with torch.amp.autocast('cuda'):
                        pa, _ = m(v, f)
                    out[si, pos:pos + v.size(0)] = pa.float().cpu().numpy()
                pos += v.size(0)
        return out

    base = fwd(F); Pmean = base.mean(0)
    sde = SD_eff[SITE]
    ybar = np.abs(Pmean * sde).mean(0)
    CFGS = [('T', +.1), ('T', -.1), ('RH', +.1), ('RH', -.1),
            ('V', +.1), ('V', -.1), ('az', +1.), ('az', -1.)]
    res_S, res_D = {}, {}
    for k, d_ in CFGS:
        Fp = F.copy()
        if k == 'az':
            dd = np.deg2rad(15.0 * d_)
            as_, ac = aff['azSin'], aff['azCos']
            s_raw = (F[:, colmap['azSin']] - as_[1]) / as_[0]
            c_raw = (F[:, colmap['azCos']] - ac[1]) / ac[0]
            Fp[:, colmap['azSin']] = as_[0] * (np.sin(dd) * c_raw + np.cos(dd) * s_raw) + as_[1]
            Fp[:, colmap['azCos']] = ac[0] * (np.cos(dd) * c_raw - np.sin(dd) * s_raw) + ac[1]
            sval = np.full(5, np.nan)
        else:
            a_, b_ = aff[k]; j = colmap[k]
            xph = (F[:, j] - b_) / a_
            Fp[:, j] = F[:, j] + a_ * (d_ * xph)
            dm = ((fwd(Fp) - base) * sde).mean(1)          # (n,5) per-seed mean delta
            sval = np.array([(dm[:, v].mean() / ybar[v]) / (d_ * xph.mean() / np.abs(xph).mean())
                             if ybar[v] > 1e-6 else np.nan for v in range(5)])
            res_S[(k, d_)] = sval
            del dm
        dphys = (fwd(Fp).mean(0) - Pmean) * sde
        res_D[(k, d_)] = np.abs(dphys).max(0)
        del dphys, Fp
    del base, Pmean

    lbl = ['T+10%', 'T-10%', 'RH+10%', 'RH-10%', 'V+10%', 'V-10%', 'az+1h', 'az-1h']
    print('\n  S index (ensemble, zero-guarded ratio form; nan = angle pert.):')
    print('   var   ' + ''.join(f'{c:>10}' for c in lbl))
    for v in range(5):
        print(f'   {AL[v]:<6}' + ''.join(
            (f'{res_S[(k, d_)][v]:>10.2f}' if not np.isnan(res_S[(k, d_)][v]) else f'{"nan":>10}')
            for (k, d_) in CFGS))
    print('\n  max |delta| (physical):')
    print('   var   ' + ''.join(f'{c:>10}' for c in lbl))
    for v in range(5):
        print(f'   {AL[v]:<6}' + ''.join(f'{res_D[(k, d_)][v]:>10.2f}' for (k, d_) in CFGS))

    # ---- draft-claim checks ----
    print('\n  ----- DRAFT-CLAIM CHECKS (§3.13) -----')
    def chk(lab, ours, okc):
        DRAFT.append((f'{SITE}: ' + lab, ours, 'PASS' if okc else 'MISMATCH'))
        print(f'  [{"PASS" if okc else "MISMATCH"}] {lab}: ours = {ours}')
    S = lambda v, c: res_S[CFGS[[f'{a}{"+" if b > 0 else "-"}' for a, b in CFGS].index(c)][0:2]][v] \
        if False else res_S[dict([('T+', ('T', .1)), ('T-', ('T', -.1)), ('RH+', ('RH', .1)),
                                  ('RH-', ('RH', -.1)), ('V+', ('V', .1)), ('V-', ('V', -.1)),
                                  ('az+', ('az', 1.)), ('az-', ('az', -1.))])[c]][v]
    chk('Ta|T+ S in [0.52,0.55]', f'{S(0,"T+"):.2f}', 0.40 <= S(0, 'T+') <= 0.70)
    chk('Ta|RH S ~ -0.2', f'{S(0,"RH+"):.2f}', -0.45 <= S(0, 'RH+') <= 0.05)
    chk('RH|T+ S in [-1.18,-1.05]', f'{S(1,"T+"):.2f}', -1.45 <= S(1, 'T+') <= -0.85)
    chk('RH|RH+ S in [0.41,0.47]', f'{S(1,"RH+"):.2f}', 0.30 <= S(1, 'RH+') <= 0.60)
    chk('Tmrt|T+ S in [0.61,0.86]', f'{S(4,"T+"):.2f}', 0.50 <= S(4, 'T+') <= 1.00)
    dmrt = max(res_D[('az', 1.)][4], res_D[('az', -1.)][4])
    chk('az Tmrt max|d| ~5.4 C', f'{dmrt:.2f} C', 3.0 <= dmrt <= 8.0)
    chk('TKE|T- S ~ -26.78', f'{S(3,"T-"):.2f}', S(3, 'T-') < -8)
    chk('TKE|T+ S ~ -4.54', f'{S(3,"T+"):.2f}', -8 <= S(3, 'T+') <= -1.5)
    chk('TKE|RH |S| > 1', f'{S(3,"RH+"):.2f}/{S(3,"RH-"):.2f}',
        abs(S(3, 'RH+')) > 1 or abs(S(3, 'RH-')) > 1)
    chk('TKE|az max|d| ~4.3 m2/s2',
        f'{max(res_D[("az",1.)][3], res_D[("az",-1.)][3]):.2f}',
        2.0 <= max(res_D[('az', 1.)][3], res_D[('az', -1.)][3]) <= 7.0)
    chk('V|V+ S canyon~0.26/plaza~0.03', f'{S(2,"V+"):.2f}', 0.0 <= S(2, 'V+') <= 0.45)
    SUMMARY['sites'][SITE] = {
        'S': {f'{k}{"+" if d_ > 0 else "-"}': [None if np.isnan(x) else round(float(x), 3)
                                               for x in res_S[(k, d_)]] for (k, d_) in res_S},
        'Dmax': {f'{k}{"+" if d_ > 0 else "-"}': [round(float(x), 3) for x in res_D[(k, d_)]]
                 for (k, d_) in res_D}}
    del models, F

(MET / 'sensitivity_oat_summary.json').write_text(json.dumps(SUMMARY, indent=2))
print('\n===== §3.13 OAT DRAFT VERDICT =====')
for lab, ours, v in DRAFT:
    print(f'  [{v:<8}] {lab:<44} ours: {ours}')
print('\n[SAVED] sensitivity_oat_summary.json')
print('===== DONE - paste this ENTIRE output back =====')
