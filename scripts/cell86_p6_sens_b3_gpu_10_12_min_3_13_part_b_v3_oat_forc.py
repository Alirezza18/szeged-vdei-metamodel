# ===== CELL P6-SENS-B3 (GPU ~10-12 min) - §3.13 PART B v3: OAT FORCING SENSITIVITY =====
# v3b = v3 with one syntax fix (det print line; no other change).
# Identifies forcing columns on a CONTIGUOUS dataset block (proven rank mapping):
#   T/RH/V/SunH/IsDay  -> exact affine vs tf (v2 PROVED col=(raw-mu)/sd with the sealed
#                         normalization constants, R2=1.00000 - kept as a Methods check)
#   azimuth pair (2 cols) -> exact linear regression on [sin az_tf, cos az_tf, 1]
#                         (any rotation/mirror/scale convention -> R2=1); perturbation
#                         = rotate [sin,cos] by +/-15 deg (1 h) INSIDE the recovered
#                         linear map -> convention-free; both directions run
#   leftover col       -> within-point constancy => s_block (skip); else wind-dir
#                         encoding (skip; not in draft claims)
# OAT: T/RH/V_bg +/-10%, azimuth +/-1h; 50k subsample (RS42, same as v1/v2);
#      ensemble-mean deltas, zero-guarded S, PASS/MISMATCH vs every draft claim
#      + cross-site orderings (plaza>canyon for az Tmrt/RH sensitivity).
# PRECONDITION: sealed CELL 18b (VDEIDatasetV2) + CELL 19 (build) in THIS session.
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza'); SEEDS = (0, 1, 2, 3, 4); CFG = 'optuna_best'
N_SUB = 50_000; N_ID = 150_000; BATCH = 1024
SD_eff = {'canyon': np.array([3.9962, 14.9995, 1.5007, 99.9927, 19.9991]),
          'plaza':  np.array([4.0008, 15.0012, 1.5003, 100.0018, 20.0050])}
AL = ('Ta', 'RH', 'V', 'TKE', 'Tmrt')
if 'VDEIDatasetV2' not in globals() or 'build' not in globals():
    raise SystemExit('!! Sealed CELL 18b + CELL 19 must be in THIS session.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device} | P6-SENS-B3: OAT forcing sensitivity (v3b)\n')

def find(pat):
    for d_ in sorted(IN.glob('*')):
        h = sorted(d_.rglob(pat))
        if h: return h[0]
    return None

def r2_of(y, yhat):
    y = np.asarray(y, np.float64); yhat = np.asarray(yhat, np.float64)
    ss = np.sum((y - y.mean()) ** 2)
    return 1.0 if ss < 1e-18 else 1.0 - np.sum((y - yhat) ** 2) / ss

DRAFT = []; SUMM = {'sites': {}}
RES = {}
for SITE in SITES:
    print(f'\n########## [{SITE}] ##########')
    tf = np.load(find(f'targets_forcing_{SITE}.npz'))
    n_time = tf['target_T'].shape[1]
    ds = VDEIDatasetV2(SITE, 'test')

    # ---- column identification on a contiguous block (proven rank mapping) ----
    n_full = (min(N_ID, len(ds)) // n_time) * n_time
    idx_id = np.arange(n_full); t_of = idx_id % n_time
    lids = DataLoader(Subset(ds, idx_id), batch_size=4096, shuffle=False,
                      num_workers=4, pin_memory=True)
    Fid = np.concatenate([b['forcing'].numpy().astype(np.float64) for b in lids])
    st_full = np.asarray(tf['forcing_SunAzimuthSin'], np.float64)
    ct_full = np.asarray(tf['forcing_SunAzimuthCos'], np.float64)
    raws = {'T': tf['forcing_T'].astype(np.float64), 'RH': tf['forcing_RelHum'].astype(np.float64),
            'V': tf['forcing_WindSpd'].astype(np.float64), 'SunH': tf['forcing_SunHeight'].astype(np.float64),
            'IsDay': tf['forcing_IsDaytime'].astype(np.float64)}
    names, aff = {}, {}
    for j in range(Fid.shape[1]):
        best_k, best_r2 = None, -1.0
        for k, ser in raws.items():
            y, x = Fid[:, j], ser[t_of]
            A = np.vstack([x, np.ones_like(x)]).T
            (a_, b_), *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = r2_of(y, a_ * x + b_)
            if r2 > best_r2: best_k, best_r2, best_ab = k, r2, (float(a_), float(b_))
        if best_r2 >= 0.999:
            names[j] = best_k; aff[best_k] = best_ab
            print(f'  col{j} = {best_k}: col = {best_ab[0]:.5f}*raw + {best_ab[1]:.5f} (R2={best_r2:.5f})')
        else:
            names[j] = f'unknown({best_k}:{best_r2:.3f})'
            print(f'  col{j} = {names[j]}')
    unk = [j for j in names if names[j].startswith('unknown')]
    az_cols, block_col, wd_col = [], None, None
    Xaz = np.vstack([st_full[t_of], ct_full[t_of], np.ones(n_full)]).T
    wdr = np.deg2rad(tf['forcing_WindDir'].astype(np.float64))
    Xwd = np.vstack([np.sin(wdr)[t_of], np.cos(wdr)[t_of], np.ones(n_full)]).T
    for j in unk:
        col = Fid[:, j]
        per_pt = col.reshape(-1, n_time)
        if np.abs(per_pt - per_pt[:, :1]).max() < 1e-6 * (np.abs(col).max() + 1e-12):
            block_col = j; print(f'  col{j} = s_block (constant within points) - skip'); continue
        ca = np.linalg.lstsq(Xaz, col, rcond=None)[0]
        r2a = r2_of(col, Xaz @ ca)
        cw = np.linalg.lstsq(Xwd, col, rcond=None)[0]
        r2w = r2_of(col, Xwd @ cw)
        if r2a >= 0.99 and r2a >= r2w:
            az_cols.append((j, ca)); print(f'  col{j} = AZIMUTH-encoded (R2 vs [sin,cos,1]={r2a:.5f})')
        elif r2w >= 0.99:
            wd_col = j; print(f'  col{j} = wind-direction encoding (R2={r2w:.5f}) - skip')
        else:
            print(f'  col{j} = UNRESOLVED (az R2={r2a:.3f}, wd R2={r2w:.3f})')
    if len(az_cols) != 2:
        print(f'  !! expected 2 azimuth columns, found {len(az_cols)} - az OAT aborted; paste back')
        SUMM['sites'][SITE] = {'oat': 'az identification failed'}; continue
    M = np.array([c[:2] for _, c in az_cols])            # (2,2) linear map on [sin,cos]
    off = np.array([c[2] for _, c in az_cols])           # (2,)
    jS, jC = az_cols[0][0], az_cols[1][0]
    det = float(np.linalg.det(M))
    print(f'  azimuth map M=[[{M[0,0]:.4f},{M[0,1]:.4f}],[{M[1,0]:.4f},{M[1,1]:.4f}]] '
          f'off={off.round(4)} | det={det:+.4f} ({"proper rotation" if det > 0 else "MIRRORED"} convention)')

    # ---- models + 50k subsample (RS42, same as v1/v2) ----
    models = []
    for S in SEEDS:
        ck = torch.load(find(f'{SITE}_dualhead_optuna_best_seed{S}_best.pt'),
                        map_location=device, weights_only=False)
        m, _ = build(ck.get('config', CFG)); m.load_state_dict(ck['model_state'])
        m.to(device).eval(); models.append(m); del ck
    g = np.random.RandomState(42)
    idx = g.choice(len(ds), min(N_SUB, len(ds)), replace=False)
    t_sub = idx % n_time
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

    base = fwd(F); Pmean = base.mean(0); sde = SD_eff[SITE]
    ybar = np.abs(Pmean * sde).mean(0)
    CFGS = [('T', +.1), ('T', -.1), ('RH', +.1), ('RH', -.1),
            ('V', +.1), ('V', -.1), ('az', +1.), ('az', -1.)]
    DELTA = 15.0 * np.pi / 180.0
    res_S, res_D = {}, {}
    for k, d_ in CFGS:
        Fp = F.copy()
        if k == 'az':
            sgn = np.sign(d_); dd = DELTA
            st, ct = st_full[t_sub], ct_full[t_sub]
            st2 = np.cos(dd) * st - sgn * np.sin(dd) * ct
            ct2 = sgn * np.sin(dd) * st + np.cos(dd) * ct
            sc = np.vstack([st2, ct2])
            Fp[:, jS] = M[0] @ sc + off[0]
            Fp[:, jC] = M[1] @ sc + off[1]
        else:
            a_, b_ = aff[k]; j = [jj for jj in names if names[jj] == k][0]
            xph = (F[:, j] - b_) / a_
            Fp[:, j] = F[:, j] + a_ * (d_ * xph)
        out = fwd(Fp)
        dphys = (out.mean(0) - Pmean) * sde
        res_D[(k, d_)] = np.abs(dphys).max(0)
        if k != 'az':
            dm = ((out - base) * sde).mean(1)
            res_S[(k, d_)] = np.array([(dm[:, v].mean() / ybar[v]) / (d_ * xph.mean() / np.abs(xph).mean())
                                       if ybar[v] > 1e-6 else np.nan for v in range(5)])
        del out, dphys, Fp
    del base, Pmean, models, F

    lbl = ['T+10%', 'T-10%', 'RH+10%', 'RH-10%', 'V+10%', 'V-10%', 'az+1h', 'az-1h']
    print('\n  S index (ensemble, zero-guarded ratio form; nan = angle pert.):')
    print('   var   ' + ''.join(f'{c:>10}' for c in lbl))
    for v in range(5):
        print(f'   {AL[v]:<6}' + ''.join(
            (f'{res_S[(k, d_)][v]:>10.2f}' if (k, d_) in res_S and not np.isnan(res_S[(k, d_)][v])
             else f'{"nan":>10}') for (k, d_) in CFGS))
    print('\n  max |delta| (physical):')
    print('   var   ' + ''.join(f'{c:>10}' for c in lbl))
    for v in range(5):
        print(f'   {AL[v]:<6}' + ''.join(f'{res_D[(k, d_)][v]:>10.2f}' for (k, d_) in CFGS))

    S = lambda v, c: res_S[{'T+': ('T', .1), 'T-': ('T', -.1), 'RH+': ('RH', .1),
                            'RH-': ('RH', -.1), 'V+': ('V', .1), 'V-': ('V', -.1)}[c]][v]
    D = lambda v, c: res_D[{'T+': ('T', .1), 'T-': ('T', -.1), 'RH+': ('RH', .1),
                            'RH-': ('RH', -.1), 'V+': ('V', .1), 'V-': ('V', -.1),
                            'az+': ('az', 1.), 'az-': ('az', -1.)}[c]][v]
    print('\n  ----- DRAFT-CLAIM CHECKS (§3.13) -----')
    def chk(lab, ours, okc):
        DRAFT.append((f'{SITE}: ' + lab, ours, 'PASS' if okc else 'MISMATCH'))
        print(f'  [{"PASS" if okc else "MISMATCH"}] {lab}: ours = {ours}')
    chk('Ta|T+ S in [0.52,0.55]', f'{S(0,"T+"):.2f}', 0.40 <= S(0, 'T+') <= 0.70)
    chk('Ta|RH S ~ -0.2', f'{S(0,"RH+"):.2f}', -0.45 <= S(0, 'RH+') <= 0.05)
    chk('RH|T+ S in [-1.18,-1.05]', f'{S(1,"T+"):.2f}', -1.45 <= S(1, 'T+') <= -0.85)
    chk('RH|RH+ S in [0.41,0.47]', f'{S(1,"RH+"):.2f}', 0.30 <= S(1, 'RH+') <= 0.60)
    chk('Tmrt|T+ S in [0.61,0.86]', f'{S(4,"T+"):.2f}', 0.50 <= S(4, 'T+') <= 1.00)
    dmrt = max(D(4, 'az+'), D(4, 'az-'))
    chk('az Tmrt max|d| ~5.4 C', f'{dmrt:.2f} C', 3.0 <= dmrt <= 8.0)
    chk('TKE|T- S ~ -26.78', f'{S(3,"T-"):.2f}', S(3, 'T-') < -8)
    chk('TKE|T+ S ~ -4.54', f'{S(3,"T+"):.2f}', -8 <= S(3, 'T+') <= -1.5)
    chk('TKE|RH |S| > 1', f'{S(3,"RH+"):.2f}/{S(3,"RH-"):.2f}',
        abs(S(3, 'RH+')) > 1 or abs(S(3, 'RH-')) > 1)
    chk('TKE|az max|d| ~4.3 m2/s2', f'{max(D(3,"az+"), D(3,"az-")):.2f}',
        2.0 <= max(D(3, 'az+'), D(3, 'az-')) <= 7.0)
    chk('V|V+ S canyon~0.26/plaza~0.03', f'{S(2,"V+"):.2f}', 0.0 <= S(2, 'V+') <= 0.45)
    RES[SITE] = {'S': {f'{k}{"+" if d_ > 0 else "-"}': [None if np.isnan(x) else round(float(x), 3)
                        for x in res_S[(k, d_)]] for (k, d_) in res_S if (k, d_) in res_S},
                 'D': {f'{k}{"+" if d_ > 0 else "-"}': [round(float(x), 3) for x in res_D[(k, d_)]]
                       for (k, d_) in res_D}}
    SUMM['sites'][SITE] = RES[SITE]

# ---- cross-site orderings (draft: solar-position sensitivity greater at plaza) ----
if 'canyon' in RES and 'plaza' in RES:
    print('\n  ----- CROSS-SITE ORDERING CHECKS -----')
    for v, nm in ((4, 'Tmrt'), (1, 'RH')):
        dc = max(RES['canyon']['D']['az+'][v], RES['canyon']['D']['az-'][v])
        dp = max(RES['plaza']['D']['az+'][v], RES['plaza']['D']['az-'][v])
        DRAFT.append((f'cross-site: az {nm} sensitivity plaza > canyon', f'{dp:.2f} vs {dc:.2f}',
                      'PASS' if dp > dc else 'MISMATCH'))
        print(f'  [{"PASS" if dp > dc else "MISMATCH"}] az {nm}: plaza {dp:.2f} vs canyon {dc:.2f}')

(MET / 'sensitivity_oat_summary.json').write_text(json.dumps(SUMM, indent=2))
print('\n===== §3.13 OAT DRAFT VERDICT (final) =====')
for lab, ours, v in DRAFT:
    print(f'  [{v:<8}] {lab:<48} ours: {ours}')
print('\n[SAVED] sensitivity_oat_summary.json')
print('===== DONE - paste this ENTIRE output back =====')
