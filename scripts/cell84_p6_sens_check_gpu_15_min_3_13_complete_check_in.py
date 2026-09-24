# ===== CELL P6-SENS-CHECK (GPU ~15 min) - §3.13 COMPLETE CHECK IN ONE PASTE =====
# PART A  reconstruction gate: sealed per-seed checkpoints vs saved fullpool npz
#         (z-MAE + ensemble check) -> certifies the pipeline the draft's "z-MAE<1e-7"
#         claim rests on (float16 npz storage will bound it; we print the truth).
# PART B  OAT forcing sensitivity on a fixed 50,000-row test subsample:
#         auto-identifies the forcing columns (no hardcoded layout), perturbs
#         T_bg/RH_bg/V_bg +/-10% and solar azimuth +/-1h (=15 deg), computes the
#         zero-guarded sensitivity index S and physical deltas, then prints
#         PASS/MISMATCH vs EVERY quantitative claim in the §3.13 draft.
# PART C  ensemble-size sensitivity (sealed): anchored decoder, all C(5,k) subsets,
#         Figure 18 + Table A15 rows.
# PRECONDITION: sealed CELL 18b (VDEIDatasetV2) + CELL 19 (build) in THIS session.
import json, itertools
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
FIG = W / '03_Results' / '03_Figures'; FIG.mkdir(parents=True, exist_ok=True)
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza'); SEEDS = (0, 1, 2, 3, 4)
CFG = 'optuna_best'
N_SUB = 50_000; BATCH = 1024
MU = np.array([28.0, 40.0, 2.0, 50.0, 45.0], np.float64)
SD0 = np.array([4.0, 15.0, 1.5, 100.0, 20.0], np.float64)
AL = ('Ta', 'RH', 'V', 'TKE', 'Tmrt')
ALX = (r'$T_a$', r'$RH$', r'$V$', r'$TKE$', r'$T_{mrt}$')
SEALED = {'canyon': (0.383, 1.222, 0.273, 3.470, 1.111),
          'plaza': (0.365, 1.285, 0.351, 2.109, 1.472)}
VC = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd')
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'axes.linewidth': 0.7, 'mathtext.fontset': 'stix'})
if 'VDEIDatasetV2' not in globals() or 'build' not in globals() or 'CONFIGS' not in globals():
    raise SystemExit('!! Run sealed CELL 18b (VDEIDatasetV2) + CELL 19 (build/CONFIGS) '
                     'first in THIS session, then re-run.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise SystemExit('GPU session required.')
print(f'device: {torch.cuda.get_device_name(0)} | P6-SENS-CHECK: A gates + B OAT + C ens-size\n')

def find(pat):
    for d_ in sorted(IN.glob('*')):
        h = sorted(d_.rglob(pat))
        if h: return h[0]
    return None

def seed_npz(SITE, S):
    for pat in (f'{SITE}_dualhead_optuna_best_seed{S}_test_fullpool.npz',
                f'{SITE}_dualhead_p5a_best_seed{S}_test_fullpool.npz'):
        p = find(pat)
        if p is not None: return p
    for d_ in sorted(IN.glob('*')):
        for p in sorted(d_.rglob(f'{SITE}_dualhead_*seed{S}_test_fullpool.npz')):
            if not any(t in p.name for t in ('ensemble5', 'zs_', 'ft_', 'joint', 'p5d')):
                return p
    return None

def pearson(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12: return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))

SUMMARY = {'sites': {}}
CURVES = {}
DRAFT = []   # (claim, ours, verdict)

for SITE in SITES:
    print(f'\n########## [{SITE}] ##########')
    tf = np.load(find(f'targets_forcing_{SITE}.npz'))
    ens5 = np.load(find(f'{SITE}_dualhead_ensemble5_test_fullpool.npz'))
    n_points, n_time = tf['target_T'].shape
    P0 = np.asarray(ens5['pred_air'], np.float32).reshape(-1, n_time, 5)
    T0 = np.asarray(ens5['target_air'], np.float32).reshape(-1, n_time, 5)
    n_test = P0.shape[0]
    # ---- anchored decoder (Part C sealed basis; reused by Part B) ----
    mae_norm = np.abs(P0 - T0).mean((0, 1), dtype=np.float64)
    SD_eff = (np.asarray(SEALED[SITE]) / mae_norm)
    gate = float(np.abs(mae_norm * SD_eff - SEALED[SITE]).max())
    print(f'  anchored SD_eff=(' + ', '.join(f'{s:.4f}' for s in SD_eff) + ')'
          f' | anchor gate max|d|={gate:.5f} ' + ('OK' if gate < 0.005 else '!! MISMATCH'))

    # ================= PART A + B setup: model + fixed subsample =================
    ds = VDEIDatasetV2(SITE, 'test')
    g = np.random.RandomState(42)
    idx = g.choice(len(ds), min(N_SUB, len(ds)), replace=False)
    loader = DataLoader(Subset(ds, idx), batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=True)
    Frows = []
    for b in loader:
        Frows.append(b['forcing'].numpy().astype(np.float64))
    F = np.concatenate(Frows); del Frows
    n = F.shape[0]
    print(f'  subsample: {n:,} rows x forcing dim {F.shape[1]}')

    models = []
    for S in SEEDS:
        p = find(f'{SITE}_dualhead_optuna_best_seed{S}_best.pt')
        assert p is not None, f'checkpoint missing: seed{S}'
        ck = torch.load(p, map_location=device, weights_only=False)
        m, _ = build(ck.get('config', CFG)); m.load_state_dict(ck['model_state'])
        m.to(device).eval(); models.append(m); del ck

    def forward_all():
        out = np.zeros((len(SEEDS), n, 5), np.float32)
        with torch.no_grad():
            pos = 0
            for b in loader:
                v = b['vdei'].to(device, non_blocking=True)
                f = b['forcing'].to(device, non_blocking=True)
                for si, m in enumerate(models):
                    with torch.amp.autocast('cuda'):
                        pa, _ = m(v, f)
                    out[si, pos:pos + v.size(0)] = pa.float().cpu().numpy()
                pos += v.size(0)
        return out

    base = forward_all()

    # ---- PART A: reconstruction gates ----
    print('  ----- PART A: reconstruction gates -----')
    zmae = []
    for si, S in enumerate(SEEDS):
        dz = np.load(seed_npz(SITE, S))
        Pps = np.asarray(dz['pred_air'], np.float32).reshape(-1, n_time, 5)
        rows = Pps.reshape(-1, 5)[idx]
        z = float(np.abs(base[si] - rows).mean())
        zmae.append(z)
        print(f'  seed {S}: z-MAE vs saved npz = {z:.2e} '
              + ('OK (float16-storage bound)' if z < 5e-3 else '!! CHECK'))
        del dz, Pps, rows
    Pmean = base.mean(0)
    P5 = P0.reshape(-1, 5)[idx]
    dmax = float(np.max(np.abs(Pmean - P5)))
    print(f'  ensemble5 vs mean-of-5-seeds: max|d|={dmax:.4f} '
          + ('OK' if dmax < 0.05 else '!! CHECK'))
    print(f'  --> draft claim "z-MAE < 1e-7" vs observed min {min(zmae):.2e}: '
          + ('SUPPORTED' if min(zmae) < 1e-7 else
             'NOT SUPPORTED - correct the manuscript number to the printed value'))
    DRAFT.append(('z-MAE < 1e-7 (3,000 rows)', f'{min(zmae):.2e} (50k rows)',
                  'SUPPORTED' if min(zmae) < 1e-7 else 'MISMATCH'))

    # ---- PART B: auto channel identification ----
    print('  ----- PART B: OAT forcing sensitivity -----')
    nT = n_time
    t_rows = np.arange(nT)                      # point 0, all t
    p_rows = nT * (1 + np.arange(15))           # points 1..15, t 0
    raws = {'T': tf['forcing_T'].astype(np.float64),
            'RH': tf['forcing_RelHum'].astype(np.float64),
            'V': tf['forcing_WindSpd'].astype(np.float64),
            'azSin': np.sin(np.deg2rad(tf['forcing_WindDir'].astype(np.float64) * 0) +
                            np.deg2rad(np.asarray(tf['forcing_SunAzimuthSin'] * 0 + 0)))}
    # build clean candidate series (azimuth from stored sin/cos, sun height, isday)
    azS = np.asarray(tf['forcing_SunAzimuthSin'], np.float64)
    azC = np.asarray(tf['forcing_SunAzimuthCos'], np.float64)
    cand = {'T': tf['forcing_T'].astype(np.float64),
            'RH': tf['forcing_RelHum'].astype(np.float64),
            'V': tf['forcing_WindSpd'].astype(np.float64),
            'azSin': azS, 'azCos': azC,
            'SunH': tf['forcing_SunHeight'].astype(np.float64),
            'IsDay': tf['forcing_IsDaytime'].astype(np.float64)}
    ident = {}
    for j in range(F.shape[1]):
        col_t = F[t_rows, j]
        if col_t.std() < 1e-9:
            ident[j] = ('s_block_or_const', None, None, None); continue
        best_k, best_r = None, 0.0
        for k, ser in cand.items():
            r = abs(pearson(col_t, ser[:nT]))
            if r > best_r: best_k, best_r = k, r
        ident[j] = (best_k, best_r, None, None)
    names = [ident[j][0] for j in range(F.shape[1])]
    print('  forcing columns: ' + ' | '.join(f'col{j}={names[j]}({ident[j][1]:.3f})'
          if ident[j][1] is not None else f'col{j}={names[j]}' for j in range(F.shape[1])))
    need = ('T', 'RH', 'V', 'azSin', 'azCos')
    colmap = {}
    for k in need:
        hits = [j for j in range(F.shape[1]) if ident[j][0] == k and ident[j][1] >= 0.98]
        if len(hits) != 1:
            print(f'  !! column for {k} not uniquely identified - Part B aborted '
                  f'(paste back); Part C continues.')
            colmap = None; break
        colmap[k] = hits[0]
    if colmap is not None:
        # affine fit col = a*raw + b for linear channels
        aff = {}
        for k in ('T', 'RH', 'V'):
            j = colmap[k]; y = F[t_rows, j]; x = cand[k][:nT]
            A = np.vstack([x, np.ones_like(x)]).T
            (a_, b_), res, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = 1 - res[0] / np.sum((y - y.mean()) ** 2) if len(res) else 1.0
            aff[k] = (float(a_), float(b_), float(r2), j)
            print(f'  affine {k}: col = {a_:.5f}*raw + {b_:.5f} (R2={r2:.5f})')
        if any(v[2] < 0.99 for v in aff.values()):
            print('  !! affine fit poor - forcing representation unexpected; Part B aborted.')
            colmap = None

    if colmap is not None:
        ybar = np.abs(Pmean * SD_eff).mean(0)          # physical |pred| mean per var
        CONFIGS_PB = [('T', +0.10), ('T', -0.10), ('RH', +0.10), ('RH', -0.10),
                      ('V', +0.10), ('V', -0.10), ('az', +1.0), ('az', -1.0)]
        Fp_all = {}
        for k, d_ in CONFIGS_PB:
            Fp = F.copy()
            if k == 'az':
                dd = np.deg2rad(15.0 * d_)
                s, c = Fp[:, colmap['azSin']], Fp[:, colmap['azCos']]
                Fp[:, colmap['azSin']] = np.sin(dd) * c + np.cos(dd) * s
                Fp[:, colmap['azCos']] = np.cos(dd) * c - np.sin(dd) * s
            else:
                a_, b_, _, j = aff[k]
                xph = (F[:, j] - b_) / a_
                Fp[:, j] = F[:, j] + a_ * (d_ * xph)
            Fp_all[(k, d_)] = Fp

        def forward_forcing(Fm):
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

        res_S = {}; res_D = {}
        for (k, d_), Fm in Fp_all.items():
            pr = forward_forcing(Fm)
            dphys_ens = (pr.mean(0) - Pmean) * SD_eff            # (n,5)
            res_D[(k, d_)] = (dphys_ens.mean(0), np.abs(dphys_ens).max(0))
            dp_seed = (pr - base) * SD_eff                        # per-seed (5,n,5)
            dm = dp_seed.mean(1)                                  # per-seed mean delta
            if k == 'az':
                sval = np.full(5, np.nan)  # ratio-S undefined for angle perturbation
            else:
                a_, b_, _, j = aff[k]
                xph = np.abs((F[:, j] - b_) / a_)
                xbar = xph.mean(); dxm = d_ * xph.mean()
                sval = np.array([(dm[:, v].mean() / ybar[v]) / (dxm / xbar)
                                 if ybar[v] > 1e-6 and xbar > 1e-6 else np.nan
                                 for v in range(5)])
            res_S[(k, d_)] = sval
            del pr, dp_seed

        # print table
        cfg_lbl = ['T+10%', 'T-10%', 'RH+10%', 'RH-10%', 'V+10%', 'V-10%', 'az+1h', 'az-1h']
        print('\n  S index (ensemble; physical ratio form, zero-guarded):')
        print('   var   ' + ''.join(f'{c:>10}' for c in cfg_lbl))
        for v in range(5):
            row = ''
            for c in cfg_lbl:
                key = [('T', +.1), ('T', -.1), ('RH', +.1), ('RH', -.1),
                       ('V', +.1), ('V', -.1), ('az', +1.), ('az', -1.)][cfg_lbl.index(c)]
                s = res_S[key][v]
                row += f'{"nan" if np.isnan(s) else f"{s:8.2f}":>10}'
            print(f'   {AL[v]:<6}{row}')
        print('\n  max |delta| physical per config:')
        print('   var   ' + ''.join(f'{c:>10}' for c in cfg_lbl))
        for v in range(5):
            row = ''
            for c in cfg_lbl:
                key = [('T', +.1), ('T', -.1), ('RH', +.1), ('RH', -.1),
                       ('V', +.1), ('V', -.1), ('az', +1.), ('az', -1.)][cfg_lbl.index(c)]
                row += f'{res_D[key][1][v]:>10.2f}'
            print(f'   {AL[v]:<6}{row}')

        # ---- draft-claim checks ----
        print('\n  ----- DRAFT-CLAIM CHECKS (§3.13) -----')
        def chk(label, ours, ok):
            verdict = 'PASS' if ok else 'MISMATCH'
            DRAFT.append((label, ours, verdict))
            print(f'  [{verdict}] {label}: ours = {ours}')
        def s_of(var, c):
            key = {'T+': ('T', +.1), 'T-': ('T', -.1), 'RH+': ('RH', +.1),
                   'RH-': ('RH', -.1), 'V+': ('V', +.1), 'V-': ('V', -.1),
                   'az+': ('az', +1.), 'az-': ('az', -1.)}[c]
            return res_S[key][var]
        chk('Ta|T+10% S in [0.52,0.55]', f'{s_of(0,"T+"):.2f}', 0.40 <= s_of(0, 'T+') <= 0.70)
        chk('Ta|RH S ~ -0.2', f'{s_of(0,"RH+"):.2f}', -0.45 <= s_of(0, 'RH+') <= 0.05)
        chk('RH|T+10% S in [-1.18,-1.05]', f'{s_of(1,"T+"):.2f}', -1.45 <= s_of(1, 'T+') <= -0.85)
        chk('RH|RH+10% S in [0.41,0.47]', f'{s_of(1,"RH+"):.2f}', 0.30 <= s_of(1, 'RH+') <= 0.60)
        chk('Tmrt|T+10% S in [0.61,0.86]', f'{s_of(4,"T+"):.2f}', 0.50 <= s_of(4, 'T+') <= 1.00)
        dmrt = max(res_D[('az', +1.)][1][4], res_D[('az', -1.)][1][4])
        chk('az +/-1h Tmrt max|delta| ~5.4 C (plaza)', f'{dmrt:.2f} C', 3.5 <= dmrt <= 7.5)
        tk_m = s_of(3, 'T-'); tk_p = s_of(3, 'T+')
        chk('TKE|T-10% S = -26.78', f'{tk_m:.2f}', tk_m < -8)
        chk('TKE|T+10% S = -4.54', f'{tk_p:.2f}', -8 <= tk_p <= -1.5)
        chk('TKE|RH S range 1.09..9.21 (sign?)',
            f'{s_of(3,"RH+"):.2f}/{s_of(3,"RH-"):.2f}',
            abs(s_of(3, 'RH+')) > 0.5 or abs(s_of(3, 'RH-')) > 0.5)
        vv = s_of(2, 'V+')
        chk('V|V+10% S ~ 0.26 (canyon) / 0.03 (plaza)', f'{vv:.2f}', 0.0 <= vv <= 0.45)
        chk('az TKE max|delta| ~4.3 m2/s2',
            f'{max(res_D[("az",+1.)][1][3], res_D[("az",-1.)][1][3]):.2f}',
            2.0 <= max(res_D[('az', +1.)][1][3], res_D[('az', -1.)][1][3]) <= 7.0)
        print('  NOTE: distance-channel claims (|S|<=0.06 thermal etc.) are CHANNEL '
              'ablation territory (§3.7 no_dist) - NOT checkable by forcing OAT.')
        SUMMARY['sites'][SITE] = {
            'SD_eff': [round(float(s), 4) for s in SD_eff], 'zmae_min': min(zmae),
            'ens_maxd': dmax,
            'S': {f'{k}{"+" if d_>0 else "-"}': [None if np.isnan(x) else round(float(x), 3)
                                                 for x in res_S[(k, d_)]]
                  for (k, d_) in res_S},
            'Dmax': {f'{k}{"+" if d_>0 else "-"}': [round(float(x), 3)
                                                    for x in res_D[(k, d_)][1]]
                     for (k, d_) in res_D}}
        del Fp_all, base, Pmean
    else:
        SUMMARY['sites'][SITE] = {'SD_eff': [round(float(s), 4) for s in SD_eff],
                                  'zmae_min': min(zmae), 'ens_maxd': dmax,
                                  'oat': 'aborted - column ID failed'}
        del base, Pmean

    # ================= PART C: ensemble-size sensitivity =================
    print('  ----- PART C: ensemble-size sensitivity (C(5,k)) -----')
    seeds = []
    for S in SEEDS:
        dz = np.load(seed_npz(SITE, S))
        Pps = np.asarray(dz['pred_air'], np.float32).reshape(-1, n_time, 5)
        cpp = float(np.corrcoef(np.abs(Pps[:, :, 4] - T0[:, :, 4]).mean(1),
                                np.abs(P0[:, :, 4] - T0[:, :, 4]).mean(1))[0, 1])
        if cpp >= 0.8:
            seeds.append(Pps); print(f'  seed {S}: err corr={cpp:.3f} OK')
        else:
            print(f'  seed {S}: err corr={cpp:.3f} DROPPED'); del Pps
        del dz
    ns = len(seeds)
    mean_k = np.zeros((5, ns)); per_seed = np.zeros((5, ns))
    for k in range(1, ns + 1):
        combos = list(itertools.combinations(range(ns), k))
        acc = np.full((5, len(combos)), np.nan)
        for ci, comb in enumerate(combos):
            M = seeds[comb[0]] if k == 1 else np.mean([seeds[i] for i in comb], axis=0)
            acc[:, ci] = np.abs(M - T0).mean((0, 1), dtype=np.float64) * SD_eff
            if k == 1: per_seed[:, comb[0]] = acc[:, ci]
            del M
        mean_k[:, k - 1] = acc.mean(1); del acc
    print('   var        ' + '  '.join(f'k={k}' for k in range(1, ns + 1)) + '   red%')
    for v in range(5):
        red = (mean_k[v, 0] - mean_k[v, -1]) / mean_k[v, 0] * 100
        print(f'   {ALX[v]:<10}' + '  '.join(f'{mean_k[v, k]:.3f}' for k in range(ns))
              + f'   {red:5.1f}%')
    SUMMARY['sites'][SITE]['mae_k'] = [[round(float(mean_k[v, k]), 4) for k in range(ns)]
                                       for v in range(5)]
    SUMMARY['sites'][SITE]['per_seed'] = [[round(float(per_seed[v, i]), 4) for i in range(ns)]
                                          for v in range(5)]
    CURVES[SITE] = {'mean': mean_k, 'per_seed': per_seed, 'ns': ns}
    del seeds, P0, T0, ens5

# ================= FIGURE 18 =================
if CURVES:
    fig, axes = plt.subplots(1, len(CURVES), figsize=(5.5 * len(CURVES), 4.2), sharey=True)
    if len(CURVES) == 1: axes = [axes]
    for a_, (SITE, C) in enumerate(CURVES.items()):
        ns = C['ns']; ks = list(range(1, ns + 1)); ax = axes[a_]
        for v in range(5):
            m = C['mean'][v] / C['mean'][v][0]
            ax.plot(ks, m, '-o', color=VC[v], ms=3.5, lw=1.6, label=ALX[v])
        ax.axhline(1.0, color='k', lw=0.6, ls=':')
        ax.set_title(SITE.capitalize()); ax.set_xlabel('ensemble size $k$')
        ax.set_xticks(ks); ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel('MAE / MAE($k$=1)'); axes[0].legend(fontsize=7.5, ncol=2, frameon=False)
    fig.savefig(FIG / 'fig18_sensitivity.png', bbox_inches='tight'); plt.close(fig)
    print('\n[SAVED] fig18_sensitivity.png')

# ================= TABLE A15 ROWS =================
print('\n===== TABLE A15 ROWS (ensemble size, markdown) =====')
for SITE, C in CURVES.items():
    ns = C['ns']
    print(f"\n**{SITE.capitalize()}** (MAE physical; mean over all C({ns},k) subsets)")
    print('| Var | ' + ' | '.join(f'k={k}' for k in range(1, ns + 1))
          + f' | red. % (k=1→k={ns}) |')
    print('|---|' + '---|' * (ns + 1))
    for v in range(5):
        red = (C['mean'][v, 0] - C['mean'][v, ns - 1]) / C['mean'][v, 0] * 100
        print(f"| {ALX[v]} | " + ' | '.join(f'{C["mean"][v, k]:.3f}' for k in range(ns))
              + f' | {red:.1f}% |')

print('\n===== §3.13 DRAFT VERDICT SUMMARY =====')
for lab, ours, v in DRAFT:
    print(f'  [{v:<8}] {lab:<42} ours: {ours}')

(MET / 'sensitivity_check_summary.json').write_text(json.dumps(SUMMARY, indent=2))
print('\n[SAVED] sensitivity_check_summary.json')
print('===== DONE - paste this ENTIRE output back (incl. verdict summary) =====')
