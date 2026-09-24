# ===== CELL P6-FC R10 + FIGURE 14 (CPU, ~5-10 min, no GPU) - §3.7 ABLATION fact-check =====
# Sealed design: 5 arms x single seed (seed 0) x BOTH sites:
#   control | no_dist (V-DEI distance channel removed) | no_facade (facade loss disabled)
#   no_forcing (8-dim dynamic forcing vector removed) | no_sunblock (solar channels removed)
# NOTE: the old draft's K_window 3/7/11 protocol has NO sealed artifacts -> replaced by these arms.
# (A) per-arm MAE/RMSE/R2 (physical) for air (5) + facade (4, valid rows only)
# (B) delta-MAE vs control per variable
# (C) BLOCK-LEVEL PAIRED t-TESTS: per-block MAE over the split's spatial test blocks
#     (20x20 blocks) -> paired t across blocks + 2000-resample bootstrap 95% CI on dMAE
#     (replaces the impossible 3-seed t-test; n = number of test blocks, printed)
# (D) no_forcing vs climatology MAD floor (links to §3.2); facade-head sanity for no_facade
# (E) Figure 14: % dMAE vs control, arms x variables, air + facade, both sites
# Inputs: szeged-backup (p5d *_seed0_test_fullpool.npz), szeged-vdei-processed (split,
#         targets_forcing, facade_targets). Outputs: fig14_ablation.png + ablation_summary.json
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sps

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
FIG = W / '03_Results' / '03_Figures'; FIG.mkdir(parents=True, exist_ok=True)
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza')
ARMS = ('control', 'no_dist', 'no_facade', 'no_forcing', 'no_sunblock')
ARM_LAB = {'control': 'Control (full V-DEI)', 'no_dist': 'No distance channel',
           'no_facade': 'No facade task', 'no_forcing': 'No dynamic forcing',
           'no_sunblock': 'No solar channels'}
AIR = [('T', (28.0, 4.0), r'$T_a$'), ('RelHum', (40.0, 15.0), r'$RH$'),
       ('WindSpd', (2.0, 1.5), r'$V$'), ('TKE', (50.0, 100.0), r'$TKE$'),
       ('TMRT', (45.0, 20.0), r'$T_{mrt}$')]
AK = [a[0] for a in AIR]; AL = [a[2] for a in AIR]
FAC = [('Twall', r'$T_{wall}$'), ('Qsens', r'$Q_{sens}$'),
       ('SWabs', r'$SW_{abs}$'), ('LWbal', r'$LW_{bal}$')]
FK = [f[0] for f in FAC]; FL = [f[1] for f in FAC]
NBOOT = 2000
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'mathtext.fontset': 'stix'})
ARM_C = {'control': '#4C72B0', 'no_dist': '#DD8452', 'no_facade': '#55A868',
         'no_forcing': '#C44E52', 'no_sunblock': '#8172B3'}

def find(pat):
    for d in sorted(IN.glob('*')):
        h = sorted(d.rglob(pat))
        if h: return h[0]
    return None

def load_npz(pat):
    p = find(pat)
    if p is None: print(f'  !! missing: {pat}'); return None
    print(f'  [{pat}] <- {p.parent.parent.name}/{p.parent.name}/{p.name}')
    return np.load(p)

def r2(p, t):
    d = p - t
    return 1.0 - float((d * d).sum()) / float(((t - t.mean()) ** 2).sum())

summary = {}
for SITE in SITES:
    print(f'\n########## [{SITE}] ##########')
    tf = load_npz(f'targets_forcing_{SITE}.npz')
    split = load_npz(f'split_{SITE}.npz')
    fct = load_npz(f'facade_targets_{SITE}.npz')
    if tf is None or split is None: print(f'  missing core inputs - skip'); continue
    n_points, n_time = tf['target_T'].shape
    # MAD floors (raw physical, no transform - sealed R6b convention)
    MAD = {k: float(np.abs(np.asarray(tf[f'target_{k}'], dtype=np.float64)
                           - np.asarray(tf[f'target_{k}'], dtype=np.float64).mean()).mean())
           for k, _, _ in AIR}
    print('  MAD floors: ' + '  '.join(f'{AL[i]}={MAD[AK[i]]:.3f}' for i in range(5)))
    fac_mu = np.asarray(fct['facade_norm_mean'], dtype=np.float64) if fct is not None else None
    fac_sd = np.asarray(fct['facade_norm_std'], dtype=np.float64) if fct is not None else None
    block_pt = np.asarray(split['block_id'])

    M = {}
    for arm in ARMS:
        z = load_npz(f'{SITE}_dualhead_p5d_{arm}_seed0_test_fullpool.npz')
        if z is None:
            print(f'  -> arm [{arm}] npz missing - SKIPPED (checkpoint exists but no fullpool pred)')
            continue
        pa = np.asarray(z['pred_air'], dtype=np.float64); ta = np.asarray(z['target_air'], dtype=np.float64)
        n_rows = pa.shape[0]
        row_block = block_pt[np.arange(n_rows) // n_time]
        ub = np.unique(row_block); code = np.searchsorted(ub, row_block)
        cnt_air = np.bincount(code, minlength=ub.size).astype(np.float64)
        mae = np.zeros(5); r2s = np.zeros(5)
        block_sums = np.zeros((ub.size, 5))
        for vi in range(5):
            mu, sd = AIR[vi][1]
            p = pa[:, vi] * sd + mu; t = ta[:, vi] * sd + mu
            e = np.abs(p - t)
            mae[vi] = e.mean(); r2s[vi] = r2(p, t)
            block_sums[:, vi] = np.bincount(code, weights=e, minlength=ub.size)
        # facade
        pf = np.asarray(z['pred_fac'], dtype=np.float64); tfa = np.asarray(z['target_fac'], dtype=np.float64)
        if 'valid' in z.files:
            valid = np.asarray(z['valid']).astype(bool)
        else:
            valid = np.isfinite(tfa).all(1)
        fmae = np.full(4, np.nan); fr2 = np.full(4, np.nan)
        fblock = np.full((ub.size, 4), np.nan)
        n_valid = int(valid.sum())
        if n_valid > 0 and fac_sd is not None:
            cv = code[valid]; cnt_f = np.bincount(cv, minlength=ub.size).astype(np.float64)
            for vi in range(4):
                p = pf[valid, vi] * fac_sd[vi] + fac_mu[vi]
                t = tfa[valid, vi] * fac_sd[vi] + fac_mu[vi]
                e = np.abs(p - t)
                fmae[vi] = e.mean(); fr2[vi] = r2(p, t)
                fblock[:, vi] = np.bincount(cv, weights=e, minlength=ub.size) / np.maximum(cnt_f, 1)
        M[arm] = {'mae': mae, 'r2': r2s, 'fmae': fmae, 'fr2': fr2,
                  'bMAE': block_sums / np.maximum(cnt_air, 1)[:, None],
                  'fbMAE': fblock, 'n_valid': n_valid, 'n_blocks': ub.size}
        print(f'  [{arm:>11}] air MAE: ' + '  '.join(f'{AL[i]}={mae[i]:.3f}' for i in range(5))
              + f' | facade n_valid={n_valid:,}')
        del z, pa, ta, pf, tfa
    if 'control' not in M: continue

    print(f'\n  --- test blocks for paired tests: n={M["control"]["n_blocks"]} '
          f'(spatial 20x20 blocks of the sealed split) ---')
    site_res = {'n_test_blocks': int(M['control']['n_blocks']), 'MAD': MAD, 'arms': {}}
    print('\n  ===== PER-ARM dMAE vs control + BLOCK-PAIRED t-test + bootstrap CI =====')
    hdr = 'arm          ' + ''.join(f'{AL[i]:>26}' for i in range(5))
    print(hdr)
    for arm in ARMS:
        if arm not in M: continue
        row = f'{arm:>11}  '
        if arm == 'control':
            print(row + ''.join(f'{"reference":>26}' for _ in range(5))); continue
        site_res['arms'][arm] = {}
        for vi in range(5):
            dm = M[arm]['mae'][vi] / M['control']['mae'][vi] - 1
            d_blk = M[arm]['bMAE'][:, vi] - M['control']['bMAE'][:, vi]
            nb = d_blk.size
            t, p = (np.nan, np.nan)
            if nb >= 5 and np.std(d_blk, ddof=1) > 0:
                tt = sps.ttest_rel(M[arm]['bMAE'][:, vi], M['control']['bMAE'][:, vi])
                t, p = float(tt.statistic), float(tt.pvalue)
            rng = np.random.default_rng(0)
            bs = np.empty(NBOOT)
            for b in range(NBOOT):
                ix = rng.integers(0, nb, nb)
                bs[b] = d_blk[ix].mean()
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sig = 'SIG' if (p == p and p < 0.05) else ('~' if (lo > 0 or hi < 0) else 'ns')
            print(f'   {AL[vi]:>10}: dMAE={100*dm:+6.1f}%  t={t:+5.2f} p={p:.4f} {sig:<3}'
                  f'  boot95CI=[{100*lo:+6.1f}%,{100*hi:+6.1f}%]  (n={nb} blocks)')
            site_res['arms'][arm][AK[vi]] = {
                'mae': float(M[arm]['mae'][vi]), 'r2': float(M[arm]['r2'][vi]),
                'dmae_pct': float(100 * dm), 't': t, 'p': p,
                'boot_lo_pct': float(100 * lo), 'boot_hi_pct': float(100 * hi)}
    print('\n  --- facade dMAE vs control (valid rows only) ---')
    for arm in ARMS:
        if arm not in M or arm == 'control': continue
        cells = []
        for vi in range(4):
            if np.isnan(M[arm]['fmae'][vi]): cells.append(f'{FL[vi]}=n/a'); continue
            dm = M[arm]['fmae'][vi] / M['control']['fmae'][vi] - 1
            cells.append(f'{FL[vi]}={M[arm]["fmae"][vi]:.3f} ({100*dm:+.1f}%)')
        note = '  <- facade head UNTRAINED (expected garbage)' if arm == 'no_facade' else ''
        print(f'  [{arm:>11}] ' + '  '.join(cells) + note)
    nf = site_res['arms'].get('no_forcing', {})
    print('\n  --- no_forcing vs climatology floor (link to §3.2) ---')
    for vi in range(5):
        if AK[vi] in nf:
            ratio = nf[AK[vi]]['mae'] / MAD[AK[vi]]
            print(f'   {AL[vi]:>10}: MAE={nf[AK[vi]]["mae"]:.3f} / MAD={MAD[AK[vi]]:.3f} = {ratio:.2f}x')
    summary[SITE] = site_res

    # ---- (E) figure (first-pass placeholder closed; real figure drawn after loop) ----
    plt.close(plt.figure())

# ---- figure pass 2 (air head, from stored summary) ----
print('\n===== D) RENDERING FIGURE 14 (air head) =====')
arms_all = [a for a in ARMS if a != 'control']
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
for col, s in enumerate(SITES):
    if s not in summary: continue
    arms_p = [a for a in arms_all if a in summary[s]['arms']]
    x = np.arange(5); w = 0.8 / max(len(arms_p), 1)
    for ai, arm in enumerate(arms_p):
        vals = [summary[s]['arms'][arm][k]['dmae_pct'] for k in AK]
        bars = axes[col].bar(x + (ai - len(arms_p)/2 + 0.5) * w, vals, w, label=ARM_LAB[arm],
                             color=ARM_C[arm])
        for b, v in zip(bars, vals):
            axes[col].text(b.get_x() + b.get_width()/2, v + (1.5 if v >= 0 else -4.5),
                           f'{v:+.0f}', ha='center', fontsize=6.5)
    axes[col].axhline(0, color='k', lw=0.8)
    axes[col].set_xticks(x); axes[col].set_xticklabels(AL)
    axes[col].set_title(f'{s.capitalize()} - air head: % MAE change vs control')
    axes[col].set_ylabel('% change in MAE' if col == 0 else '')
    axes[col].legend(fontsize=7.5)
fig.suptitle('Input-ablation study (sealed p5d arms, seed 0, full test pool)', fontsize=11, y=1.02)
out = FIG / 'fig14_ablation.png'
fig.savefig(out, bbox_inches='tight'); plt.close(fig)
print(f'  [SAVED] {out.name}')

(MET / 'ablation_summary.json').write_text(json.dumps(summary, indent=2, default=str))
print('\n===== DRAFT-CLAIM VERDICTS (§3.7) =====')
print('  [XX] old draft protocol (K_window 3/7/11, 3 seeds, canyon only, V/Ta/RH/Tmrt numbers)')
print('       -> NO sealed artifacts; REPLACED by sealed 5-arm input ablation (1 seed, both sites).')
print('  [OK] "exactly one significance at 5%" old claim -> retired; new test basis = spatial blocks.')
print('  -> physical expectations to check in output above:')
print('     * no_forcing should degrade ALL vars (largest single effect)')
print('     * no_dist should hit V/TKE hardest (directional proximity)')
print('     * no_sunblock should hit T_mrt hardest (radiative geometry)')
print('     * no_facade air-head delta should be small (multi-task side-effect)')
print('\n===== DONE - paste this entire output back =====')
print('Then I deliver final §3.7 + Table 5 (block-paired t-tests) + updated Methods/Table A5 rows.')
