# ===== CELL P6-FC R12c + FIGURES A1/A2/A3 (CPU, ~6-8 min) - §3.9 TEMPORAL COHERENCE =====
# Fixes vs R12b: (1) A2 NameError (del dP,dT inside row loop) -> deleted after both rows;
#                (2) Figure A1 rebuilt in the OLD-MANUSCRIPT style: 2x3 panels, one per
#                    variable, 4 curves per panel (Canyon/Plaza x full-pool/pedestrian),
#                    x = HOURS FROM SOLAR NOON (dt = 24/(n_time-1)), night shading;
#                    panel (f) = sun-exposed fraction from SEALED sun_hit flags.
#                (3) NEW Figure A3: sun-exposed vs shaded MAE profiles (T_a, T_mrt x sites).
# Sun_hit note: sealed columns = DAYLIGHT timesteps only (canyon 15 of 25, plaza 29 of 49);
# the daylight window is recovered data-driven as the contiguous window with the HIGHEST
# mean reference T_mrt (printed for audit). x-mapping assumption dt=24/(n_time-1) -> 1.0h
# canyon / 0.5h plaza (validated by the printed implied daylight duration).
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
FIG = W / '03_Results' / '03_Figures'; FIG.mkdir(parents=True, exist_ok=True)
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza')
AIR = [('T', 28.0, 4.0, r'$T_a$'), ('RelHum', 40.0, 15.0, r'$RH$'),
       ('WindSpd', 2.0, 1.5, r'$V$'), ('TKE', 50.0, 100.0, r'$TKE$'),
       ('TMRT', 45.0, 20.0, r'$T_{mrt}$')]
AK = [a[0] for a in AIR]; AL = [a[3] for a in AIR]
MU = np.array([a[1] for a in AIR], dtype=np.float32)
SD = np.array([a[2] for a in AIR], dtype=np.float32)
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.titlesize': 10,
                     'mathtext.fontset': 'stix'})
CURVES = {}          # site -> {'full': (5, n_time), 'ped': (5, n_ped, n_time) means}
META = {}            # site -> dict(dt_h, t_noon, lab(n_time), daylight_idx, x(n_time))

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

def f3(x):
    return f'{x:.3f}' if x == x else '  n/a'

summary = {}
for SITE in SITES:
    print(f'\n########## [{SITE}] ##########')
    tf = load_npz(f'targets_forcing_{SITE}.npz')
    split = load_npz(f'split_{SITE}.npz')
    d = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    if any(x is None for x in (tf, split, d)): continue
    n_points, n_time = tf['target_T'].shape
    spl = np.asarray(split['split']).astype(str)
    ix_te = np.where(spl == 'test')[0]
    n_test = len(ix_te)
    pa = np.asarray(d['pred_air']); ta = np.asarray(d['target_air'])
    print(f'  pool rows={pa.shape[0]:,} | n_test={n_test:,} x n_time={n_time} '
          f'-> reshape exact: {pa.shape[0] == n_test * n_time}')
    P = pa.reshape(n_test, n_time, 5).astype(np.float32) * SD + MU
    T = ta.reshape(n_test, n_time, 5).astype(np.float32) * SD + MU
    E = np.abs(P - T)
    print('  cross-check MAE (must match §3.1.1): '
          + '  '.join(f'{AL[v]}={float(E[:, :, v].mean()):.3f}' for v in range(5)))
    dP = np.diff(P, axis=1).astype(np.float32); dT = np.diff(T, axis=1).astype(np.float32)
    d2P = (P[:, 2:, :] - 2 * P[:, 1:-1, :] + P[:, :-2, :]).astype(np.float32)
    d2T = (T[:, 2:, :] - 2 * T[:, 1:-1, :] + T[:, :-2, :]).astype(np.float32)
    refT = np.asarray(tf['target_TMRT'], dtype=np.float64).mean(0)
    q = np.quantile(refT, [1/3, 2/3]); lab = np.digitize(refT, q)
    inc_reg = lab[1:]
    # x-axis mapping: hours from solar noon
    dt_h = 24.0 / (n_time - 1)
    t_noon = int(np.argmax(refT))
    x_h = (np.arange(n_time) - t_noon) * dt_h
    print(f'  x-mapping: dt={dt_h:.2f} h/step, solar noon t={t_noon}, '
          f'x range [{x_h[0]:+.1f},{x_h[-1]:+.1f}] h')
    # metrics table (Table A12) - identical basis to R12b
    reg_flat = np.tile(inc_reg.astype(np.int8), n_test)
    site_res = {'n_pairs_per_pt': int(n_time - 1), 'dt_h': dt_h, 't_noon': t_noon,
                'regime_q': [float(q[0]), float(q[1])], 'vars': {}}
    print('\n  ===== TEMPORAL-COHERENCE METRICS (Table A12 rows) =====')
    print('  var      SDref    SDpred  varRat  SA_raw  SA_db    night  trans   peak   '
          'roughRat   r(dy)')
    for v in range(5):
        aP = dP[:, :, v].ravel().astype(np.float64)
        aT = dT[:, :, v].ravel().astype(np.float64)
        sdT, sdP = aT.std(), aP.std()
        sa_raw = float((np.sign(aP) == np.sign(aT)).mean())
        eps = 0.1 * sdT
        m = np.abs(aT) > eps
        sa_db = float((np.sign(aP[m]) == np.sign(aT[m])).mean())
        sa_reg = []
        for r in range(3):
            mr = (reg_flat == r) & m
            sa_reg.append(float((np.sign(aP[mr]) == np.sign(aT[mr])).mean())
                          if mr.sum() else np.nan)
        rr = float(np.abs(d2P[:, :, v].ravel().astype(np.float64)).mean()
                   / np.abs(d2T[:, :, v].ravel().astype(np.float64)).mean())
        r_dy = float(np.corrcoef(aT, aP)[0, 1])
        print(f'  {AL[v]:>8} {sdT:7.3f} {sdP:7.3f} {sdP/sdT:6.3f}  {sa_raw:5.3f}  {sa_db:5.3f}  '
              f'{f3(sa_reg[0])} {f3(sa_reg[1])} {f3(sa_reg[2])}   {rr:6.3f}  {r_dy:+.3f}')
        site_res['vars'][AK[v]] = {
            'sd_ref_dy': float(sdT), 'sd_pred_dy': float(sdP),
            'var_ratio': float(sdP / sdT), 'SA_raw': sa_raw, 'SA_deadband': sa_db,
            'SA_night': None if sa_reg[0] != sa_reg[0] else sa_reg[0],
            'SA_transition': None if sa_reg[1] != sa_reg[1] else sa_reg[1],
            'SA_peak': None if sa_reg[2] != sa_reg[2] else sa_reg[2],
            'rough_ratio': rr, 'r_dy': r_dy}
        del aP, aT
    summary[SITE] = site_res
    # ---- curves for Figures A1/A3 (ped mask within test points) ----
    k0 = np.asarray(tf['k0'])[ix_te]
    ped = (k0 == 4)
    CURVES[SITE] = {'full': np.stack([E[:, :, v].mean(0) for v in range(5)]),
                    'ped': np.stack([E[ped, :, v].mean(0) for v in range(5)]),
                    'n_ped': int(ped.sum())}
    # ---- daylight window from sealed sun_hit (data-driven: highest-T_mrt window) ----
    if 'sun_hit' in tf.files:
        sh = np.asarray(tf['sun_hit'])[ix_te]
        n_sun = sh.shape[1]
        wins = [(s, float(refT[s:s + n_sun].mean()))
                for s in range(n_time - n_sun + 1)]
        s0 = max(wins, key=lambda w: w[1])[0]
        day_idx = np.arange(s0, s0 + n_sun)
        exp_frac = (sh > 0).mean(0)                     # per daylight column
        META[SITE] = {'dt_h': dt_h, 't_noon': t_noon, 'lab': lab, 'x': x_h,
                      'day_idx': day_idx, 'exp_frac': exp_frac, 'n_sun': n_sun}
        print(f'  sun_hit: shape={sh.shape} values={np.unique(sh)[:6]} | daylight window '
              f't=[{s0}..{s0+n_sun-1}] ({n_sun*dt_h:.1f} h daylight - physical check) | '
              f'mean exposed frac={exp_frac.mean():.3f}')
        del sh
    del E, P, T, dP, dT, d2P, d2T, pa, ta, d

# ================= FIGURE A1 (old-manuscript style, 2x3) =================
print('\n===== FIGURE A1 (diurnal MAE profiles, 4 curves/panel) =====')
fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.4))
STY = {'canyon': ('#1f5fa8', '-'), 'plaza': ('#c0392b', '-')}
PSTY = {'canyon': ('#7fa8d0', '--'), 'plaza': ('#e08a7a', '--')}
for v, (ax, nm) in enumerate(zip(axes.ravel()[:5], AL)):
    for SITE in SITES:
        if SITE not in CURVES: continue
        x = META[SITE]['x']; lab = META[SITE]['lab']
        for r in (0,):                     # night shading per site
            for t in np.where(lab == 0)[0]:
                ax.axvspan(x[t] - META[SITE]['dt_h']/2, x[t] + META[SITE]['dt_h']/2,
                           color='0.55', alpha=0.15, lw=0, zorder=0)
        ax.plot(x, CURVES[SITE]['full'][v], STY[SITE][1], color=STY[SITE][0], lw=1.7,
                marker='o', ms=2.5, label=f'{SITE.capitalize()} (full pool)')
        ax.plot(x, CURVES[SITE]['ped'][v], PSTY[SITE][1], color=PSTY[SITE][0], lw=1.5,
                marker='s', ms=2.5, label=f'{SITE.capitalize()} (pedestrian)')
    ax.set_yscale('log')
    ax.set_title(f'({chr(97+v)}) {nm} MAE', fontsize=9.5)
    ax.set_xlabel('t (hours from solar noon)')
    ax.set_ylabel(f'{nm} MAE (log)' if v % 3 == 0 else '')
    ax.grid(True, ls=':', lw=0.4, alpha=0.6)
axes.ravel()[0].legend(fontsize=6.6, ncol=1)
# panel (f): sun-exposed fraction
axf = axes.ravel()[5]
for SITE in SITES:
    if SITE not in META: continue
    M = META[SITE]
    xs = M['x'][M['day_idx']]
    axf.plot(xs, M['exp_frac'], STY[SITE][1], color=STY[SITE][0], lw=1.7, marker='o',
             ms=2.5, label=f'{SITE.capitalize()}')
    axf.fill_between(xs, 0, M['exp_frac'], color=STY[SITE][0], alpha=0.10)
axf.set_title('(f) Sun-exposed fraction (daylight steps)', fontsize=9.5)
axf.set_xlabel('t (hours from solar noon)'); axf.set_ylabel('fraction of points')
axf.set_ylim(0, 1); axf.grid(True, ls=':', lw=0.4, alpha=0.6); axf.legend(fontsize=7)
fig.suptitle('Diurnal MAE profiles - Canyon vs Plaza (night shaded); pedestrian = k=4 test points',
             fontsize=11, y=0.99)
fig.tight_layout()
fig.savefig(FIG / 'figA1_diurnal_MAE_profiles.png', bbox_inches='tight')
plt.close(fig); print('  [SAVED] figA1_diurnal_MAE_profiles.png')

# ================= FIGURE A2 (fixed del bug) =================
print('\n===== FIGURE A2 (increment coherence) =====')
fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.4))
rng = np.random.default_rng(0)
for col, SITE in enumerate(SITES):
    d = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    split = load_npz(f'split_{SITE}.npz')
    tf = load_npz(f'targets_forcing_{SITE}.npz')
    n_test = int((np.asarray(split['split']).astype(str) == 'test').sum())
    n_time = tf['target_T'].shape[1]
    P = np.asarray(d['pred_air'], dtype=np.float32).reshape(n_test, n_time, 5) * SD + MU
    T = np.asarray(d['target_air'], dtype=np.float32).reshape(n_test, n_time, 5) * SD + MU
    dP = np.diff(P, axis=1); dT = np.diff(T, axis=1)
    del P, T, d, tf, split
    for row, (v, nm) in enumerate(((0, r'$T_a$'), (4, r'$T_{mrt}$'))):
        aT = dT[:, :, v].ravel().astype(np.float64)
        aP = dP[:, :, v].ravel().astype(np.float64)
        lim = float(np.percentile(np.abs(np.concatenate([aT, aP])), 99))
        sub = rng.choice(aT.size, min(aT.size, 300_000), replace=False)
        ax = axes[row, col]
        ax.hexbin(aT[sub], aP[sub], gridsize=60, mincnt=1, cmap='viridis', bins='log',
                  extent=[-lim, lim, -lim, lim])
        ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
        eps = 0.1 * aT.std()
        m = np.abs(aT) > eps
        sa = float((np.sign(aP[m]) == np.sign(aT[m])).mean())
        vr = float(aP.std() / aT.std()); rd = float(np.corrcoef(aT, aP)[0, 1])
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(rf'reference $\Delta${nm}' if row == 1 else '')
        ax.set_ylabel(rf'predicted $\Delta${nm}' if col == 0 else '')
        ax.set_title(f'{SITE.capitalize()} {nm}: SA={100*sa:.0f}%, '
                     f'SD-ratio={vr:.2f}, r={rd:.2f}', fontsize=9)
        del aT, aP
    del dP, dT                                   # FIX: outside the row loop
fig.suptitle('Temporal coherence: consecutive-step increments (hexbin log-count, '
             'dashed = 1:1)', fontsize=10.5, y=0.995)
fig.savefig(FIG / 'figA2_temporal_coherence.png', bbox_inches='tight')
plt.close(fig); print('  [SAVED] figA2_temporal_coherence.png')

# ================= FIGURE A3 (sun-exposed vs shaded; old-manuscript companion) =====
print('\n===== FIGURE A3 (sun-exposed vs shaded MAE) =====')
fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6))
for col, SITE in enumerate(SITES):
    if SITE not in META: continue
    d = load_npz(f'{SITE}_dualhead_ensemble5_test_fullpool.npz')
    split = load_npz(f'split_{SITE}.npz')
    tf = load_npz(f'targets_forcing_{SITE}.npz')
    n_test = int((np.asarray(split['split']).astype(str) == 'test').sum())
    n_time = tf['target_T'].shape[1]
    P = np.asarray(d['pred_air'], dtype=np.float32).reshape(n_test, n_time, 5) * SD + MU
    T = np.asarray(d['target_air'], dtype=np.float32).reshape(n_test, n_time, 5) * SD + MU
    E = np.abs(P - T); del P, T
    sh = np.asarray(tf['sun_hit'])[ixs] if False else np.asarray(tf['sun_hit'])[np.where(
        np.asarray(split['split']).astype(str) == 'test')[0]]
    del d, tf, split
    M = META[SITE]; x = M['x'][M['day_idx']]
    for row, (v, nm) in enumerate(((0, r'$T_a$'), (4, r'$T_{mrt}$'))):
        me, ms = [], []
        for j, t in enumerate(M['day_idx']):
            exp = sh[:, j] > 0
            me.append(float(E[exp, t, v].mean()))
            ms.append(float(E[~exp, t, v].mean()))
        ax = axes[row, col]
        ax.plot(x, me, '-o', color='#c0392b', lw=1.6, ms=3, label='Sun-exposed')
        ax.plot(x, ms, '--s', color='#1f5fa8', lw=1.6, ms=3, label='Shaded')
        ax.set_title(f'({chr(97+row+2*col)}) {nm} MAE - {SITE.capitalize()}', fontsize=9.5)
        ax.set_xlabel('t (hours from solar noon)')
        ax.set_ylabel(f'{nm} MAE (°C)' if col == 0 else '')
        ax.grid(True, ls=':', lw=0.4, alpha=0.6); ax.legend(fontsize=7)
    del E, sh
fig.suptitle('Sun-exposed vs shaded MAE profiles (daylight steps, sealed sun_hit flags)',
             fontsize=10.5, y=0.99)
fig.tight_layout()
fig.savefig(FIG / 'figA3_sun_shaded_MAE.png', bbox_inches='tight')
plt.close(fig); print('  [SAVED] figA3_sun_shaded_MAE.png')

(MET / 'temporal_coherence_summary.json').write_text(json.dumps(summary, indent=2))
print('\n===== VERDICT HINTS =====')
print('  * var_ratio < 1 + rough_ratio < 1 (thermals) -> temporal over-smoothing (§3.1.4 twin)')
print('  * canyon V roughRat 2.69 -> watch plaza: if >1, predicted V jitter is a real finding')
print('  * TKE night deadband n/a expected (quiet-night increments below 10% of SD)')
print('\n===== DONE - paste this entire output back =====')
print('Then I deliver final §3.9 + Figure A1/A2(/A3) captions + Table A12.')
