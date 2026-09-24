# ===== CELL P6-FC R2 (CPU, zero GPU quota) - §3.1.2 morphology-gradient fact-check =====
# Verifies every number in the §3.1.2 draft. Reads the SAME sealed ensemble5 npz the §3.1.1
# cell used (so all cross-site numbers are consistent), plus the sealed targets_forcing +
# svf npz for the reference-range and SVF claims.
# Outputs: printed cross-site gradient table + draft-claim verdicts. No files written.
import json
from pathlib import Path
import numpy as np

IN = Path('/kaggle/input')
SITES = ('canyon', 'plaza')
AIR = [('T', (28.0, 4.0)), ('RelHum', (40.0, 15.0)), ('WindSpd', (2.0, 1.5)),
       ('TKE', (50.0, 100.0)), ('TMRT', (45.0, 20.0))]
AIR_KEYS = [a[0] for a in AIR]
AIR_LABELS = [r'$T_a$', r'$RH$', r'$V$', r'$TKE$', r'$T_{mrt}$']
AIR_UNITS = ['degC', '%', 'm/s', 'm2/s2', 'degC']


def find(pat):
    for d in sorted(IN.glob('*')):
        h = sorted(d.rglob(pat))
        if h:
            return h[0]
    return None


def load_npz(pat):
    p = find(pat)
    if p is None:
        print(f'  !! missing: {pat}')
        return None
    print(f'  [{pat}] <- {p.parent.parent.name}/{p.parent.name}/{p.name}')
    return np.load(p)


def metrics(p, t):
    p = p.astype(np.float64); t = t.astype(np.float64)
    d = p - t
    ss_res = float((d * d).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(np.abs(d).mean()), float(np.sqrt((d * d).mean())), r2, float(t.mean())


print('device: CPU | inputs: szeged-backup (ensemble5) + szeged-vdei-processed (targets_forcing, svf)\n')
print('NOTE: all cross-site metrics from the SAME sealed ensemble5 npz as Figure 6.')

# ---- A) cross-site air metrics from sealed ensemble5 npz ----
ens = {}
for site in SITES:
    d = load_npz(f'{site}_dualhead_ensemble5_test_fullpool.npz')
    if d is None:
        continue
    rec = {}
    for vi, (k, (mu, sd)) in enumerate(AIR):
        p = d['pred_air'][:, vi].astype(np.float64) * sd + mu
        t = d['target_air'][:, vi].astype(np.float64) * sd + mu
        mae, rmse, r2, tmean = metrics(p, t)
        # test-pool physical range of the TARGET (for the "wider range" claim)
        rec[k] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'tmean': tmean,
                  'tmin': float(t.min()), 'tmax': float(t.max()),
                  'rel': mae / tmean * 100.0}
    ens[site] = rec
    d.close()

print('\n=== A) CROSS-SITE AIR GRADIENTS (ensemble5, physical units) ===')
hdr = f'{"var":7}' + f'{"canyon_MAE":>11}{"plaza_MAE":>10}{"dMAE":>9}' \
      + f'{"canyon_R2":>11}{"plaza_R2":>10}{"dR2":>8}' \
      + f'{"canyon_rel%":>12}{"plaza_rel%":>11}'
print(hdr)
for vi, k in enumerate(AIR_KEYS):
    c, p = ens['canyon'][k], ens['plaza'][k]
    dmae = c['mae'] - p['mae']
    dr2 = c['r2'] - p['r2']
    print(f'{AIR_LABELS[vi]:7}{c["mae"]:>11.3f}{p["mae"]:>10.3f}{dmae:>+9.3f}'
          f'{c["r2"]:>11.4f}{p["r2"]:>10.4f}{dr2:>+8.4f}'
          f'{c["rel"]:>11.2f}%{p["rel"]:>10.2f}%')
print('  dMAE = canyon - plaza  (+ = canyon worse / plaza better;  - = canyon better)')
print('  dR2  = canyon - plaza  (+ = canyon better;                   - = plaza better)')

print('\n=== B) TMRT + all-air TEST-POOL TARGET RANGE (physical, for "wider range" claim) ===')
for site in SITES:
    rec = ens[site]
    print(f'  [{site}] ' + ' | '.join(
        f'{AIR_LABELS[vi]} [{rec[k]["tmin"]:.1f}, {rec[k]["tmax"]:.1f}] (span {rec[k]["tmax"]-rec[k]["tmin"]:.1f})'
        for vi, k in enumerate(AIR_KEYS)))

print('\n=== C) MEAN SVF (sealed svf npz) - for the "0.87 vs 0.43" claim ===')
for site in SITES:
    d = load_npz(f'svf_{site}.npz')
    if d is None:
        continue
    ev = d['svf_envimet'].astype(np.float64); ra = d['svf_rays'].astype(np.float64)
    ok = np.isfinite(ev) & np.isfinite(ra)
    print(f'  [{site}] ENVI-met mean SVF = {ev[ok].mean():.3f} | ray mean SVF = {ra[ok].mean():.3f} '
          f'| n={ok.sum():,} | pearson_r={float(d["pearson_r"]):.4f}')
    d.close()

# ================= DRAFT-CLAIM VERDICTS =================
print('\n================= DRAFT-CLAIM VERDICTS (§3.1.2) =================')
def chk(label, cond, detail):
    print(f'  [{"OK " if cond else "XX "}] {label}: {detail}')

c, p = ens['canyon'], ens['plaza']
chk('canyon V R2 ~ 0.978 (draft) vs 0.941 plaza',
    abs(c['WindSpd']['r2'] - 0.978) < 0.005 and abs(p['WindSpd']['r2'] - 0.941) < 0.005,
    f'canyon={c["WindSpd"]["r2"]:.4f} (draft 0.978), plaza={p["WindSpd"]["r2"]:.4f} (draft 0.941)')
chk('V MAE 0.14 m/s lower at canyon (draft)',
    abs((p['WindSpd']['mae'] - c['WindSpd']['mae']) - 0.14) < 0.02,
    f'actual diff (canyon lower) = {p["WindSpd"]["mae"]-c["WindSpd"]["mae"]:.3f} m/s (draft 0.14)')
chk('TKE MAE 2.45 m2/s2 LOWER at canyon (draft)',
    (c['TKE']['mae'] - p['TKE']['mae']) < 0,
    f'canyon={c["TKE"]["mae"]:.2f}, plaza={p["TKE"]["mae"]:.2f} -> canyon {c["TKE"]["mae"]-p["TKE"]["mae"]:+.2f} HIGHER, not lower')
chk('TKE R2: canyon 0.916 vs plaza 0.941 (draft)',
    abs(c['TKE']['r2'] - 0.916) < 0.005 and abs(p['TKE']['r2'] - 0.941) < 0.005,
    f'canyon={c["TKE"]["r2"]:.4f} (draft 0.916), plaza={p["TKE"]["r2"]:.4f} (draft 0.941)')
chk('plaza Tmrt MAE 1.715 < canyon 2.648 (draft)',
    p['TMRT']['mae'] < c['TMRT']['mae'],
    f'canyon={c["TMRT"]["mae"]:.3f}, plaza={p["TMRT"]["mae"]:.3f} -> canyon LOWER, opposite of draft')
chk('relative Tmrt error 2.9% (plaza) vs 5.9% (canyon) (draft)',
    abs(p['TMRT']['rel'] - 2.9) < 0.4 and abs(c['TMRT']['rel'] - 5.9) < 0.4,
    f'canyon={c["TMRT"]["rel"]:.2f}%, plaza={p["TMRT"]["rel"]:.2f}% (draft canyon 5.9 / plaza 2.9)')
chk('plaza TMRT range wider (35-74) than canyon (draft)',
    (p['TMRT']['tmax'] - p['TMRT']['tmin']) > (c['TMRT']['tmax'] - c['TMRT']['tmin']),
    f'canyon span={c["TMRT"]["tmax"]-c["TMRT"]["tmin"]:.1f}, plaza span={p["TMRT"]["tmax"]-p["TMRT"]["tmin"]:.1f} -> canyon wider')
chk('Ta/RH R2 > 0.992 both sites (draft)',
    c['T']['r2'] > 0.992 and p['T']['r2'] > 0.992 and c['RelHum']['r2'] > 0.992 and p['RelHum']['r2'] > 0.992,
    f'canyon T={c["T"]["r2"]:.4f} RH={c["RelHum"]["r2"]:.4f} | plaza T={p["T"]["r2"]:.4f} RH={p["RelHum"]["r2"]:.4f} (draft >0.992)')
chk('Ta/RH MAE diff only 0.10 C / 0.15% (draft)',
    abs(c['T']['mae'] - p['T']['mae']) < 0.10 and abs(c['RelHum']['mae'] - p['RelHum']['mae']) < 0.15,
    f'T diff={abs(c["T"]["mae"]-p["T"]["mae"]):.3f} C, RH diff={abs(c["RelHum"]["mae"]-p["RelHum"]["mae"]):.3f}% (draft 0.10/0.15)')

print('\n===== DONE - paste this entire output back =====')
print('Then I rewrite §3.1.2 with the correct morphology-gradient narrative.')
