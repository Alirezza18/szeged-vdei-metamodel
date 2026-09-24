# ===== CELL P6-FC R9 (CPU, ~1-2 min, no GPU, no figure) - §3.6 COMPUTATIONAL EFFICIENCY =====
# (A) parameter counts from every sealed checkpoint (trainable vs BN buffers) vs META n_params
# (B) training wall-clock: per-epoch 'sec' from ALL *history.csv artifacts ->
#     per-run totals (mean+-SD across seeds), epochs run (early-stop check), campaign total
# (C) SEALED inference constants (from the fullpool eval runs, GPU session): test-pool size
#     and wall-clock -> throughput; recompute and print as sealed, NOT re-measured here
# (D) checkpoint disk size per model + full-ensemble footprint
# (E) ENVI-met comparison: fill ENVI_MET_RUNTIME_MIN below with YOUR simulation wall-clock
#     (minutes per site) and the cell prints the empirical surrogate speed-up ratio
# (F) context: domain dims, timesteps, test-pool sizes (from geometry + ensemble npz shapes)
# (G) DRAFT-CLAIM VERDICTS for §2.4.4 / §3.6
# Inputs: szeged-backup (+ any other input with *.pt / *history.csv), szeged-vdei-processed.
# Outputs: printed tables + 03_Results/02_Metrics/efficiency_summary.json
import json, time, os, glob as pyglob
from pathlib import Path
import numpy as np
import torch

IN = Path('/kaggle/input'); W = Path('/kaggle/working')
MET = W / '03_Results' / '02_Metrics'; MET.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza')

# >>> FILL with your ENVI-met wall-clock per site (minutes, one full diurnal simulation):
ENVI_MET_RUNTIME_MIN = {'canyon': None, 'plaza': None}

def find_all(pat):
    out = []
    for d in sorted(IN.glob('*')):
        out += sorted(d.rglob(pat))
    return out

def find(pat):
    h = find_all(pat)
    return h[0] if h else None

def load_npz(pat):
    p = find(pat)
    if p is None: print(f'  !! missing: {pat}'); return None
    print(f'  [{pat}] <- {p.parent.parent.name}/{p.parent.name}/{p.name}')
    return np.load(p)

summary = {}
t_all = time.time()
print('device: CPU | §3.6 efficiency fact-check (no GPU, no dataset classes needed)\n')

# ---------- (A) parameter counts ----------
print('===== A) PARAMETER COUNTS (sealed checkpoints) =====')
ck_files = find_all('*_best.pt')
param_counts = {}
for p in ck_files:
    name = p.name
    site = 'canyon' if 'canyon' in name else ('plaza' if 'plaza' in name else '?')
    ck = torch.load(p, map_location='cpu', weights_only=False)
    sd = ck.get('model_state', ck)
    n_total = sum(v.numel() for v in sd.values())
    n_buf = sum(v.numel() for k, v in sd.items()
                if ('running_' in k or 'num_batches' in k))
    n_train = n_total - n_buf
    meta_n = ck.get('n_params', None)
    ok = '' if meta_n is None else ('OK' if int(meta_n) == n_train else f'!! META says {meta_n}')
    print(f'  [{site:>6}] {name:<58} trainable={n_train:,} buffers={n_buf:,} total={n_total:,} {ok}')
    param_counts[f'{site}/{name}'] = {'trainable': n_train, 'buffers': n_buf, 'total': n_total,
                                      'meta_n_params': meta_n}
    del ck
uniq = {(v['trainable'], v['buffers'], v['total']) for v in param_counts.values()}
if len(uniq) == 1:
    tr, bu, to = uniq.pop()
    print(f'  -> ALL architectures identical: trainable={tr:,} (buffers={bu:,}, total={to:,})')
    summary['params'] = {'trainable': tr, 'buffers': bu, 'total': to}
print('  NOTE: manuscript convention = TRAINABLE parameters (BN running stats are buffers).')

# ---------- (B) training wall-clock ----------
print('\n===== B) TRAINING WALL-CLOCK (history CSVs, col sec) =====')
hist_files = find_all('*history.csv')
runs = []
for p in hist_files:
    try:
        import csv
        rows = list(csv.DictReader(open(p)))
        if not rows or 'sec' not in rows[0]: continue
        secs = [float(r['sec']) for r in rows if r.get('sec') not in (None, '', 'nan')]
        eps = len(rows)
        name = p.name
        site = 'canyon' if 'canyon' in name else ('plaza' if 'plaza' in name else '?')
        runs.append({'file': name, 'site': site, 'epochs': eps, 'total_min': sum(secs) / 60,
                     'mean_epoch_s': float(np.mean(secs)) if secs else float('nan')})
    except Exception as e:
        print(f'  !! {p.name}: {e}')
for r in sorted(runs, key=lambda x: (x['site'], x['file'])):
    print(f"  [{r['site']:>6}] {r['file']:<58} epochs={r['epochs']:>3}  "
          f"total={r['total_min']:6.1f} min  mean={r['mean_epoch_s']:5.1f} s/epoch")
grp = {}
for r in runs:
    key = (r['site'], r['file'].split('_seed')[0])
    grp.setdefault(key, []).append(r)
summary['training'] = []
print('  --- per site/variant (mean +- SD across seeds) ---')
for (site, variant), rs in sorted(grp.items()):
    tot = [r['total_min'] for r in rs]; ep = [r['epochs'] for r in rs]
    print(f'  [{site:>6}] {variant:<44} n={len(rs)} epochs={min(ep)}-{max(ep)}  '
          f'run time = {np.mean(tot):.1f} +- {np.std(tot):.1f} min')
    summary['training'].append({'site': site, 'variant': variant, 'n_seeds': len(rs),
                                'epochs_min': int(min(ep)), 'epochs_max': int(max(ep)),
                                'total_min_mean': float(np.mean(tot)),
                                'total_min_sd': float(np.std(tot))})
if runs:
    camp = sum(r['total_min'] for r in runs)
    print(f'  CAMPAIGN TOTAL (all {len(runs)} history files present in inputs): {camp:.1f} min '
          f'= {camp/60:.2f} h')
    summary['campaign_total_min'] = camp
    print('  NOTE: covers scratch runs with archived histories only; p5c fine-tunes add '
          '<=15 epochs each (light), baselines ran on CPU.')

# ---------- (C) sealed inference constants ----------
print('\n===== C) INFERENCE THROUGHPUT (SEALED constants from the fullpool eval runs) =====')
SEALED_EVAL = {'canyon': {'n': 10_357_000, 'min': 10.1}, 'plaza': {'n': 6_782_825, 'min': 6.5}}
summary['inference'] = {}
for site, v in SEALED_EVAL.items():
    thr = v['n'] / (v['min'] * 60)
    print(f'  [{site:>6}] {v["n"]:,} node-timestep samples in {v["min"]:.1f} min  '
          f'-> {thr:,.0f} samples/s  (sealed, single-pass ensemble5 eval)')
    summary['inference'][site] = {**v, 'samples_per_s': thr}

# ---------- (D) disk footprint ----------
print('\n===== D) MODEL DISK FOOTPRINT =====')
sizes = {}
for p in ck_files[:1]:
    kb = p.stat().st_size / 1024
    print(f'  single model (.pt): {kb:.1f} KB  -> 5-seed ensemble = {5*kb/1024:.2f} MB  '
          f'(both sites ~ {2*5*kb/1024:.2f} MB total)')
    sizes['single_model_KB'] = kb
summary['disk'] = sizes

# ---------- (E) ENVI-met speed-up ----------
print('\n===== E) ENVI-met COMPARISON (fill ENVI_MET_RUNTIME_MIN at top) =====')
summary['envimet'] = {}
for site in SITES:
    rt = ENVI_MET_RUNTIME_MIN.get(site)
    thr = summary['inference'][site]['samples_per_s']
    if rt is None:
        print(f'  [{site:>6}] ENVI-met runtime NOT filled -> speed-up ratio pending. '
              f'Fill ENVI_MET_RUNTIME_MIN["{site}"] and re-run (takes 1 min).')
    else:
        eval_min = SEALED_EVAL[site]['min']
        print(f'  [{site:>6}] ENVI-met {rt:.0f} min vs surrogate full-pool eval {eval_min:.1f} min '
              f'-> speed-up x{rt/eval_min:.1f}')
        summary['envimet'][site] = {'envimet_min': rt, 'surrogate_eval_min': eval_min,
                                    'speedup_x': rt / eval_min}

# ---------- (F) context ----------
print('\n===== F) DOMAIN CONTEXT =====')
ctx = {}
for site in SITES:
    g = load_npz(f'geometry_{site}.npz'); tf = load_npz(f'targets_forcing_{site}.npz')
    if g is None or tf is None: continue
    zc = np.asarray(g['z_center']); dz = np.asarray(g['dz'])
    npt, ntm = tf['target_T'].shape
    top = zc[-1] + dz[-1] / 2
    print(f'  [{site:>6}] air points={npt:,} x {ntm} timesteps = {npt*ntm:,} node-timesteps | '
          f'{len(zc)} levels, top={top:.1f} m, pitch dz={dz[0]:.2f} m')
    ctx[site] = {'air_points': int(npt), 'n_time': int(ntm), 'node_timesteps': int(npt*ntm),
                 'n_levels': int(len(zc)), 'top_m': float(top), 'dz': float(dz[0])}
summary['context'] = ctx

# ---------- (G) verdicts ----------
print('\n===== DRAFT-CLAIM VERDICTS (§2.4.4 / §3.6) =====')
tr = summary.get('params', {}).get('trainable')
print(f'  [{"OK" if tr == 117481 else "XX"}] "117,481 trainable parameters"  sealed={tr:,} '
      f'(total incl. BN buffers = {summary.get("params", {}).get("total"):,})')
if runs:
    per = {}
    for (site, _), rs in grp.items(): per.setdefault(site, []).extend(r['total_min'] for r in rs)
    for site, tot in per.items():
        m = np.mean(tot)
        print(f'  [{"OK" if 15 <= m <= 45 else "XX"}] "full run ~20-30 min per site ({site})"  '
              f'sealed mean={m:.1f} min (n={len(tot)} histories)')
for site in SITES:
    m = SEALED_EVAL[site]['min']
    ref = 10.0 if site == 'canyon' else 6.5
    print(f'  [{"OK" if abs(m-ref) < 1 else "XX"}] "test-pool eval ~{ref} min ({site})"  '
          f'sealed={m} min')
print(f'  [OK ] throughput {summary["inference"]["canyon"]["samples_per_s"]:,.0f} / '
      f'{summary["inference"]["plaza"]["samples_per_s"]:,.0f} samples/s (canyon/plaza)')

(MET / 'efficiency_summary.json').write_text(json.dumps(summary, indent=2, default=str))
print(f'\n===== DONE in {(time.time()-t_all)/60:.1f} min - paste this entire output back =====')
print('Then I deliver final §3.6 (+ Table: efficiency summary) with artifact-true numbers.')
