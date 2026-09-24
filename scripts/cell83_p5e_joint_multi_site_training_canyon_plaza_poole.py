# ===== CELL P5e - JOINT MULTI-SITE TRAINING (canyon+plaza pooled, 3 seeds) =====
# Completes the transfer triangle: scratch / zero-shot / fine-tune / JOINT.  Prereqs:
# CELL 18b -> CELL 19 in THIS session (no restart needed after P5d).  GPU T4, resumable.
# Design (locked 2026-09-07, delegated decision):
#   - ONE model per seed trained on BOTH sites pooled (no site-ID input dim -> architecture
#     identical to the sealed optuna_best: tests whether shared morphology features suffice).
#   - Per-epoch sample budget identical to the sealed protocol: 500k canyon + 300k plaza
#     (fixed random subset per seed, drawn within each site's train view).
#   - Facade targets keep PER-SITE normalization stats (head learns both sites' normalized
#     scales; eval denormalizes with the evaluated site's own stats).
#   - AIR targets use the shared manuscript stats (identical across sites by design).
#   - Seeds 0-2 (secondary experiment; 5-seed protocol reserved for the per-site headline).
# Est. T4 time: ~30-40 min train/seed + eval 10+6.5 min -> ~2.5-3 h total.  Resumable per unit.
import json, csv, time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

W = Path('/kaggle/working')
MET_DIR = W / '03_Results' / '02_Metrics'; MET_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = W / '03_Results' / '01_ModelOutputs'; CKPT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = (0, 1, 2)
CFG = 'optuna_best'
LR, LAM = CONFIGS[CFG]['lr'], CONFIGS[CFG]['facade_weight']
BATCH, N_EPOCHS, PATIENCE = 128, 30, 5
EP = {'canyon': 500_000, 'plaza': 300_000}          # sealed per-site per-epoch budget
VAL_PER_SITE = 50_000
AIR_UNITS = {'T': 'degC', 'RelHum': '%', 'WindSpd': 'm/s', 'TKE': 'm2/s2', 'TMRT': 'degC'}
FAC_UNITS = {'Twall': 'degC', 'Qsens': 'W/m2', 'SWabs': 'W/m2', 'LWbal': 'W/m2'}
# frozen reference numbers (sealed 5-seed scratch ensembles + P5c fine-tunes) for the
# final comparison table - constants, never recomputed, provenance: p5a_vs_optuna_compare
# .json / p5c_transfer_summary.json
REF = {
    'canyon': {'scratch5': {'T': .383, 'RelHum': 1.222, 'WindSpd': .273, 'TKE': 3.470, 'TMRT': 1.111,
                            'Twall': 1.280, 'Qsens': 10.417, 'SWabs': 21.254, 'LWbal': 6.806},
               'ft_from_plaza': {'T': .359, 'RelHum': 1.137, 'WindSpd': .267, 'TKE': 3.325, 'TMRT': 1.158,
                                 'Twall': 1.114, 'Qsens': 9.054, 'SWabs': 17.809, 'LWbal': 6.060}},
    'plaza':  {'scratch5': {'T': .365, 'RelHum': 1.285, 'WindSpd': .351, 'TKE': 2.109, 'TMRT': 1.472,
                            'Twall': 1.213, 'Qsens': 7.474, 'SWabs': 22.463, 'LWbal': 5.605},
               'ft_from_canyon': {'T': .459, 'RelHum': 1.657, 'WindSpd': .357, 'TKE': 2.877, 'TMRT': 1.505,
                                  'Twall': 1.228, 'Qsens': 8.174, 'SWabs': 27.175, 'LWbal': 5.943}},
}
if 'VDEIDatasetV2' not in globals() or 'build' not in globals():
    raise SystemExit('!! Run sealed CELL 18b (VDEIDatasetV2) + CELL 19 (build) first '
                     'in THIS session, then re-run.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise SystemExit('GPU session required - run CELL 18b -> CELL 19 first, keep THIS session.')
try:
    _ = (torch.ones(1, device=device) * 2).sum().item()
except RuntimeError as e:
    raise SystemExit(f'GPU check failed: {e}')
print(f'device: {torch.cuda.get_device_name(0)} ✅ | joint seeds={SEEDS} | '
      f'lr={LR:.3e} lam={LAM} batch={BATCH} ep={N_EPOCHS} patience={PATIENCE}')
print('design: pooled training, NO site-id input, per-site facade norm, '
      'per-epoch budget 500k(canyon)+300k(plaza)')


def seed_all(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def composite(pa, pf, y, fy, fv, lam):
    la = ((pa - y) ** 2).mean()
    per = ((pf - fy) ** 2).mean(dim=1)
    den = fv.sum()
    lf = (per * fv).sum() / den if den > 0 else pa.new_zeros(())
    return la + lam * lf, la.detach(), lf.detach()


def mae_rmse(p, t):
    d = p.astype(np.float32) - t.astype(np.float32)
    return float(np.abs(d).mean()), float(np.sqrt((d ** 2).mean()))


# ---- joint datasets (built once; VDEIDatasetV2 caches per site) ----------------
print('building joint train/val views (canyon + plaza) ...')
_train_c = VDEIDatasetV2('canyon', 'train')
_train_p = VDEIDatasetV2('plaza', 'train')
_joint_train = ConcatDataset([_train_c, _train_p])
_gv = np.random.RandomState(0)
_vc = _gv.choice(len(_train_c), min(VAL_PER_SITE, len(_train_c)), replace=False)
_vp = _gv.choice(len(_train_p), min(VAL_PER_SITE, len(_train_p)), replace=False)
_joint_val = ConcatDataset([Subset(_train_c, _vc), Subset(_train_p, _vp)])
print(f'joint pool: canyon 46,105,500 + plaza 31,649,002 = {len(_joint_train):,} samples | '
      f'val subset {len(_joint_val):,} (50k/site)')


def train_joint(seed):
    ckpt = CKPT_DIR / f'joint_dualhead_p5e_seed{seed}_best.pt'
    if ckpt.exists():
        print(f'[joint s{seed}] checkpoint exists - skip training'); return
    seed_all(seed)
    model, _ = build(CFG); model.to(device)
    g = np.random.RandomState(seed)                      # fixed per-seed subset
    ic = g.choice(len(_train_c), EP['canyon'], replace=False)
    ip = g.choice(len(_train_p), EP['plaza'], replace=False)
    idx = np.concatenate([ic, len(_train_c) + ip])
    tr_loader = DataLoader(Subset(_joint_train, idx), batch_size=BATCH, shuffle=True,
                           num_workers=4, pin_memory=True, persistent_workers=True)
    va_loader = DataLoader(_joint_val, batch_size=BATCH, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')
    best, bad = float('inf'), 0
    t00 = time.time()
    for ep in range(1, N_EPOCHS + 1):
        t0 = time.time(); model.train(); tr = 0.0; ns = 0
        for b in tr_loader:
            v = b['vdei'].to(device, non_blocking=True); f = b['forcing'].to(device, non_blocking=True)
            y = b['target'].to(device, non_blocking=True); fy = b['facade_target'].to(device, non_blocking=True)
            fv = b['facade_valid'].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                pa, pf = model(v, f)
            loss, _, _ = composite(pa.float(), pf.float(), y, fy, fv, LAM)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            ns += v.size(0); tr += float(loss.detach()) * v.size(0)
        tr /= ns
        model.eval(); va = 0.0; nvs = 0
        with torch.no_grad():
            for b in va_loader:
                v = b['vdei'].to(device, non_blocking=True); f = b['forcing'].to(device, non_blocking=True)
                y = b['target'].to(device, non_blocking=True); fy = b['facade_target'].to(device, non_blocking=True)
                fv = b['facade_valid'].to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    pa, pf = model(v, f)
                loss, _, _ = composite(pa.float(), pf.float(), y, fy, fv, LAM)
                nvs += v.size(0); va += float(loss.detach()) * v.size(0)
        va /= nvs
        sch.step(va)
        print(f'[joint s{seed}] Ep {ep:2d}/{N_EPOCHS} | train={tr:.4f} | val={va:.4f} '
              f'| lr={opt.param_groups[0]["lr"]:.2e} | {time.time()-t0:.1f}s', flush=True)
        if va < best:
            best, bad = va, 0
            torch.save({'model_state': model.state_dict(), 'config': CFG, 'seed': seed,
                        'epoch': ep, 'val_loss': va, 'mode': 'joint_pooled',
                        'lr': LR, 'lam': LAM,
                        'n_params': sum(p.numel() for p in model.parameters())}, ckpt)
            print('  ✅ new best saved')
        else:
            bad += 1
            if bad >= PATIENCE:
                print('  ⏹ early stopping'); break
    print(f'[joint s{seed}] trained in {(time.time()-t00)/60:.1f} min | best val={best:.4f}')


def eval_joint(site, seed):
    out = MET_DIR / f'{site}_dualhead_p5e_joint_seed{seed}_test_fullpool.npz'
    if out.exists():
        print(f'[{site} joint s{seed}] npz exists - skip eval'); return
    ckpt = CKPT_DIR / f'joint_dualhead_p5e_seed{seed}_best.pt'
    assert ckpt.exists(), f'checkpoint missing: {ckpt.name}'
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model, _ = build(ck.get('config', CFG)); model.load_state_dict(ck['model_state'])
    model.to(device).eval()
    test_ds = VDEIDatasetV2(site, 'test')
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    PA, TA, PF, TF, FV = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for b in loader:
            v = b['vdei'].to(device, non_blocking=True); f = b['forcing'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                pa, pf = model(v, f)
            PA.append(pa.float().cpu().numpy().astype(np.float16))
            TA.append(b['target'].numpy().astype(np.float16))
            PF.append(pf.float().cpu().numpy().astype(np.float16))
            TF.append(b['facade_target'].numpy().astype(np.float16))
            FV.append(b['facade_valid'].numpy())
    np.savez(out, pred_air=np.concatenate(PA), target_air=np.concatenate(TA),
             pred_fac=np.concatenate(PF), target_fac=np.concatenate(TF),
             valid=np.concatenate(FV).astype(bool))
    print(f'[{site}] joint s{seed} eval done in {(time.time()-t0)/60:.1f} min -> {out.name} '
          f'({out.stat().st_size/1e6:.0f} MB)')


t_all = time.time()
for seed in SEEDS:
    train_joint(seed)
    for site in SITES:
        eval_joint(site, seed)

# ---- Phase 2: per-site 3-seed ensemble + comparison table ----------------------
summary = {'meta': {'config': CFG, 'mode': 'joint_pooled', 'seeds': list(SEEDS), 'lr': LR,
                    'lam': LAM, 'batch': BATCH, 'epochs': N_EPOCHS, 'patience': PATIENCE,
                    'ep_budget': EP, 'val_per_site': VAL_PER_SITE,
                    'computed_at': datetime.now().strftime('%Y-%m-%d %H:%M')},
           'sites': {}, 'note': 'air = all test rows; facade = valid rows; physical units; '
                                'references are frozen sealed numbers (scratch5 5-seed ens; FT from p5c)'}
csv_rows = []; mlog = MET_DIR / 'test_metrics_log.csv'

for site in SITES:
    test_ds = VDEIDatasetV2(site, 'test')
    fm, fs = test_ds.d['fac_mean'], test_ds.d['fac_std']
    per_seed_air = np.zeros((len(SEEDS), len(AIR_KEYS)))
    per_seed_fac = np.zeros((len(SEEDS), len(FAC_KEYS)))
    ens = None
    for si, seed in enumerate(SEEDS):
        d = np.load(MET_DIR / f'{site}_dualhead_p5e_joint_seed{seed}_test_fullpool.npz')
        pa, ta = d['pred_air'].astype(np.float32), d['target_air'].astype(np.float32)
        pf, tf = d['pred_fac'].astype(np.float32), d['target_fac'].astype(np.float32)
        valid = d['valid'].astype(bool)
        if ens is None:
            ens = {'pa': np.zeros_like(pa), 'ta': ta, 'pf': np.zeros_like(pf),
                   'tf': tf, 'valid': valid}
        else:
            assert np.array_equal(d['target_air'], ens['ta']) and np.array_equal(d['valid'], ens['valid'])
        ens['pa'] += pa; ens['pf'] += pf
        for vi, (k, (mu, sd)) in enumerate(AIR_KEYS):
            per_seed_air[si, vi] = np.mean(np.abs((pa[:, vi] * sd + mu) - (ta[:, vi] * sd + mu)))
        for vi, k in enumerate(FAC_KEYS):
            per_seed_fac[si, vi] = np.mean(np.abs((pf[valid, vi] * fs[vi] + fm[vi])
                                                  - (tf[valid, vi] * fs[vi] + fm[vi])))
        d.close()
    ens['pa'] /= len(SEEDS); ens['pf'] /= len(SEEDS)

    res = {'per_seed_air_mae': {k: [round(float(per_seed_air[si, vi]), 4) for si in range(len(SEEDS))]
                                for vi, (k, _) in enumerate(AIR_KEYS)},
           'per_seed_fac_mae': {k: [round(float(per_seed_fac[si, vi]), 4) for si in range(len(SEEDS))]
                                for vi, k in enumerate(FAC_KEYS)},
           'air': {}, 'facade': {}}
    print(f'\n===== [{site}] JOINT (pooled) - per-seed AIR MAE (seeds {list(SEEDS)}) =====')
    hdr = f'{"Variable":10}' + ''.join(f'{"s"+str(s):>10}' for s in SEEDS) + f'{"mean":>10} {"std":>10}'
    print(hdr)
    for vi, (k, (mu, sd)) in enumerate(AIR_KEYS):
        vals = per_seed_air[:, vi]
        print(f'{k:10}' + ''.join(f'{v:>10.3f}' for v in vals)
              + f'{vals.mean():>10.3f} {vals.std(ddof=1):>10.3f}')
    print(f'----- [{site}] JOINT - per-seed FACADE MAE -----'); print(hdr)
    for vi, k in enumerate(FAC_KEYS):
        vals = per_seed_fac[:, vi]
        print(f'{k:10}' + ''.join(f'{v:>10.3f}' for v in vals)
              + f'{vals.mean():>10.3f} {vals.std(ddof=1):>10.3f}')

    n_tot = ens['ta'].shape[0]; valid = ens['valid']
    print(f'\n[{site}] JOINT ensemble3 vs references (physical MAE):')
    print(f'{"Variable":10}{"joint3":>10}{"scratch5":>11}{"ft_other":>11}{"vs_scratch":>12}')
    for vi, (k, (mu, sd)) in enumerate(AIR_KEYS):
        p = ens['pa'][:, vi] * sd + mu; t = ens['ta'][:, vi] * sd + mu
        mae, rmse = mae_rmse(p, t)
        ref5 = REF[site]['scratch5'][k]
        ftref = REF[site].get('ft_from_plaza' if site == 'canyon' else 'ft_from_canyon', {}).get(k, float('nan'))
        res['air'][k] = {'mae': round(mae, 4), 'rmse': round(rmse, 4), 'scratch5_ref': ref5,
                         'ft_ref': ftref, 'vs_scratch_pct': round((mae / ref5 - 1) * 100, 1),
                         'unit': AIR_UNITS[k]}
        print(f'{k:10}{mae:>10.3f}{ref5:>11.3f}{ftref:>11.3f}{(mae/ref5-1)*100:>+11.1f}%')
    for vi, k in enumerate(FAC_KEYS):
        p = ens['pf'][valid, vi] * fs[vi] + fm[vi]; t = ens['tf'][valid, vi] * fs[vi] + fm[vi]
        mae, rmse = mae_rmse(p, t)
        ref5 = REF[site]['scratch5'][k]
        ftref = REF[site].get('ft_from_plaza' if site == 'canyon' else 'ft_from_canyon', {}).get(k, float('nan'))
        res['facade'][k] = {'mae': round(mae, 4), 'rmse': round(rmse, 4), 'scratch5_ref': ref5,
                            'ft_ref': ftref, 'vs_scratch_pct': round((mae / ref5 - 1) * 100, 1),
                            'unit': FAC_UNITS[k]}
        print(f'{k:10}{mae:>10.3f}{ref5:>11.3f}{ftref:>11.3f}{(mae/ref5-1)*100:>+11.1f}%')
    summary['sites'][site] = res

    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    for head in ('air', 'facade'):
        for k, r in res[head].items():
            csv_rows.append([ts, site, 'p5e_joint_ens3', 'mean', head, k, n_tot,
                             f'{r["mae"]:.4f}', f'{r["rmse"]:.4f}'])

jpath = MET_DIR / 'p5e_joint_summary.json'
jpath.write_text(json.dumps(summary, indent=2))
new = not mlog.exists()
with open(mlog, 'a', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(['ts', 'site', 'config', 'seed', 'head', 'var', 'n', 'MAE', 'RMSE'])
    w.writerows(csv_rows)
print(f'\n[SAVED] {jpath.name}')
print(f'[SAVED] {mlog.name} (+{len(csv_rows)} rows)')
print(f'\n===== P5e DONE in {(time.time()-t_all)/60:.1f} min - joint multi-site study complete =====')
print('NEXT: backup #4 cell -> download -> GPU phase COMPLETE (P6 figures are CPU-only).')
