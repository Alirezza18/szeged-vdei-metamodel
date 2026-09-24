# ===== CELL P5c (S6) - CROSS-SITE TRANSFER LEARNING: zero-shot + fine-tune =====
# Headline config: optuna_best (locked 2026-09-07). Prereqs: CELL 18b -> CELL 19 -> THIS. GPU T4.
# Reads the 10 sealed optuna_best checkpoints from the szeged_backup Input (zero-shot = no training).
# Per direction (canyon->plaza, plaza->canyon) x seeds 0..4:
#   ZS zero-shot : source model evaluated on the TARGET full test pool
#   FT fine-tune : same weights, lr/10, <=15 epochs (patience 3), S3e subset protocol on TARGET train
# Resumable: every existing artifact is skipped. Facade MAE: ZS preds denormed with SOURCE stats,
#   FT with TARGET stats; targets always TARGET stats. Air uses shared manuscript stats.
# Est. T4 time: ZS ~1.4 h + FT ~3.5 h = ~5 h total.

import json, csv, time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

W = Path('/kaggle/working'); IN = Path('/kaggle/input')
MET_DIR = W / '03_Results' / '02_Metrics'; MET_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = W / '03_Results' / '01_ModelOutputs'; CKPT_DIR.mkdir(parents=True, exist_ok=True)
SITES = ('canyon', 'plaza'); SEEDS = (0, 1, 2, 3, 4)
HEAD_CFG = 'optuna_best'
FT_EPOCHS, FT_PATIENCE, FT_LR_DIV, FT_BATCH = 15, 3, 10, 256
TRAIN_SAMPLES = {'canyon': 500_000, 'plaza': 300_000}
VAL_SUBSET = 100_000
AIR_KEYS = [('T', (28.0, 4.0)), ('RelHum', (40.0, 15.0)), ('WindSpd', (2.0, 1.5)),
            ('TKE', (50.0, 100.0)), ('TMRT', (45.0, 20.0))]
FAC_KEYS = ['Twall', 'Qsens', 'SWabs', 'LWbal']
AIR_MU = np.array([mu for _, (mu, sd) in AIR_KEYS], np.float32)
AIR_SD = np.array([sd for _, (mu, sd) in AIR_KEYS], np.float32)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise SystemExit('GPU session required (Settings -> Accelerator -> GPU T4 x2).')
_ = (torch.ones(1, device=DEVICE) * 2).sum().item()
print(f'device: {torch.cuda.get_device_name(0)} OK', flush=True)

def _find(pattern):
    for base in (MET_DIR, CKPT_DIR):
        hits = sorted(base.glob(pattern))
        if hits: return hits[0]
    for d in sorted(IN.glob('*')):
        hits = sorted(d.rglob(pattern))
        if hits: return hits[0]
    raise FileNotFoundError(f'{pattern} not found - check attached Inputs!')

def seed_all(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

FSTATS = {}
for site in SITES:
    fc = np.load(_find(f'facade_targets_{site}.npz'))
    FSTATS[site] = (fc['facade_norm_mean'].astype(np.float32),
                    fc['facade_norm_std'].astype(np.float32))
    fc.close()
print('facade norm stats loaded for both sites', flush=True)

def composite(pa, pf, y, fy, fv, lam):
    la = ((pa - y) ** 2).mean()
    per = ((pf - fy) ** 2).mean(dim=1)
    den = fv.sum()
    lf = (per * fv).sum() / den if den > 0 else pa.new_zeros(())
    return la + lam * lf

def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model, cfg = build(ck.get('config', HEAD_CFG))
    model.load_state_dict(ck['model_state'])
    return model.to(DEVICE).eval(), cfg

def eval_fullpool(target, model, tag, seed):
    out = MET_DIR / f'{target}_dualhead_{tag}_seed{seed}_test_fullpool.npz'
    if out.exists():
        print(f'[{target}] {tag} s{seed}: npz exists - skip eval'); return
    t0 = time.time()
    test_ds = VDEIDatasetV2(target, 'test')
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    PA, TA, PF, TF, FV = [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            v = b['vdei'].to(DEVICE, non_blocking=True)
            f = b['forcing'].to(DEVICE, non_blocking=True)
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
    print(f'[{target}] {tag} s{seed}: eval done in {(time.time()-t0)/60:.1f} min -> {out.name}', flush=True)

def finetune(src, tgt, seed):
    ckpt = CKPT_DIR / f'{tgt}_dualhead_p5c_ft_from{src}_seed{seed}_best.pt'
    if ckpt.exists():
        print(f'[{tgt}] ft_from{src} s{seed}: ckpt exists - skip training'); return ckpt
    t0 = time.time(); seed_all(seed)
    model, cfg = load_model(_find(f'{src}_dualhead_{HEAD_CFG}_seed{seed}_best.pt'))
    lam, lr = cfg['facade_weight'], cfg['lr'] / FT_LR_DIV
    train_ds = VDEIDatasetV2(tgt, 'train'); val_ds = VDEIDatasetV2(tgt, 'val')
    ep_samples = min(TRAIN_SAMPLES[tgt], len(train_ds))
    g = np.random.RandomState(0)
    val_idx = g.choice(len(val_ds), size=min(VAL_SUBSET, len(val_ds)), replace=False)
    val_loader = DataLoader(torch.utils.data.Subset(val_ds, val_idx), batch_size=FT_BATCH,
                            shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    g = np.random.RandomState(seed)
    sub = g.choice(len(train_ds), size=ep_samples, replace=False)
    train_loader = DataLoader(torch.utils.data.Subset(train_ds, sub), batch_size=FT_BATCH,
                              shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=1)
    scaler = torch.amp.GradScaler('cuda')
    best, bad = float('inf'), 0
    for ep in range(1, FT_EPOCHS + 1):
        model.train(); tr = 0.0
        for b in train_loader:
            v = b['vdei'].to(DEVICE, non_blocking=True)
            f = b['forcing'].to(DEVICE, non_blocking=True)
            y = b['target'].to(DEVICE, non_blocking=True)
            fy = b['facade_target'].to(DEVICE, non_blocking=True)
            fv = b['facade_valid'].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                pa, pf = model(v, f)
            loss = composite(pa.float(), pf.float(), y, fy, fv, lam)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); tr += float(loss)
        model.eval(); va = 0.0
        with torch.no_grad():
            for b in val_loader:
                v = b['vdei'].to(DEVICE, non_blocking=True)
                f = b['forcing'].to(DEVICE, non_blocking=True)
                y = b['target'].to(DEVICE, non_blocking=True)
                fy = b['facade_target'].to(DEVICE, non_blocking=True)
                fv = b['facade_valid'].to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    pa, pf = model(v, f)
                va += float(composite(pa.float(), pf.float(), y, fy, fv, lam))
        va /= len(val_loader); sched.step(va)
        print(f'[{tgt} ft_from{src} s{seed}] Ep {ep:2d}/{FT_EPOCHS} | train={tr:.4f} '
              f'| val={va:.4f} | lr={opt.param_groups[0]["lr"]:.2e} | {time.time()-t0:.0f}s', flush=True)
        if va < best:
            best, bad = va, 0
            torch.save({'model_state': model.state_dict(), 'config': HEAD_CFG, 'epoch': ep,
                        'val_loss': va, 'seed': seed, 'fine_tuned_from': src,
                        'n_params': sum(p.numel() for p in model.parameters())}, ckpt)
            print('  ✅ new best saved')
        else:
            bad += 1
            if bad >= FT_PATIENCE:
                print('  ⏹ early stopping'); break
    print(f'[{tgt}] ft_from{src} s{seed}: trained in {(time.time()-t0)/60:.1f} min | best val={best:.4f}')
    return ckpt

# ---- Phase 1: zero-shot eval + fine-tune + eval, all seeds, both directions ----
t00 = time.time()
for src in SITES:
    tgt = 'plaza' if src == 'canyon' else 'canyon'
    print(f'\n########## DIRECTION {src} -> {tgt} ##########', flush=True)
    for s in SEEDS:
        zs_tag, ft_tag = f'p5c_zs_from{src}', f'p5c_ft_from{src}'
        if not (MET_DIR / f'{tgt}_dualhead_{zs_tag}_seed{s}_test_fullpool.npz').exists():
            model, _ = load_model(_find(f'{src}_dualhead_{HEAD_CFG}_seed{s}_best.pt'))
            eval_fullpool(tgt, model, zs_tag, s)
            del model; torch.cuda.empty_cache()
        ft_ck = finetune(src, tgt, s)
        if not (MET_DIR / f'{tgt}_dualhead_{ft_tag}_seed{s}_test_fullpool.npz').exists():
            model, _ = load_model(ft_ck)
            eval_fullpool(tgt, model, ft_tag, s)
            del model; torch.cuda.empty_cache()

# ---- Phase 2: per-direction ensembles + mean±std + summary ----
mlog = MET_DIR / 'test_metrics_log.csv'
ts = datetime.now().strftime('%Y-%m-%d %H:%M')
summary = {'meta': {'headline_config': HEAD_CFG, 'seeds': list(SEEDS),
                    'ft': {'epochs': FT_EPOCHS, 'patience': FT_PATIENCE, 'lr_divisor': FT_LR_DIV,
                           'batch': FT_BATCH}, 'computed_at': ts}}
csv_rows = []

def summarize(tgt, src, mode):
    tag = f'p5c_{mode}_from{src}'
    ds = [np.load(MET_DIR / f'{tgt}_dualhead_{tag}_seed{s}_test_fullpool.npz') for s in SEEDS]
    valid = ds[0]['valid'].astype(bool)
    for d in ds[1:]:
        assert np.array_equal(d['valid'], ds[0]['valid']), f'{tag}: valid mismatch'
    targ_a, targ_f = ds[0]['target_air'], ds[0]['target_fac']
    ta = targ_a.astype(np.float32) * AIR_SD + AIR_MU
    tf = targ_f[valid].astype(np.float32) * FSTATS[tgt][1] + FSTATS[tgt][0]
    n_tot, n_fac = targ_a.shape[0], int(valid.sum())
    pm = FSTATS[src] if mode == 'zs' else FSTATS[tgt]
    psa = np.zeros((len(SEEDS), 5)); psf = np.zeros((len(SEEDS), 4))
    ens_a = np.zeros_like(targ_a, np.float32); ens_f = np.zeros_like(targ_f, np.float32)
    for i, d in enumerate(ds):
        pa = d['pred_air'].astype(np.float32) * AIR_SD + AIR_MU
        psa[i] = np.abs(pa - ta).mean(axis=0)
        pf = d['pred_fac'][valid].astype(np.float32) * pm[1] + pm[0]
        psf[i] = np.abs(pf - tf).mean(axis=0)
        ens_a += d['pred_air'].astype(np.float32); ens_f += d['pred_fac'].astype(np.float32)
    ens_a /= len(SEEDS); ens_f /= len(SEEDS)
    for d in ds:
        d.close()

    print(f'\n===== [{tgt}] {mode.upper()} from {src} - per-seed AIR MAE =====')
    print(f'{"Variable":10}' + ''.join(f'{"s"+str(s):>9}' for s in SEEDS) + f'{"mean":>9} {"std":>9}')
    for vi, (var, _) in enumerate(AIR_KEYS):
        v = psa[:, vi]
        print(f'{var:10}' + ''.join(f'{x:>9.3f}' for x in v) + f'{v.mean():>9.3f} {v.std(ddof=1):>9.3f}')
    print(f'----- [{tgt}] {mode.upper()} from {src} - per-seed FACADE MAE -----')
    print(f'{"Variable":10}' + ''.join(f'{"s"+str(s):>9}' for s in SEEDS) + f'{"mean":>9} {"std":>9}')
    for vi, var in enumerate(FAC_KEYS):
        v = psf[:, vi]
        print(f'{var:10}' + ''.join(f'{x:>9.3f}' for x in v) + f'{v.mean():>9.3f} {v.std(ddof=1):>9.3f}')

    pa = ens_a * AIR_SD + AIR_MU
    pf = ens_f[valid] * pm[1] + pm[0]
    print(f'\n[{tgt}] {mode.upper()} from {src} - ENSEMBLE5 (physical units):')
    block = {'air': {}, 'facade': {}}
    for vi, (var, _) in enumerate(AIR_KEYS):
        diff = pa[:, vi] - ta[:, vi]
        mae, rmse = float(np.abs(diff).mean()), float(np.sqrt((diff ** 2).mean()))
        unit = {'T': 'degC', 'RelHum': '%', 'WindSpd': 'm/s', 'TKE': 'm2/s2', 'TMRT': 'degC'}[var]
        print(f'  {var:8} MAE={mae:8.3f}  RMSE={rmse:8.3f}  {unit}')
        block['air'][var] = {'mae': round(mae, 4), 'rmse': round(rmse, 4),
                             'per_seed_mae_mean': round(float(psa[:, vi].mean()), 4),
                             'per_seed_mae_std': round(float(psa[:, vi].std(ddof=1)), 4)}
    for vi, var in enumerate(FAC_KEYS):
        diff = pf[:, vi] - tf[:, vi]
        mae, rmse = float(np.abs(diff).mean()), float(np.sqrt((diff ** 2).mean()))
        unit = {'Twall': 'degC', 'Qsens': 'W/m2', 'SWabs': 'W/m2', 'LWbal': 'W/m2'}[var]
        print(f'  {var:8} MAE={mae:8.3f}  RMSE={rmse:8.3f}  {unit}')
        block['facade'][var] = {'mae': round(mae, 4), 'rmse': round(rmse, 4),
                                'per_seed_mae_mean': round(float(psf[:, vi].mean()), 4),
                                'per_seed_mae_std': round(float(psf[:, vi].std(ddof=1)), 4)}

    out = MET_DIR / f'{tgt}_dualhead_{tag}_ensemble5_test_fullpool.npz'
    np.savez(out, pred_air=ens_a.astype(np.float16), target_air=targ_a,
             pred_fac=ens_f.astype(np.float16), target_fac=targ_f, valid=valid)
    print(f'[SAVED] {out.name}')

    for head, keys, mat, n in (('air', AIR_KEYS, psa, n_tot), ('facade', FAC_KEYS, psf, n_fac)):
        for si, s in enumerate(SEEDS):
            for vi, key in enumerate(keys):
                var = key if isinstance(key, str) else key[0]
                csv_rows.append([ts, tgt, f'{tag}', s, head, var, n, f'{mat[si, vi]:.4f}', ''])
    for vi, (var, _) in enumerate(AIR_KEYS):
        csv_rows.append([ts, tgt, f'{tag}_ens5', 'mean', 'air', var, n_tot,
                         f'{block["air"][var]["mae"]:.4f}', f'{block["air"][var]["rmse"]:.4f}'])
    for vi, var in enumerate(FAC_KEYS):
        csv_rows.append([ts, tgt, f'{tag}_ens5', 'mean', 'facade', var, n_fac,
                         f'{block["facade"][var]["mae"]:.4f}', f'{block["facade"][var]["rmse"]:.4f}'])
    return block

for src in SITES:
    tgt = 'plaza' if src == 'canyon' else 'canyon'
    summary[f'{src}->{tgt}'] = {'zs': summarize(tgt, src, 'zs'),
                                'ft': summarize(tgt, src, 'ft')}

jpath = MET_DIR / 'p5c_transfer_summary.json'
jpath.write_text(json.dumps(summary, indent=2))
new = not mlog.exists()
with open(mlog, 'a', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(['ts', 'site', 'config', 'seed', 'head', 'var', 'n', 'MAE', 'RMSE'])
    w.writerows(csv_rows)
print(f'\n[SAVED] {jpath.name}')
print(f'[SAVED] {mlog.name}')
print(f'\n===== P5c DONE in {(time.time()-t00)/60:.1f} min - transfer study complete (both directions, ZS+FT) =====')
print('NEXT: backup cell -> download -> P5d ablations (next GPU session).')
