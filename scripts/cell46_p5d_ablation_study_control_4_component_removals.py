# ===== CELL P5d - ABLATION STUDY (control + 4 component removals, seed 0, both sites) =====
# Prereqs: CELL 18b -> CELL 19 in THIS SAME session -> THIS.  GPU T4.  Resumable per unit.
# Arms (all identical protocol = S3e scratch: optuna_best config lr=1.542e-3 fw=0.3105,
#       30 epochs patience 5, 500k/300k subset per epoch, batch 128, seed 0):
#   control     : full model (reference for all deltas)
#   no_dist     : V-DEI distance channel zeroed      -> value of geometry-decay info
#   no_sunblock : sun-block forcing dim zeroed       -> value of shading awareness
#   no_forcing  : all 8 forcing dims zeroed          -> spatial-only model (no temporal context)
#   no_facade   : facade loss weight = 0             -> multi-task benefit (facade head untrained)
# Est. T4 time: ~3.5-4.5 h. Resumable: checkpoint -> skip train; npz -> skip eval.
import csv, json, time, glob
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

AIR_KEYS = [('T', (28.0, 4.0)), ('RelHum', (40.0, 15.0)), ('WindSpd', (2.0, 1.5)),
            ('TKE', (50.0, 100.0)), ('TMRT', (45.0, 20.0))]
FAC_KEYS = ['Twall', 'Qsens', 'SWabs', 'LWbal']

SITES   = ('canyon', 'plaza')
ARMS    = ('control', 'no_dist', 'no_sunblock', 'no_forcing', 'no_facade')
SEED    = 0
CFG     = 'optuna_best'
N_EPOCHS, PATIENCE, VAL_SUBSET = 30, 5, 100_000
LR      = CONFIGS['optuna_best']['lr']
LAM     = CONFIGS['optuna_best']['facade_weight']
BATCH   = 128
EP_BASE = {'canyon': 500_000, 'plaza': 300_000}   # S3e per-epoch subsets (full base)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise SystemExit('GPU session required - stay in THIS GPU session (18b + 19 already run).')
try:
    _ = (torch.ones(1, device=device) * 2).sum().item()
except RuntimeError as e:
    raise SystemExit(f'GPU check failed: {e}')
print(f'device: {torch.cuda.get_device_name(0)} ✅ | arms: {ARMS}')
print(f'protocol: lr={LR:.3e} fw={LAM} batch={BATCH} ep={N_EPOCHS} patience={PATIENCE} '
      f'subset/epoch={EP_BASE} seed={SEED}')


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


def apply_arm(arm, v, f):
    """Ablation transform on a fresh GPU batch (collate makes new tensors, safe)."""
    if arm == 'control':
        return v, f
    if arm == 'no_dist':
        v = v.clone(); v[..., 4] = 0.0
        return v, f
    if arm == 'no_sunblock':
        f = f.clone(); f[..., 7] = 0.0
        return v, f
    if arm == 'no_forcing':
        return v, torch.zeros_like(f)
    if arm == 'no_facade':
        return v, f          # inputs unchanged - the ablation is lam=0 in the loss
    raise ValueError(arm)


def facade_stats(site):
    fc = np.load(processed_dir(site) / f'facade_targets_{site}.npz')
    mu = fc['facade_norm_mean'].astype(np.float32)
    sd = fc['facade_norm_std'].astype(np.float32)
    fc.close()
    return mu, sd


FSTAT = {s: facade_stats(s) for s in SITES}
print('facade norm stats loaded for both sites')


def train_arm(SITE, arm):
    lam = 0.0 if arm == 'no_facade' else LAM
    ckpt = output(f'03_Results/01_ModelOutputs/{SITE}_dualhead_p5d_{arm}_seed{SEED}_best.pt')
    if ckpt.exists():
        print(f'[{SITE} {arm}] checkpoint exists - skip training')
        return
    seed_all(SEED)
    model, _ = build(CFG)
    model.to(device)
    train_ds = VDEIDatasetV2(SITE, 'train')
    val_ds   = VDEIDatasetV2(SITE, 'val')
    ep_samples = min(EP_BASE[SITE], len(train_ds))
    g = np.random.RandomState(0)
    val_idx = g.choice(len(val_ds), size=min(VAL_SUBSET, len(val_ds)), replace=False)
    val_loader = DataLoader(torch.utils.data.Subset(val_ds, val_idx), batch_size=BATCH,
                            shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    g = np.random.RandomState(SEED)
    subset_idx = g.choice(len(train_ds), size=ep_samples, replace=False)
    train_loader = DataLoader(torch.utils.data.Subset(train_ds, subset_idx), batch_size=BATCH,
                              shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')
    best, bad = float('inf'), 0
    t00 = time.time()
    for ep in range(1, N_EPOCHS + 1):
        t0 = time.time(); model.train(); tr = 0.0
        for b in train_loader:
            v = b['vdei'].to(device, non_blocking=True)
            f = b['forcing'].to(device, non_blocking=True)
            v, f = apply_arm(arm, v, f)
            y  = b['target'].to(device, non_blocking=True)
            fy = b['facade_target'].to(device, non_blocking=True)
            fv = b['facade_valid'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                pa, pf = model(v, f)
            loss, _, _ = composite(pa.float(), pf.float(), y, fy, fv, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            tr += float(loss.detach())
        tr /= len(train_loader)
        model.eval(); va = 0.0
        with torch.no_grad():
            for b in val_loader:
                v = b['vdei'].to(device, non_blocking=True)
                f = b['forcing'].to(device, non_blocking=True)
                v, f = apply_arm(arm, v, f)
                y  = b['target'].to(device, non_blocking=True)
                fy = b['facade_target'].to(device, non_blocking=True)
                fv = b['facade_valid'].to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    pa, pf = model(v, f)
                loss, _, _ = composite(pa.float(), pf.float(), y, fy, fv, lam)
                va += float(loss.detach())
        va /= len(val_loader)
        scheduler.step(va)
        print(f'[{SITE} {arm}] Ep {ep:2d}/{N_EPOCHS} | train={tr:.4f} | val={va:.4f} '
              f'| lr={optimizer.param_groups[0]["lr"]:.2e} | {time.time()-t0:.1f}s', flush=True)
        if va < best:
            best, bad = va, 0
            torch.save({'model_state': model.state_dict(), 'config': CFG, 'arm': arm,
                        'epoch': ep, 'val_loss': va, 'seed': SEED, 'lr': LR, 'lam': lam,
                        'n_params': sum(p.numel() for p in model.parameters())}, ckpt)
            print('  ✅ new best saved')
        else:
            bad += 1
            if bad >= PATIENCE:
                print('  ⏹ early stopping'); break
    print(f'[{SITE}] {arm} trained in {(time.time()-t00)/60:.1f} min | best val={best:.4f}')


def eval_arm(SITE, arm):
    out = output(f'03_Results/02_Metrics/{SITE}_dualhead_p5d_{arm}_seed{SEED}_test_fullpool.npz')
    if out.exists():
        print(f'[{SITE} {arm}] fullpool npz exists - skip eval')
        return
    ckpt = output(f'03_Results/01_ModelOutputs/{SITE}_dualhead_p5d_{arm}_seed{SEED}_best.pt')
    assert ckpt.exists(), f'checkpoint missing: {ckpt.name} - train first!'
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model, _ = build(ck.get('config', CFG))
    model.load_state_dict(ck['model_state'])
    model.to(device).eval()
    test_ds = VDEIDatasetV2(SITE, 'test')
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    PA, TA, PF, TF, FV = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for b in loader:
            v = b['vdei'].to(device, non_blocking=True)
            f = b['forcing'].to(device, non_blocking=True)
            v, f = apply_arm(arm, v, f)
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
    print(f'[{SITE}] {arm} eval done in {(time.time()-t0)/60:.1f} min -> {out.name} '
          f'({out.stat().st_size/1e6:.0f} MB)')


# ---- Phase 1: train + eval every arm ------------------------------------------
t_all = time.time()
for SITE in SITES:
    for arm in ARMS:
        train_arm(SITE, arm)
        eval_arm(SITE, arm)


# ---- Phase 2: summary tables (physical MAE, deltas vs control) -----------------
def phys_site(site, arm):
    p = output(f'03_Results/02_Metrics/{site}_dualhead_p5d_{arm}_seed{SEED}_test_fullpool.npz')
    d = np.load(p)
    pa, ta = d['pred_air'].astype(np.float32), d['target_air'].astype(np.float32)
    valid = d['valid'].astype(bool)
    pf, tf = d['pred_fac'][valid].astype(np.float32), d['target_fac'][valid].astype(np.float32)
    d.close()
    a = {k: float(np.abs(pa[:, i] - ta[:, i]).mean()) * float(sd)
         for i, (k, (mu, sd)) in enumerate(AIR_KEYS)}
    if arm == 'no_facade':
        f = {k: float('nan') for k in FAC_KEYS}     # facade head untrained in this arm
    else:
        fmu, fsd = FSTAT[site]
        f = {k: float(np.abs(pf[:, i] - tf[:, i]).mean()) * float(fsd[i])
             for i, k in enumerate(FAC_KEYS)}
    return a, f


def find_input(pat):
    for d in sorted(Path('/kaggle/input').glob('*')):
        h = sorted(d.rglob(pat))
        if h:
            return h[0]
    return None


summary = {'meta': {'config': CFG, 'seed': SEED, 'lr': LR, 'lam': LAM, 'batch': BATCH,
                    'epochs': N_EPOCHS, 'patience': PATIENCE, 'ep_base': EP_BASE,
                    'arms': list(ARMS), 'computed_at': datetime.now().strftime('%Y-%m-%d %H:%M')},
           'sites': {}}
csv_rows = []
mlog = output('03_Results/02_Metrics/test_metrics_log.csv')

for site in SITES:
    res = {arm: phys_site(site, arm) for arm in ARMS}
    summary['sites'][site] = res
    ca, cf = res['control']
    print(f'\n===== [{site}] P5d ABLATION - AIR MAE (physical units, seed {SEED}) =====')
    print(f'{"arm":12}' + ''.join(f'{k:>12}' for k, _ in AIR_KEYS))
    for arm in ARMS:
        a, _ = res[arm]
        row = f'{arm:12}'
        for k, _ in AIR_KEYS:
            v = a[k]
            if arm == 'control':
                row += f'{v:>12.3f}'
            else:
                row += f'{v:>7.3f}{(v / ca[k] - 1) * 100:+5.1f}%'
        print(row)
    print(f'----- [{site}] P5d ABLATION - FACADE MAE (n/a for no_facade) -----')
    print(f'{"arm":12}' + ''.join(f'{k:>12}' for k in FAC_KEYS))
    for arm in ARMS:
        _, f = res[arm]
        row = f'{arm:12}'
        for k in FAC_KEYS:
            if np.isnan(f[k]):
                row += f'{"n/a":>12}'
            else:
                v = f[k]
                if arm == 'control':
                    row += f'{v:>12.3f}'
                else:
                    row += f'{v:>7.3f}{(v / cf[k] - 1) * 100:+5.1f}%'
        print(row)
    ref = find_input(f'{site}_dualhead_optuna_best_seed0_test_fullpool.npz')
    if ref is not None:
        d = np.load(ref)
        pa, ta = d['pred_air'].astype(np.float32), d['target_air'].astype(np.float32)
        valid = d['valid'].astype(bool)
        pf, tf = d['pred_fac'][valid].astype(np.float32), d['target_fac'][valid].astype(np.float32)
        d.close()
        ra = {k: float(np.abs(pa[:, i] - ta[:, i]).mean()) * sd for i, (k, (mu, sd)) in enumerate(AIR_KEYS)}
        fmu, fsd = FSTAT[site]
        rf = {k: float(np.abs(pf[:, i] - tf[:, i]).mean()) * fsd[i] for i, k in enumerate(FAC_KEYS)}
        print(f'{"native s0":12}' + ''.join(f'{ra[k]:>12.3f}' for k, _ in AIR_KEYS)
              + '   (S3e scratch reference)')
        print(f'{"native s0":12}' + ''.join(f'{rf[k]:>12.3f}' for k in FAC_KEYS) + '   (facade)')
    for arm in ARMS:
        a, f = res[arm]
        for k, v in a.items():
            csv_rows.append([datetime.now().isoformat(timespec='seconds'), site, f'p5d_{arm}', SEED, 'air', k, '', f'{v:.4f}', ''])
        for k, v in f.items():
            if not np.isnan(v):
                csv_rows.append([datetime.now().isoformat(timespec='seconds'), site, f'p5d_{arm}', SEED, 'facade', k, '', f'{v:.4f}', ''])

jpath = output('03_Results/02_Metrics/p5d_ablation_summary.json')
jpath.write_text(json.dumps(summary, indent=2))
new = not mlog.exists()
with open(mlog, 'a', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(['ts', 'site', 'config', 'seed', 'head', 'var', 'n', 'MAE', 'RMSE'])
    w.writerows(csv_rows)
print(f'\n[SAVED] {jpath.name}')
print(f'[SAVED] {mlog.name} (+{len(csv_rows)} rows)')
print(f'\n===== P5d DONE in {(time.time()-t_all)/60:.1f} min - ablation study complete (5 arms x 2 sites) =====')
print('NEXT: backup #3 cell -> download -> then P6 figures (CPU) or optional joint training.')
