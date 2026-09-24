# ===== CELL 26 (S3e) - seed-0 recovery + 5-seed ensemble + mean±std =====
# Rebuilds missing seed-0 artifacts (train ~23 min canyon / ~12 min plaza ONLY if
# the checkpoint is gone; eval ~10/~6.5 min if only the npz is gone), then runs
# the 5-seed ensemble. All steps skip when the artifact already exists.
import csv, time
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

SEEDS = (0, 1, 2, 3, 4)
CONFIG = 'optuna_best'
BATCH = 256
PATIENCE = 5
N_EPOCHS = 30
VAL_SUBSET = 100_000

AIR_KEYS = [('T', (28.0, 4.0)), ('RelHum', (40.0, 15.0)), ('WindSpd', (2.0, 1.5)),
            ('TKE', (50.0, 100.0)), ('TMRT', (45.0, 20.0))]
FAC_KEYS = ['Twall', 'Qsens', 'SWabs', 'LWbal']
AIR_UNITS = {'T': 'degC', 'RelHum': '%', 'WindSpd': 'm/s', 'TKE': 'm2/s2', 'TMRT': 'degC'}
FAC_UNITS = {'Twall': 'degC', 'Qsens': 'W/m2', 'SWabs': 'W/m2', 'LWbal': 'W/m2'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise SystemExit('GPU session required (Settings -> Accelerator -> GPU T4 x2).')
try:
    _ = (torch.ones(1, device=device) * 2).sum().item()
    print(f'device: {torch.cuda.get_device_name(0)} ✅ CUDA kernels OK')
except RuntimeError as e:
    raise SystemExit('❌ GPU not supported by installed PyTorch (e.g., P100/sm_60). Use GPU T4 x2.') from e

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

def train_seed0(SITE):
    """Rebuild the seed-0 checkpoint (same protocol as CELL 24/CELL 21/22)."""
    ckpt_path = output(f'03_Results/01_ModelOutputs/{SITE}_dualhead_{CONFIG}_seed0_best.pt')
    if ckpt_path.exists():
        print(f'[{SITE}] seed 0 checkpoint exists - skip training')
        return
    print(f'[{SITE}] seed 0 checkpoint MISSING - retraining (~similar to CELL 24 timing)')
    seed_all(0)
    model, cfg = build(CONFIG)
    model.to(device)
    lam = cfg['facade_weight']
    train_ds = VDEIDatasetV2(SITE, 'train')
    val_ds = VDEIDatasetV2(SITE, 'val')
    EP_SAMPLES = {'canyon': 500_000, 'plaza': 300_000}[SITE]
    g = np.random.RandomState(0)
    val_idx = g.choice(len(val_ds), size=min(VAL_SUBSET, len(val_ds)), replace=False)
    val_loader = DataLoader(torch.utils.data.Subset(val_ds, val_idx), batch_size=BATCH,
                            shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    g = np.random.RandomState(SEED)
    subset_idx = g.choice(len(train_ds), size=min(EP_SAMPLES, len(train_ds)), replace=False)
    subset = torch.utils.data.Subset(train_ds, subset_idx)
    train_loader = DataLoader(subset, batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')
    best, bad = float('inf'), 0
    t00 = time.time()
    for ep in range(1, N_EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr, tr_a, tr_f = 0.0, 0.0, 0.0
        for b in train_loader:
            v = b['vdei'].to(device, non_blocking=True)
            f = b['forcing'].to(device, non_blocking=True)
            y = b['target'].to(device, non_blocking=True)
            fy = b['facade_target'].to(device, non_blocking=True)
            fv = b['facade_valid'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                pa, pf = model(v, f)
            loss, la, lf = composite(pa.float(), pf.float(), y, fy, fv, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            tr += float(loss); tr_a += float(la); tr_f += float(lf)
        nb = len(train_loader)
        tr, tr_a, tr_f = tr / nb, tr_a / nb, tr_f / nb
        model.eval()
        va, va_a, va_f = 0.0, 0.0, 0.0
        with torch.no_grad():
            for b in val_loader:
                v = b['vdei'].to(device, non_blocking=True)
                f = b['forcing'].to(device, non_blocking=True)
                y = b['target'].to(device, non_blocking=True)
                fy = b['facade_target'].to(device, non_blocking=True)
                fv = b['facade_valid'].to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    pa, pf = model(v, f)
                loss, la, lf = composite(pa.float(), pf.float(), y, fy, fv, lam)
                va += float(loss); va_a += float(la); va_f += float(lf)
        nv = len(val_loader)
        va, va_a, va_f = va / nv, va_a / nv, va_f / nv
        scheduler.step(va)
        print(f'Epoch {ep:2d}/{N_EPOCHS} | train={tr:.4f} | val={va:.4f} (air={va_a:.4f} fac={va_f:.4f}) '
              f'| lr={optimizer.param_groups[0]["lr"]:.2e} | {time.time()-t0:.1f}s', flush=True)
        if va < best:
            best = va
            bad = 0
            torch.save({'model_state': model.state_dict(), 'config': CONFIG, 'epoch': ep,
                        'val_loss': va, 'seed': 0, 'n_params': sum(p.numel() for p in model.parameters())},
                       ckpt_path)
            print(f'  ✅ new best saved (val_loss={va:.4f})')
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f'  ⏹ early stopping after {ep} epochs')
                break
    print(f'[{SITE}] seed 0 retrained in {(time.time()-t00)/60:.1f} min | best val={best:.4f}')

def eval_seed(SITE, SEED):
    """Full-pool eval (same as CELL 21/23/25). Skips when the npz exists."""
    out = output(f'03_Results/02_Metrics/{SITE}_dualhead_{CONFIG}_seed{SEED}_test_fullpool.npz')
    if out.exists():
        print(f'[{SITE}] seed {SEED} fullpool npz exists - skip eval')
        return
    ckpt = output(f'03_Results/01_ModelOutputs/{SITE}_dualhead_{CONFIG}_seed{SEED}_best.pt')
    assert ckpt.exists(), f'checkpoint missing: {ckpt.name}'
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model, _ = build(ck.get('config', CONFIG))
    model.load_state_dict(ck['model_state'])
    model.to(device).eval()
    print(f'[{SITE}] seed {SEED} eval | epoch={ck["epoch"]}  val={ck["val_loss"]:.4f}')
    test_ds = VDEIDatasetV2(SITE, 'test')
    n_tot = len(test_ds)
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    PA, TA, PF, TF, FV = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for b in loader:
            v = b['vdei'].to(device, non_blocking=True)
            f = b['forcing'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                pa, pf = model(v, f)
            PA.append(pa.float().cpu().numpy().astype(np.float16))
            TA.append(b['target'].numpy().astype(np.float16))
            PF.append(pf.float().cpu().numpy().astype(np.float16))
            TF.append(b['facade_target'].numpy().astype(np.float16))
            FV.append(b['facade_valid'].numpy())
    pred = np.concatenate(PA); targ = np.concatenate(TA)
    pfall = np.concatenate(PF); tfall = np.concatenate(TF)
    valid = np.concatenate(FV).astype(bool)
    el = time.time() - t0
    print(f'[{SITE}] seed {SEED} eval done in {el/60:.1f} min | {n_tot/el:,.0f} samples/s')
    np.savez(out, pred_air=pred, target_air=targ, pred_fac=pfall, target_fac=tfall, valid=valid)
    print(f'[SAVED] {out.name} ({out.stat().st_size/1e6:.0f} MB)')

# ---- Phase 1: rebuild missing seed-0 artifacts --------------------------------
for SITE in SITES:
    train_seed0(SITE)
    eval_seed(SITE, 0)

# ---- Phase 2: ensemble + mean±std (unchanged CELL 26 logic) -------------------
def load_seed(site, seed):
    path = output(f'03_Results/02_Metrics/{site}_dualhead_{CONFIG}_seed{seed}_test_fullpool.npz')
    assert path.exists(), f'missing {path.name}'
    return np.load(path)

def mae_rmse(pred, targ):
    diff = pred.astype(np.float32) - targ.astype(np.float32)
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff ** 2)))

mlog = output('03_Results/02_Metrics/test_metrics_log.csv')

for SITE in SITES:
    print(f'\n===== [{SITE.upper()}] 5-seed ensemble + mean±std =====')
    per_seed = []
    target_air_ref = target_fac_ref = valid_ref = None
    for SEED in SEEDS:
        d = load_seed(SITE, SEED)
        if target_air_ref is None:
            target_air_ref = d['target_air']; target_fac_ref = d['target_fac']; valid_ref = d['valid']
        else:
            assert np.array_equal(d['target_air'], target_air_ref), f'{SITE} seed {SEED}: target_air differs!'
            assert np.array_equal(d['target_fac'], target_fac_ref), f'{SITE} seed {SEED}: target_fac differs!'
            assert np.array_equal(d['valid'], valid_ref), f'{SITE} seed {SEED}: valid mask differs!'
        per_seed.append(d)
        print(f'  ✅ seed {SEED} loaded')
    n_tot = target_air_ref.shape[0]
    valid = valid_ref.astype(bool)
    n_valid = int(valid.sum())
    test_ds = VDEIDatasetV2(SITE, 'test')
    fm, fs = test_ds.d['fac_mean'], test_ds.d['fac_std']

    per_seed_air_mae = np.zeros((len(SEEDS), len(AIR_KEYS)))
    per_seed_fac_mae = np.zeros((len(SEEDS), len(FAC_KEYS)))
    for si, d in enumerate(per_seed):
        for vi, (var, (mu, sd)) in enumerate(AIR_KEYS):
            p = d['pred_air'][:, vi].astype(np.float32) * sd + mu
            t = target_air_ref[:, vi].astype(np.float32) * sd + mu
            per_seed_air_mae[si, vi] = np.mean(np.abs(p - t))
        for vi, var in enumerate(FAC_KEYS):
            p = d['pred_fac'][valid, vi].astype(np.float32) * fs[vi] + fm[vi]
            t = target_fac_ref[valid, vi].astype(np.float32) * fs[vi] + fm[vi]
            per_seed_fac_mae[si, vi] = np.mean(np.abs(p - t))

    header = f'{"Variable":10}' + ''.join(f'{"s" + str(s):>10}' for s in SEEDS) + f'{"mean":>10} {"std":>10}'
    print(f'\n[{SITE}] per-seed AIR MAE:'); print(header)
    air_rows = []
    for vi, (var, _) in enumerate(AIR_KEYS):
        vals = per_seed_air_mae[:, vi]
        print(f'{var:10}' + ''.join(f'{v:>10.3f}' for v in vals) + f'{vals.mean():>10.3f} {vals.std(ddof=1):>10.3f}')
        for si, s in enumerate(SEEDS):
            air_rows.append(('air', var, n_tot, per_seed_air_mae[si, vi], s))
    print(f'\n[{SITE}] per-seed FACADE MAE:'); print(header)
    fac_rows = []
    for vi, var in enumerate(FAC_KEYS):
        vals = per_seed_fac_mae[:, vi]
        print(f'{var:10}' + ''.join(f'{v:>10.3f}' for v in vals) + f'{vals.mean():>10.3f} {vals.std(ddof=1):>10.3f}')
        for si, s in enumerate(SEEDS):
            fac_rows.append(('facade', var, n_valid, per_seed_fac_mae[si, vi], s))

    ens_air = np.zeros_like(target_air_ref, dtype=np.float32)
    ens_fac = np.zeros_like(target_fac_ref, dtype=np.float32)
    for d in per_seed:
        ens_air += d['pred_air'].astype(np.float32)
        ens_fac += d['pred_fac'].astype(np.float32)
    ens_air /= len(SEEDS); ens_fac /= len(SEEDS)

    print(f'\n[{SITE}] ENSEMBLE (mean of {len(SEEDS)} seeds) AIR MAE:')
    ens_air_rows = []
    for vi, (var, (mu, sd)) in enumerate(AIR_KEYS):
        p = ens_air[:, vi] * sd + mu
        t = target_air_ref[:, vi].astype(np.float32) * sd + mu
        mae, rmse = mae_rmse(p, t)
        print(f'{var:10} {mae:>10.3f} {rmse:>10.3f} {AIR_UNITS[var]:>8}')
        ens_air_rows.append((var, n_tot, mae, rmse))
    print(f'\n[{SITE}] ENSEMBLE FACADE MAE:')
    ens_fac_rows = []
    for vi, var in enumerate(FAC_KEYS):
        p = ens_fac[valid, vi] * fs[vi] + fm[vi]
        t = target_fac_ref[valid, vi].astype(np.float32) * fs[vi] + fm[vi]
        mae, rmse = mae_rmse(p, t)
        print(f'{var:10} {mae:>10.3f} {rmse:>10.3f} {FAC_UNITS[var]:>8}')
        ens_fac_rows.append((var, n_valid, mae, rmse))

    out = output(f'03_Results/02_Metrics/{SITE}_dualhead_ensemble5_test_fullpool.npz')
    np.savez(out, pred_air=ens_air.astype(np.float16), target_air=target_air_ref,
             pred_fac=ens_fac.astype(np.float16), target_fac=target_fac_ref, valid=valid)
    print(f'[SAVED] {out.name} ({out.stat().st_size/1e6:.0f} MB) - ensemble bootstrap input')

    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    new = not mlog.exists()
    with open(mlog, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['ts', 'site', 'config', 'seed', 'head', 'var', 'n', 'MAE', 'RMSE'])
        for head, var, n, mae, seed in air_rows + fac_rows:
            w.writerow([ts, SITE, CONFIG, seed, head, var, n, f'{mae:.4f}', ''])
        for var, n, mae, rmse in ens_air_rows:
            w.writerow([ts, SITE, 'ensemble5', 'mean', 'air', var, n, f'{mae:.4f}', f'{rmse:.4f}'])
        for var, n, mae, rmse in ens_fac_rows:
            w.writerow([ts, SITE, 'ensemble5', 'mean', 'facade', var, n, f'{mae:.4f}', f'{rmse:.4f}'])
    print(f'[SAVED] {mlog.name}')

print('\n===== CELL 26 DONE - 5-seed ensemble + mean±std complete for both sites.')
print('NEXT: S5e block-bootstrap CIs, then P5a Optuna re-tune.')
