# ===== CELL 22 (S3c-plaza) - FULL dual-head training on PLAZA (same protocol as canyon seed0) =====
import csv, shutil, time, datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SITE   = 'plaza'           # <-- plaza
CONFIG = 'optuna_best'
SMOKE  = False             # full run
SEED   = 0

def seed_all(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise SystemExit('GPU session required (Settings -> Accelerator -> GPU T4 x2).')
try:
    _ = (torch.ones(1, device=device) * 2).sum().item()
    print(f'device: {torch.cuda.get_device_name(0)} ✅ CUDA kernels OK')
except RuntimeError as e:
    raise SystemExit('❌ GPU not supported by installed PyTorch (e.g., P100/sm_60). '
                     'Switch: Settings -> Accelerator -> GPU T4 x2, then restart.') from e
seed_all(SEED)

def run_tag(prefix):
    return f"{prefix}_{datetime.datetime.now():%Y%m%d_%H%M%S}"

model, cfg = build(CONFIG)
model.to(device)
lam = cfg['facade_weight']
n_params = sum(p.numel() for p in model.parameters())
print(f'[{SITE}] config={CONFIG}  params={n_params:,}  lambda_facade={lam}  lr={cfg["lr"]:.4e}')

train_ds = VDEIDatasetV2(SITE, 'train')
val_ds   = VDEIDatasetV2(SITE, 'val')

EP_SAMPLES = {'canyon': 500_000, 'plaza': 300_000}[SITE]   # D048: ~1% pool/epoch (fair vs canyon)N_EPOCHS   = 30
VAL_SUBSET = 100_000
BATCH      = 256
PATIENCE   = 5

g = np.random.RandomState(0)
val_idx = g.choice(len(val_ds), size=min(VAL_SUBSET, len(val_ds)), replace=False)
val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)
g = np.random.RandomState(SEED)
tr_idx = g.choice(len(train_ds), size=min(EP_SAMPLES, len(train_ds)), replace=False)
train_loader = DataLoader(Subset(train_ds, tr_idx), batch_size=BATCH, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
print(f'train samples/epoch={len(tr_idx):,}  val subset={len(val_idx):,}  batch={BATCH}')

optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
scaler    = torch.amp.GradScaler('cuda')

ckpt_path = output(f'03_Results/01_ModelOutputs/{SITE}_dualhead_{CONFIG}_seed{SEED}_best.pt')
hist_path = output(f'03_Results/01_ModelOutputs/{SITE}_dualhead_{CONFIG}_seed{SEED}_history.csv')
if ckpt_path.exists():
    bkp = ckpt_path.with_name(run_tag(f'{SITE}_{CONFIG}_seed{SEED}_backup') + '.pt')
    shutil.copy2(ckpt_path, bkp)
    print(f'📦 backed up previous best -> {bkp.name}')

def composite(pa, pf, y, fy, fv):
    la = ((pa - y) ** 2).mean()
    per = ((pf - fy) ** 2).mean(dim=1)
    den = fv.sum()
    lf = (per * fv).sum() / den if den > 0 else pa.new_zeros(())
    return la + lam * lf, la.detach(), lf.detach()

def run_epoch(loader, train=True):
    model.train(train)
    tot = ta = tfl = 0.0
    ns = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in loader:
            v  = b['vdei'].to(device, non_blocking=True)
            f  = b['forcing'].to(device, non_blocking=True)
            y  = b['target'].to(device, non_blocking=True)
            fy = b['facade_target'].to(device, non_blocking=True)
            fv = b['facade_valid'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                pa, pf = model(v, f)
            loss, la, lfl = composite(pa.float(), pf.float(), y, fy, fv)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            bs = v.size(0)
            ns += bs
            tot += loss.item() * bs
            ta  += la.item() * bs
            tfl += lfl.item() * bs
    return tot / ns, ta / ns, tfl / ns

best, bad = float('inf'), 0
rows = []
t00 = time.time()
for ep in range(1, N_EPOCHS + 1):
    t0 = time.time()
    tr, tr_a, tr_f = run_epoch(train_loader, train=True)
    va, va_a, va_f = run_epoch(val_loader, train=False)
    scheduler.step(va)
    elapsed = time.time() - t0
    rows.append([ep, tr, tr_a, tr_f, va, va_a, va_f, optimizer.param_groups[0]['lr'], elapsed])
    print(f'Epoch {ep:2}/{N_EPOCHS} | train={tr:.4f} (air={tr_a:.4f} fac={tr_f:.4f}) | '
          f'val={va:.4f} (air={va_a:.4f} fac={va_f:.4f}) | '
          f'lr={optimizer.param_groups[0]["lr"]:.2e} | {elapsed:.1f}s')
    if va < best:
        best = va
        bad = 0
        torch.save({'model_state': model.state_dict(), 'config': CONFIG, 'seed': SEED,
                    'site': SITE, 'epoch': ep, 'val_loss': va, 'val_air': va_a, 'val_fac': va_f,
                    'n_params': n_params, 'amp': True,
                    'forcing_dim': 8, 'n_air': 5, 'n_fac': 4}, ckpt_path)
        print(f'  ✅ new best saved (val_loss={va:.4f})')
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f'  ⏹ early stopping after {ep} epochs')
            break

tot_min = (time.time() - t00) / 60
print(f'\n[{SITE}] done in {tot_min:.1f} min | best val={best:.4f}')
with open(hist_path, 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['epoch', 'train_loss', 'train_air', 'train_fac', 'val_loss',
                'val_air', 'val_fac', 'lr', 'sec'])
    w.writerows(rows)
print(f'[SAVED] {ckpt_path.name} + {hist_path.name}')
print('NEXT: CELL 23 (S3d-plaza full-pool eval).')
