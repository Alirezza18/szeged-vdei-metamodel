# ===== CELL 24 (S3e) - MULTI-SEED training: seeds 1..4 x both sites (D047) =====
# Protocol identical to seed-0 (CELL 21/22): ONE 500k/300k train subset per seed reused
# across epochs, val subset = RandomState(0), AMP, grad-clip, early stop patience 5.
# Seed 0 already trained + evaluated for both sites. Resumable: completed seeds skipped.
import csv, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SEEDS      = [1, 2, 3, 4]    # seed 0 = done
BATCH      = 256
PATIENCE   = 5
N_EPOCHS   = 30
VAL_SUBSET = 100_000

def seed_all(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
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
                     'Use GPU T4 x2.') from e

def composite(pa, pf, y, fy, fv, lam):
    la  = ((pa - y) ** 2).mean()
    per = ((pf - fy) ** 2).mean(dim=1)
    den = fv.sum()
    lf  = (per * fv).sum() / den if den > 0 else pa.new_zeros(())
    return la + lam * lf, la.detach(), lf.detach()

summary = []
for SITE in SITES:
    EP_SAMPLES = {'canyon': 500_000, 'plaza': 300_000}[SITE]        # D048 fair sampling
    train_ds = VDEIDatasetV2(SITE, 'train')
    val_ds   = VDEIDatasetV2(SITE, 'val')
    g = np.random.RandomState(0)                                     # same val pool as seed 0
    val_idx = g.choice(len(val_ds), size=min(VAL_SUBSET, len(val_ds)), replace=False)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=BATCH, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    for SEED in SEEDS:
        ckpt_path = output(f'03_Results/01_ModelOutputs/{SITE}_dualhead_optuna_best_seed{SEED}_best.pt')
        hist_path = output(f'03_Results/01_ModelOutputs/{SITE}_dualhead_optuna_best_seed{SEED}_history.csv')
        if ckpt_path.exists() and hist_path.exists():
            print(f'\n⏭️ [{SITE}] seed {SEED} already complete - skip')
            continue
        seed_all(SEED)
        print(f'\n===== [{SITE.upper()}] seed {SEED} =====')
        model, cfg = build('optuna_best')
        model.to(device)
        lam = cfg['facade_weight']
        n_params = sum(p.numel() for p in model.parameters())

        g = np.random.RandomState(SEED)                              # ONE subset per seed (seed-0 protocol)
        tr_idx = g.choice(len(train_ds), size=min(EP_SAMPLES, len(train_ds)), replace=False)
        train_loader = DataLoader(Subset(train_ds, tr_idx), batch_size=BATCH, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True)
        print(f'train samples/epoch={len(tr_idx):,}  val subset={len(val_idx):,}  batch={BATCH}')

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        scaler    = torch.amp.GradScaler('cuda')

        best, bad, rows = float('inf'), 0, []
        t00 = time.time()
        for ep in range(1, N_EPOCHS + 1):
            t0 = time.time()
            model.train()
            tot = ta = tf_ = 0.0; ns = 0
            for b in train_loader:
                v  = b['vdei'].to(device, non_blocking=True)
                f  = b['forcing'].to(device, non_blocking=True)
                y  = b['target'].to(device, non_blocking=True)
                fy = b['facade_target'].to(device, non_blocking=True)
                fv = b['facade_valid'].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    pa, pf = model(v, f)
                loss, la, lfl = composite(pa.float(), pf.float(), y, fy, fv, lam)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                bs = v.size(0); ns += bs
                tot += loss.item() * bs; ta += la.item() * bs; tf_ += lfl.item() * bs

            model.eval()
            vt = va_ = vf_ = 0.0; nvs = 0
            with torch.no_grad():
                for b in val_loader:
                    v  = b['vdei'].to(device); f  = b['forcing'].to(device)
                    y  = b['target'].to(device); fy = b['facade_target'].to(device)
                    fv = b['facade_valid'].to(device)
                    with torch.amp.autocast('cuda'):
                        pa, pf = model(v, f)
                    loss, la, lfl = composite(pa.float(), pf.float(), y, fy, fv, lam)
                    bs = v.size(0); nvs += bs
                    vt += loss.item() * bs; va_ += la.item() * bs; vf_ += lfl.item() * bs
            va, va_a, va_f = vt/nvs, va_/nvs, vf_/nvs
            scheduler.step(va)
            rows.append([ep, tot/ns, ta/ns, tf_/ns, va, va_a, va_f,
                         optimizer.param_groups[0]['lr'], time.time()-t0])
            print(f'Epoch {ep:2}/{N_EPOCHS} | train={tot/ns:.4f} | val={va:.4f} '
                  f'(air={va_a:.4f} fac={va_f:.4f}) | '
                  f'lr={optimizer.param_groups[0]["lr"]:.2e} | {time.time()-t0:.1f}s')
            if va < best:
                best, bad = va, 0
                torch.save({'model_state': model.state_dict(), 'config': 'optuna_best',
                            'seed': SEED, 'site': SITE, 'epoch': ep, 'val_loss': va,
                            'val_air': va_a, 'val_fac': va_f, 'n_params': n_params,
                            'amp': True, 'forcing_dim': 8, 'n_air': 5, 'n_fac': 4}, ckpt_path)
                print(f'  ✅ new best saved (val_loss={va:.4f})')
            else:
                bad += 1
                if bad >= PATIENCE:
                    print(f'  ⏹ early stopping after {ep} epochs')
                    break

        mins = (time.time() - t00) / 60
        with open(hist_path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['epoch','train_loss','train_air','train_fac',
                        'val_loss','val_air','val_fac','lr','sec'])
            w.writerows(rows)
        print(f'[{SITE}] seed {SEED} done in {mins:.1f} min | best val={best:.4f}')
        print(f'[SAVED] {ckpt_path.name} + {hist_path.name}')
        summary.append((SITE, SEED, best, mins))

print('\n===== MULTI-SEED SUMMARY =====')
print('(seed 0, previous session: canyon 0.0447 | plaza 0.0441)')
for site, seed, best, mins in summary:
    print(f'{site:>7} seed {seed}: best_val={best:.4f}  ({mins:.1f} min)')
print('\nNEXT: paste the summary to Buffy -> per-seed full-pool eval cell.')
