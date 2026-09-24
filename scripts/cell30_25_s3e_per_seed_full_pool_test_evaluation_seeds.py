# ===== CELL 25 (S3e) - per-seed FULL-POOL test evaluation, seeds 1-4 x both sites =====
# (seed 0 already evaluated in S3d - NOT redone here)
# For each (site, seed): loads best checkpoint from CELL 24, evaluates ENTIRE test pool,
# saves float16 preds (ensemble + bootstrap input), appends to test_metrics_log.csv.
# Resumable: (site, seed) with existing fullpool npz is skipped.
import csv, time
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

SITES  = ('canyon', 'plaza')
SEEDS  = [1, 2, 3, 4]
CONFIG = 'optuna_best'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise SystemExit('GPU session required (Settings -> Accelerator -> GPU T4 x2).')
try:
    _ = (torch.ones(1, device=device) * 2).sum().item()
    print(f'device: {torch.cuda.get_device_name(0)} ✅ CUDA kernels OK')
except RuntimeError as e:
    raise SystemExit('❌ GPU not supported by installed PyTorch (e.g., P100/sm_60). '
                     'Use GPU T4 x2.') from e

AIR_UNITS = {'T': 'degC', 'RelHum': '%', 'WindSpd': 'm/s', 'TKE': 'm2/s2', 'TMRT': 'degC'}
FAC_UNITS = {'Twall': 'degC', 'Qsens': 'W/m2', 'SWabs': 'W/m2', 'LWbal': 'W/m2'}

for SITE in SITES:
    test_ds = VDEIDatasetV2(SITE, 'test')
    n_tot = len(test_ds)
    loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                        num_workers=4, pin_memory=True)
    nb = len(loader)

    for SEED in SEEDS:
        out = output(f'03_Results/02_Metrics/{SITE}_dualhead_{CONFIG}_seed{SEED}_test_fullpool.npz')
        if out.exists():
            print(f'\n⏭️ [{SITE}] seed {SEED}: fullpool npz exists - skip')
            continue
        ckpt = output(f'03_Results/01_ModelOutputs/{SITE}_dualhead_{CONFIG}_seed{SEED}_best.pt')
        assert ckpt.exists(), f'checkpoint missing: {ckpt.name} - run CELL 24 first!'

        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model, _ = build(ck.get('config', CONFIG))
        model.load_state_dict(ck['model_state'])
        model.to(device).eval()
        print(f'\n===== [{SITE.upper()}] seed {SEED} =====')
        print(f'loaded {ckpt.name} | epoch={ck["epoch"]}  val={ck["val_loss"]:.4f}  '
              f'params={ck.get("n_params"):,}')
        print(f'[{SITE}] FULL test pool = {n_tot:,} samples (no subsampling)')

        PA, TA, PF, TF, FV = [], [], [], [], []
        t0 = time.time()
        with torch.no_grad():
            for bi, b in enumerate(loader, 1):
                v = b['vdei'].to(device, non_blocking=True)
                f = b['forcing'].to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    pa, pf = model(v, f)
                PA.append(pa.float().cpu().numpy().astype(np.float16))
                TA.append(b['target'].numpy().astype(np.float16))
                PF.append(pf.float().cpu().numpy().astype(np.float16))
                TF.append(b['facade_target'].numpy().astype(np.float16))
                FV.append(b['facade_valid'].numpy())
                if bi % 500 == 0 or bi == nb:
                    el = time.time() - t0
                    print(f'  batch {bi}/{nb} | {el:.0f}s | {bi * 512 / el:,.0f} samples/s', flush=True)

        pred  = np.concatenate(PA);  targ  = np.concatenate(TA)
        pfall = np.concatenate(PF);  tfall = np.concatenate(TF)
        valid = np.concatenate(FV).astype(bool)
        el = time.time() - t0
        print(f'[{SITE}] seed {SEED} eval done in {el/60:.1f} min | {n_tot/el:,.0f} samples/s')

        print(f'\n{"="*66}\n[{SITE.upper()}] seed {SEED} FULL TEST - AIR HEAD (n={n_tot:,})\n{"="*66}')
        print(f'{"Variable":10} {"MAE":>10} {"RMSE":>10} {"Unit":>8}')
        air_rows = []
        for i, (var, (mu, sd)) in enumerate(AIR_KEYS):
            p = pred[:, i].astype(np.float32) * sd + mu
            t = targ[:, i].astype(np.float32) * sd + mu
            mae = float(np.mean(np.abs(p - t)))
            rmse = float(np.sqrt(np.mean((p - t) ** 2)))
            air_rows.append((var, n_tot, mae, rmse))
            print(f'{var:10} {mae:>10.3f} {rmse:>10.3f} {AIR_UNITS[var]:>8}')

        n_valid = int(valid.sum())
        print(f'\n{"="*66}\n[{SITE.upper()}] seed {SEED} FULL TEST - FACADE HEAD (n_valid={n_valid:,})\n{"="*66}')
        print(f'{"Variable":10} {"MAE":>10} {"RMSE":>10} {"Unit":>8}')
        fac_rows = []
        fm, fs = test_ds.d['fac_mean'], test_ds.d['fac_std']
        for i, var in enumerate(FAC_KEYS):
            p = pfall[valid, i].astype(np.float32) * fs[i] + fm[i]
            t = tfall[valid, i].astype(np.float32) * fs[i] + fm[i]
            mae = float(np.mean(np.abs(p - t)))
            rmse = float(np.sqrt(np.mean((p - t) ** 2)))
            fac_rows.append((var, n_valid, mae, rmse))
            print(f'{var:10} {mae:>10.3f} {rmse:>10.3f} {FAC_UNITS[var]:>8}')

        np.savez(out, pred_air=pred, target_air=targ,
                 pred_fac=pfall, target_fac=tfall, valid=valid)
        print(f'[SAVED] {out.name} ({out.stat().st_size/1e6:.0f} MB)')

        mlog = output('03_Results/02_Metrics/test_metrics_log.csv')
        new = not mlog.exists()
        ts = datetime.now().strftime('%Y-%m-%d %H:%M')
        with open(mlog, 'a', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(['ts', 'site', 'config', 'seed', 'head', 'var', 'n', 'MAE', 'RMSE'])
            for var, n, mae, rmse in air_rows:
                w.writerow([ts, SITE, CONFIG, SEED, 'air', var, n, f'{mae:.4f}', f'{rmse:.4f}'])
            for var, n, mae, rmse in fac_rows:
                w.writerow([ts, SITE, CONFIG, SEED, 'facade', var, n, f'{mae:.4f}', f'{rmse:.4f}'])
        print(f'[SAVED] {mlog.name}')

print('\n===== CELL 25 DONE - 8 full-pool evals complete (seeds 1-4 x 2 sites).')
print('NEXT: CELL 26 (5-seed ensemble + mean±std table).')
