# ===== CELL 23 (S3d-plaza) - FULL-POOL test evaluation on PLAZA =====
import csv, time
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

SITE   = 'plaza'         # <-- plaza
CONFIG = 'optuna_best'
SEED   = 0

CKPT = output(f'03_Results/01_ModelOutputs/{SITE}_dualhead_{CONFIG}_seed{SEED}_best.pt')
assert CKPT.exists(), (f'checkpoint not found: {CKPT.name} - run CELL 22 (plaza training) first!')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    _ = (torch.ones(1, device=device) * 2).sum().item()
except RuntimeError as e:
    raise SystemExit('❌ GPU not supported by installed PyTorch (e.g., P100/sm_60). Use GPU T4 x2.') from e
ck = torch.load(CKPT, map_location=device, weights_only=False)
model, _ = build(ck.get('config', CONFIG))
model.load_state_dict(ck['model_state'])
model.to(device).eval()
print(f'[{SITE}] loaded {CKPT.name} | epoch={ck["epoch"]}  val={ck["val_loss"]:.4f}  '
      f'seed={ck.get("seed")}  params={ck.get("n_params"):,}')

test_ds = VDEIDatasetV2(SITE, 'test')
n_tot = len(test_ds)
print(f'[{SITE}] FULL test pool = {n_tot:,} samples (no subsampling)')

loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

PA, TA, PF, TF, FV = [], [], [], [], []
t0 = time.time()
nb = len(loader)
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
print(f'\n[{SITE}] eval done in {el/60:.1f} min | throughput {n_tot/el:,.0f} samples/s')

AIR_UNITS = {'T': 'degC', 'RelHum': '%', 'WindSpd': 'm/s', 'TKE': 'm2/s2', 'TMRT': 'degC'}
FAC_UNITS = {'Twall': 'degC', 'Qsens': 'W/m2', 'SWabs': 'W/m2', 'LWbal': 'W/m2'}

print(f'\n{"="*66}\n[{SITE.upper()}] FULL TEST POOL - AIR HEAD (n={n_tot:,})\n{"="*66}')
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
print(f'\n{"="*66}\n[{SITE.upper()}] FULL TEST POOL - FACADE HEAD (n_valid={n_valid:,})\n{"="*66}')
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

out = output(f'03_Results/02_Metrics/{SITE}_dualhead_{CONFIG}_seed{SEED}_test_fullpool.npz')
np.savez(out, pred_air=pred, target_air=targ, pred_fac=pfall, target_fac=tfall, valid=valid)
print(f'[SAVED] {out.name} ({out.stat().st_size/1e6:.0f} MB) - input for S5e bootstrap')

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
print(f'[SAVED] {mlog.name} (metrics log, paper tables)')
print('\nS3d-PLAZA DONE - both headline evals complete. Now: Quick Save, then sleep.')
