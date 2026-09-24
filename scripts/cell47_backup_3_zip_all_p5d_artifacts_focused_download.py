# ===== BACKUP #3 - zip all P5d artifacts (focused) -> download =====
# Run ONLY after you see "===== P5d DONE =====". Verifies all 10 units before zipping.
import os, glob, zipfile
from pathlib import Path
from IPython.display import FileLink
W = Path('/kaggle/working'); R = W / '03_Results'
pts  = sorted(glob.glob(str(R / '01_ModelOutputs' / '*p5d*')))
npzs = sorted(glob.glob(str(R / '02_Metrics' / '*p5d*')))
log  = R / '02_Metrics' / 'test_metrics_log.csv'
n_ckpt = len([p for p in pts if p.endswith('.pt')])
n_seed = len([p for p in npzs if 'seed' in p and p.endswith('.npz')])
print(f'checkpoints : {n_ckpt}/10')
print(f'fullpool npz: {n_seed}/10')
print(f'summary json: {"OK" if any("ablation_summary" in p for p in npzs) else "MISSING"}')
print(f'metrics log : {"OK" if log.exists() else "MISSING"}')
assert n_ckpt == 10 and n_seed == 10, \
    'INCOMPLETE - wait for P5d DONE (all 10 units + summary tables) before zipping!'
zp = W / 'p5d_ablation_backup.zip'
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_STORED) as z:
    for p in pts + npzs:
        z.write(p, str(Path(p).relative_to(W)))
    if log.exists():
        z.write(str(log), str(log.relative_to(W)))
print(f'\n[READY] {zp.name} -> {zp.stat().st_size/1e9:.2f} GB')
FileLink('p5d_ablation_backup.zip')
