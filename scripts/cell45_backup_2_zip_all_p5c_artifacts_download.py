# ===== BACKUP #2 - zip all P5c artifacts -> download =====
import shutil, os, glob
from pathlib import Path
from IPython.display import FileLink
W = Path('/kaggle/working')
pts  = sorted(glob.glob(f'{W}/03_Results/01_ModelOutputs/*p5c*'))
npzs = sorted(glob.glob(f'{W}/03_Results/02_Metrics/*p5c*'))
print(f'FT checkpoints : {len(pts)}/10')
print(f'per-seed npz   : {len([p for p in npzs if "seed" in p])}/20')
print(f'ensemble npz   : {len([p for p in npzs if "ensemble" in p])}/4')
base = shutil.make_archive(str(W / 'p5c_transfer_backup'), 'zip', root_dir=W, base_dir='03_Results')
print(f'\n[READY] p5c_transfer_backup.zip -> {os.path.getsize(base)/1e9:.2f} GB')
FileLink('p5c_transfer_backup.zip')
