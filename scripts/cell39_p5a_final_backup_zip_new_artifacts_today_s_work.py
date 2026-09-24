# ===== P5a-FINAL BACKUP - zip new artifacts (today's work) =====
import shutil, os, glob
from pathlib import Path
W = Path('/kaggle/working')
pts  = sorted(glob.glob(f'{W}/03_Results/01_ModelOutputs/*p5a_best*'))
npzs = sorted(glob.glob(f'{W}/03_Results/02_Metrics/*p5a_best*'))
print(f'p5a checkpoints: {len([p for p in pts if p.endswith(".pt")])}/10')
print(f'p5a fullpool npz: {len([p for p in npzs if "seed" in p])}/10 + {len([p for p in npzs if "ensemble" in p])} ensembles')
base = shutil.make_archive(str(W / 'p5a_final_backup'), 'zip',
                           root_dir=W, base_dir='03_Results')
print(f'\n[READY] p5a_final_backup.zip -> {os.path.getsize(base)/1e9:.2f} GB')
print('Files panel -> tick p5a_final_backup.zip -> Download')
