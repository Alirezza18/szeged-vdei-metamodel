# ===== BACKUP - zip everything into ONE file for download =====
import shutil, zipfile, os, glob

W = '/kaggle/working'
pts  = glob.glob(f'{W}/03_Results/01_ModelOutputs/*.pt')
npzs = glob.glob(f'{W}/03_Results/02_Metrics/*fullpool.npz')
print(f'checkpoints: {len(pts)}/10   fullpool npz: {len(npzs)}/12')

base = shutil.make_archive(f'{W}/szeged_backup', 'zip',
                           root_dir=W, base_dir='03_Results')
sz = os.path.getsize(base)
print(f'\n[READY] szeged_backup.zip -> {sz/1e9:.2f} GB')
print('Files panel -> tick the zip -> Download')
