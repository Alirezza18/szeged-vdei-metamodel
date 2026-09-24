# ===== P5b OUTPUT BACKUP - tiny, download in seconds =====
import shutil, os
from pathlib import Path
W = Path('/kaggle/working')
src = W / '03_Results' / '02_Metrics'
base = shutil.make_archive(str(W / 'p5b_output_backup'), 'zip',
                           root_dir=src, base_dir='.')
print(f'[READY] p5b_output_backup.zip -> {os.path.getsize(W/"p5b_output_backup.zip")/1e3:.0f} KB')
print('Files panel -> tick p5b_output_backup.zip -> Download')
