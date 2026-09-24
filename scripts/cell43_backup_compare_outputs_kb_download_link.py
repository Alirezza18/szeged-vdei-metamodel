# ===== BACKUP - compare outputs (KB) + download link =====
import shutil, os
from IPython.display import FileLink
base = shutil.make_archive('/kaggle/working/p5a_compare_backup', 'zip',
                           root_dir='/kaggle/working/03_Results/02_Metrics', base_dir='.')
print(f'p5a_compare_backup.zip -> {os.path.getsize(base)/1e3:.0f} KB')
FileLink('p5a_compare_backup.zip')
