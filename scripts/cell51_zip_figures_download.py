# ===== zip figures -> download =====
import shutil, os
from IPython.display import FileLink
z = shutil.make_archive('/kaggle/working/p6_figures', 'zip',
                        root_dir='/kaggle/working/03_Results/03_Figures')
print(f'[READY] p6_figures.zip -> {os.path.getsize(z)/1e6:.1f} MB')
FileLink('p6_figures.zip')
