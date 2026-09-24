from pathlib import Path
root = Path('/kaggle/input/datasets/alirezzakarimi/szeged-vdei-processed')
for p in sorted(root.rglob('*'))[:40]:
    print(p.relative_to(root), '(folder)' if p.is_dir() else f'{p.stat().st_size/1e6:.1f} MB')
print('total files:', sum(1 for _ in root.rglob('*') if _.is_file()))
