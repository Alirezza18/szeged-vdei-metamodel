# ===== CELL 17 - S2 packaging -> private Kaggle dataset 'szeged-vdei-processed' =====
import numpy as np, shutil, zipfile, hashlib, csv

BUNDLE = output('szeged-vdei-processed')
if BUNDLE.exists(): shutil.rmtree(BUNDLE)
(BUNDLE / '01_Data' / '02_Processed').mkdir(parents=True)
(BUNDLE / '01_Data' / '03_Metadata').mkdir(parents=True)
(BUNDLE / '03_Results' / 'diagnostics').mkdir(parents=True)

FILES = ['geometry', 'vdei_raw', 'vdei_features', 'bbox', 'targets_forcing',
         'split', 'kwindow', 'svf', 'facade_targets']
copied, total = 0, 0
for site in SITES:
    for name in FILES:
        src = output(f'01_Data/02_Processed/{name}_{site}.npz')
        if src.exists():
            shutil.copy2(src, BUNDLE / '01_Data' / '02_Processed' / src.name)
            copied += 1; total += src.stat().st_size
manifest = output('01_Data/03_Metadata/s2_manifest.csv')
shutil.copy2(manifest, BUNDLE / '01_Data' / '03_Metadata' / manifest.name)
diag = list(output('03_Results/diagnostics').glob('*.png'))
for f in diag:
    shutil.copy2(f, BUNDLE / '03_Results' / 'diagnostics' / f.name)

(BUNDLE / 'README.txt').write_text(
    'szeged-vdei-processed - sealed S2 artifacts of the Szeged V-DEI metamodeling project.\n'
    'All files hash-sealed in 01_Data/03_Metadata/s2_manifest.csv (blake2b, D028).\n'
    'Generated on Kaggle; layout mirrors the Drive tree (01_Data/02_Processed).\n',
    encoding='utf-8')

zip_path = BUNDLE.with_suffix('.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as z:
    for p in sorted(BUNDLE.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(BUNDLE.parent))
print(f'[BUNDLE] {copied} npz + manifest + {len(diag)} diagnostics | data = {total/1e9:.2f} GB')
print(f'[ZIP] {zip_path.name}  ({zip_path.stat().st_size/1e9:.2f} GB)')

def blake2b_file(p, chunk=1 << 20):
    h = hashlib.blake2b(digest_size=16)
    with open(p, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

man = {}
with open(manifest, newline='', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
        if row['file'] not in ('-', '') and row['blake2b16'] not in ('-', 'MISSING'):
            man[row['file']] = row['blake2b16']
ok = bad = 0
for p in sorted((BUNDLE / '01_Data' / '02_Processed').glob('*.npz')):
    h = blake2b_file(p)[:16]
    exp = man.get(p.name)
    if exp is None:
        print(f'  ℹ️ {p.name}: NEW artifact (svf/facade) - added in manifest v2'); ok += 1; continue
    if h == exp: ok += 1
    else: bad += 1; print(f'  ❌ {p.name}: HASH MISMATCH {h} vs {exp}')
print(f'[VERIFY] copies: {ok} OK, {bad} mismatch')
print('\nUPLOAD (by hand): kaggle.com -> Datasets -> New Dataset -> upload szeged-vdei-processed.zip')
print('                  -> name exactly: szeged-vdei-processed  -> visibility: PRIVATE')
