# ===== CELL 0b v3 - mount adapter + zip-wrapper unwrap + inline dataset verify =====
import os, sys, csv, hashlib
from pathlib import Path

INP = Path('/kaggle/input')

def find_dataset(slug):
    for cand in (INP / slug, *sorted((INP / 'datasets').glob(f'*/{slug}')),
                 *sorted(INP.glob(f'*/{slug}'))):
        if cand.exists():
            return cand
    return None

def real_root(d):
    if (d / '01_Data').exists():
        return d
    inner = d / d.name
    if (inner / '01_Data').exists():
        print(f'↳ zip wrapper detected -> real root: {inner.name}/')
        return inner
    return d

RAW_NAMES = ['Urban_Canyon.nc', 'Urban Canopy.nc', 'Main_Plaza.nc']

RAW  = real_root(find_dataset('szeged-envimet-raw') or Path('/nonexistent'))
PROC = real_root(find_dataset('szeged-vdei-processed') or Path('/nonexistent'))
print('raw dataset      :', RAW)
print('processed dataset:', PROC)

raw_ok = any((RAW / n).exists() for n in RAW_NAMES) or \
         any((RAW / '01_Data' / '01_Raw' / n).exists() for n in RAW_NAMES)
print('raw nc files     :', 'FOUND ✅' if raw_ok else 'NOT FOUND ❌')
assert raw_ok, f'raw nc files not found under {RAW}'
assert (PROC / '01_Data').exists(), 'szeged-vdei-processed content not found!'

def _proc_root(site):
    for cand in (PROC / '01_Data' / '02_Processed' / site, PROC / site,
                 PROC / '01_Data' / '02_Processed'):
        if cand.exists():
            return cand
    return PROC / '01_Data' / '02_Processed'

def _raw_file(site):
    names = {'canyon': ['Urban_Canyon.nc', 'Urban Canopy.nc'], 'plaza': ['Main_Plaza.nc']}[site]
    for n in names:
        for cand in (RAW / n, RAW / '01_Data' / '01_Raw' / n):
            if cand.exists():
                return cand
    raise FileNotFoundError(f'{site}: raw nc not found under {RAW}')

pm = None
for name, mod in list(sys.modules.items()):
    if hasattr(mod, 'kaggle_input_root') and hasattr(mod, 'processed_dir'):
        pm = mod
        break
if pm is not None:
    pm.kaggle_input_root = lambda: RAW
    pm.processed_dir = _proc_root
    print(f'✅ patched paths module: {getattr(pm, "__name__", "?")}')

processed_dir = _proc_root
raw_file = _raw_file
print('✅ session globals adapted (processed_dir, raw_file)')

def blake2b_file(p, chunk=1 << 20):
    h = hashlib.blake2b(digest_size=16)
    with open(p, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

man_path = PROC / '01_Data' / '03_Metadata' / 's2_manifest.csv'
assert man_path.exists(), f'manifest missing: {man_path}'
man = {}
with open(man_path, newline='', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
        if row['file'] not in ('-', '') and row['blake2b16'] not in ('-', 'MISSING'):
            man[row['file']] = row['blake2b16']

ok = bad = missing = 0
pdir = PROC / '01_Data' / '02_Processed'
for f, exp in sorted(man.items()):
    p = pdir / f
    if not p.exists():
        missing += 1; print(f'  ❌ {f}: MISSING'); continue
    if blake2b_file(p)[:16] == exp:
        ok += 1
    else:
        bad += 1; print(f'  ❌ {f}: HASH MISMATCH')
print(f'\n[ATTACH-VERIFY] {ok} OK | {bad} mismatch | {missing} missing  ->  '
      f'{"DATASET VALID ✅ ready for S3" if bad == 0 and missing == 0 else "PROBLEM ❌ - tell Buffy"}')
print('\nNEXT: CELL 15c (I will give it right after this passes).')
