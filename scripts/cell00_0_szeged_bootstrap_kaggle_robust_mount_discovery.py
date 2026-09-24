# ===== CELL 0 — Szeged bootstrap (Kaggle, robust mount discovery) =====
from pathlib import Path

KAGGLE_RAW_DATASET = 'szeged-envimet-raw'
RAW_FILES = {'canyon': ['Urban_Canyon.nc', 'Urban Canopy.nc', 'Canopy.nc'],
             'plaza':  ['Main_Plaza.nc']}
SITES = ('canyon', 'plaza')
INPUT = Path('/kaggle/input')

def find_nc(name_candidates):
    # 1) known mount layouts (old + new editor)
    bases = [INPUT / KAGGLE_RAW_DATASET,
             INPUT / KAGGLE_RAW_DATASET / '01_Data' / '01_Raw',
             INPUT / 'datasets' / KAGGLE_RAW_DATASET,
             INPUT / 'datasets' / 'alirezzakarimi' / KAGGLE_RAW_DATASET]
    for base in bases:
        for n in name_candidates:
            p = base / n
            if p.exists():
                return p
    # 2) recursive fallback — bulletproof against any mount-layout change
    for n in name_candidates:
        hits = list(INPUT.rglob(n))
        if hits:
            return hits[0]
    raise FileNotFoundError(f'None of {name_candidates} found under {INPUT} — check Add Input')

def raw_file(site):
    return find_nc(RAW_FILES[site])

def output(subpath):
    p = Path('/kaggle/working') / subpath     # mirror of the Drive tree
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

print('[Szeged] attached inputs:', sorted(p.name for p in INPUT.iterdir()))
for s in SITES:
    try:
        p = raw_file(s)
        print(f'  {s:<8}: OK  {p}  ({p.stat().st_size/1e9:.2f} GB)')
    except FileNotFoundError as e:
        print(f'  {s:<8}: MISSING - {e}')
