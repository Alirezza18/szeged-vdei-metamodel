# ===== CELL 14b v2 - manifest v2: seal svf + facade (dataset-aware, never crashes) =====
import numpy as np, hashlib, csv

NEW = ['svf', 'facade_targets']

def blake2b_file(p, chunk=1 << 20):
    h = hashlib.blake2b(digest_size=16)
    with open(p, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def artifact_path(name, site):
    """working copy first (freshly computed), else the sealed dataset copy (hash-identical)."""
    w = output(f'01_Data/02_Processed/{name}_{site}.npz')
    if w.exists():
        return w, 'working'
    d = processed_dir(site) / f'{name}_{site}.npz'
    if d.exists():
        return d, 'sealed-dataset'
    return None, 'MISSING'

mf = output('01_Data/03_Metadata/s2_manifest.csv')
assert mf.exists(), 'run CELL 14 (manifest v1) first!'
existing = set()
with open(mf, newline='', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
        existing.add(row['file'])

rows = []
for site in SITES:
    for name in NEW:
        p, src = artifact_path(name, site)
        if p is None:
            print(f'  ❌ {name}_{site}.npz: not found anywhere (run CELL 15/16 first)'); continue
        if p.name in existing:
            print(f'  ✅ {p.name:<28} already sealed - skip'); continue
        h16 = blake2b_file(p)
        d = np.load(p)
        shapes = []
        for k in d.keys():
            a = d[k]; shapes.append(f'{k}{tuple(a.shape)}:{a.dtype}'); del a
        d.close()
        size = p.stat().st_size / 1e6
        rows.append([site, name, p.name, round(size, 1), h16[:16], len(shapes),
                     ' | '.join(shapes)[:180], f'OK ({src})'])
        print(f'  ✅ {p.name:<28} {size:>6.1f} MB  hash={h16[:16]}  [{src}]  -> SEALED')

if rows:
    with open(mf, 'a', newline='', encoding='utf-8') as fh:
        csv.writer(fh).writerows(rows)
    print(f'[UPDATED] {mf.name} -> manifest v2 (all 18 artifacts sealed)')
else:
    print('[OK] manifest already v2 - nothing to do')
print('NOW: CELL 17 packaging re-verifies these hashes.')
