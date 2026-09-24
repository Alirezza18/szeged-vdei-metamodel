# ===== CELL 14 - S2 manifest + integrity check (upgraded from your Colab corruption-diagnostic) =====
# Every S2 data artifact: size + blake2b hash + keys + shapes/dtypes + target NaN%/range
# -> manifest CSV (data-card backbone, D012/D028). Run any time after CELL 13.
import numpy as np, hashlib, csv

ARTIFACTS = {
    'geometry':        '01_Data/02_Processed/geometry_{site}.npz',
    'vdei_raw':        '01_Data/02_Processed/vdei_raw_{site}.npz',
    'vdei_features':   '01_Data/02_Processed/vdei_features_{site}.npz',
    'bbox':            '01_Data/02_Processed/bbox_{site}.npz',
    'targets_forcing': '01_Data/02_Processed/targets_forcing_{site}.npz',
    'split':           '01_Data/02_Processed/split_{site}.npz',
    'kwindow':         '01_Data/02_Processed/kwindow_{site}.npz',
}

def blake2b_file(p, chunk=1 << 20):
    h = hashlib.blake2b(digest_size=16)
    with open(p, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

rows = []
for site in SITES:
    print(f'\n===== [{site.upper()}] S2 integrity =====')
    for name, tpl in ARTIFACTS.items():
        p = output(tpl.format(site=site))
        if not p.exists():
            rows.append([site, name, '-', 0.0, 'MISSING', '', '', 'MISSING'])
            print(f'  ❌ {name}: MISSING'); continue
        h16 = blake2b_file(p)
        d = np.load(p)
        keys = list(d.keys())
        shapes, nan_note = [], ''
        for k in keys:
            try:
                a = d[k]
                shapes.append(f'{k}{tuple(a.shape)}:{a.dtype}')
                if k.startswith('target_'):
                    nan_note += f'{k}:nan={100*np.isnan(a).mean():.2f}% '
                del a
            except Exception as e:
                shapes.append(f'{k}:ERR:{type(e).__name__}')
        d.close()
        size = p.stat().st_size / 1e6
        status = 'OK' if not any(':ERR' in s for s in shapes) else 'CHECK'
        rows.append([site, name, p.name, round(size, 1), h16[:16], len(keys),
                     ' | '.join(shapes)[:180], (nan_note + status).strip()])
        print(f'  ✅ {name:<16} {size:>8.1f} MB  hash={h16[:16]}  keys={len(keys)}  {nan_note}{status}')

    for f in sorted(output('03_Results/diagnostics').glob(f'{site}_*.png')):
        rows.append([site, 'diagnostic_png', f.name, round(f.stat().st_size / 1e6, 2),
                     '-', '-', '-', 'regenerated-from-code'])

mf = output('01_Data/03_Metadata/s2_manifest.csv')
with open(mf, 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['site', 'artifact', 'file', 'size_MB', 'blake2b16', 'n_keys', 'keys_shapes', 'notes'])
    w.writerows(rows)
print(f'\n[SAVED] {mf.name}  ({len(rows)} rows)')
print('S2 manifest DONE - artifacts hash-sealed; Kaggle packaging re-verifies these hashes.')
