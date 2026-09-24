# ===== CELL 18b (S3-BOOTSTRAP) - self-contained setup for ANY fresh Kaggle session =====
# Replaces CELL 0 / CELL 0b / CELL 19 in S3 sessions: defines SITES/output/processed_dir/
# raw_file (flat + nested + zip-wrapper aware) AND the complete VDEIDatasetV2.
# After every GPU switch: run CELL 18b -> CELL 20 (S3b) -> CELL 21 v2 (S3c).  C52.
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------- environment
KAGGLE = Path('/kaggle').exists()
WORK   = Path('/kaggle/working') if KAGGLE else Path.cwd()
SITES  = ('canyon', 'plaza')

def output(subpath):
    q = WORK / subpath
    q.parent.mkdir(parents=True, exist_ok=True)
    return q

# ---------------------------------------------------------------- dataset mounts
def _find(slug):
    base = Path('/kaggle/input') if KAGGLE else None
    if base is None:
        return None
    for cand in (base / slug, *sorted((base / 'datasets').glob(f'*/{slug}')),
                 *sorted(base.glob(f'*/{slug}'))):
        if cand.exists():
            return cand
    return None

def _real_root(d):
    if d is None:
        return None
    if (d / '01_Data').exists():
        return d
    inner = d / d.name
    if (inner / '01_Data').exists():
        print(f'↳ zip wrapper detected -> real root: {inner.name}/')
        return inner
    return d

PROC = _real_root(_find('szeged-vdei-processed'))
RAW  = _find('szeged-envimet-raw')
if PROC is None or not (PROC / '01_Data' / '02_Processed').exists():
    raise SystemExit('❌ szeged-vdei-processed not attached/extracted! '
                     'Add Input -> szeged-vdei-processed, restart.')
print('processed dataset:', PROC)
print('raw dataset      :', RAW)

def processed_dir(site):
    return PROC / '01_Data' / '02_Processed'

def raw_file(site):
    if RAW is None:
        raise FileNotFoundError('szeged-envimet-raw not attached (only CELL 15c needs it).')
    names = {'canyon': ['Urban_Canyon.nc', 'Urban Canopy.nc'], 'plaza': ['Main_Plaza.nc']}[site]
    for n in names:
        for cand in (RAW / n, RAW / '01_Data' / '01_Raw' / n):
            if cand.exists():
                return cand
    raise FileNotFoundError(f'{site}: raw nc not found under {RAW}')

# ---------------------------------------------------------------- dataset (S3a content)
AIR_KEYS = [('T', (28.0, 4.0)), ('RelHum', (40.0, 15.0)), ('WindSpd', (2.0, 1.5)),
            ('TKE', (50.0, 100.0)), ('TMRT', (45.0, 20.0))]   # fixed manuscript stats (D006)
FAC_KEYS = ['Twall', 'Qsens', 'SWabs', 'LWbal']
MAX_RANGE = 100.0
SPLIT_CODE = {'train': 0, 'val': 1, 'test': 2}

_CACHE = {}

def _load_site(site):
    if site in _CACHE:
        return _CACHE[site]
    proc, src = processed_dir(site), 'sealed-dataset'
    if not (proc / f'vdei_features_{site}.npz').exists():
        proc = output('01_Data/02_Processed'); src = 'working-fallback'
        assert (proc / f'vdei_features_{site}.npz').exists(), f'no artifacts for {site!r}'
    print(f'[{site}] loading artifacts from {proc}  [{src}]')
    vf = np.load(proc / f'vdei_features_{site}.npz')
    kw = np.load(proc / f'kwindow_{site}.npz')
    tf = np.load(proc / f'targets_forcing_{site}.npz')
    sp = np.load(proc / f'split_{site}.npz')
    fc = np.load(proc / f'facade_targets_{site}.npz')
    raw_split = sp['split']
    if raw_split.dtype.kind in 'US':            # CELL 12 saved string labels ('<U5')
        codes = np.full(len(raw_split), -1, np.int8)
        codes[raw_split == 'train'] = 0
        codes[raw_split == 'val'] = 1
        codes[raw_split == 'test'] = 2
        split_codes = codes
    else:
        split_codes = raw_split.astype(np.int8)
    d = {
        'cls':     vf['cls'],                                   # (n,144) int8
        'dist':    vf['dist'],                                  # (n,144) float16
        'nbr_idx': kw['nbr_idx'],                               # (n,7) int32, D027-filled
        'targets': {k: tf[f'target_{k}'] for k, _ in AIR_KEYS}, # (n,T) float32
        'forcing': {k: tf[f'forcing_{k}'] for k in ['T', 'RelHum', 'WindSpd', 'SunHeight', 'IsDaytime']},
        'sun_az_grid': tf['sun_az_grid_deg'],                   # (n_day,) grid frame (D021)
        'sun_times':   tf['sun_times'].astype(np.int64),
        'sun_hit':     tf['sun_hit'],                           # (n, n_day) int8 (>0 = blocked)
        'split':   split_codes,                                 # (n,) 0/1/2 normalized
        'fac':     {k: fc[f'facade_{k}'] for k in FAC_KEYS},    # (n_f,T) float32
        'fac_mean': fc['facade_norm_mean'], 'fac_std': fc['facade_norm_std'],
    }
    n = d['cls'].shape[0]
    n_time = d['targets']['T'].shape[1]
    d['fac_row'] = np.full(n, -1, np.int64)
    d['fac_row'][fc['facade_point_idx']] = np.arange(len(fc['facade_point_idx']))
    d['day_pos'] = np.full(n_time, -1, np.int64)
    d['day_pos'][d['sun_times']] = np.arange(len(d['sun_times']))
    d['n_time'] = n_time
    for h in (vf, kw, tf, sp, fc):
        h.close()
    _CACHE[site] = d
    print(f'  ✅ [{site}] cached: n={n:,}  n_time={n_time}  '
          f'facade-adjacent={int((d["fac_row"] >= 0).sum()):,}  '
          f'split: train={int((split_codes==0).sum()):,} val={int((split_codes==1).sum()):,} '
          f'test={int((split_codes==2).sum()):,}')
    return d

class VDEIDatasetV2(Dataset):
    def __init__(self, site, split_name, use_sun_block=True):
        d = _load_site(site)
        self.d = d
        self.use_sun_block = use_sun_block
        self.point_idx = np.where(d['split'] == SPLIT_CODE[split_name])[0]
        self.n_time = d['n_time']
        n_f = int((d['fac_row'][self.point_idx] >= 0).sum())
        print(f'[{site}] {split_name:5} view: {len(self.point_idx):,} points '
              f'({n_f:,} facade-adjacent) x {self.n_time} t = {len(self):,} samples')

    def __len__(self):
        return len(self.point_idx) * self.n_time

    def __getitem__(self, idx):
        d = self.d
        pos, t = divmod(idx, self.n_time)
        p = int(self.point_idx[pos])
        rows = d['nbr_idx'][p]                                       # (7,) D027-filled
        cw = d['cls'][rows].reshape(7, 16, 9)                        # int8
        dw = d['dist'][rows].reshape(7, 16, 9).astype(np.float32)
        v = np.empty((7, 16, 9, 5), np.float32)
        v[..., :4] = (cw[..., None] == np.arange(4, dtype=cw.dtype))
        v[..., 4] = np.clip(dw / MAX_RANGE, 0.0, 1.0)

        f = d['forcing']
        dp = d['day_pos'][t]
        if dp >= 0:
            az = np.radians(float(d['sun_az_grid'][dp]))
            saz, caz = np.float32(np.sin(az)), np.float32(np.cos(az))
        else:
            saz = caz = np.float32(0.0)
        sb = np.float32(0.0)
        if self.use_sun_block and dp >= 0:
            sb = np.float32(d['sun_hit'][p, dp] > 0)
        fv = np.array([(f['T'][t] - 28.0) / 4.0,
                       (f['RelHum'][t] - 40.0) / 15.0,
                       (f['WindSpd'][t] - 2.0) / 1.5,
                       saz, caz,
                       f['SunHeight'][t] / 90.0,
                       f['IsDaytime'][t],
                       sb], np.float32)

        tv = np.array([(d['targets'][k][p, t] - mu) / sd for k, (mu, sd) in AIR_KEYS], np.float32)

        fr = d['fac_row'][p]
        if fr >= 0:
            fcv = np.array([(d['fac'][k][fr, t] - m) / s
                            for k, m, s in zip(FAC_KEYS, d['fac_mean'], d['fac_std'])], np.float32)
            fmask = np.float32(1.0)
        else:
            fcv = np.zeros(4, np.float32)
            fmask = np.float32(0.0)
        return {'vdei': torch.from_numpy(v), 'forcing': torch.from_numpy(fv),
                'target': torch.from_numpy(tv), 'facade_target': torch.from_numpy(fcv),
                'facade_valid': torch.tensor(fmask)}

# ------------------------------------------------------------------ self-test
for site in SITES:
    for s in ('train', 'val', 'test'):
        VDEIDatasetV2(site, s)

ds = VDEIDatasetV2('canyon', 'train')
s0 = ds[0]
print('\nsample shapes:', {k: tuple(x.shape) for k, x in s0.items()})
print('one-hot sum per level (must be 144):',
      [round(float(s0['vdei'][w, :, :, :4].sum())) for w in range(7)])
fr = ds.d['fac_row'][ds.point_idx]
q = int(np.argmax(fr >= 0))
s1 = ds[q * ds.n_time + 5]
print(f'facade sample (row pos {q}): valid={float(s1["facade_valid"])}  '
      f'facade_target={s1["facade_target"].numpy().round(2)}')

t0 = time.time()
for i in range(500):
    _ = ds[i]
dt = (time.time() - t0) / 500 * 1000
print(f'getitem speed: {dt:.2f} ms/sample  -> ~{1000/dt:,.0f} samples/s single-thread')
print('\nCELL 18b DONE - NEXT: CELL 20 (S3b) -> CELL 21 v2 (SMOKE).')
