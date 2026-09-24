# ===== CELL 10 — bbox-filtered V-DEI feature set (manuscript design: 16 az x 9 el x 5 ch) =====
# Compact storage: cls(int8) + dist(float16) per point; the (16,9,5) tensor is reshaped at
# TRAIN time (materializing ~2.6M x 720 float32 ~ 7.5 GB is unnecessary and slow).
import numpy as np

for site in SITES:
    d = np.load(output(f'01_Data/02_Processed/vdei_raw_{site}.npz'))
    cls, dist = d['cls'], d['dist']
    i0, j0, k0 = d['i0'], d['j0'], d['k0']
    bbox = np.load(output(f'01_Data/02_Processed/bbox_{site}.npz'))
    jmn, jmx, imn, imx = (int(bbox[k]) for k in ('j_min', 'j_max', 'i_min', 'i_max'))
    keep = (i0 >= imn) & (i0 < imx) & (j0 >= jmn) & (j0 < jmx)
    print(f'\n[{site}] bbox J[{jmn}:{jmx}] I[{imn}:{imx}]  -> points kept: '
          f'{keep.sum():,} / {cls.shape[0]:,} ({100*keep.mean():.1f}%)')

    out = output(f'01_Data/02_Processed/vdei_features_{site}.npz')
    np.savez_compressed(out, cls=cls[keep].astype(np.int8),
                        dist=dist[keep].astype(np.float16),
                        i0=i0[keep], j0=j0[keep], k0=k0[keep],
                        n_az=16, n_el=9, max_range=100.0)
    print(f'  [SAVED] {out.name} ({out.stat().st_size/1e6:.0f} MB)')

    s = keep.nonzero()[0][0]                    # one-sample preview of the real tensor
    t = np.zeros((16, 9, 5), np.float32)
    t[..., :4] = (cls[s].reshape(16, 9)[..., None] == np.arange(4)).astype(np.float32)
    t[..., 4] = np.clip(dist[s].reshape(16, 9) / 100.0, 0, 1)
    print(f'  sample tensor preview: {t.shape}  one-hot sum={t[...,:4].sum():.0f}/144')
