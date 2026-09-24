# ===== CELL 13 - S2c: vertical K-window index (K=7; sorted-key lookup, no dict/pickle) =====
# D027: holes (domain edges AND building-occupied levels - sample set is AIR-only) are
# filled with the NEAREST VALID AIR level of the same column; ties -> BELOW.
# nbr_gap (0 = exact) kept for QC and future ablations.
import numpy as np

K_WINDOW = 7
HALF_K = K_WINDOW // 2

for site in SITES:
    print(f"\n{'='*60}\n[{site}] K-window vertical index (K={K_WINDOW})\n{'='*60}")
    d = np.load(output(f'01_Data/02_Processed/vdei_features_{site}.npz'))
    i0, j0, k0 = d['i0'].astype(np.int64), d['j0'].astype(np.int64), d['k0'].astype(np.int64)
    n = len(i0)

    g = np.load(output(f'01_Data/02_Processed/geometry_{site}.npz'))
    is_air = g['is_air']
    K, J, I = is_air.shape

    key = (i0 * J + j0) * K + k0
    order = np.argsort(key, kind='stable')
    sorted_key = key[order]

    def row_of(i, j, k):
        q = (i * J + j) * K + k
        pos = np.searchsorted(sorted_key, q)
        assert np.all(sorted_key[np.minimum(pos, n - 1)] == q), 'lookup key missing!'
        return order[pos]

    air_col = is_air[:, j0, i0].T          # (n, K) valid-air mask per column
    ks = np.arange(K)[None, :]

    nbr_idx = np.zeros((n, K_WINDOW), np.int32)
    nbr_gap = np.zeros((n, K_WINDOW), np.int8)
    for w, dk in enumerate(range(-HALF_K, HALF_K + 1)):
        k_t = np.clip(k0 + dk, 0, K - 1)                     # clamp to domain (D027)
        up_ok = air_col & (ks >= k_t[:, None])
        up = np.where(up_ok.any(1), up_ok.argmax(1), K)      # first valid air level >= k_t
        dn_ok = air_col & (ks <= k_t[:, None])
        dn = np.where(dn_ok.any(1), K - 1 - dn_ok[:, ::-1].argmax(1), -1)
        dist_up = np.where(up < K, up - k_t, K + 100)
        dist_dn = np.where(dn >= 0, k_t - dn, K + 100)
        nbr_k = np.where(dist_up < dist_dn, up, dn)          # tie -> BELOW wins (D027)
        nbr_idx[:, w] = row_of(i0, j0, nbr_k)
        nbr_gap[:, w] = np.abs(nbr_k - (k0 + dk))
        ex = 100 * (nbr_gap[:, w] == 0).mean()
        print(f'  dk={dk:+d}: exact {ex:5.1f}%  mean_gap {nbr_gap[:, w].mean():.2f}  '
              f'max_gap {nbr_gap[:, w].max()}')

    full = 100 * (nbr_gap.max(axis=1) == 0).mean()
    print(f'[{site}] points with ALL {K_WINDOW} levels exact (no hole): {full:.1f}%')

    out = output(f'01_Data/02_Processed/kwindow_{site}.npz')
    np.savez_compressed(out, nbr_idx=nbr_idx, nbr_gap=nbr_gap,
                        k_window=K_WINDOW, half_k=HALF_K)
    print(f'[SAVED] {out.name} ({out.stat().st_size/1e6:.0f} MB)')

print('\nS2c DONE - sample-set artifacts complete; next: facade targets.')
