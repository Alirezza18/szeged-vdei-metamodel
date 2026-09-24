# ===== CELL 28 v3 (P5b) - Classical ML baselines WITH hyperparameter tuning (CPU-only) =====
# Paper comparison table: LR / RF / XGBoost / CatBoost / MLP-SVF on the simple
# exposure features [SVF, BuildingHeight] (2 features - matches the manuscript's
# "Comparison with Simple Exposure-Based Baseline" protocol; the deep V-DEI model
# sees the full 3D neighbourhood + forcing, these baselines only see SVF + BH).
# v3 fixes vs v2: MLP-SVF's first Linear layer is sized from the data
# (Xtr.shape[1]) instead of hardcoded 9, which crashed with
# "mat1 and mat2 shapes cannot be multiplied (2048x2 and 9x32)".
# Results are saved to json AFTER EVERY MODEL -> re-running skips finished models.
# Each tree model: 10-config random search PER VARIABLE, selected on the sealed
# VAL split (40k pairs), then a final fit on the 200k train subsample, evaluated
# on the FULL test pool. MLP-SVF: joint 5-output net, same scheme.
# Runs entirely on CPU - no GPU, no quota. ~2-3 h total.
# Prereq: CELL 18b (PROC/RAW/SITES/output/raw_file) + BOTH inputs attached.
import json, time, csv
from datetime import datetime
import numpy as np
import xarray as xr
import torch
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import catboost as cb

TARGET_VARS = ['T', 'RelHum', 'WindSpd', 'TKE', 'TMRT']
N_SAMPLES   = 200_000     # train subsample per site (matches old protocol)
TUNE_TRAIN  = 100_000     # pairs used for the tuning fits (of the 200k)
TUNE_VAL    = 40_000      # (point,time) pairs from sealed VAL split for selection
N_CFG       = 10          # random configs per model
SEED        = 0
UNITS = {'T': 'degC', 'RelHum': '%', 'WindSpd': 'm/s', 'TKE': 'm2/s2', 'TMRT': 'degC'}
MU = {'T': 28.0, 'RelHum': 40.0, 'WindSpd': 2.0, 'TKE': 50.0, 'TMRT': 45.0}
SD = {'T': 4.0, 'RelHum': 15.0, 'WindSpd': 1.5, 'TKE': 100.0, 'TMRT': 20.0}
MODELS = ['LinearRegression', 'RandomForest', 'XGBoost', 'CatBoost', 'MLP-SVF']
MU_A = np.array([MU[v] for v in TARGET_VARS], np.float32)
SD_A = np.array([SD[v] for v in TARGET_VARS], np.float32)

# ------------------------------------------------------------------ features
def build_features(site):
    """Returns X (n, 2)=[SVF, BuildingHeight], Y (n, n_time, 5) physical,
    split (n,) int codes, in sealed point order."""
    proc = PROC / '01_Data' / '02_Processed'
    vf = np.load(proc / f'vdei_features_{site}.npz')
    tf = np.load(proc / f'targets_forcing_{site}.npz')
    sp = np.load(proc / f'split_{site}.npz')
    cls, i0, j0 = vf['cls'], vf['i0'], vf['j0']
    n = cls.shape[0]
    n_time = tf['target_T'].shape[1]

    # --- SVF feature (1) ---
    svp = proc / f'svf_{site}.npz'
    if svp.exists():
        sv = np.load(svp)
        svf = sv['svf_envimet'] if 'svf_envimet' in sv else sv['svf_rays']
        print(f'[{site}] svf: sealed svf_envimet (pearson_r vs rays = {float(sv["pearson_r"]):.3f})')
    else:
        cls3 = cls.reshape(n, 16, 9)
        svf = (cls3[:, :, 5:9] == 0).mean(axis=(1, 2)).astype(np.float32)
        print(f'[{site}] ⚠️ svf_{site}.npz not sealed -> computed svf_rays from V-DEI geometry')
    svf = np.nan_to_num(svf.astype(np.float32), nan=0.0)

    # --- BuildingHeight (2) from attached raw nc; grid (J, I) -> bh[j0, i0] ---
    bh = None
    try:
        with xr.open_dataset(raw_file(site), decode_times=False) as ds:
            b = ds['BuildingHeight']
            b = b.isel(Time=0).values if b.ndim == 3 else b.values
        bh = np.nan_to_num(b[j0, i0].astype(np.float32), nan=0.0)
        print(f'[{site}] BuildingHeight: OK  mean={bh.mean():.2f} m')
    except Exception as e:
        print(f'[{site}] ⚠️ BuildingHeight failed ({type(e).__name__}) -> proxy = height class / 15')
        cls3 = cls.reshape(n, 16, 9)
        bh = (np.argmax((cls3 != 0).any(axis=2), axis=1) / 15.0).astype(np.float32)

    # --- targets (physical units, n x n_time x 5) + split codes ---
    Y = np.stack([tf[f'target_{k}'] for k in TARGET_VARS], axis=-1).astype(np.float32)
    raw_split = sp['split']
    if raw_split.dtype.kind in 'US':
        codes = np.full(len(raw_split), -1, np.int8)
        for i, lab in enumerate(('train', 'val', 'test')):
            codes[raw_split == lab] = i
        split = codes
    else:
        split = raw_split.astype(np.int8)

    X = np.concatenate([svf[:, None], bh[:, None]], axis=1).astype(np.float32)   # (n, 2)
    print(f'[{site}] features: n={n:,}  n_time={n_time}  | '
          f'split train={int((split==0).sum()):,} val={int((split==1).sum()):,} '
          f'test={int((split==2).sum()):,}')
    return X, Y, split, n_time

# ------------------------------------------------------------------ helpers
def mae_rmse(y_true, y_pred):
    return (float(mean_absolute_error(y_true, y_pred)),
            float(np.sqrt(mean_squared_error(y_true, y_pred))))

def _pick(rng, vals):
    x = rng.choice(vals)
    return int(x) if isinstance(vals[0], int) else float(x)

def draw_configs(name):
    rng = np.random.RandomState(SEED)
    if name == 'RandomForest':
        grid = {'n_estimators': [100, 200, 250], 'max_depth': [5, 8, 12],
                'min_samples_leaf': [1, 2, 5]}
    elif name == 'XGBoost':
        grid = {'n_estimators': [100, 200, 400], 'max_depth': [5, 8],
                'learning_rate': [0.03, 0.15, 0.3], 'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0]}
    elif name == 'CatBoost':
        grid = {'iterations': [100, 150, 200], 'depth': [4, 6],
                'learning_rate': [0.03, 0.1, 0.2, 0.3]}
    else:  # MLP-SVF
        grid = {'lr': [3e-4, 1e-3, 3e-3], 'epochs': [15, 25], 'hidden': [32, 64]}
    return [{k: _pick(rng, v) for k, v in grid.items()} for _ in range(N_CFG)]

CFGS = {m: draw_configs(m) for m in MODELS if m != 'LinearRegression'}

def make_estimator(name, cfg):
    if name == 'RandomForest':
        return RandomForestRegressor(n_estimators=cfg['n_estimators'],
                                     max_depth=cfg['max_depth'],
                                     min_samples_leaf=cfg['min_samples_leaf'],
                                     n_jobs=-1, random_state=SEED)
    if name == 'XGBoost':
        return xgb.XGBRegressor(n_estimators=cfg['n_estimators'],
                                max_depth=cfg['max_depth'],
                                learning_rate=cfg['learning_rate'],
                                subsample=cfg['subsample'],
                                colsample_bytree=cfg['colsample_bytree'],
                                n_jobs=-1, random_state=SEED, verbosity=0)
    if name == 'CatBoost':
        return cb.CatBoostRegressor(iterations=cfg['iterations'], depth=cfg['depth'],
                                    learning_rate=cfg['learning_rate'], verbose=0,
                                    random_state=SEED)
    raise ValueError(name)

# ------------------------------------------------------------------ MLP-SVF
def fit_mlp(Xtr, Ytr, cfg):
    """Joint multi-output MLP (n_in -> hidden -> 5), normalized targets.
    FIXED vs v2: first Linear layer uses Xtr.shape[1] (== 2 here), not 9."""
    torch.manual_seed(SEED)
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr).astype(np.float32)
    Ytr_n = ((Ytr - MU_A) / SD_A).astype(np.float32)
    n_in = Xtr.shape[1]                                    # <-- THE FIX (was 9)
    net = torch.nn.Sequential(
        torch.nn.Linear(n_in, cfg['hidden']), torch.nn.ReLU(inplace=True),
        torch.nn.Linear(cfg['hidden'], 5))
    opt = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
    lossf = torch.nn.MSELoss()
    Xt = torch.from_numpy(Xtr_s); Yt = torch.from_numpy(Ytr_n)
    B = 4096
    for ep in range(cfg['epochs']):
        net.train()
        perm = torch.randperm(Xt.size(0))
        for i in range(0, Xt.size(0), B):
            idx = perm[i:i + B]
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(Xt[idx]), Yt[idx])
            loss.backward(); opt.step()
    return net, sc

def mlp_mean_mae(net, sc, Xva, Yva):
    """Mean (over vars AND samples) of the denormalized MAE on the val set."""
    Xva_s = sc.transform(Xva).astype(np.float32)
    net.eval()
    with torch.no_grad():
        p = net(torch.from_numpy(Xva_s)).numpy()
    p = p * SD_A + MU_A
    return float(np.mean(np.abs(p - Yva)))

def mlp_predict(net, sc, X, B=500_000):
    """Batched prediction -> physical units (memory-safe on the full test pool)."""
    Xs = sc.transform(X).astype(np.float32)
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), B):
            out.append(net(torch.from_numpy(Xs[i:i + B])).numpy())
    p = np.concatenate(out)
    return p * SD_A + MU_A

# ------------------------------------------------------------------ main
t00 = time.time()
results = {}
out = output('03_Results/02_Metrics/p5b_classical_baselines.json')
if out.exists():
    try:
        results = json.loads(out.read_text()).get('results', {})
        print(f'resuming: {len(results)} site(s) already in json')
    except Exception:
        results = {}

for site in SITES:
    X, Y, split, n_time = build_features(site)
    rng = np.random.RandomState(SEED)
    n_flat = X.shape[0] * n_time
    flat = rng.choice(n_flat, size=min(N_SAMPLES, n_flat), replace=False)
    p_tr, t_tr = flat // n_time, flat % n_time
    keep = split[p_tr] == 0
    Xtr = X[p_tr[keep]]
    Ytr = np.stack([Y[p_tr[keep], t_tr[keep], i] for i in range(5)], axis=1).astype(np.float32)
    n_tr = Xtr.shape[0]

    # tuning sets (subsample of the train pairs + sealed VAL split pairs)
    tune_idx = rng.choice(n_tr, size=min(TUNE_TRAIN, n_tr), replace=False)
    Xtr_t, Ytr_t = Xtr[tune_idx], Ytr[tune_idx]
    val_pts = np.where(split == 1)[0]
    vflat = rng.choice(len(val_pts) * n_time, size=min(TUNE_VAL, len(val_pts) * n_time),
                       replace=False)
    p_va, t_va = vflat // n_time, vflat % n_time
    Xva = X[val_pts[p_va]]
    Yva = np.stack([Y[val_pts[p_va], t_va, i] for i in range(5)], axis=1).astype(np.float32)

    # full test pool, flattened to (n_test * n_time,)
    test_pts = np.where(split == 2)[0]
    Xte = np.repeat(X[test_pts], n_time, axis=0)
    Yte = Y[test_pts].reshape(-1, 5)
    n_te = Xte.shape[0]
    print(f'[{site}] train={n_tr:,}  tune_train={len(Xtr_t):,}  tune_val={len(Xva):,}  '
          f'test_flat={n_te:,}', flush=True)

    site_res = results.get(site, {})
    for var in TARGET_VARS:
        site_res.setdefault(var, {})
    vi = {v: i for i, v in enumerate(TARGET_VARS)}

    for m in MODELS:
        if all(m in site_res[var] for var in TARGET_VARS):
            print(f'[{site}] {m}: already in json - skip', flush=True)
            continue
        t0 = time.time()
        if m == 'LinearRegression':
            for var in TARGET_VARS:
                lr = LinearRegression().fit(Xtr, Ytr[:, vi[var]])
                pred = lr.predict(Xte)
                ma, rs = mae_rmse(Yte[:, vi[var]], pred)
                site_res[var][m] = {'mae': ma, 'rmse': rs, 'cfg': None}
                print(f'  {m} {var:5} FINAL MAE={ma:.3f} RMSE={rs:.3f} ({time.time()-t0:.0f}s)',
                      flush=True)
        elif m == 'MLP-SVF':
            print(f'[{site}] tuning MLP-SVF: {N_CFG} configs (joint net, val-selected)...',
                  flush=True)
            best_m, best_c = float('inf'), None
            for c in CFGS['MLP-SVF']:
                net, sc = fit_mlp(Xtr_t, Ytr_t, c)
                vm = mlp_mean_mae(net, sc, Xva, Yva)
                if vm < best_m:
                    best_m, best_c = vm, c
            print(f'    best val MAE={best_m:.4f}  {best_c}', flush=True)
            net, sc = fit_mlp(Xtr, Ytr, best_c)
            pred = mlp_predict(net, sc, Xte)
            for var in TARGET_VARS:
                ma, rs = mae_rmse(Yte[:, vi[var]], pred[:, vi[var]])
                site_res[var][m] = {'mae': ma, 'rmse': rs, 'cfg': best_c}
                print(f'  {m} {var:5} FINAL MAE={ma:.3f} RMSE={rs:.3f} ({time.time()-t0:.0f}s)',
                      flush=True)
        else:
            print(f'[{site}] tuning {m}: {N_CFG} configs x 5 vars (val-selected)...', flush=True)
            best_cfg = {}
            for var in TARGET_VARS:
                best_vm, bc = float('inf'), None
                for c in CFGS[m]:
                    est = make_estimator(m, c).fit(Xtr_t, Ytr_t[:, vi[var]])
                    vm = mean_absolute_error(Yva[:, vi[var]], est.predict(Xva))
                    if vm < best_vm:
                        best_vm, bc = vm, c
                best_cfg[var] = bc
                print(f'    {var:7} best val MAE={best_vm:.4f}  {bc}', flush=True)
            for var in TARGET_VARS:
                est = make_estimator(m, best_cfg[var]).fit(Xtr, Ytr[:, vi[var]])
                pred = est.predict(Xte)
                ma, rs = mae_rmse(Yte[:, vi[var]], pred)
                site_res[var][m] = {'mae': ma, 'rmse': rs, 'cfg': best_cfg[var]}
                print(f'  {m} {var:5} FINAL MAE={ma:.3f} RMSE={rs:.3f} ({time.time()-t0:.0f}s)',
                      flush=True)

        # incremental save after EVERY model -> a crash loses at most one model
        results[site] = site_res
        out.write_text(json.dumps({'site': site, 'n_samples': N_SAMPLES, 'tuned': True,
                                   'results': results}, indent=2), encoding='utf-8')
        print(f'[SAVED] p5b_classical_baselines.json ({site} / {m})', flush=True)

    # summary table
    print(f'\n[{site.upper()}] CLASSICAL BASELINES - FULL TEST POOL (physical units)')
    print(f'{"Variable":8}' + ''.join(f'{m:>17}' for m in MODELS))
    for var in TARGET_VARS:
        print(f'{var:8}' + ''.join(f'{site_res[var][m]["mae"]:>17.3f}' for m in MODELS))

    # append to the shared metrics log
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    mlog = output('03_Results/02_Metrics/test_metrics_log.csv')
    new = not mlog.exists()
    with open(mlog, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['ts', 'site', 'config', 'seed', 'head', 'var', 'n', 'MAE', 'RMSE'])
        for var in TARGET_VARS:
            for m in MODELS:
                r = site_res[var][m]
                w.writerow([ts, site, f'p5b_{m}', SEED, 'air', var, n_te,
                            f'{r["mae"]:.4f}', f'{r["rmse"]:.4f}'])
    print(f'[SAVED] test_metrics_log.csv ({site})')

print(f'\n===== P5b v2 DONE in {(time.time()-t00)/60:.1f} min - tuned baselines '
      f'(LR/RF/XGBoost/CatBoost/MLP-SVF), full test pool, physical MAE/RMSE. '
      f'Next: P5a-final (GPU).')
