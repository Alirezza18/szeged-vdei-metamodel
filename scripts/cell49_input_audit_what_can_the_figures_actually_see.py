# ===== input audit: what can the figures actually see? =====
from pathlib import Path
IN = Path('/kaggle/input')
print('attached datasets:')
for d in sorted(IN.glob('*')):
    print(f'  {d.name:<28} files: {sum(1 for _ in d.rglob("*"))}')
NEED = ['*_dualhead_optuna_best_seed0_test_fullpool.npz',
        '*_dualhead_ensemble5_test_fullpool.npz',
        '*p5c_zs_*ensemble5*.npz', '*p5c_ft_*ensemble5*.npz',
        'p5b_classical_baselines.json', 'facade_targets_canyon.npz', 'svf_canyon.npz']
print('\nkey files:')
for pat in NEED:
    hits = [p for d in IN.glob('*') for p in d.rglob(pat)]
    print(f'  {pat:<52} {"OK  x" + str(len(hits)) if hits else "MISSING ❌"}')
