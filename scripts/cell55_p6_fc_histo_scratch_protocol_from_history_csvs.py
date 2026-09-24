# ===== CELL P6-FC HISTO - scratch protocol from history CSVs =====
import csv
from pathlib import Path
IN = Path('/kaggle/input')
for name in ('canyon_dualhead_optuna_best_seed0_history.csv',
             'plaza_dualhead_optuna_best_seed0_history.csv'):
    hits = sorted(IN.rglob(name))
    if not hits:
        print(f'{name}: not found'); continue
    rows = list(csv.DictReader(open(hits[0])))
    print(f'\n{hits[0].name}')
    print('  columns:', list(rows[0].keys()))
    print('  epochs recorded:', len(rows))
    for r in rows[:2] + rows[-2:]:
        print('  ', dict(list(r.items())[:6]))
