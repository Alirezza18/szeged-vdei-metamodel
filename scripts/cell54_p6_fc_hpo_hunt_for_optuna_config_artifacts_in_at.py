# ===== CELL P6-FC HPO - hunt for Optuna/config artifacts in attached inputs =====
from pathlib import Path
import json
IN = Path('/kaggle/input')

def find_all(pat):
    out = []
    for d in sorted(IN.glob('*')):
        out += sorted(d.rglob(pat))
    return out

seen = set()
for pat in ['*optuna*', '*study*', '*hpo*', '*config*.json', '*best_param*', '*trial*']:
    for p in find_all(pat):
        if p in seen or p.stat().st_size > 50e6:
            continue
        seen.add(p)
        print(f'\n{p}  ({p.stat().st_size/1e3:.1f} KB)')
        if p.suffix == '.json':
            try:
                j = json.loads(p.read_text())
                print('  keys:', list(j.keys())[:15])
                for k in ('best_params', 'best_value', 'n_trials', 'config', 'configs'):
                    if isinstance(j, dict) and k in j:
                        print(f'  {k}:', json.dumps(j[k])[:400])
            except Exception as e:
                print('  <unreadable>', e)
