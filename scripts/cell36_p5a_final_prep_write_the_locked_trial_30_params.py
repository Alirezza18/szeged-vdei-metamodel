# ===== P5a-final PREP - write the LOCKED trial-30 params =====
# The study's params JSON exists only in the PC backup; paste the locked winner
# here so P5a-final reads exactly these (NOT its trial-22 fallback).
import json
from pathlib import Path
P = {'study_name': 'p5a_vdei_retune', 'n_trials': 40,
     'best_score': 0.0859, 'best_trial_number': 30,
     'best_params': {'lr': 4.48e-4, 'facade_weight': 0.30,
                     'batch': 128, 'ep_samples_frac': 0.76},
     'finished_at': 'locked on PC 2026-09-05'}
out = Path('/kaggle/working/03_Results/01_ModelOutputs/p5a_optuna_best_params.json')
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(P, open(out, 'w'), indent=2)
print('[SAVED]', out)
print('WINNING PARAMS:', P['best_params'])
