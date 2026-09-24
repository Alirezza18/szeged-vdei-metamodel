# ===== CELL P6-FC SCRATCH-META - scratch protocol from the sealed checkpoint =====
import torch
from pathlib import Path
p = next(Path('/kaggle/input').rglob('canyon_dualhead_optuna_best_seed0_best.pt'))
ck = torch.load(p, map_location='cpu', weights_only=False)
print({k: v for k, v in ck.items() if k != 'model_state'})
