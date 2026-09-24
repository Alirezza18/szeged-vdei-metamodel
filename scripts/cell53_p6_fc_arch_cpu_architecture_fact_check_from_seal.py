# ===== CELL P6-FC ARCH (CPU) - architecture fact-check from sealed checkpoints =====
# Reveals: conv channels/kernels, BN, pooling, head Linear shapes (fused dim!),
# exact parameter count, and the checkpoint metadata. Runtime < 1 min.
import torch
from pathlib import Path

IN = Path('/kaggle/input')

def find_all(pat):
    out = []
    for d in sorted(IN.glob('*')):
        out += sorted(d.rglob(pat))
    return out

cands = find_all('*dualhead*seed0*.pt') or find_all('*.pt')
assert cands, 'no checkpoint found - attach szeged-backup or p5d-ablation-backup'
p = sorted(cands)[0]
print(f'checkpoint: {p}\n')
ck = torch.load(p, map_location='cpu', weights_only=False)
sd = ck['model_state']
print('--- LAYERS (name -> shape -> params) ---')
total = 0
for k, v in sd.items():
    n = v.numel()
    total += n
    print(f'  {k:<52}{str(tuple(v.shape)):<24}{n:>10,}')
print(f'\nTOTAL PARAMETERS: {total:,}')
meta = {k: (v if not hasattr(v, "shape") else tuple(v.shape)) for k, v in ck.items() if k != 'model_state'}
print(f'\nMETA: {meta}')

# quick derived facts
linears = [(k, tuple(v.shape)) for k, v in sd.items() if 'weight' in k and v.ndim == 2]
print('\n--- LINEAR LAYERS (in -> out) ---')
for k, s in linears:
    print(f'  {k:<52}{s[1]:>5} <- {s[0]}')
conv3d = [(k, tuple(v.shape)) for k, v in sd.items() if 'weight' in k and v.ndim == 5]
print('\n--- CONV3D LAYERS (out_ch, in_ch, k,k,k) ---')
for k, s in conv3d:
    print(f'  {k:<52}{s}')
