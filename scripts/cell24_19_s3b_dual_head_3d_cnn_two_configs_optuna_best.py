# ===== CELL 19 (S3b) - Dual-head 3D-CNN, two configs (optuna_best headline / nb3_ref context) =====
import torch
import torch.nn as nn

CONFIGS = {
    'optuna_best': dict(filters=(48, 32, 64), dropout=0.1227, lr=1.542e-3, facade_weight=0.3105),
    'nb3_ref':     dict(filters=(32, 64, 128), dropout=0.20,  lr=1.0e-3,  facade_weight=1.0),
}

class VDEI3DCNN_DualHead(nn.Module):
    def __init__(self, filters=(48, 32, 64), dropout=0.1227, forcing_dim=8, n_air=5, n_fac=4):
        super().__init__()
        f1, f2, f3 = filters
        self.conv1 = nn.Sequential(nn.Conv3d(5, f1, 3, padding=1),
                                   nn.BatchNorm3d(f1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(f1, f2, 3, padding=1),
                                   nn.BatchNorm3d(f2), nn.ReLU(inplace=True),
                                   nn.MaxPool3d(2, 2))
        self.conv3 = nn.Sequential(nn.Conv3d(f2, f3, 3, padding=1),
                                   nn.BatchNorm3d(f3), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool3d(1)

        def head(n_out):
            return nn.Sequential(nn.Linear(f3 + forcing_dim, 64), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU(inplace=True),
                                 nn.Linear(32, n_out))
        self.head_air = head(n_air)
        self.head_fac = head(n_fac)

    def forward(self, vdei, forcing):
        x = vdei.permute(0, 4, 1, 2, 3).contiguous()      # (B,5,7,16,9)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        z = torch.cat([self.pool(x).flatten(1), forcing], dim=1)
        return self.head_air(z), self.head_fac(z)

def build(config_name='optuna_best'):
    cfg = dict(CONFIGS[config_name])
    model = VDEI3DCNN_DualHead(filters=cfg['filters'], dropout=cfg['dropout'])
    return model, cfg

for name in CONFIGS:
    m, c = build(name)
    print(f'{name:12}: params={sum(p.numel() for p in m.parameters()):,}  {c}')
m, _ = build('optuna_best')
a, f = m(torch.randn(4, 7, 16, 9, 5), torch.randn(4, 8))
print('forward shapes: air', tuple(a.shape), '| facade', tuple(f.shape))
print('\nCELL 20 (S3b) DONE - paste this output to Buffy.')
