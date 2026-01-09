import torch.nn as nn

class RNANet(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_p=0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, out_dim) 
        )
        self.out_dim = out_dim 
        
    def forward(self, x):
        out = self.net(x)
        return out         