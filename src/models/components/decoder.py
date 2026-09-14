import torch.nn as nn


class FeatureDecoder(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, *, mlp_hidden: int, dropout: float):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        self.mlp_hidden = int(mlp_hidden)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(self.hidden_channels, self.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.mlp_hidden, self.out_channels),
        )

    def forward(self, z):
        return self.net(z)

    def clone(self):
        cloned = FeatureDecoder(
            self.hidden_channels,
            self.out_channels,
            mlp_hidden=self.mlp_hidden,
            dropout=self.dropout,
        )
        cloned.load_state_dict(self.state_dict())
        return cloned
