import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    重力波の特徴量（相対遅延、相互相関、振幅比、位相差など）を入力とする全結合ネットワーク。
    """
    def __init__(self, n_sectors: int, use_KAGRA: bool = False):
        super(MLP, self).__init__()
        in_features = 42 if use_KAGRA else 21
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, n_sectors)

        self.dropout1 = nn.Dropout(p=0.2)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()

        self.net = nn.Sequential(
            self.fc1, self.prelu1, self.dropout1,
            self.fc2, self.prelu2, self.dropout1,
            self.fc3, self.prelu3, self.dropout1,
            self.fc4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)