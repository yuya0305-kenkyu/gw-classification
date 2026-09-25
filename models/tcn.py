import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride,
            padding=int(padding / 2), dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(n_outputs, eps=1e-03, momentum=0.01)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride,
            padding=int(padding / 2), dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(n_outputs, eps=1e-03, momentum=0.01)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.bn1, self.prelu1, self.dropout1,
            self.conv2, self.bn2, self.prelu2, self.dropout2
        )

        if n_inputs != n_outputs:
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
        else:
            self.downsample = None
        self.prelu = nn.PReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.prelu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: list, kernel_size: int = 2, dropout: float = 0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size, stride=1,
                    dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCN(nn.Module):
    """
    時間波形(歪みデータ)を直接入力とする時系列畳み込みネットワーク。
    時間軸方向にGlobal Max Poolingを行い、特徴量を集約して分類します。
    """
    def __init__(self, input_size: int, output_size: int,
                 num_channels: list, kernel_size: int = 7, dropout: float = 0.0):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = torch.max(x, dim=2)[0]
        x = self.linear(x)
        return x