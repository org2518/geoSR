# Source: https://github.com/yjn870/SRCNN-pytorch

from torch import nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=1, scale_factor=2, n_feats=128):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_channels = num_channels
        self.n_feats = n_feats
        self.conv1 = nn.Conv2d(num_channels, n_feats, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(n_feats, n_feats // 2, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(
            n_feats // 2, num_channels, kernel_size=5, padding=5 // 2
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = nn.functional.interpolate(
            x,
            size=None,
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=None,
            recompute_scale_factor=None,
            antialias=False,
        )
        identity = x  # do residual
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x += identity  # do residual
        x = self.relu(x)  # do residual
        return x
