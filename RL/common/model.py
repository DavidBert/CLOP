from .misc_util import orthogonal_init, xavier_uniform_init
import torch.nn as nn
from .layer import CLOPLayer
import torch.nn.functional as F
from torch.distributions import Categorical


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaModel(nn.Module):
    def __init__(self, in_channels=3, task_dim=32, **kwargs):
        super(ImpalaModel, self).__init__()

        self.task_dim = task_dim
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=self.task_dim)
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, encoder, action_size, dropout, clop):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(Policy, self).__init__()
        self.encoder = encoder
        self.dropout = dropout
        self.clop = clop

        # small scale weight-initialization in policy enhances the stability
        self.fc = xavier_uniform_init(
            nn.Linear(in_features=32 * 8 * 8, out_features=256)
        )
        self.fc2 = xavier_uniform_init(
            nn.Linear(in_features=32 * 8 * 8, out_features=256)
        )
        self.fc_policy = orthogonal_init(nn.Linear(256, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(256, 1), gain=1.0)
        self.clop_layer = CLOPLayer(p=clop)
        self.dp = nn.Dropout(p=dropout)

    def forward(self, x):
        z_task = self.encoder(x)
        x = Flatten()(z_task)
        xv = self.fc(x)
        xv = F.relu(xv)
        v = self.fc_value(xv).reshape(-1)

        if self.clop > 0:
            x = self.clop_layer(z_task)
            x = Flatten()(x)

        if self.dropout > 0:
            x = self.dp(x)

        x = self.fc2(x)
        x = F.relu(x)

        logits = self.fc_policy(x)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)

        return p, v
