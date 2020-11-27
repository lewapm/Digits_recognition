import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, activation_fn):
        super(ResidualBlock, self).__init__()
        self.conv2d1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2d2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation_fn = activation_fn()

    def forward(self, x):
        residual = x
        out = self.activation_fn(self.conv2d1(x))
        out = self.activation_fn(self.conv2d2(out))

        out += residual
        return out