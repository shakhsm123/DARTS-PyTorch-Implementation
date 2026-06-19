import torch
import torch.nn as nn


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv_depthwise = nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in, bias=False)
        self.conv_pointwise = nn.Conv2d(C_in, C_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2_depthwise = nn.Conv2d(C_out, C_out, kernel_size, stride=1, padding=padding, groups=C_out, bias=False)
        self.conv2_pointwise = nn.Conv2d(C_out, C_out, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv_depthwise(x)
        x = self.conv_pointwise(x)
        x = self.bn(x)
        x = self.relu2(x)
        x = self.conv2_depthwise(x)
        x = self.conv2_pointwise(x)
        x = self.bn2(x)
        return x


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv_depthwise = nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False)
        self.conv_pointwise = nn.Conv2d(C_in, C_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv_depthwise(x)
        x = self.conv_pointwise(x)
        x = self.bn(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        x = x[:, :, ::self.stride, ::self.stride]
        x = x * 0
        return x


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv_1_by_1 = nn.Conv2d(C_in, C_out // 2, kernel_size=1, stride=2, bias=False)
        self.conv_1_by_1_2 = nn.Conv2d(C_in, C_out // 2, kernel_size=1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        x_new_1 = self.conv_1_by_1(x)
        x_new_2 = self.conv_1_by_1_2(x[:, :, 1:, 1:])
        x_resultant = torch.cat([x_new_1, x_new_2], 1)
        x_resultant = self.bn(x_resultant)
        return x_resultant


OPS = {
    'none':         lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.Sequential(nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False), nn.BatchNorm2d(C, affine=affine)),
    'max_pool_3x3': lambda C, stride, affine: nn.Sequential(nn.MaxPool2d(3, stride=stride, padding=1), nn.BatchNorm2d(C, affine=affine)),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine),
}
