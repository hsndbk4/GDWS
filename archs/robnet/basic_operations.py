import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.stat_modules import Conv2dStat, LinearStat, BatchNorm2dStat
operation_canditates = {
    '00': lambda C, stride, use_stat_layers: Zero(stride),
    '01': lambda C, stride, use_stat_layers: SepConv(C, C, 3, stride, 1, use_stat_layers),
    '10': lambda C, stride, use_stat_layers: Identity() if stride == 1 else FactorizedReduce(C, C, use_stat_layers),
    '11': lambda C, stride, use_stat_layers: ResSepConv(C, C, 3, stride, 1, use_stat_layers),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, use_stat_layers=False):
        super(ReLUConvBN, self).__init__()
        Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
        Linear = LinearStat if use_stat_layers else nn.Linear
        BatchNorm2d = BatchNorm2dStat if use_stat_layers else nn.BatchNorm2d
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            BatchNorm2d(C_out, affine=False)
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding,use_stat_layers=False):
        super(SepConv, self).__init__()
        Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
        Linear = LinearStat if use_stat_layers else nn.Linear
        BatchNorm2d = BatchNorm2dStat if use_stat_layers else nn.BatchNorm2d
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=False),
            nn.ReLU(inplace=False),
            Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(C_out, affine=False),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, use_stat_layers=False):
        super(FactorizedReduce, self).__init__()
        Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
        Linear = LinearStat if use_stat_layers else nn.Linear
        BatchNorm2d = BatchNorm2dStat if use_stat_layers else nn.BatchNorm2d
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = BatchNorm2d(C_out, affine=False)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class ResSepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, use_stat_layers=False):
        super(ResSepConv, self).__init__()
        self.conv = SepConv(C_in, C_out, kernel_size, stride, padding, use_stat_layers=use_stat_layers)
        self.res = Identity() if stride == 1 else FactorizedReduce(C_in, C_out,  use_stat_layers=use_stat_layers)

    def forward(self, x):
        return sum([self.conv(x), self.res(x)])
