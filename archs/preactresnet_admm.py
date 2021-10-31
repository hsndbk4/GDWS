'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.stat_modules import Conv2dStat, LinearStat, BatchNorm2dStat

class PreActBlock_ADMM(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, Conv2d = nn.Conv2d, BatchNorm2d = nn.BatchNorm2d):
        super(PreActBlock_ADMM, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        #print(out.shape)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck_ADMM(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, Conv2d = nn.Conv2d, BatchNorm2d = nn.BatchNorm2d):
        super(PreActBottleneck_ADMM, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet_ADMM(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_stat_layers=False):
        super(PreActResNet_ADMM, self).__init__()
        self.in_planes = 64
        Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
        Linear = LinearStat if use_stat_layers else nn.Linear
        BatchNorm2d = BatchNorm2dStat if use_stat_layers else nn.BatchNorm2d
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,  Conv2d = Conv2d, BatchNorm2d = BatchNorm2d)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,  Conv2d = Conv2d, BatchNorm2d = BatchNorm2d)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,  Conv2d = Conv2d, BatchNorm2d = BatchNorm2d)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, Conv2d = Conv2d, BatchNorm2d = BatchNorm2d)
        self.linear = Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, Conv2d = nn.Conv2d, BatchNorm2d = nn.BatchNorm2d):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, Conv2d = Conv2d, BatchNorm2d = BatchNorm2d))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18_ADMM(use_stat_layers=False):
    return PreActResNet_ADMM(PreActBlock_ADMM, [2,2,2,2], use_stat_layers=use_stat_layers)

def PreActResNet34_ADMM():
    return PreActResNet_ADMM(PreActBlock_ADMM, [3,4,6,3])

def PreActResNet50_ADMM():
    return PreActResNet_ADMM(PreActBottleneck_ADMM, [3,4,6,3])

def PreActResNet101_ADMM():
    return PreActResNet_ADMM(PreActBottleneck_ADMM, [3,4,23,3])

def PreActResNet152_ADMM():
    return PreActResNet_ADMM(PreActBottleneck_ADMM, [3,8,36,3])


def test():
    net = PreActResNet18_ADMM()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
