import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.functional import interpolate
import torch.nn.functional as F
from archs.resnet import ResNet18
from archs.preactresnet import PreActResNet18, PreActResNet50
from archs.preactresnet_admm import PreActResNet18_ADMM
from archs.vggnet import VGG
from archs.vgg_hydra import vgg16_bn as vgg16_hydra
from archs.wideresnet import wrn_28_4 as wideresnet_28_4
from archs.stat_modules import ModuleStat, Conv2dStat, LinearStat, BatchNorm2dStat
from archs.robnetfree import RobNetFree
from utils import log, rgetattr, rsetattr
import functools


def get_num_classes(dataset: str):
    if dataset == 'cifar10':
        return 10
    elif dataset == 'svhn':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'vww':
        return 2
    else:
        raise ValueError('Invalid dataset name.')

def get_input_size(dataset: str):
    if dataset == 'cifar10':
        return 32
    elif dataset == 'svhn':
        return 32
    elif dataset == 'cifar100':
        return 32
    elif dataset == 'vww':
        return 224
    else:
        raise ValueError('Invalid dataset name.')

def get_architecture(arch: str, dataset: str, use_stat_layers: bool) -> torch.nn.Module:
    num_classes = get_num_classes(dataset)
    if arch == 'preactresnet18':
        model = PreActResNet18(num_classes=num_classes, use_stat_layers=use_stat_layers)
    elif arch == 'preactresnet50':
        model = PreActResNet50(num_classes=num_classes, use_stat_layers=use_stat_layers)
    elif arch == 'vgg16_hydra':
        if use_stat_layers:
            model = vgg16_hydra(conv_layer = Conv2dStat, linear_layer=LinearStat)
        else:
            model = vgg16_hydra()
    elif arch == 'wideresnet_28_4':
        model = wideresnet_28_4(use_stat_layers=use_stat_layers)
        model.sub_block1 = None
    elif arch == 'wideresnetdws_28_4':
        model = wideresnetdws_28_4(use_stat_layers=use_stat_layers)
        model.sub_block1 = None
    elif arch == 'robnetfree':
        model = RobNetFree(use_stat_layers=use_stat_layers)
    elif arch =='preactresnet18_admm':
        model = PreActResNet18_ADMM(use_stat_layers=use_stat_layers)
    elif arch == 'resnet18':
        model = ResNet18(num_classes=num_classes, use_stat_layers=use_stat_layers)
    elif arch.startswith('vgg'):
        model = VGG(vgg_name=arch, num_classes=num_classes, use_stat_layers=use_stat_layers)
    else:
        raise ValueError('Invalid model name.')
    return model
