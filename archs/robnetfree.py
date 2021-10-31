import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
from archs.stat_modules import Conv2dStat, LinearStat, BatchNorm2dStat
from archs.robnet import model_entry_v2

# loading the robnet-free model for CIFAR-10
# code from (https://github.com/gmh14/RobNets)

model = 'robnet_free'
model_param = dict(C=36,
                   num_classes=10,
                   layers=20,
                   steps=4,
                   multiplier=4,
                   stem_multiplier=3,
                   share=False,
                   AdPoolSize=1)

arch_code = [['11', '10',
                '11', '11',
                '10', '11',
                '01', '01',
                '10', '11',
                '11', '00',
                '11', '01'],
               ['11', '00',
                '01', '10',
                '01', '00',
                '10', '10',
                '00', '01',
                '11', '10',
                '11', '00'],
               ['11', '11',
                '11', '01',
                '01', '11',
                '01', '00',
                '10', '00',
                '10', '10',
                '00', '11'],
               ['10', '11',
                '10', '01',
                '01', '10',
                '10', '01',
                '10', '11',
                '00', '00',
                '01', '10'],
               ['11', '11',
                '01', '00',
                '10', '10',
                '10', '01',
                '10', '01',
                '00', '10',
                '01', '11'],
               ['11', '10',
                '11', '11',
                '11', '01',
                '10', '11',
                '00', '10',
                '01', '11',
                '01', '11'],
               ['11', '11',
                '11', '10',
                '10', '01',
                '11', '10',
                '01', '10',
                '10', '10',
                '01', '10'],
               ['01', '11',
                '11', '11',
                '01', '11',
                '11', '11',
                '01', '11',
                '01', '11',
                '10', '00'],
               ['11', '11',
                '11', '11',
                '11', '01',
                '01', '11',
                '10', '01',
                '00', '10',
                '01', '11'],
               ['10', '11',
                '01', '00',
                '11', '11',
                '10', '11',
                '01', '11',
                '11', '11',
                '11', '00'],
               ['11', '10',
                '11', '00',
                '00', '00',
                '11', '00',
                '01', '10',
                '00', '01',
                '10', '11'],
               ['01', '11',
                '01', '11',
                '11', '10',
                '10', '11',
                '10', '11',
                '01', '11',
                '10', '00'],
               ['11', '11',
                '10', '10',
                '01', '00',
                '10', '11',
                '11', '01',
                '10', '10',
                '00', '01'],
               ['01', '11',
                '11', '11',
                '01', '01',
                '11', '01',
                '01', '11',
                '11', '01',
                '10', '10'],
               ['01', '11',
                '11', '11',
                '11', '01',
                '00', '01',
                '10', '10',
                '11', '10',
                '10', '11'],
               ['11', '11',
                '00', '11',
                '11', '01',
                '00', '01',
                '10', '00',
                '11', '01',
                '11', '11'],
               ['11', '11',
                '01', '11',
                '11', '10',
                '11', '10',
                '10', '10',
                '10', '10',
                '11', '11'],
               ['10', '11',
                '01', '11',
                '11', '01',
                '11', '00',
                '11', '11',
                '00', '10',
                '00', '01'],
               ['11', '10',
                '11', '11',
                '11', '11',
                '11', '10',
                '11', '00',
                '11', '01',
                '11', '11'],
               ['11', '11',
                '01', '01',
                '11', '11',
                '01', '00',
                '00', '10',
                '00', '01',
                '01', '11']]


def RobNetFree(use_stat_layers=False):
    #arch_code = eval('architecture_code.{}'.format(model))
    net = model_entry_v2(model_param, arch_code,use_stat_layers=use_stat_layers)
    return net
