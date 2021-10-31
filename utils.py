import numpy as np
from collections import namedtuple
import torch
from torch import nn
import torchvision
import os
import shutil
import functools


#################
## utils
#################
def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()

def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()

def read_logfile(filename: str):
    if not os.path.isfile(filename):
        raise ValueError("=> no logfile found at '{}'".format(filename))
    f = open(filename,'r')
    line = f.readline()
    cnt = 1
    return_lists = {}
    keys = line.strip('\n').split('\t')
    for i in range(len(keys)):
        return_lists[keys[i]] = []
    line = f.readline()
    while line:
        #print(line)
        s = line.split('\t')
        for i in range(len(s)):
            return_lists[keys[i]].append(float(s[i]))
        line = f.readline()
        cnt += 1
    f.close()
    return return_lists

def read_logfile_str(filename: str):
    if not os.path.isfile(filename):
        raise ValueError("=> no logfile found at '{}'".format(filename))
    f = open(filename,'r')
    line = f.readline()
    cnt = 1
    return_lists = {}
    keys = line.strip('\n').split('\t')
    for i in range(len(keys)):
        return_lists[keys[i]] = []
    line = f.readline()
    while line:
        #print(line)
        s = line.split('\t')
        for i in range(len(s)):
            return_lists[keys[i]].append((s[i]))
        line = f.readline()
        cnt += 1
    f.close()
    return return_lists


########## setters and getters


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
