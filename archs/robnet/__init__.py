from archs.robnet.robnet import *

def model_entry_v2(model_param, genotype_list, use_stat_layers=False):
    return globals()['robnet'](genotype_list, use_stat_layers, **model_param)
