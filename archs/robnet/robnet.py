from .basic_model import Network



def robnet(genotype_list, use_stat_layers, **kwargs):
    return Network(genotype_list=genotype_list,use_stat_layers=use_stat_layers, **kwargs)
