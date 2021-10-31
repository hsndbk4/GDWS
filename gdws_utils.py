import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.functional import interpolate
import torch.nn.functional as F
from archs.stat_modules import ModuleStat, Conv2dStat, LinearStat, BatchNorm2dStat
from utils import log, rgetattr, rsetattr


class ChannelRepeater(nn.Module):
    def __init__(self, x_indx,*kargs, **kwargs):
        super(ChannelRepeater, self).__init__(*kargs, **kwargs)
        self.register_buffer('x_indx', x_indx)
        self.G = x_indx.size()[0]
        self.C = x_indx.max().item()+1
    def forward(self, x):
        return x[:,self.x_indx,:,:]

    def extra_repr(self):
        return '{C}=>{G}'.format(**self.__dict__)


# Algorithm 3 in our paper
def apply_gdws_approx_LEGO(model: torch.nn.Module, beta: float, alphas: dict, use_stat_layers: bool=False, explicit: bool=True, skip_first: bool=True) -> None:
    #first fetch the layer
    Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
    first_layer = skip_first
    for n,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            ## assumes bias=False as well
            print(n)
            if first_layer == True:
                first_layer = False
                print('First conv layer skipped')
                continue
            W = m.weight.data.clone()
            M,C,K,K = W.shape
            if K==1:
                print('PW layer skipped')
                continue
            alpha = alphas[n+'.weight']
            W = W.view(M,-1)
            W_pw, W_gdw, G_dist,G_eff = LEGO(W,K*K,alpha,beta)
            if explicit: ## explicit decomposition, replace the conv layer with a GDW layer and a PW layer
                conv_gdw = Conv2d(G_eff,G_eff,K,groups=G_eff,stride=m.stride, padding=m.padding, bias=False).to(W.device)
                if m.bias is None:
                    conv_pw = Conv2d(G_eff,M,1,bias=False).to(W.device)
                else:
                    conv_pw = Conv2d(G_eff,M,1,bias=True).to(W.device)
                    conv_pw.bias = m.bias
                W_gdw_s = torch.zeros(conv_gdw.weight.shape).to(W.device)
                x_indx = torch.LongTensor(G_eff).to(W.device).to(W.device)
                g=0
                for c in range(C):
                    g_len = G_dist[c]
                    for l in range(g_len):
                        W_gdw_s[g,0,:,:] = W_gdw[g,c*K*K:(c+1)*K*K].reshape(K,K)
                        x_indx[g]=c
                        g = g+1
                channel_repeater = ChannelRepeater(x_indx)
                conv_gdw.weight.data = W_gdw_s
                conv_pw.weight.data = W_pw.reshape(M,G_eff,1,1)
                conv_gdws = nn.Sequential(channel_repeater,conv_gdw,conv_pw)
                rsetattr(model, n, conv_gdws)
            print('MACs savings per output channel = ', (M*C*K*K)/(M*G_eff + G_eff*K*K))
            if not explicit:
                W_gdws = W_pw@W_gdw
                W_gdws = W_gdws.view(M,C,K,K)
                m.weight.data = W_gdws
    print(model)

# MEGO: Complexity-constrained minimum-error GDWS
# W: M x CK^2 weight matrix of a standard 2D convolution
# K2: K*K, total number of filer elements per channel
# alpha: weight error vector
# G: the upper bound for the number of GDW kernels
def MEGO(W, K2, alpha, G):
    [M,N] = W.size()
    #N = K*K*C
    device = W.device
    assert(N%K2==0)
    W_splits = torch.split(W.data.clone(),K2,dim=1)
    U_splits = []
    S_splits = []
    V_splits = []

    idx=0
    for W_s in W_splits:
        U_s,S_s,V_s = torch.svd(W_s)
        U_splits.append(U_s)
        S_splits.append(S_s)
        V_splits.append(V_s)

    S_splits = torch.stack(S_splits,dim=0)
    [m,n] = S_splits.size()
    S_mat = S_splits**2
    S_mat = S_mat * alpha.unsqueeze(1)
    S_arr = S_mat.view(1,-1)
    #S_arr = S_splits.view(1,-1)
    _,top_indx = torch.topk(S_arr,k=G)

    top_indx = top_indx.view(-1).tolist()

    S_arr_dup = S_arr.data.clone().view(-1)
    S_arr_dup[top_indx] = 0
    top_indx = [(i//n,i%n) for i in top_indx]
    top_indx_sorted = sorted(top_indx,key= lambda x: x[0])
    #DW kernel distribution across the channels
    C = N//K2
    G_dist = torch.LongTensor(C).fill_(0).to(device)
    for e in top_indx_sorted:
        G_dist[e[0]]+=1

    ##given the top indices, need to construct the corresponding W_pw and W_dw
    W_dw = torch.zeros(G,N).to(device)
    W_pw = torch.zeros(M,G).to(device)
    for g,i_pair in enumerate(top_indx_sorted):
        U_g = U_splits[i_pair[0]]
        u = U_g[:,i_pair[1]]*S_splits[i_pair]
        V_g = V_splits[i_pair[0]]
        v = V_g[:,i_pair[1]].reshape(V_g.size(0),-1)

        W_pw[:,g] = u
        W_dw[g,i_pair[0]*L:(i_pair[0]+1)*L] = v.t()
    return W_pw, W_dw, G_dist

# LEGO: Error-constrained minimum-complexity GDWS
# W: M x CK^2 weight matrix of a standard 2D convolution
# K2: K*K, total number of filer elements per channel
# alpha: weight error vector
# beta: the upper bound for the squared error, note that this is a slight abuse of notation in the paper
def LEGO(W, K2, alpha, beta):
    [M,N] = W.size()
    #N = K*K*C
    device = W.device
    assert(N%K2==0)
    W_splits = torch.split(W.data.clone(),K2,dim=1)
    U_splits = []
    S_splits = []
    V_splits = []
    C = len(W_splits)

    for W_s in W_splits:
        U_s,S_s,V_s = torch.svd(W_s)
        U_splits.append(U_s)
        S_splits.append(S_s)
        V_splits.append(V_s)
    S_splits = torch.stack(S_splits,dim=0) #matrix of the singular values
    [m,n] = S_splits.size()
    S_mat = S_splits**2
    S_mat = S_mat * alpha.unsqueeze(1)
    S_arr = S_mat.view(1,-1)
    S_arr_sorted,indices = torch.sort(S_arr,descending=True)
    budget_gap = beta
    G = K2*C

    S_arr_sorted = S_arr_sorted.squeeze()
    indices = indices.squeeze()
    budget_so_far=0
    g=G
    #limit = 1.3
    for i in range(G):
        if budget_so_far+S_arr_sorted[g-1]>beta:
            break
        budget_so_far+=S_arr_sorted[g-1]
        g-=1
    G=g
    top_indx = indices[0:G]


    top_indx = top_indx.view(-1).tolist()
    top_indx = [(i//n,i%n) for i in top_indx]
    top_indx_sorted = sorted(top_indx,key= lambda x: x[0])
    #GDW channel distribution across the channels
    G_dist = torch.LongTensor(C).fill_(0).to(device)
    for e in top_indx_sorted:
        G_dist[e[0]]+=1

    ##given the top indices, need to construct the corresponding W_pw and W_gdw
    W_gdw = torch.zeros(G,N).to(device)
    W_pw = torch.zeros(M,G).to(device)
    for g,i_pair in enumerate(top_indx_sorted):
        U_g = U_splits[i_pair[0]]
        u = U_g[:,i_pair[1]]*S_splits[i_pair]
        V_g = V_splits[i_pair[0]]
        v = V_g[:,i_pair[1]].reshape(V_g.size(0),-1)

        W_pw[:,g] = u
        W_gdw[g,i_pair[0]*K2:(i_pair[0]+1)*K2] = v.t()

    return W_pw, W_gdw, G_dist,G
