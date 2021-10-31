import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

from utils import *
from architectures import get_architecture
import config
from datasets import get_normalize, get_dataloaders


upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False, normalize=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta






def main():
    args = config.get_args()

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)
    # Copies files to the outdir to store complete script with each experiment

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.batch_size!=1:
        raise ValueError('Noise gains can only be computed with batch size equal to 1')

    test_loader, train_loader = get_dataloaders(args)
    normalize = get_normalize(args)
    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)


    model = get_architecture(args.model, args.dataset, args.use_stat_layers)

    model = nn.DataParallel(model).cuda()
    model.train()

    ##loading the model
    if 'hydra' in args.fname:
        ckpt = torch.load(os.path.join(args.fname, f'model_best_dense.pth'),map_location='cuda:0')
        state_dict = {}
        for key in ckpt['state_dict']:
            state_dict['module.'+key] = ckpt['state_dict'][key]
        model.load_state_dict(state_dict)
        print('HYDRA model loaded from: ',os.path.join(args.fname, f'model_best_dense.pth'))
    elif 'admm' in args.fname:
        ckpt_filename = os.path.join(args.fname, f'pretrained.pt')
        if not os.path.isfile(ckpt_filename):
            ckpt_filename = os.path.join(args.fname, f'retrained_bn.pt')
        ckpt = torch.load(ckpt_filename)
        new_ckpt = ckpt.copy()
        for key in ckpt.keys():
            if 'basic_model' in key:
                new_key = key.split('.')
                new_key.remove('basic_model')
                new_key='.'.join(new_key)
                new_ckpt[new_key] = new_ckpt.pop(key)
        model.load_state_dict(new_ckpt)
        print('ADMM model loaded from: ', ckpt_filename)
    elif 'msd' in args.fname:
        ckpt_filename = os.path.join(args.fname, f'MSD_V0.pt')
        ckpt = torch.load(ckpt_filename)
        state_dict = {}
        for key in ckpt:
            state_dict['module.'+key] = ckpt[key]
        model.load_state_dict(state_dict)
        print('MSD model loaded from: ', ckpt_filename)
    elif 'nas' in args.fname:
        ckpt_filename = os.path.join(args.fname, f'RobNet_free_cifar10.pth.tar')
        def map_func(storage, location):
            return storage.cuda()

        print("=> loading checkpoint '{}'".format(ckpt_filename))

        checkpoint = torch.load(ckpt_filename, map_location=map_func)
        state_dict = {}
        for key in checkpoint['state_dict']:
            state_dict['module.'+key] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict, strict=False)

        ckpt_keys = set(state_dict.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print('caution: missing keys from checkpoint {}: {}'.format(ckpt_filename, k))
        print('NAS model loaded from: ', ckpt_filename)
    else:
        ckpt = torch.load(os.path.join(args.fname, f'model_best.pth'))
        model.load_state_dict(ckpt['state_dict'])

    params = model.parameters()
    noise_gains = {} #need to populate this dict
    weight_dict = {}
    #structure:
    #noise_gains['weight_name_in_model_dict'] is an array of noise gains, since each weight matrix is divided into multiple ranks

    first_layer=True
    for name,param in model.named_parameters():
        sh = param.shape
        if len(sh)==4: # conv layer
            M,C,K,K = param.shape
            if first_layer:
                first_layer = False
                continue

            elif K==1: ## PW layer
                continue
            else: ## K*K conv layer
                noise_gains[name] = torch.zeros(C).to(param.device)
                weight_dict[name] = param

    if args.attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    for i, (X, y) in enumerate(train_loader):
        if i >= args.sample_size:
            break
        print('progress {}/{}'.format(i,args.sample_size))
        X, y = X.cuda(), y.cuda()
        if args.attack == 'pgd':
            # Random initialization
            delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, normalize=normalize)
            delta = delta.detach()
        elif args.attack == 'fgsm':
            delta = attack_pgd(model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm, normalize=normalize)
        # Standard training
        elif args.attack == 'none':
            delta = torch.zeros_like(X)
        robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
        robust_output = robust_output.view(-1)
        y_top = robust_output.max()
        c_top = robust_output.argmax()
        for c in range(len(robust_output)):
            if c==c_top:
                continue
            output_diff = y_top - robust_output[c]
            output_diff.backward(retain_graph=True)
            with torch.no_grad():
                denominator=2*(output_diff**2)
                for key in noise_gains.keys():
                    grad = weight_dict[key].grad
                    M,C,K,K = grad.shape
                    grad = grad.view(M,-1)
                    g_splits = torch.split(grad,K*K,dim=1)
                    ng_splits = torch.stack([(g**2).sum()/(denominator*M*K*K) for g in g_splits])
                    noise_gains[key].add_(ng_splits)
                    weight_dict[key].grad.zero_()
    for key in noise_gains.keys():
        noise_gains[key].div_(args.sample_size)
    filename =args.alphas_filename + '.pth'
    torch.save({
            'noise_gains':noise_gains,
            'attack':args.attack,
            'attack-iters':args.attack_iters,
            'epsilon':args.epsilon,
            'sample_size': args.sample_size}, os.path.join(args.fname, filename))

if __name__ == "__main__":
    main()
