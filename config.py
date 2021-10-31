
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar2','cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--num-workers', default=16, type=int)
    parser.add_argument('--data-dir', default='/scratch/CIFAR10', type=str)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--use-stat-layers', action='store_true', help='if true, replaces nn modules with stat modules')
    parser.add_argument('--no-norm', action='store_true', help='if true, no data normalization would be used for evaluating the model')
    parser.add_argument('--apply-gdws', action='store_true')


    parser.add_argument('--alphas-filename', default='', help='path to filename containing the pre-computed weight error vectors $\alpha_l$')
    parser.add_argument('--beta',default=0,type=float, help='choose the per-layer error constraint for LEGO algorithm' )
    parser.add_argument('--logfilename', default='pgd-100_linf', type=str, help='choose the output filename')
    parser.add_argument('--sample-size', default=1000, type=int)

    return parser.parse_args()
