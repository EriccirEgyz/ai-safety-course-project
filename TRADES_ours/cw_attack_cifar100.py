from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *

try:
    import torchattacks
except ImportError:
    print("Error: torchattacks not installed. Please run: pip install torchattacks")
    exit(1)


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 C&W Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--c', type=float, default=1.0,
                    help='c parameter in C&W attack')
parser.add_argument('--kappa', type=float, default=0.0,
                    help='confidence parameter')
parser.add_argument('--num-steps', type=int, default=1000,
                    help='number of optimization steps')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate for C&W optimizer')
parser.add_argument('--model-path',
                    default='./checkpoints/model_cifar100_wrn.pt',
                    help='model for white-box attack evaluation')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def _cw_whitebox(model, X, y):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    # C&W attack
    attack = torchattacks.CW(model, c=args.c, kappa=args.kappa, steps=args.num_steps, lr=args.lr)
    X_cw = attack(X, y)

    err_cw = (model(X_cw).data.max(1)[1] != y.data).float().sum()
    print('err cw (white-box): ', err_cw)
    return err, err_cw


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # cw attack
        X, y = data.clone().detach(), target.clone().detach()
        err_natural, err_robust = _cw_whitebox(model, X, y)
        robust_err_total += err_robust.item()
        natural_err_total += err_natural.item()
        if use_cuda:
            torch.cuda.empty_cache()
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def main():
    # white-box attack
    print('cw white-box attack')
    model = WideResNet(num_classes=100).to(device)
    model.load_state_dict(torch.load(args.model_path))

    eval_adv_test_whitebox(model, device, test_loader)


if __name__ == '__main__':
    main()
