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
    from autoattack import AutoAttack
except ImportError:
    print("Error: autoattack not installed. Please run: pip install autoattack")
    exit(1)


parser = argparse.ArgumentParser(description='PyTorch CIFAR AutoAttack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--version', default='standard',
                    help='AutoAttack version: standard, plus, rand')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def eval_adv_test_autoattack(model, device, test_loader):
    """
    evaluate model by AutoAttack
    """
    model.eval()

    # Prepare data
    x_test = []
    y_test = []
    for data, target in test_loader:
        x_test.append(data)
        y_test.append(target)

    x_test = torch.cat(x_test, dim=0).to(device)
    y_test = torch.cat(y_test, dim=0).to(device)

    # Natural accuracy
    with torch.no_grad():
        out = model(x_test)
        natural_err_total = (out.data.max(1)[1] != y_test.data).float().sum()

    # AutoAttack
    print('running AutoAttack...')
    adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, version=args.version, device=device)
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.test_batch_size)

    # Robust accuracy
    with torch.no_grad():
        out_adv = model(x_adv)
        robust_err_total = (out_adv.data.max(1)[1] != y_test.data).float().sum()

    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def main():
    # white-box attack
    print('autoattack')
    model = WideResNet().to(device)
    model.load_state_dict(torch.load(args.model_path))

    eval_adv_test_autoattack(model, device, test_loader)


if __name__ == '__main__':
    main()
