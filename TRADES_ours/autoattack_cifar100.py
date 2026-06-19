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
    print("Error: autoattack not installed. Please run: pip install git+https://github.com/fra31/auto-attack")
    exit(1)


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 AutoAttack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing and AutoAttack (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', type=float, default=0.031,
                    help='perturbation')
parser.add_argument('--model-path',
                    default='./checkpoints/model_cifar100_wrn.pt',
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
testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def eval_adv_test_autoattack(model, device, test_loader):
    """
    evaluate model by AutoAttack (batch-wise to avoid OOM)
    """
    model.eval()

    natural_err_total = 0
    robust_err_total = 0
    total_samples = 0

    # Initialize AutoAttack once
    print('Initializing AutoAttack...')
    adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, version=args.version, device=device)

    # Process the test set batch by batch to avoid evaluating all 10k samples at once.
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        X, y = data.clone().detach(), target.clone().detach()

        # Natural accuracy for this batch
        with torch.no_grad():
            out = model(X)
            natural_err_total += (out.data.max(1)[1] != y.data).float().sum().item()

        # AutoAttack on this batch
        print(f'Running AutoAttack on batch {batch_idx+1}, samples {total_samples} to {total_samples + X.size(0)}...')
        x_adv = adversary.run_standard_evaluation(
            X,
            y,
            bs=min(args.test_batch_size, X.size(0)),
        )

        # Robust accuracy for this batch
        with torch.no_grad():
            out_adv = model(x_adv)
            robust_err_total += (out_adv.data.max(1)[1] != y.data).float().sum().item()

        total_samples += X.size(0)

        # Clear cache after each batch
        if use_cuda:
            torch.cuda.empty_cache()

    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)
    print(f'Total samples: {total_samples}')
    print(f'Natural accuracy: {100.0 - 100.0 * natural_err_total / total_samples:.2f}%')
    print(f'Robust accuracy: {100.0 - 100.0 * robust_err_total / total_samples:.2f}%')


def main():
    # white-box attack
    print('autoattack')
    model = WideResNet(num_classes=100).to(device)
    model.load_state_dict(torch.load(args.model_path))

    eval_adv_test_autoattack(model, device, test_loader)


if __name__ == '__main__':
    main()
