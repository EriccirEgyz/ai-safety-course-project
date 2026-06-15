from __future__ import print_function
import os
import argparse
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from models.wideresnet import *
from models.resnet import *
from trades import trades_loss

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=76, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', type=float, default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', type=float, default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--margin-tau', type=float, default=0.3,
                    help='margin threshold tau for margin-based beta_i')
parser.add_argument('--margin-temperature', type=float, default=0.15,
                    help='temperature T for margin-based beta_i')
parser.add_argument('--beta-i-min', type=float, default=1.0,
                    help='lower bound for clipped per-sample beta_i')
parser.add_argument('--beta-i-max', type=float, default=10.0,
                    help='upper bound for clipped per-sample beta_i')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--metrics-file', default=None,
                    help='CSV file for epoch-level margin/beta_i metrics')
# Save frequency is counted in epochs, not optimizer steps.
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def aggregate_batch_stats(batch_stats):
    total_samples = sum(item['num_samples'] for item in batch_stats)
    summary = {}
    weighted_fields = ['loss_total', 'loss_natural', 'loss_robust']
    for field in weighted_fields:
        summary[field] = sum(item[field] * item['num_samples'] for item in batch_stats) / total_samples
    summary.update(summarize_values('margin', [item['margin_values'] for item in batch_stats]))
    summary.update(summarize_values('beta_i', [item['beta_i_values'] for item in batch_stats]))
    return summary


def summarize_values(prefix, tensors):
    values = torch.cat(tensors).float()
    sorted_values = torch.sort(values)[0]
    last_idx = values.numel() - 1
    return {
        prefix + '_mean': values.mean().item(),
        prefix + '_std': values.std(unbiased=False).item(),
        prefix + '_min': values.min().item(),
        prefix + '_max': values.max().item(),
        prefix + '_p10': sorted_values[int(0.10 * last_idx)].item(),
        prefix + '_p50': sorted_values[int(0.50 * last_idx)].item(),
        prefix + '_p90': sorted_values[int(0.90 * last_idx)].item(),
    }


def append_metrics_row(metrics_path, row):
    file_exists = os.path.exists(metrics_path)
    with open(metrics_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    batch_stats = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss, stats = trades_loss(model=model,
                                  x_natural=data,
                                  y=target,
                                  optimizer=optimizer,
                                  step_size=args.step_size,
                                  epsilon=args.epsilon,
                                  perturb_steps=args.num_steps,
                                  beta=args.beta,
                                  margin_tau=args.margin_tau,
                                  margin_temperature=args.margin_temperature,
                                  beta_i_min=args.beta_i_min,
                                  beta_i_max=args.beta_i_max,
                                  return_stats=True)
        batch_stats.append(stats)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return aggregate_batch_stats(batch_stats)


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # MODIFIED: size_average=False 已废弃，改为 reduction='sum'
            train_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # MODIFIED: size_average=False 已废弃，改为 reduction='sum'
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, ResNet18() can be also used here for training
    model = WideResNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    metrics_path = args.metrics_file or os.path.join(model_dir, 'margin_beta_metrics.csv')

    total_start_time = time.perf_counter()
    print('Training started at: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print('Metrics CSV: {}'.format(metrics_path))

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()

        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train_start_time = time.perf_counter()
        training_stats = train(args, model, device, train_loader, optimizer, epoch)
        train_time = time.perf_counter() - train_start_time

        # evaluation on natural examples
        eval_start_time = time.perf_counter()
        print('================================================================')
        clean_train_loss, clean_train_acc = eval_train(model, device, train_loader)
        clean_test_loss, clean_test_acc = eval_test(model, device, test_loader)
        print('================================================================')
        eval_time = time.perf_counter() - eval_start_time

        # save checkpoint
        save_time = 0.0
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            save_start_time = time.perf_counter()
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))
            save_time = time.perf_counter() - save_start_time

        epoch_time = time.perf_counter() - epoch_start_time
        elapsed_time = time.perf_counter() - total_start_time
        row = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'clean_train_loss': clean_train_loss,
            'clean_train_acc': clean_train_acc,
            'clean_test_loss': clean_test_loss,
            'clean_test_acc': clean_test_acc,
            'beta': args.beta,
            'margin_tau': args.margin_tau,
            'margin_temperature': args.margin_temperature,
            'beta_i_min_bound': args.beta_i_min,
            'beta_i_max_bound': args.beta_i_max,
            'train_time_sec': train_time,
            'eval_time_sec': eval_time,
            'save_time_sec': save_time,
            'epoch_time_sec': epoch_time,
            'elapsed_time_sec': elapsed_time,
        }
        row.update(training_stats)
        append_metrics_row(metrics_path, row)
        print('Margin/Beta Epoch {}: margin mean {:.4f}, p10 {:.4f}, p50 {:.4f}, p90 {:.4f}; '
              'beta_i mean {:.4f}, std {:.4f}, min {:.4f}, max {:.4f}'.format(
                  epoch,
                  training_stats['margin_mean'],
                  training_stats['margin_p10'],
                  training_stats['margin_p50'],
                  training_stats['margin_p90'],
                  training_stats['beta_i_mean'],
                  training_stats['beta_i_std'],
                  training_stats['beta_i_min'],
                  training_stats['beta_i_max']))
        print('Time Epoch {}: train {:.2f}s, eval {:.2f}s, save {:.2f}s, epoch {:.2f}s, elapsed {:.2f}s'.format(
            epoch, train_time, eval_time, save_time, epoch_time, elapsed_time))

    total_time = time.perf_counter() - total_start_time
    print('Training finished at: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print('Total training time: {:.2f}s ({:.2f}h)'.format(total_time, total_time / 3600.0))


if __name__ == '__main__':
    main()
