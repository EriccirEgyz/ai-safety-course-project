"""Q1: does TRADES leave far-from-boundary samples still vulnerable under PGD-20?

For every test sample we compute the clean adversarial margin
    m_i = f_{y_i}(x_i) - max_{j != y_i} f_j(x_i)
then bin all samples into --num_bins equal-population quantile buckets and
report clean accuracy and PGD-20 robust accuracy per bin.

Self-contained: WideResNet is imported from TRADES_ours/models via sys.path.
The PGD-20 attack mirrors TRADES_ours/pgd_attack_cifar10.py:_pgd_whitebox
(random uniform init inside the eps-ball, sign-step, L_inf eps-ball projection
around x_natural, [0,1] box clamp) but is inlined here rather than imported,
because pgd_attack_cifar10.py bakes its defaults from a module-level
argparse.parse_args() that fires at import time.

Run from repo root:
    python eval/margin_stratified_robust_acc.py --ckpt PATH [--dataset cifar10]
"""
from __future__ import print_function

import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRADES_OURS = os.path.join(_REPO_ROOT, 'TRADES_ours')
if _TRADES_OURS not in sys.path:
    sys.path.insert(0, _TRADES_OURS)
from models.wideresnet import WideResNet


def parse_args():
    p = argparse.ArgumentParser(description='Q1: margin-stratified clean vs PGD-20 robust accuracy')
    p.add_argument('--ckpt', type=str, required=True,
                   help='checkpoint path (.pt state_dict for WideResNet)')
    p.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    p.add_argument('--eps', type=float, default=0.031, help='L_inf perturbation budget')
    p.add_argument('--step', type=float, default=0.003, help='PGD step size')
    p.add_argument('--num_steps', type=int, default=20, help='PGD steps')
    p.add_argument('--num_bins', type=int, default=5, help='equal-population margin bins')
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--out_csv', type=str,
                   default=os.path.join(_REPO_ROOT, 'results', 'q1_margin_stratified.csv'))
    p.add_argument('--data-dir', type=str,
                   default=os.path.join(_REPO_ROOT, '..', 'data'),
                   help='torchvision dataset root')
    p.add_argument('--no-cuda', action='store_true', default=False)
    return p.parse_args()


def load_test_loader(dataset, batch_size, data_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == 'cifar10':
        ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    else:
        ds = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1)


def build_model(dataset, device):
    num_classes = 100 if dataset == 'cifar100' else 10
    return WideResNet(num_classes=num_classes).to(device)


def pgd_whitebox(model, X, y, epsilon, num_steps, step_size):
    """L_inf PGD on cross-entropy. Mirrors TRADES_ours/pgd_attack_cifar10.py:_pgd_whitebox."""
    X_pgd = X.clone().detach()
    # Random uniform init inside the eps-ball, then clamp to the [0,1] box.
    noise = torch.empty_like(X_pgd).uniform_(-epsilon, epsilon)
    X_pgd = (X_pgd + noise).clamp(0.0, 1.0).detach().requires_grad_(True)
    for _ in range(num_steps):
        with torch.enable_grad():
            loss = F.cross_entropy(model(X_pgd), y)
        grad, = torch.autograd.grad(loss, [X_pgd])
        eta = step_size * grad.sign()
        X_pgd = X_pgd.detach() + eta
        # Project back into the L_inf eps-ball around X.
        X_pgd = torch.max(torch.min(X_pgd, X + epsilon), X - epsilon)
        # Clamp to the valid pixel box and re-enable grad for the next step.
        X_pgd = torch.clamp(X_pgd, 0.0, 1.0).detach().requires_grad_(True)
    return X_pgd.detach()


def clean_margin(logits, y, num_classes):
    """m_i = f_{y_i}(x_i) - max_{j != y_i} f_j(x_i)."""
    true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
    other_logits = logits.masked_fill(
        F.one_hot(y, num_classes=num_classes).bool(), float('-inf'))
    max_other = other_logits.max(dim=1)[0]
    return true_logits - max_other


def equal_population_bins(margins, num_bins):
    """Return (num_bins+1,) bin edges from equal-population quantiles of margins."""
    q = torch.linspace(0.0, 1.0, num_bins + 1, dtype=margins.dtype)
    return torch.quantile(margins, q).tolist()


def assign_bins(margins, edges):
    """Bin index per sample. Internal edges belong to the upper bin; outer edges
    are inclusive so no sample is dropped at the population extremes."""
    n_bins = len(edges) - 1
    bin_idx = torch.full_like(margins, -1, dtype=torch.long)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == 0:
            mask = (margins >= lo) & (margins <= hi)
        else:
            mask = (margins > lo) & (margins <= hi)
        bin_idx[mask] = b
    return bin_idx


def main():
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    num_classes = 100 if args.dataset == 'cifar100' else 10
    test_loader = load_test_loader(args.dataset, args.batch_size, args.data_dir)
    model = build_model(args.dataset, device)
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()

    margins_all, clean_correct_all, pgd_correct_all = [], [], []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            logits = model(data)
            margins = clean_margin(logits, target, num_classes)
            clean_correct_all.append((logits.argmax(1) == target).float().cpu())
            margins_all.append(margins.cpu())

        x_adv = pgd_whitebox(model, data, target, args.eps, args.num_steps, args.step)
        with torch.no_grad():
            adv_logits = model(x_adv)
            pgd_correct_all.append((adv_logits.argmax(1) == target).float().cpu())

    margins = torch.cat(margins_all)
    clean_correct = torch.cat(clean_correct_all)
    pgd_correct = torch.cat(pgd_correct_all)

    edges = equal_population_bins(margins, args.num_bins)
    bin_idx = assign_bins(margins, edges)

    rows = []
    for b in range(args.num_bins):
        mask = (bin_idx == b)
        n = int(mask.sum().item())
        if n > 0:
            c_acc = float(clean_correct[mask].mean().item())
            p_acc = float(pgd_correct[mask].mean().item())
        else:
            c_acc = 0.0
            p_acc = 0.0
        rows.append((b, edges[b], edges[b + 1], n, c_acc, p_acc))

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bin', 'margin_low', 'margin_high', 'n', 'clean_acc', 'pgd20_acc'])
        for b, lo, hi, n, c_acc, p_acc in rows:
            w.writerow([b, f'{lo:.6f}', f'{hi:.6f}', n, f'{c_acc:.6f}', f'{p_acc:.6f}'])

    print('Wrote {}'.format(args.out_csv))
    print()
    print('{:>3} {:>12} {:>12} {:>7} {:>10} {:>10}'.format(
        'bin', 'margin_low', 'margin_high', 'n', 'clean_acc', 'pgd20_acc'))
    for b, lo, hi, n, c_acc, p_acc in rows:
        print('{:>3} {:>12.6f} {:>12.6f} {:>7} {:>10.4f} {:>10.4f}'.format(
            b, lo, hi, n, c_acc, p_acc))


if __name__ == '__main__':
    main()
