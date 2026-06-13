"""Q2: during short TRADES training, is KL acting as a gradient obstacle for
near-boundary samples?

Trades-style training with a SCALAR beta (not per-sample beta_i) for a few
epochs. Every --log_every batches we:
  1. Bin the current batch's clean margins using edges fixed at start-of-run
     by a small calibration pass.
  2. Stratify-sample up to --grad_subsample examples across bins.
  3. For each sampled i, compute the two per-sample gradients w.r.t. theta:
       g_CE_i = grad of CE(f(x_adv_i), y_i)
       g_KL_i = grad of KL(p(x_i) || p(x_adv_i))
     and record cos(g_CE_i, g_KL_i).
  4. Also record per-bin variance of the per-sample TRADES loss in that batch.

=== Design choices documented per task spec ===

Per-sample gradients: autograd + retain_graph, NOT torch.func.vmap+grad.
Reason: WideResNet uses BatchNorm, and vmap+functional_call over a BN module
in train mode would compute per-sample batch statistics, diverging from the
model's actual training-time behavior. The autograd+retain_graph path reuses
one batched forward pass and just runs N scalar backward passes through it,
which keeps BN semantics intact (modulo the eval-mode approximation below).

BatchNorm during measurement: the per-sample grad pass runs in model.eval()
mode so the per-sample forward passes do not pollute the running stats.
Dropout is also disabled. This is a documented approximation: we trade
exact train-mode BN/dropout behavior for clean per-sample gradients and
unpolluted running stats. The qualitative "is KL an obstacle" signal is
robust to BN mode.

Bin edges: one calibration pass at start-of-run (first few batches in eval
mode), then frozen. Simpler than maintaining a running quantile estimator;
the cost is that early-run edges may not match late-run margin distributions,
so per-bin populations can become uneven -- the per-bin `n` is recoverable
from the per-(epoch,batch,bin) cos_phi counts.

Usage (run from repo root):
    python train/log_grad_angle.py [--init_ckpt PATH] [--epochs 5] [--beta 6]
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
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
    p = argparse.ArgumentParser(description='Q2: log per-sample CE-vs-KL grad angle during short TRADES training')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--init_ckpt', type=str, default=None,
                   help='optional checkpoint; default is random init matching the natural baseline config')
    p.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    p.add_argument('--beta', type=float, default=6.0,
                   help='SCALAR beta (not per-sample beta_i); clean CE-vs-KL conflict signal at uniform weight')
    p.add_argument('--num_bins', type=int, default=5)
    p.add_argument('--log_every', type=int, default=20,
                   help='compute per-sample grad angles once every K batches')
    p.add_argument('--grad_subsample', type=int, default=32,
                   help='within a logged batch, sample up to this many examples, stratified across bins')
    p.add_argument('--out_npz', type=str,
                   default=os.path.join(_REPO_ROOT, 'results', 'q2_grad_angle.npz'))
    # Reproduce TRADES_ours/train_natural_cifar10.py / train_trades_cifar10.py defaults.
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=2e-4)
    p.add_argument('--eps', type=float, default=0.031)
    p.add_argument('--step', type=float, default=0.007,
                   help='TRADES PGD step size on KL (matches train_trades_cifar10.py default)')
    p.add_argument('--num_steps', type=int, default=10,
                   help='TRADES PGD steps on KL (matches train_trades_cifar10.py default)')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--data_dir', type=str, default=os.path.join(_REPO_ROOT, 'data'))
    p.add_argument('--cal_batches', type=int, default=5,
                   help='number of batches used by the start-of-run bin-edge calibration pass')
    p.add_argument('--no_cuda', action='store_true', default=False)
    return p.parse_args()


def clean_margin(logits, y, num_classes):
    """m_i = f_{y_i}(x_i) - max_{j != y_i} f_j(x_i)."""
    true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
    other_logits = logits.masked_fill(
        F.one_hot(y, num_classes=num_classes).bool(), float('-inf'))
    max_other = other_logits.max(dim=1)[0]
    return true_logits - max_other


def assign_bins(margins, edges):
    """Bin index per sample given (num_bins+1,) edges. Internal edges belong
    to the upper bin; outer edges are inclusive so extremes are not dropped."""
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


def calibrate_bin_edges(model, train_loader, device, num_classes, num_bins, num_batches):
    """One-time calibration: equal-population quantile edges from margins
    observed on the first num_batches batches in eval mode."""
    was_training = model.training
    model.eval()
    margins = []
    with torch.no_grad():
        for b, (data, target) in enumerate(train_loader):
            if b >= num_batches:
                break
            data, target = data.to(device), target.to(device)
            logits = model(data)
            margins.append(clean_margin(logits, target, num_classes).cpu())
    if was_training:
        model.train()
    margins = torch.cat(margins).float()
    q = torch.linspace(0.0, 1.0, num_bins + 1, dtype=margins.dtype)
    edges = torch.quantile(margins, q)
    return edges.to(device)


def generate_trades_adv(model, x_natural, step_size, epsilon, perturb_steps):
    """TRADES adversarial example: PGD on KL(p(adv) || p(natural)).
    Mirrors TRADES_ours/trades.py:trades_loss l_inf branch."""
    criterion_kl = torch.nn.KLDivLoss(reduction='sum')
    was_training = model.training
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    if was_training:
        model.train()
    return x_adv.detach()


def compute_per_sample_trades_loss(model, x_nat, x_adv, y, beta):
    """Per-sample TRADES loss: CE(natural) + beta * KL(p_nat || p_adv).
    No-grad; used for per-bin variance only."""
    with torch.no_grad():
        logits_nat = model(x_nat)
        logits_adv = model(x_adv)
        ce = F.cross_entropy(logits_nat, y, reduction='none')
        kl = F.kl_div(F.log_softmax(logits_adv, dim=1),
                      F.softmax(logits_nat, dim=1),
                      reduction='none').sum(dim=1)
        return ce + beta * kl


def compute_per_sample_grad_angles(model, x_nat, x_adv, y, sampled_idx):
    """For each i in sampled_idx, return cos(g_CE_i, g_KL_i) where
       g_CE_i = grad of CE(f(x_adv_i), y_i)         w.r.t. theta
       g_KL_i = grad of KL(p(x_i) || p(x_adv_i))    w.r.t. theta
    Uses autograd + retain_graph (see file header for rationale vs torch.func).
    Runs in eval mode to avoid polluting BatchNorm running stats.
    """
    was_training = model.training
    model.eval()
    try:
        params = list(model.parameters())
        idx = sampled_idx
        # One batched forward pass over the sampled subset; the graph is reused
        # across all per-sample scalar backward passes via retain_graph=True.
        logits_nat = model(x_nat[idx])
        logits_adv = model(x_adv[idx])
        y_sub = y[idx]

        cos_phis = []
        n = int(idx.shape[0])
        for i in range(n):
            ce_i = F.cross_entropy(logits_adv[i:i + 1], y_sub[i:i + 1], reduction='sum')
            grad_ce = torch.autograd.grad(ce_i, params, retain_graph=True)

            p_nat_i = F.softmax(logits_nat[i:i + 1], dim=1)
            log_p_adv_i = F.log_softmax(logits_adv[i:i + 1], dim=1)
            kl_i = F.kl_div(log_p_adv_i, p_nat_i, reduction='sum')
            grad_kl = torch.autograd.grad(kl_i, params, retain_graph=True)

            g_ce_flat = torch.cat([g.detach().reshape(-1) for g in grad_ce])
            g_kl_flat = torch.cat([g.detach().reshape(-1) for g in grad_kl])
            cos = (g_ce_flat * g_kl_flat).sum() / \
                  (g_ce_flat.norm() * g_kl_flat.norm() + 1e-12)
            cos_phis.append(float(cos.item()))
        return cos_phis
    finally:
        if was_training:
            model.train()


def stratified_sample(bin_idx, num_bins, k):
    """Up to k indices spread evenly across the num_bins bins present in bin_idx.
    Returns a 1D LongTensor of indices into the batch (CPU)."""
    if k <= 0:
        return torch.empty(0, dtype=torch.long)
    per_bin = max(1, k // num_bins)
    out = []
    for b in range(num_bins):
        idx_b = torch.nonzero(bin_idx == b, as_tuple=False).reshape(-1)
        if idx_b.numel() == 0:
            continue
        take = min(per_bin, idx_b.numel())
        perm = torch.randperm(idx_b.numel())[:take]
        out.append(idx_b[perm])
    if not out:
        return torch.empty(0, dtype=torch.long)
    sampled = torch.cat(out)
    # Top up to k from any remaining unsampled index.
    if sampled.numel() < k:
        remaining = k - sampled.numel()
        mask = torch.ones(bin_idx.numel(), dtype=torch.bool)
        mask[sampled] = False
        avail = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if avail.numel() > 0:
            take = min(remaining, avail.numel())
            perm = torch.randperm(avail.numel())[:take]
            sampled = torch.cat([sampled, avail[perm]])
    return sampled[:k]


def main():
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.manual_seed(args.seed)

    num_classes = 100 if args.dataset == 'cifar100' else 10

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                                download=True, transform=transform_train)
    else:
        trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True,
                                                 download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=1)

    model = WideResNet(num_classes=num_classes).to(device)
    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)
        print('Loaded init checkpoint from {}'.format(args.init_ckpt))
    else:
        print('Using random init (no --init_ckpt)')

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    print('Calibrating bin edges on first {} batches...'.format(args.cal_batches))
    bin_edges = calibrate_bin_edges(model, train_loader, device, num_classes,
                                    args.num_bins, args.cal_batches)
    print('Bin edges: {}'.format(bin_edges.tolist()))

    cos_phi_log = []
    bin_log = []
    epoch_log = []
    batch_log = []
    loss_var_by_bin_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # --- Standard TRADES step with SCALAR beta (no per-sample beta_i) ---
            optimizer.zero_grad()
            x_adv = generate_trades_adv(model, data, args.step, args.eps, args.num_steps)
            logits_nat = model(data)
            logits_adv = model(x_adv)
            loss_natural = F.cross_entropy(logits_nat, target)
            robust_kl = F.kl_div(F.log_softmax(logits_adv, dim=1),
                                 F.softmax(logits_nat, dim=1),
                                 reduction='none').sum(dim=1)
            loss = loss_natural + args.beta * robust_kl.mean()
            loss.backward()
            optimizer.step()

            # --- Logging ---
            if batch_idx % args.log_every == 0:
                with torch.no_grad():
                    logits_cal = model(data)
                    margins = clean_margin(logits_cal, target, num_classes)
                    bin_idx_full = assign_bins(margins, bin_edges)

                # Per-sample TRADES loss for the full batch -> per-bin variance.
                x_adv_meas = generate_trades_adv(model, data, args.step, args.eps, args.num_steps)
                per_sample_loss = compute_per_sample_trades_loss(
                    model, data, x_adv_meas, target, args.beta)
                var_per_bin = []
                for b in range(args.num_bins):
                    mask = (bin_idx_full == b)
                    if int(mask.sum()) >= 2:
                        var_per_bin.append(float(per_sample_loss[mask].var().item()))
                    else:
                        var_per_bin.append(0.0)
                loss_var_by_bin_log.append(var_per_bin)

                # Per-sample grad angles on a stratified subset.
                sampled_idx = stratified_sample(bin_idx_full, args.num_bins,
                                                args.grad_subsample).to(device)
                if sampled_idx.numel() > 0:
                    cos_phis = compute_per_sample_grad_angles(
                        model, data, x_adv_meas, target, sampled_idx)
                    bins_of_sampled = bin_idx_full[sampled_idx].tolist()
                    for i, cos in enumerate(cos_phis):
                        cos_phi_log.append(cos)
                        bin_log.append(int(bins_of_sampled[i]))
                        epoch_log.append(epoch)
                        batch_log.append(batch_idx)

                print('[epoch {} batch {}] loss={:.4f} | n_sampled={} | loss_var_by_bin={}'.format(
                    epoch, batch_idx, loss.item(), int(sampled_idx.numel()),
                    [round(v, 4) for v in var_per_bin]))

    cos_phi = np.asarray(cos_phi_log, dtype=np.float32)
    bin_idx_arr = np.asarray(bin_log, dtype=np.int8)
    epoch_arr = np.asarray(epoch_log, dtype=np.int16)
    batch_arr = np.asarray(batch_log, dtype=np.int32)
    loss_var = np.asarray(loss_var_by_bin_log, dtype=np.float32)
    edges_arr = bin_edges.detach().cpu().numpy().astype(np.float32)

    out_dir = os.path.dirname(args.out_npz)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out_npz,
             cos_phi=cos_phi,
             bin_idx=bin_idx_arr,
             epoch=epoch_arr,
             batch_idx=batch_arr,
             loss_var_by_bin=loss_var,
             bin_edges=edges_arr)
    print('Wrote {} (N_logged={}, N_batches_logged={})'.format(
        args.out_npz, cos_phi.shape[0], loss_var.shape[0]))


if __name__ == '__main__':
    main()
