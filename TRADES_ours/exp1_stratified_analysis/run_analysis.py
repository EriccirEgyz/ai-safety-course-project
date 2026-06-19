"""
Experiment 1: Stratified Accuracy Analysis (Fixed Strata)
使用 fixed 模型的 margin 定义样本难度分层，三种方法在同一批样本上对比

Usage:
    python run_analysis.py \
        --fixed path/to/fixed.pt \
        --easy path/to/easy.pt \
        --hard path/to/hard.pt \
        --dataset cifar100 \
        --save-json results.json
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.wideresnet import WideResNet


def pgd_attack(model, x, y, device, eps=0.031, steps=20, step_size=0.003):
    """PGD-20 攻击"""
    model.eval()
    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad = True
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


def compute_margins(model, loader, device):
    """
    计算所有测试样本的 margin
    margin = true_class_logit - max_other_class_logit
    """
    model.eval()
    all_margins = []

    print('  Computing margins...')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)

            # 计算 margin
            true_logits = logits.gather(1, target.view(-1, 1)).squeeze(1)
            mask = F.one_hot(target, logits.size(1)).bool()
            other_logits = logits.masked_fill(mask, float('-inf'))
            max_other = other_logits.max(dim=1)[0]
            margins = (true_logits - max_other).cpu().numpy()

            all_margins.append(margins)

            if (batch_idx + 1) % 20 == 0:
                print(f'    Batch {batch_idx + 1}/{len(loader)}')

    all_margins = np.concatenate(all_margins)
    print(f'  Margin stats: min={all_margins.min():.3f}, max={all_margins.max():.3f}, '
          f'mean={all_margins.mean():.3f}, std={all_margins.std():.3f}')

    return all_margins


def compute_strata_boundaries(margins, num_quantiles=4):
    """
    根据 margin 计算分位数边界，并为每个样本分配 quantile 标签
    返回：boundaries, sample_to_quantile
    """
    quantile_points = np.linspace(0, 100, num_quantiles + 1)
    boundaries = np.percentile(margins, quantile_points)

    # 为每个样本分配 quantile (0-based index)
    sample_to_quantile = np.searchsorted(boundaries[1:], margins, side='left')
    sample_to_quantile = np.clip(sample_to_quantile, 0, num_quantiles - 1)

    return boundaries, sample_to_quantile


def evaluate_on_fixed_strata(model, loader, device, sample_to_quantile, num_quantiles=4):
    """
    在固定的分层上评估模型性能
    sample_to_quantile: 每个样本所属的 quantile (0-indexed)
    """
    model.eval()

    # 初始化统计
    stats = {
        i: {'nat_correct': 0, 'rob_correct': 0, 'total': 0}
        for i in range(num_quantiles)
    }

    sample_idx = 0
    print('  Evaluating on fixed strata...')

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        batch_size = len(data)

        # 自然准确率
        with torch.no_grad():
            nat_pred = model(data).argmax(dim=1)
            nat_correct = (nat_pred == target).cpu().numpy()

        # 鲁棒准确率 (PGD-20)
        adv_data = pgd_attack(model, data, target, device)
        with torch.no_grad():
            rob_pred = model(adv_data).argmax(dim=1)
            rob_correct = (rob_pred == target).cpu().numpy()

        # 根据预先计算的 quantile 分配
        for i in range(batch_size):
            q_idx = sample_to_quantile[sample_idx + i]
            stats[q_idx]['nat_correct'] += int(nat_correct[i])
            stats[q_idx]['rob_correct'] += int(rob_correct[i])
            stats[q_idx]['total'] += 1

        sample_idx += batch_size

        if (batch_idx + 1) % 20 == 0:
            print(f'    Batch {batch_idx + 1}/{len(loader)}')

    # 计算准确率
    results = {}
    for q_idx in range(num_quantiles):
        data = stats[q_idx]
        if data['total'] > 0:
            results[f'Q{q_idx+1}'] = {
                'natural_acc': 100.0 * data['nat_correct'] / data['total'],
                'robust_acc': 100.0 * data['rob_correct'] / data['total'],
                'total': data['total']
            }

    return results


def print_comparison(all_results, boundaries):
    """打印对比表格"""
    quantiles = sorted(
        [k for k in all_results['fixed'].keys() if k.startswith('Q')],
        key=lambda name: int(name[1:])
    )

    print('\n' + '='*100)
    print('STRATIFIED ACCURACY ANALYSIS')
    print('Sample strata defined by FIXED model\'s margin')
    print('='*100)

    # 打印边界信息
    print('\nStrata Boundaries (based on Fixed model):')
    for i in range(len(boundaries) - 1):
        total = all_results['fixed'][f'Q{i+1}']['total']
        print(f'  Q{i+1}: margin ∈ [{boundaries[i]:6.3f}, {boundaries[i+1]:6.3f}]  '
              f'n = {total} samples')

    # 对比表格
    print('\n' + '='*100)
    print(f'{"Quantile":<10} {"Metric":<12} {"Fixed":<12} {"Margin-Easy":<14} {"Margin-Hard":<14} '
          f'{"ΔEasy":<10} {"ΔHard":<10}')
    print('-'*100)

    for q in quantiles:
        for metric_name, metric_key in [('Natural', 'natural_acc'), ('Robust', 'robust_acc')]:
            fixed_val = all_results['fixed'][q][metric_key]
            easy_val = all_results['margin_easy'][q][metric_key]
            hard_val = all_results['margin_hard'][q][metric_key]

            easy_diff = easy_val - fixed_val
            hard_diff = hard_val - fixed_val

            line = (f'{q:<10} {metric_name:<12} {fixed_val:>6.2f}%      '
                   f'{easy_val:>6.2f}%        {hard_val:>6.2f}%        '
                   f'{easy_diff:>+6.2f}%     {hard_diff:>+6.2f}%')
            print(line)

        print()  # 空行分隔

    # 关键发现
    print('='*100)
    print('KEY FINDINGS')
    print('='*100)

    hardest_q = quantiles[0]
    easiest_q = quantiles[-1]

    # 最大 margin 分层（默认 Q4）代表最容易样本。
    easy_q4_rob = all_results['margin_easy'][easiest_q]['robust_acc']
    fixed_q4_rob = all_results['fixed'][easiest_q]['robust_acc']
    diff_q4 = easy_q4_rob - fixed_q4_rob
    print(f'✓ Margin-Easy on {easiest_q} (easiest samples): Robust accuracy {diff_q4:+.2f}%')

    # 最小 margin 分层（默认 Q1）代表最难样本。
    easy_q1_nat = all_results['margin_easy'][hardest_q]['natural_acc']
    hard_q1_nat = all_results['margin_hard'][hardest_q]['natural_acc']
    fixed_q1_nat = all_results['fixed'][hardest_q]['natural_acc']
    diff_easy_q1 = easy_q1_nat - fixed_q1_nat
    diff_hard_q1 = hard_q1_nat - fixed_q1_nat

    print(f'✓ Margin-Easy on {hardest_q} (hardest samples): Natural accuracy {diff_easy_q1:+.2f}%')

    if diff_hard_q1 < -1.0:
        print(f'⚠️  Margin-Hard on {hardest_q} (hardest samples): Natural accuracy {diff_hard_q1:+.2f}% '
              f'(degradation due to over-regularization)')
    else:
        print(f'✓ Margin-Hard on {hardest_q} (hardest samples): Natural accuracy {diff_hard_q1:+.2f}%')

    print('='*100)


def main():
    parser = argparse.ArgumentParser(description='Stratified Accuracy Analysis with Fixed Strata')
    parser.add_argument('--fixed', required=True, help='Fixed baseline checkpoint')
    parser.add_argument('--easy', required=True, help='Margin-easy checkpoint')
    parser.add_argument('--hard', required=True, help='Margin-hard checkpoint')
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--num-quantiles', type=int, default=4, help='Number of quantiles')
    parser.add_argument('--save-json', type=str, help='Save results to JSON file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据集
    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'cifar10':
        testset = datasets.CIFAR10(root='../../data', train=False, download=False, transform=transform)
        num_classes = 10
    else:
        testset = datasets.CIFAR100(root='../../data', train=False, download=False, transform=transform)
        num_classes = 100

    testloader = DataLoader(testset, batch_size=200, shuffle=False, num_workers=2)

    print(f'\nDataset: {args.dataset.upper()}, Num samples: {len(testset)}')
    print('Data directory: ../../data')

    # ==================== Step 1: 用 Fixed 模型定义分层 ====================
    print('\n' + '='*80)
    print('STEP 1: Computing strata boundaries using FIXED model')
    print('='*80)

    fixed_model = WideResNet(num_classes=num_classes).to(device)
    fixed_model.load_state_dict(torch.load(args.fixed, map_location=device))
    fixed_model.eval()

    margins = compute_margins(fixed_model, testloader, device)
    boundaries, sample_to_quantile = compute_strata_boundaries(margins, args.num_quantiles)

    print(f'\nStrata boundaries:')
    for i in range(args.num_quantiles):
        count = np.sum(sample_to_quantile == i)
        print(f'  Q{i+1}: [{boundaries[i]:6.3f}, {boundaries[i+1]:6.3f}]  '
              f'n = {count} samples ({100*count/len(margins):.1f}%)')

    # ==================== Step 2: 在固定分层上评估三个模型 ====================
    all_results = {}
    checkpoints = [
        ('fixed', args.fixed),
        ('margin_easy', args.easy),
        ('margin_hard', args.hard)
    ]

    for method_name, ckpt_path in checkpoints:
        print('\n' + '='*80)
        print(f'STEP 2.{len(all_results)+1}: Evaluating {method_name.upper()} on fixed strata')
        print('='*80)

        model = WideResNet(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        results = evaluate_on_fixed_strata(model, testloader, device,
                                            sample_to_quantile, args.num_quantiles)

        # 添加 margin range 信息
        for i in range(args.num_quantiles):
            q_name = f'Q{i+1}'
            if q_name in results:
                results[q_name]['margin_range'] = (float(boundaries[i]), float(boundaries[i+1]))

        all_results[method_name] = results

        print(f'\n  Results for {method_name}:')
        for q_name in sorted(results.keys()):
            r = results[q_name]
            print(f'    {q_name}: Natural {r["natural_acc"]:5.2f}%  Robust {r["robust_acc"]:5.2f}%')

    # ==================== Step 3: 打印对比 ====================
    print_comparison(all_results, boundaries)

    # ==================== Step 4: 保存 JSON ====================
    if args.save_json:
        output_data = {
            'dataset': args.dataset,
            'num_quantiles': args.num_quantiles,
            'strata_method': 'fixed',  # 标记使用了固定分层
            'data_dir': '../../data',
            'checkpoints': {
                'fixed': args.fixed,
                'margin_easy': args.easy,
                'margin_hard': args.hard,
            },
            'boundaries': boundaries.tolist(),
            'results': all_results
        }
        with open(args.save_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f'\n✓ Results saved to {args.save_json}')


if __name__ == '__main__':
    main()
