"""
Experiment 1: Stratified Accuracy Analysis (Self-defined Strata)
每个方法使用自己的 margin 定义样本难度分层

Usage:
    python run_analysis_self_strata.py \
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


def evaluate_on_self_strata(model, loader, device, sample_to_quantile, num_quantiles=4):
    """
    在自己定义的分层上评估模型性能
    sample_to_quantile: 每个样本所属的 quantile (0-based index)
    """
    model.eval()

    # 初始化统计
    stats = {
        i: {'nat_correct': 0, 'rob_correct': 0, 'total': 0}
        for i in range(num_quantiles)
    }

    sample_idx = 0
    print('  Evaluating on self-defined strata...')

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

        # 根据自己的 quantile 分配
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


def print_comparison(all_results, all_boundaries):
    """打印对比表格"""
    quantiles = ['Q1', 'Q2', 'Q3', 'Q4']

    print('\n' + '='*100)
    print('STRATIFIED ACCURACY ANALYSIS (SELF-DEFINED STRATA)')
    print('Each method uses its own margin to define sample strata')
    print('='*100)

    # 打印每个方法的边界信息
    print('\nStrata Boundaries (per method):')
    for method_name, boundaries in all_boundaries.items():
        print(f'\n  {method_name.upper()}:')
        for i in range(len(boundaries) - 1):
            total = all_results[method_name][f'Q{i+1}']['total']
            print(f'    Q{i+1}: margin ∈ [{boundaries[i]:6.3f}, {boundaries[i+1]:6.3f}]  '
                  f'n = {total} samples')

    # 对比表格：每个方法在自己的分层上的表现
    print('\n' + '='*100)
    print('PERFORMANCE ON SELF-DEFINED STRATA')
    print('='*100)

    for q in quantiles:
        print(f'\n{q} (Hardest → Easiest based on each method\'s own margin)')
        print('-'*100)
        print(f'{"Method":<15} {"Natural Acc":<15} {"Robust Acc":<15} {"Margin Range":<30}')
        print('-'*100)

        for method_name in ['fixed', 'margin_easy', 'margin_hard']:
            if q in all_results[method_name]:
                nat_acc = all_results[method_name][q]['natural_acc']
                rob_acc = all_results[method_name][q]['robust_acc']
                margin_range = all_results[method_name][q]['margin_range']
                range_str = f'[{margin_range[0]:6.3f}, {margin_range[1]:6.3f}]'

                print(f'{method_name:<15} {nat_acc:>6.2f}%        {rob_acc:>6.2f}%        {range_str:<30}')

    # 关键发现：对比各方法在自己的Q1和Q4上的表现
    print('\n' + '='*100)
    print('KEY FINDINGS')
    print('='*100)

    print('\n1. Performance on HARDEST samples (Q1, lowest margin for each method):')
    print('-'*100)
    print(f'{"Method":<15} {"Natural Acc":<15} {"Robust Acc":<15}')
    print('-'*100)
    for method_name in ['fixed', 'margin_easy', 'margin_hard']:
        nat_acc = all_results[method_name]['Q1']['natural_acc']
        rob_acc = all_results[method_name]['Q1']['robust_acc']
        print(f'{method_name:<15} {nat_acc:>6.2f}%        {rob_acc:>6.2f}%')

    print('\n2. Performance on EASIEST samples (Q4, highest margin for each method):')
    print('-'*100)
    print(f'{"Method":<15} {"Natural Acc":<15} {"Robust Acc":<15}')
    print('-'*100)
    for method_name in ['fixed', 'margin_easy', 'margin_hard']:
        nat_acc = all_results[method_name]['Q4']['natural_acc']
        rob_acc = all_results[method_name]['Q4']['robust_acc']
        print(f'{method_name:<15} {nat_acc:>6.2f}%        {rob_acc:>6.2f}%')

    # 分析：谁在"自认为的easy样本"上表现最好？
    print('\n3. Analysis:')
    print('-'*100)

    # Q4 robust accuracy (easiest samples should have highest robust acc)
    q4_rob_accs = {
        'fixed': all_results['fixed']['Q4']['robust_acc'],
        'margin_easy': all_results['margin_easy']['Q4']['robust_acc'],
        'margin_hard': all_results['margin_hard']['Q4']['robust_acc']
    }
    best_method = max(q4_rob_accs, key=q4_rob_accs.get)
    print(f'✓ On easiest samples (Q4): {best_method.upper()} achieves highest robust accuracy '
          f'({q4_rob_accs[best_method]:.2f}%)')

    # Q1 natural accuracy (hardest samples)
    q1_nat_accs = {
        'fixed': all_results['fixed']['Q1']['natural_acc'],
        'margin_easy': all_results['margin_easy']['Q1']['natural_acc'],
        'margin_hard': all_results['margin_hard']['Q1']['natural_acc']
    }
    best_method_q1 = max(q1_nat_accs, key=q1_nat_accs.get)
    print(f'✓ On hardest samples (Q1): {best_method_q1.upper()} achieves highest natural accuracy '
          f'({q1_nat_accs[best_method_q1]:.2f}%)')

    # Margin range comparison
    print('\n✓ Margin range comparison:')
    for method_name in ['fixed', 'margin_easy', 'margin_hard']:
        boundaries = all_boundaries[method_name]
        margin_span = boundaries[-1] - boundaries[0]
        print(f'  {method_name:<15}: [{boundaries[0]:6.3f}, {boundaries[-1]:6.3f}]  '
              f'span = {margin_span:.3f}')

    print('='*100)


def main():
    parser = argparse.ArgumentParser(description='Stratified Accuracy Analysis with Self-defined Strata')
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

    # ==================== Step 1: 每个模型计算自己的分层 ====================
    all_results = {}
    all_boundaries = {}

    checkpoints = [
        ('fixed', args.fixed),
        ('margin_easy', args.easy),
        ('margin_hard', args.hard)
    ]

    for method_name, ckpt_path in checkpoints:
        print('\n' + '='*80)
        print(f'STEP {len(all_results)+1}: Processing {method_name.upper()}')
        print('='*80)

        # 加载模型
        model = WideResNet(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # 计算该模型的 margin
        margins = compute_margins(model, testloader, device)

        # 根据该模型自己的 margin 计算分层
        boundaries, sample_to_quantile = compute_strata_boundaries(margins, args.num_quantiles)
        all_boundaries[method_name] = boundaries

        print(f'\n  Strata boundaries for {method_name}:')
        for i in range(args.num_quantiles):
            count = np.sum(sample_to_quantile == i)
            print(f'    Q{i+1}: [{boundaries[i]:6.3f}, {boundaries[i+1]:6.3f}]  '
                  f'n = {count} samples ({100*count/len(margins):.1f}%)')

        # 在自己的分层上评估性能
        results = evaluate_on_self_strata(model, testloader, device,
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

    # ==================== Step 2: 打印对比 ====================
    print_comparison(all_results, all_boundaries)

    # ==================== Step 3: 保存 JSON ====================
    if args.save_json:
        output_data = {
            'dataset': args.dataset,
            'num_quantiles': args.num_quantiles,
            'strata_method': 'self_defined',  # 标记使用了自定义分层
            'data_dir': '../../data',
            'checkpoints': {
                'fixed': args.fixed,
                'margin_easy': args.easy,
                'margin_hard': args.hard,
            },
            'boundaries': {k: v.tolist() for k, v in all_boundaries.items()},
            'results': all_results
        }
        with open(args.save_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f'\n✓ Results saved to {args.save_json}')


if __name__ == '__main__':
    main()
