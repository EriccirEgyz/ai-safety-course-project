"""
Experiment 1: Stratified Accuracy Analysis
分层准确率分析：按 margin 分组，对比三种策略在不同难度样本上的表现

Usage:
    # 分析单个 checkpoint
    python exp1_stratified_accuracy.py --checkpoint path/to/model.pt --method fixed --epoch 80

    # 对比三种方法（分别指定路径）
    python exp1_stratified_accuracy.py \
        --checkpoint-fixed path/to/fixed_epoch80.pt \
        --checkpoint-easy path/to/margin_easy_epoch80.pt \
        --checkpoint-hard path/to/margin_hard_epoch80.pt \
        --epoch 80
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import json
from models.wideresnet import WideResNet


def _pgd_whitebox(model, X, y, device, epsilon=0.031, num_steps=20, step_size=0.003, random=True):
    """PGD-20 attack for robust accuracy evaluation"""
    model.eval()
    X_pgd = X.clone().detach()

    if random:
        X_pgd = X_pgd + torch.empty_like(X_pgd).uniform_(-epsilon, epsilon)
        X_pgd = torch.clamp(X_pgd, 0.0, 1.0)

    for _ in range(num_steps):
        X_pgd.requires_grad = True
        with torch.enable_grad():
            loss = F.cross_entropy(model(X_pgd), y)
        grad = torch.autograd.grad(loss, [X_pgd])[0]
        X_pgd = X_pgd.detach() + step_size * torch.sign(grad.detach())
        X_pgd = torch.max(torch.min(X_pgd, X + epsilon), X - epsilon)
        X_pgd = torch.clamp(X_pgd, 0.0, 1.0)

    return X_pgd


def compute_margins(model, loader, device):
    """计算每个样本的 clean margin: true_class_logit - max_other_class_logit"""
    model.eval()
    all_margins = []

    print('  Computing margins...')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)

            # margin = true_class_logit - max_other_class_logit
            true_logits = logits.gather(1, target.view(-1, 1)).squeeze(1)
            other_logits = logits.masked_fill(
                F.one_hot(target, num_classes=logits.size(1)).bool(),
                float('-inf')
            )
            max_other = other_logits.max(dim=1)[0]
            margins = (true_logits - max_other).cpu().numpy()

            all_margins.append(margins)

            if (batch_idx + 1) % 20 == 0:
                print(f'    Processed {batch_idx + 1}/{len(loader)} batches')

    all_margins = np.concatenate(all_margins)
    print(f'  Margin statistics: min={all_margins.min():.3f}, max={all_margins.max():.3f}, '
          f'mean={all_margins.mean():.3f}, std={all_margins.std():.3f}')
    return all_margins


def evaluate_stratified(model, loader, device, margins, num_quantiles=4, epsilon=0.031):
    """按 margin 分层评估自然准确率和鲁棒准确率"""
    model.eval()

    # 计算分位点边界
    quantile_points = np.linspace(0, 100, num_quantiles + 1)
    boundaries = np.percentile(margins, quantile_points)

    # 初始化统计
    stats = {
        f'Q{i+1}': {
            'natural_correct': 0,
            'robust_correct': 0,
            'total': 0,
            'margin_range': (boundaries[i], boundaries[i+1])
        }
        for i in range(num_quantiles)
    }

    sample_idx = 0
    print('  Evaluating stratified accuracy...')

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        batch_size = len(data)

        # 自然准确率
        with torch.no_grad():
            pred_natural = model(data).argmax(dim=1)
            natural_correct = (pred_natural == target).cpu().numpy()

        # 鲁棒准确率 (PGD-20)
        data_adv = _pgd_whitebox(model, data, target, device, epsilon=epsilon,
                                  num_steps=20, step_size=0.003, random=True)
        with torch.no_grad():
            pred_adv = model(data_adv).argmax(dim=1)
            robust_correct = (pred_adv == target).cpu().numpy()

        # 分配到对应分位组
        for i in range(batch_size):
            margin = margins[sample_idx + i]

            # 找到所属分位 (使用 searchsorted 更准确)
            q_idx = np.searchsorted(boundaries[1:], margin, side='left')
            q_idx = min(q_idx, num_quantiles - 1)  # 确保不越界

            q_name = f'Q{q_idx + 1}'
            stats[q_name]['natural_correct'] += int(natural_correct[i])
            stats[q_name]['robust_correct'] += int(robust_correct[i])
            stats[q_name]['total'] += 1

        sample_idx += batch_size

        if (batch_idx + 1) % 20 == 0:
            print(f'    Processed {batch_idx + 1}/{len(loader)} batches')

    # 计算准确率
    results = {}
    for q_name, data in stats.items():
        if data['total'] > 0:
            results[q_name] = {
                'natural_acc': 100.0 * data['natural_correct'] / data['total'],
                'robust_acc': 100.0 * data['robust_correct'] / data['total'],
                'total': data['total'],
                'margin_range': data['margin_range']
            }

    return results


def analyze_checkpoint(checkpoint_path, method_name, dataset, num_classes,
                       testloader, device, num_quantiles=4):
    """分析单个 checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f'⚠️  Checkpoint not found: {checkpoint_path}')
        return None

    print(f'\n{"="*80}')
    print(f'Analyzing: {method_name.upper()}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'{"="*80}')

    # 加载模型
    model = WideResNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 计算 margins
    margins = compute_margins(model, testloader, device)

    # 分层评估
    results = evaluate_stratified(model, testloader, device, margins,
                                   num_quantiles=num_quantiles)

    # 打印结果
    print(f'\nResults for {method_name}:')
    print(f'{"Quantile":<10} {"Margin Range":<25} {"Samples":<10} {"Natural Acc":<15} {"Robust Acc"}')
    print('-' * 85)
    for q_name in sorted(results.keys()):
        data = results[q_name]
        print(f'{q_name:<10} [{data["margin_range"][0]:6.3f}, {data["margin_range"][1]:6.3f}]      '
              f'{data["total"]:<10} {data["natural_acc"]:>6.2f}%         {data["robust_acc"]:>6.2f}%')

    return results


def print_comparison(all_results, epoch):
    """打印对比总结"""
    if len(all_results) == 0:
        return

    print('\n' + '='*100)
    print(f'COMPARISON SUMMARY - Epoch {epoch}')
    print('='*100)

    # 获取所有 quantiles
    quantiles = sorted(list(next(iter(all_results.values())).keys()))

    # 表头
    header = f'{"Quantile":<10} {"Metric":<15}'
    for method in ['fixed', 'margin_easy', 'margin_hard']:
        if method in all_results:
            header += f'{method.replace("_", "-").title():<15}'

    # 添加差值列（如果有 fixed 作为 baseline）
    if 'fixed' in all_results:
        for method in ['margin_easy', 'margin_hard']:
            if method in all_results:
                header += f'Δ{method.split("_")[1][:4].title():<12}'

    print(header)
    print('-' * len(header))

    # 数据行
    for q_name in quantiles:
        for metric, key in [('Natural', 'natural_acc'), ('Robust', 'robust_acc')]:
            line = f'{q_name:<10} {metric:<15}'

            values = {}
            for method in ['fixed', 'margin_easy', 'margin_hard']:
                if method in all_results and q_name in all_results[method]:
                    val = all_results[method][q_name][key]
                    values[method] = val
                    line += f'{val:>6.2f}%        '
                else:
                    line += f'{"N/A":<15}'

            # 添加差值
            if 'fixed' in values:
                for method in ['margin_easy', 'margin_hard']:
                    if method in values:
                        diff = values[method] - values['fixed']
                        sign = '+' if diff >= 0 else ''
                        line += f'{sign}{diff:>5.2f}%      '

            print(line)
        print()  # 空行分隔不同 quantile

    # 关键发现总结
    print('\n' + '='*100)
    print('KEY FINDINGS')
    print('='*100)

    if 'fixed' in all_results and 'margin_easy' in all_results:
        # Q4 (easiest) robust improvement for margin_easy
        if 'Q4' in all_results['margin_easy']:
            easy_q4_robust = all_results['margin_easy']['Q4']['robust_acc']
            fixed_q4_robust = all_results['fixed']['Q4']['robust_acc']
            diff = easy_q4_robust - fixed_q4_robust
            print(f'✓ Margin-Easy improves Q4 (easiest samples) robust accuracy: {diff:+.2f}%')

    if 'fixed' in all_results and 'margin_hard' in all_results:
        # Q1 (hardest) natural degradation for margin_hard
        if 'Q1' in all_results['margin_hard']:
            hard_q1_natural = all_results['margin_hard']['Q1']['natural_acc']
            fixed_q1_natural = all_results['fixed']['Q1']['natural_acc']
            diff = hard_q1_natural - fixed_q1_natural
            if diff < -1.0:
                print(f'⚠️  Margin-Hard degrades Q1 (hardest samples) natural accuracy: {diff:+.2f}%')
            else:
                print(f'✓ Margin-Hard Q1 natural accuracy change: {diff:+.2f}%')

    print('='*100)


def main():
    parser = argparse.ArgumentParser(description='Stratified Accuracy Analysis')

    # 单个 checkpoint 模式
    parser.add_argument('--checkpoint', type=str, help='Single checkpoint path')
    parser.add_argument('--method', type=str, choices=['fixed', 'margin_easy', 'margin_hard'],
                        help='Method name for single checkpoint')

    # 对比模式（三个 checkpoints）
    parser.add_argument('--checkpoint-fixed', type=str, help='Fixed method checkpoint')
    parser.add_argument('--checkpoint-easy', type=str, help='Margin-easy method checkpoint')
    parser.add_argument('--checkpoint-hard', type=str, help='Margin-hard method checkpoint')

    # 通用参数
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number for labeling')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--num-quantiles', type=int, default=4, help='Number of margin quantiles')
    parser.add_argument('--save-results', type=str, help='Save results to JSON file')

    args = parser.parse_args()

    # 检查参数
    single_mode = args.checkpoint is not None and args.method is not None
    compare_mode = any([args.checkpoint_fixed, args.checkpoint_easy, args.checkpoint_hard])

    if not single_mode and not compare_mode:
        parser.error('Either provide --checkpoint and --method, or at least one of '
                     '--checkpoint-fixed/easy/hard')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载测试集
    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
        num_classes = 10
    else:
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                 download=True, transform=transform)
        num_classes = 100

    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print('\n' + '='*100)
    print(f'STRATIFIED ACCURACY ANALYSIS - Epoch {args.epoch} - {args.dataset.upper()}')
    print('='*100)

    all_results = {}

    # 分析 checkpoints
    if single_mode:
        results = analyze_checkpoint(args.checkpoint, args.method, args.dataset,
                                     num_classes, testloader, device, args.num_quantiles)
        if results:
            all_results[args.method] = results

    if compare_mode:
        checkpoints = [
            ('fixed', args.checkpoint_fixed),
            ('margin_easy', args.checkpoint_easy),
            ('margin_hard', args.checkpoint_hard)
        ]

        for method, ckpt_path in checkpoints:
            if ckpt_path:
                results = analyze_checkpoint(ckpt_path, method, args.dataset,
                                             num_classes, testloader, device, args.num_quantiles)
                if results:
                    all_results[method] = results

    # 打印对比总结
    if len(all_results) > 1:
        print_comparison(all_results, args.epoch)

    # 保存结果到 JSON
    if args.save_results:
        output_data = {
            'epoch': args.epoch,
            'dataset': args.dataset,
            'num_quantiles': args.num_quantiles,
            'results': all_results
        }
        with open(args.save_results, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f'\n✓ Results saved to {args.save_results}')


if __name__ == '__main__':
    main()
