"""
Beta-Margin Correlation Analysis
展示三种策略如何根据样本 margin 分配 beta 权重

Usage:
    python beta_margin_analysis.py \
        --checkpoint path/to/fixed_epoch80.pt \
        --epoch 80 \
        --dataset cifar100
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.wideresnet import WideResNet

# 论文风格设置
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.sans-serif'] = ['Arial']


def compute_margins(model, loader, device):
    """计算样本的 margin: true_logit - max_other_logit"""
    model.eval()
    all_margins = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)

            # margin = true_class_logit - max_other_class_logit
            true_logits = logits.gather(1, target.view(-1, 1)).squeeze(1)
            mask = F.one_hot(target, logits.size(1)).bool()
            other_logits = logits.masked_fill(mask, float('-inf'))
            max_other = other_logits.max(dim=1)[0]
            margins = (true_logits - max_other).cpu().numpy()

            all_margins.append(margins)

    return np.concatenate(all_margins)


def compute_beta_fixed(margins, beta=6.0):
    """Fixed: 常数 beta"""
    return np.full_like(margins, beta, dtype=np.float32)


def compute_beta_margin_easy(margins, beta=6.0, tau=0.0, temperature=0.5,
                              beta_min=1.0, beta_max=10.0):
    """Margin-Easy: 大 margin → 高 beta"""
    margins_tensor = torch.from_numpy(margins).float()

    # margin_easy: sigmoid((margin - tau) / T)
    margin_weights = torch.sigmoid((margins_tensor - tau) / temperature)

    # 归一化保持平均 beta 不变
    beta_i = beta * margin_weights / (margin_weights.mean() + 1e-12)

    # Clamp 到范围
    beta_i = torch.clamp(beta_i, min=beta_min, max=beta_max)

    return beta_i.numpy()


def compute_beta_margin_hard(margins, beta=6.0, tau=0.0, temperature=0.5,
                              beta_min=1.0, beta_max=10.0):
    """Margin-Hard: 小 margin → 高 beta (反向)"""
    margins_tensor = torch.from_numpy(margins).float()

    # margin_hard: sigmoid((tau - margin) / T)  注意是反过来的
    margin_weights = torch.sigmoid((tau - margins_tensor) / temperature)

    # 归一化
    beta_i = beta * margin_weights / (margin_weights.mean() + 1e-12)

    # Clamp
    beta_i = torch.clamp(beta_i, min=beta_min, max=beta_max)

    return beta_i.numpy()


def safe_pearsonr(x, y):
    """Pearson r is undefined when either input is constant."""
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None, None
    return pearsonr(x, y)


def plot_beta_margin_correlation(margins, betas_dict, output_path, epoch):
    """
    绘制 Beta-Margin 相关性图
    betas_dict: {'fixed': beta_array, 'margin_easy': beta_array, 'margin_hard': beta_array}
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    strategies = [
        ('fixed', '(a) Fixed', '#0072B2'),
        ('margin_easy', '(b) Margin-Easy', '#009E73'),
        ('margin_hard', '(c) Margin-Hard', '#D55E00')
    ]

    for ax, (key, title, color) in zip(axes, strategies):
        beta_values = betas_dict[key]

        # 散点图
        ax.scatter(margins, beta_values, alpha=0.3, s=10, color=color, edgecolors='none')

        # 趋势线（使用排序后的数据绘制平滑曲线）
        sorted_indices = np.argsort(margins)
        sorted_margins = margins[sorted_indices]
        sorted_betas = beta_values[sorted_indices]

        # 使用滑动窗口平均绘制趋势
        if len(sorted_margins) > 50:
            window = max(51, len(sorted_margins) // 20)
            if window % 2 == 0:
                window += 1
            smoothed_betas = savgol_filter(sorted_betas, window_length=window, polyorder=3)
            ax.plot(sorted_margins, smoothed_betas, color='black', linewidth=2, alpha=0.8)

        # 标题和标注
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 11])

    fig.supxlabel('Clean margin', fontsize=11)
    fig.supylabel(r'Weight $\beta_i$', fontsize=11)
    plt.tight_layout(rect=[0.03, 0.05, 1.0, 1.0])
    plt.savefig(output_path, bbox_inches='tight')
    print(f'✓ Figure saved: {output_path}')
    plt.close()


def print_statistics(margins, betas_dict):
    """打印统计信息"""
    print('\n' + '='*80)
    print('BETA-MARGIN CORRELATION ANALYSIS')
    print('='*80)

    print(f'\nMargin statistics (n={len(margins)}):')
    print(f'  Range: [{np.min(margins):.3f}, {np.max(margins):.3f}]')
    print(f'  Mean: {np.mean(margins):.3f}, Std: {np.std(margins):.3f}')
    print(f'  Quantiles: p10={np.percentile(margins, 10):.3f}, '
          f'p50={np.percentile(margins, 50):.3f}, p90={np.percentile(margins, 90):.3f}')

    for strategy in ['fixed', 'margin_easy', 'margin_hard']:
        beta_values = betas_dict[strategy]
        r, p_value = safe_pearsonr(margins, beta_values)

        print(f'\n{strategy.upper().replace("_", "-")}:')
        if r is None:
            print('  Pearson correlation: N/A (constant beta allocation)')
        else:
            print(f'  Pearson correlation: r = {r:.4f} (p = {p_value:.2e})')
        print(f'  Beta_i statistics:')
        print(f'    Mean: {np.mean(beta_values):.3f}, Std: {np.std(beta_values):.3f}')
        print(f'    Range: [{np.min(beta_values):.2f}, {np.max(beta_values):.2f}]')
        print(f'    Quantiles: p10={np.percentile(beta_values, 10):.2f}, '
              f'p50={np.percentile(beta_values, 50):.2f}, p90={np.percentile(beta_values, 90):.2f}')

    print('\n' + '='*80)
    print('INTERPRETATION')
    print('='*80)

    r_easy, _ = safe_pearsonr(margins, betas_dict['margin_easy'])
    r_hard, _ = safe_pearsonr(margins, betas_dict['margin_hard'])

    if r_easy is not None and r_easy > 0.5:
        print(f'✓ Margin-Easy shows strong positive correlation (r={r_easy:.3f}):')
        print(f'  → Samples with larger margins receive higher beta (stronger regularization)')

    if r_hard is not None and r_hard < -0.5:
        print(f'⚠️  Margin-Hard shows strong negative correlation (r={r_hard:.3f}):')
        print(f'  → Samples with smaller margins receive higher beta (risk of over-regularization)')

    print('='*80)


def main():
    parser = argparse.ArgumentParser(description='Beta-Margin Correlation Analysis')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint (typically fixed baseline)')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number (for labeling)')
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--num-samples', type=int, default=2000,
                       help='Number of samples to analyze (default: 2000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Beta 策略超参数
    parser.add_argument('--beta', type=float, default=6.0, help='Base beta value')
    parser.add_argument('--margin-tau', type=float, default=0.0, help='Margin threshold')
    parser.add_argument('--margin-temperature', type=float, default=0.5, help='Sigmoid temperature')
    parser.add_argument('--beta-min', type=float, default=1.0, help='Min beta_i')
    parser.add_argument('--beta-max', type=float, default=10.0, help='Max beta_i')

    parser.add_argument('--output', type=str, default='beta_margin_correlation.png',
                       help='Output figure path')

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据集
    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'cifar10':
        testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        num_classes = 10
    else:
        testset = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
        num_classes = 100

    # 随机抽样
    total_samples = len(testset)
    sample_indices = np.random.choice(total_samples, size=args.num_samples, replace=False)
    subset = Subset(testset, sample_indices)
    loader = DataLoader(subset, batch_size=200, shuffle=False, num_workers=2)

    print(f'\nDataset: {args.dataset.upper()}')
    print(f'Total test samples: {total_samples}')
    print(f'Analyzing {args.num_samples} randomly sampled images')
    print(f'Seed: {args.seed}')

    # 加载模型
    print(f'\nLoading checkpoint: {args.checkpoint}')
    model = WideResNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 计算 margins
    print('Computing margins...')
    margins = compute_margins(model, loader, device)
    print(f'✓ Computed {len(margins)} margins')

    # 计算三种策略的 beta
    print('\nComputing beta allocation for three strategies...')
    print(f'  Hyperparameters: beta={args.beta}, tau={args.margin_tau}, '
          f'T={args.margin_temperature}, range=[{args.beta_min}, {args.beta_max}]')

    betas_dict = {
        'fixed': compute_beta_fixed(margins, beta=args.beta),
        'margin_easy': compute_beta_margin_easy(margins, beta=args.beta, tau=args.margin_tau,
                                                temperature=args.margin_temperature,
                                                beta_min=args.beta_min, beta_max=args.beta_max),
        'margin_hard': compute_beta_margin_hard(margins, beta=args.beta, tau=args.margin_tau,
                                                temperature=args.margin_temperature,
                                                beta_min=args.beta_min, beta_max=args.beta_max)
    }

    # 打印统计
    print_statistics(margins, betas_dict)

    # 绘图
    print(f'\nGenerating figure...')
    plot_beta_margin_correlation(margins, betas_dict, args.output, args.epoch)

    print(f'\n✓ Analysis complete!')


if __name__ == '__main__':
    main()
