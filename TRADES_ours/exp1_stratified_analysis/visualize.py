"""
Visualize Stratified Accuracy Analysis results
从 exp1_stratified_accuracy.py 保存的 JSON 绘制图表

Usage:
    python exp1_visualize.py --input results_epoch80.json --output figures/
"""
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 中文字体设置（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 论文风格设置
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_results(json_path):
    """加载 JSON 结果"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def plot_stratified_bars(data, output_dir):
    """
    Figure 1: 柱状图对比三种方法在不同 quantile 的表现
    左图: Natural Accuracy, 右图: Robust Accuracy
    """
    results = data['results']
    methods = ['fixed', 'margin_easy', 'margin_hard']
    method_labels = ['Fixed', 'Margin-Easy', 'Margin-Hard']
    colors = ['#1f77b4', '#ff7f0e', '#d62728']

    quantiles = sorted([k for k in results[methods[0]].keys() if k.startswith('Q')])
    x = np.arange(len(quantiles))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Natural Accuracy
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        if method not in results:
            continue
        natural_acc = [results[method][q]['natural_acc'] for q in quantiles]
        ax1.bar(x + i * width, natural_acc, width, label=label, color=color, alpha=0.8)

    ax1.set_xlabel('Margin Quantile (Q1=Hardest, Q4=Easiest)', fontsize=11)
    ax1.set_ylabel('Natural Accuracy (%)', fontsize=11)
    ax1.set_title('Natural Accuracy by Sample Difficulty', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(quantiles)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])

    # Robust Accuracy
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        if method not in results:
            continue
        robust_acc = [results[method][q]['robust_acc'] for q in quantiles]
        ax2.bar(x + i * width, robust_acc, width, label=label, color=color, alpha=0.8)

    ax2.set_xlabel('Margin Quantile (Q1=Hardest, Q4=Easiest)', fontsize=11)
    ax2.set_ylabel('Robust Accuracy (%)', fontsize=11)
    ax2.set_title('Robust Accuracy by Sample Difficulty', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(quantiles)
    ax2.legend(loc='lower right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'stratified_bars_epoch{data["epoch"]}.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f'✓ Saved: {output_path}')
    plt.close()


def plot_stratified_lines(data, output_dir):
    """
    Figure 2: 折线图展示 accuracy 随 quantile 的变化趋势
    """
    results = data['results']
    methods = ['fixed', 'margin_easy', 'margin_hard']
    method_labels = ['Fixed', 'Margin-Easy', 'Margin-Hard']
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    markers = ['o', 's', '^']

    quantiles = sorted([k for k in results[methods[0]].keys() if k.startswith('Q')])
    x = np.arange(len(quantiles))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Natural Accuracy
    for method, label, color, marker in zip(methods, method_labels, colors, markers):
        if method not in results:
            continue
        natural_acc = [results[method][q]['natural_acc'] for q in quantiles]
        ax1.plot(x, natural_acc, marker=marker, color=color, label=label,
                 linewidth=2, markersize=8, alpha=0.8)

    ax1.set_xlabel('Margin Quantile', fontsize=11)
    ax1.set_ylabel('Natural Accuracy (%)', fontsize=11)
    ax1.set_title('Natural Accuracy Trend', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(quantiles)
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 100])

    # Robust Accuracy
    for method, label, color, marker in zip(methods, method_labels, colors, markers):
        if method not in results:
            continue
        robust_acc = [results[method][q]['robust_acc'] for q in quantiles]
        ax2.plot(x, robust_acc, marker=marker, color=color, label=label,
                 linewidth=2, markersize=8, alpha=0.8)

    ax2.set_xlabel('Margin Quantile', fontsize=11)
    ax2.set_ylabel('Robust Accuracy (%)', fontsize=11)
    ax2.set_title('Robust Accuracy Trend', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(quantiles)
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'stratified_lines_epoch{data["epoch"]}.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f'✓ Saved: {output_path}')
    plt.close()


def plot_improvement_heatmap(data, output_dir):
    """
    Figure 3: 热力图展示 margin_easy/hard 相对 fixed 的改进
    """
    results = data['results']
    if 'fixed' not in results:
        print('⚠️  No "fixed" baseline found, skipping heatmap')
        return

    quantiles = sorted([k for k in results['fixed'].keys() if k.startswith('Q')])
    methods = []
    method_labels = []

    if 'margin_easy' in results:
        methods.append('margin_easy')
        method_labels.append('Margin-Easy')
    if 'margin_hard' in results:
        methods.append('margin_hard')
        method_labels.append('Margin-Hard')

    if len(methods) == 0:
        print('⚠️  No comparison methods found, skipping heatmap')
        return

    # 计算差值
    metrics = ['natural_acc', 'robust_acc']
    metric_labels = ['Natural Acc', 'Robust Acc']

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    for ax, method, method_label in zip(axes, methods, method_labels):
        diff_matrix = []
        for metric in metrics:
            diffs = []
            for q in quantiles:
                fixed_val = results['fixed'][q][metric]
                method_val = results[method][q][metric]
                diffs.append(method_val - fixed_val)
            diff_matrix.append(diffs)

        diff_matrix = np.array(diff_matrix)

        # 绘制热力图
        im = ax.imshow(diff_matrix, cmap='RdYlGn', aspect='auto',
                       vmin=-5, vmax=5, interpolation='nearest')
        ax.set_xticks(np.arange(len(quantiles)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(quantiles)
        ax.set_yticklabels(metric_labels)
        ax.set_xlabel('Margin Quantile', fontsize=11)
        ax.set_title(f'{method_label} vs Fixed (Δ%)', fontsize=12, fontweight='bold')

        # 添加数值标注
        for i in range(len(metrics)):
            for j in range(len(quantiles)):
                val = diff_matrix[i, j]
                color = 'white' if abs(val) > 2.5 else 'black'
                ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                       color=color, fontsize=10, fontweight='bold')

        # 添加 colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy Difference (%)', rotation=270, labelpad=20)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'improvement_heatmap_epoch{data["epoch"]}.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f'✓ Saved: {output_path}')
    plt.close()


def plot_margin_distribution(data, output_dir):
    """
    Figure 4: 展示 margin 的分布范围
    """
    results = data['results']
    method = list(results.keys())[0]  # 使用第一个方法的数据
    quantiles = sorted([k for k in results[method].keys() if k.startswith('Q')])

    ranges = [results[method][q]['margin_range'] for q in quantiles]
    totals = [results[method][q]['total'] for q in quantiles]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Margin range visualization
    for i, (q, (min_m, max_m)) in enumerate(zip(quantiles, ranges)):
        ax1.barh(i, max_m - min_m, left=min_m, height=0.6,
                color=plt.cm.viridis(i / len(quantiles)), alpha=0.7, edgecolor='black')
        ax1.text(min_m + (max_m - min_m) / 2, i, f'{min_m:.2f} ~ {max_m:.2f}',
                ha='center', va='center', fontsize=9, fontweight='bold')

    ax1.set_yticks(range(len(quantiles)))
    ax1.set_yticklabels(quantiles)
    ax1.set_xlabel('Margin Value', fontsize=11)
    ax1.set_ylabel('Quantile', fontsize=11)
    ax1.set_title('Margin Range Distribution', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Sample count
    colors_bar = [plt.cm.viridis(i / len(quantiles)) for i in range(len(quantiles))]
    ax2.bar(quantiles, totals, color=colors_bar, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Quantile', fontsize=11)
    ax2.set_ylabel('Number of Samples', fontsize=11)
    ax2.set_title('Sample Count per Quantile', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for i, (q, total) in enumerate(zip(quantiles, totals)):
        ax2.text(i, total + 50, str(total), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'margin_distribution_epoch{data["epoch"]}.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f'✓ Saved: {output_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Stratified Accuracy Analysis')
    parser.add_argument('--input', type=str, required=True, help='JSON file from exp1')
    parser.add_argument('--output', type=str, default='./figures', help='Output directory')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载数据
    print(f'Loading results from {args.input}...')
    data = load_results(args.input)
    print(f'Dataset: {data["dataset"].upper()}, Epoch: {data["epoch"]}, '
          f'Methods: {", ".join(data["results"].keys())}')

    # 绘制图表
    print('\nGenerating figures...')
    plot_stratified_bars(data, args.output)
    plot_stratified_lines(data, args.output)
    plot_improvement_heatmap(data, args.output)
    plot_margin_distribution(data, args.output)

    print(f'\n✓ All figures saved to {args.output}/')


if __name__ == '__main__':
    main()
