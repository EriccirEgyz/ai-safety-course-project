"""
Visualize Stratified Accuracy Analysis results
从 run_analysis.py 保存的 JSON 绘制论文主图：
2-panel line plot, 左图 Clean/Natural Acc，右图 PGD-20 Robust Acc

Usage:
    python visualize.py --input results.json --output figures/
"""
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 论文风格设置
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


METHOD_SPECS = [
    ('fixed', ['fixed'], 'Fixed', '#0072B2', 'o'),
    ('margin_easy', ['margin_easy', 'easy'], 'Margin-Easy', '#009E73', 's'),
    ('margin_hard', ['margin_hard', 'hard'], 'Margin-Hard', '#D55E00', '^'),
]


def load_results(json_path):
    """加载 JSON 结果"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def resolve_methods(results):
    """Return available methods while supporting old easy/hard JSON keys."""
    methods = []
    for canonical_key, aliases, label, color, marker in METHOD_SPECS:
        for key in aliases:
            if key in results:
                methods.append((key, label, color, marker))
                break
    return methods


def get_quantiles(method_results):
    """Return Q1, Q2, ... in numeric order."""
    return sorted(
        [key for key in method_results.keys() if key.startswith('Q')],
        key=lambda key: int(key[1:])
    )


def get_output_filename(data):
    """Use epoch suffix for old JSON files; otherwise use a stable filename."""
    if 'epoch' in data:
        return f'stratified_lines_epoch{data["epoch"]}.png'
    return 'stratified_lines.png'


def set_accuracy_ylim(ax, values):
    min_val = min(values)
    max_val = max(values)
    padding = max(2.0, (max_val - min_val) * 0.18)
    lower = max(0, np.floor((min_val - padding) / 5) * 5)
    upper = min(100, np.ceil((max_val + padding) / 5) * 5)

    if upper - lower < 10:
        center = (upper + lower) / 2
        lower = max(0, center - 5)
        upper = min(100, center + 5)

    ax.set_ylim([lower, upper])


def plot_stratified_lines(data, output_dir):
    """
    Main figure: 2-panel line plot for clean and robust accuracy by stratum.
    """
    results = data['results']
    methods = resolve_methods(results)
    if len(methods) == 0:
        raise ValueError('No supported methods found in JSON results.')

    quantiles = get_quantiles(results[methods[0][0]])
    x = np.arange(len(quantiles))
    all_natural_acc = []
    all_robust_acc = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.8), sharex=True)
    tick_labels = quantiles.copy()
    if len(tick_labels) >= 2:
        tick_labels[0] = f'{tick_labels[0]}\nhardest'
        tick_labels[-1] = f'{tick_labels[-1]}\neasiest'

    for method, label, color, marker in methods:
        natural_acc = [results[method][q]['natural_acc'] for q in quantiles]
        all_natural_acc.extend(natural_acc)
        ax1.plot(x, natural_acc, marker=marker, color=color, label=label,
                 linewidth=2.0, markersize=5.5, markeredgecolor='white',
                 markeredgewidth=0.7, alpha=0.95)

    ax1.set_ylabel('Clean accuracy (%)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tick_labels)
    ax1.grid(alpha=0.3)
    set_accuracy_ylim(ax1, all_natural_acc)
    ax1.text(0.02, 0.96, '(a)', transform=ax1.transAxes, ha='left', va='top',
             fontsize=11, fontweight='bold')

    for method, label, color, marker in methods:
        robust_acc = [results[method][q]['robust_acc'] for q in quantiles]
        all_robust_acc.extend(robust_acc)
        ax2.plot(x, robust_acc, marker=marker, color=color, label=label,
                 linewidth=2.0, markersize=5.5, markeredgecolor='white',
                 markeredgewidth=0.7, alpha=0.95)

    ax2.set_ylabel('PGD-20 robust accuracy (%)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tick_labels)
    ax2.grid(alpha=0.3)
    set_accuracy_ylim(ax2, all_robust_acc)
    ax2.text(0.02, 0.96, '(b)', transform=ax2.transAxes, ha='left', va='top',
             fontsize=11, fontweight='bold')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(methods),
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.supxlabel('Fixed-model margin stratum', fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.9])
    output_path = os.path.join(output_dir, get_output_filename(data))
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
    print(f'Dataset: {data["dataset"].upper()}, Methods: {", ".join(data["results"].keys())}')

    # 绘制论文主图：2-panel line plot
    print('\nGenerating main figure...')
    plot_stratified_lines(data, args.output)

    print(f'\n✓ Main figure saved to {args.output}/')


if __name__ == '__main__':
    main()
