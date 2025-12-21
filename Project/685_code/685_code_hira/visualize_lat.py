import numpy as np
import matplotlib.pyplot as plt

# Data for three methods and two datasets
methods = ['LoRA', 'SparseLora', 'HIRA']
datasets = ['SST2', 'IMDB']

data = {
    'LoRA, SST2': {'mean': 0.3042, 'std': 0.0010},
    'LoRA, IMDB': {'mean': 0.7925, 'std': 0.0007},
    'SparseLora, SST2': {'mean': 0.2890, 'std': 0.0004},
    'SparseLora, IMDB': {'mean': 0.7531, 'std': 0.0004},
    'HIRA, SST2': {'mean': 0.3350, 'std': 0.0736},
    'HIRA, IMDB': {'mean': 0.7753, 'std': 0.0006},
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Create separate plots for each dataset
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    labels = [f'{m}' for m in methods]
    means = [data[f'{m}, {dataset}']['mean'] for m in methods]
    stds = [data[f'{m}, {dataset}']['std'] for m in methods]
    
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
    
    ax.set_ylabel('Inference Latency (seconds)', fontsize=12)
    ax.set_title(f'{dataset}', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Inference Latency Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('latency_comparison.png', dpi=300, bbox_inches='tight')
plt.show()