import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Data
systems = ['Traditional\nCentralized', 'DecenTrack\nPoW', 
           'DecenTrack\nPoW+ML', 'BI-WDRS\n(Blockchain+ML)']
fpr = [7.8, 5.0, 2.5, 2.3]  # False Positive Rate (%)
std_dev = [0.8, 1.2, 0.8, 0.5]  # Experimental variance

# Colors: consistent with Graphs 1-2
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Create bars with error bars
x_pos = np.arange(len(systems))
bars = ax.bar(x_pos, fpr, yerr=std_dev, capsize=8,
              color=colors, alpha=0.85, edgecolor='black', linewidth=1.5,
              error_kw={'elinewidth': 2, 'capthick': 2})

# Customize axes
ax.set_ylabel('False Positive Rate (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('System Type', fontsize=13, fontweight='bold')
ax.set_title('False Positive Rate Trend Analysis\nDecenTrack (Internal Evaluation) with Literature Reference', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(systems, fontsize=11)
ax.set_ylim(0, 10)

# Add horizontal line for baseline (traditional system)
ax.axhline(y=7.8, color='red', linestyle='--', linewidth=2, 
           alpha=0.4, label='Traditional System Baseline')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, fpr)):
    height = bar.get_height()
    label_y = height + std_dev[i] + 0.3
    
    # Main value
    ax.text(bar.get_x() + bar.get_width()/2., label_y,
            f'{val:.1f}%', ha='center', va='bottom', 
            fontsize=12, fontweight='bold')
    
    # Improvement percentage (vs traditional)
    if i > 0:
        improvement = ((fpr[0] - fpr[i]) / fpr[0]) * 100
        ax.text(bar.get_x() + bar.get_width()/2., 0.3,
                f'-{improvement:.1f}%', ha='center', va='bottom',
                fontsize=9, style='italic', color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))

# Add green zone shading for "acceptable" FPR
ax.axhspan(0, 3.0, alpha=0.05, color='green', label='Acceptable FPR Zone')

# Grid styling
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Add comparison annotation
textstr_comparison = (
    f'DecenTrack PoW+ML vs BI-WDRS:\n'
    f'Difference: {abs(fpr[2] - fpr[3]):.1f}%\n'
    f'(2.5% vs 2.3%, statistically\nequivalent with overlap)'
)
props_comparison = dict(boxstyle='round', facecolor='lightyellow', alpha=0.5, pad=0.8)
ax.text(0.98, 0.70, textstr_comparison, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=props_comparison, fontweight='bold')

# Add methodology annotation
textstr_method = (
    'Derived from Fig6 precision data:\n'
    'FPR = (1 - Precision) × Validator_Weight'
)
props_method = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
ax.text(0.02, 0.98, textstr_method, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='left',
        bbox=props_method, style='italic')

# Add legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('fpr_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved as 'fpr_comparison.png'")
plt.show()