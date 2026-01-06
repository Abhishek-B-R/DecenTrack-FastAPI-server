import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Create figure
fig, ax = plt.subplots(figsize=(15, 8))

# === VALIDATOR COHORT LATENCIES ===
np.random.seed(42)

# Define validator types
validator_types = ['Reliable', 'Average', 'Unreliable']

# PoW: Accept all
pow_latencies = [
    199,      # Reliable
    598,      # Average
    2444      # Unreliable
]

# PoW+ML: ML filtering at 500ms threshold
pow_ml_latencies = [
    180,      # Reliable
    495,      # Average
    1950      # Unreliable
]

# BI-WDRS comparison
bi_wdrs_latencies = [
    179,
    581,
    2246
]

# arXiv 2407.00015 baseline
arxiv_latencies = [
    380,      # Reliable
    890,      # Average
    2650      # Unreliable
]

# Bar positions
x = np.arange(len(validator_types))
width = 0.2

# Create bars
bars1 = ax.bar(x - 1.5*width, pow_latencies, width, label='PoW', 
               color='#3498DB', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x - 0.5*width, pow_ml_latencies, width, label='PoW + ML', 
               color='#2ECC71', edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + 0.5*width, bi_wdrs_latencies, width, label='BI-WDRS', 
               color='#9B59B6', edgecolor='black', linewidth=1.2)
bars4 = ax.bar(x + 1.5*width, arxiv_latencies, width, label='arXiv 2407.00015 Baseline', 
               color='#E74C3C', edgecolor='black', linewidth=1.2)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}ms',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)
add_value_labels(bars4)

# Customize
ax.set_xlabel('Validator Cohort', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Latency of Accepted Ticks (ms)', fontsize=13, fontweight='bold')
ax.set_title('Latency of Accepted Ticks Across Validator Cohorts (Internal Evaluation with Literature Reference)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(validator_types, fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax.set_ylim(0, 3000)
ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

# Add improvement annotations (PoW+ML vs arXiv)
for i, (ml_val, arxiv_val) in enumerate(zip(pow_ml_latencies, arxiv_latencies)):
    improvement = ((arxiv_val - ml_val) / arxiv_val) * 100
    if improvement > 0:
        # Draw arrow
        ax.annotate('', xy=(i - 0.5*width, ml_val), xytext=(i + 1.5*width, arxiv_val),
                    arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2.5))
        # Add improvement text
        ax.text(i + 0.5*width, (ml_val + arxiv_val) / 2, f'↓{improvement:.0f}%',
                ha='center', va='center', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='#90EE90', alpha=0.85, edgecolor='darkgreen', linewidth=1.5))

plt.tight_layout()
plt.savefig('latency_by_validator_cohort_optimized.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved as 'latency_by_validator_cohort_optimized.png'")
plt.show()