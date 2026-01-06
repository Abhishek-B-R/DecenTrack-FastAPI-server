import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(12, 7))

# Data: Correct decisions per block
systems = ['Traditional\nCentralized', 'DecenTrack\nPoW', 
           'DecenTrack\nPoW+ML', 'BI-WDRS\n(Blockchain+ML)']
throughput = [0.60, 0.82, 0.97, 0.75]  # decisions per block
std_dev = [0.08, 0.10, 0.06, 0.08]

colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']

bars = ax.bar(systems, throughput, yerr=std_dev, capsize=8,
              color=colors, alpha=0.85, edgecolor='black', linewidth=1.5,
              error_kw={'elinewidth': 2, 'capthick': 2})

ax.set_ylabel('Correct Decisions Per Block', fontsize=12, fontweight='bold')
ax.set_title('Effective Consensus Throughput\nDecenTrack (Internal Evaluation) with Literature Reference',
             fontsize=13, fontweight='bold', pad=20)
ax.set_ylim(0, 1.2)
ax.grid(axis='y', alpha=0.3)

# Add baseline line
ax.axhline(y=0.60, color='red', linestyle='--', linewidth=2, alpha=0.4, label='Traditional Baseline')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, throughput)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement percentage
    if i > 0:
        improvement = ((val - throughput[0]) / throughput[0]) * 100
        ax.text(bar.get_x() + bar.get_width()/2., 0.02,
                f'+{improvement:.1f}%', ha='center', va='bottom',
                fontsize=9, style='italic', color='green', fontweight='bold')

ax.legend(fontsize=10, loc='upper left')

# # Add annotation
# ax.text(0.5, -0.22, 
#         'DecenTrack PoW+ML achieves 62% higher throughput than traditional systems',
#         transform=ax.transAxes, ha='center', fontsize=10, style='italic',
#         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

plt.tight_layout()
plt.savefig('throughput_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Throughput graph saved")
plt.show()