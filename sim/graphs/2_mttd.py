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
response_times = [6.2, 4.5, 2.5, 3.1]
std_dev = [0.5, 0.6, 0.4, 0.45]  # Experimental variance

# Colors: consistent with Graph 1
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Create bars with error bars
x_pos = np.arange(len(systems))
bars = ax.bar(x_pos, response_times, yerr=std_dev, capsize=8,
              color=colors, alpha=0.85, edgecolor='black', linewidth=1.5,
              error_kw={'elinewidth': 2, 'capthick': 2})

# Customize axes
ax.set_ylabel('Average Response Time (seconds)', fontsize=13, fontweight='bold')
ax.set_xlabel('System Type', fontsize=13, fontweight='bold')
ax.set_title('Average Response Time Comparison\nDecenTrack vs BI-WDRS', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(systems, fontsize=11)
ax.set_ylim(0, 7.5)

# Add horizontal line for baseline (traditional system)
ax.axhline(y=6.2, color='red', linestyle='--', linewidth=2, 
           alpha=0.4, label='Traditional System Baseline')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, response_times)):
    height = bar.get_height()
    label_y = height + std_dev[i] + 0.2
    
    # Main value
    ax.text(bar.get_x() + bar.get_width()/2., label_y,
            f'{val:.1f}s', ha='center', va='bottom', 
            fontsize=12, fontweight='bold')
    
    # Performance improvement percentage (vs traditional)
    if i > 0:
        improvement = ((response_times[0] - response_times[i]) / response_times[0]) * 100
        ax.text(bar.get_x() + bar.get_width()/2., 0.3,
                f'-{improvement:.1f}%', ha='center', va='bottom',
                fontsize=9, style='italic', color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))

# Grid styling
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Add comparison annotation
textstr_comparison = (
    f'DecenTrack PoW+ML vs BI-WDRS:\n'
    f'{((response_times[2] - response_times[3]) / response_times[3] * 100):.1f}% faster\n'
    f'(2.5s vs 3.1s)'
)
props_comparison = dict(boxstyle='round', facecolor='lightyellow', alpha=0.5, pad=0.8)
ax.text(0.98, 0.50, textstr_comparison, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=props_comparison, fontweight='bold', color='darkgreen')

# Add methodology annotation
textstr_method = (
    'Converted from MTTD:\n'
    'Response = MTTD_rounds × 2s + 0.5s overhead'
)
props_method = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
ax.text(0.02, 0.98, textstr_method, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='left',
        bbox=props_method, style='italic')

# Add legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('response_time_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved as 'response_time_comparison.png'")
plt.show()