import matplotlib.pyplot as plt
import numpy as np
from math import pi
import pandas as pd

# Data
categories = ['Detection\nAccuracy', 'Response\nSpeed', 
              'False Positive\nRejection', 'Data\nIntegrity', 
              'Throughput\nEfficiency']

systems = {
    'Traditional\nCentralized': [0, 0, 0, 17, 60],
    'DecenTrack\nPoW+ML': [83, 100, 96, 75, 97],
    'BI-WDRS\n(Blockchain+ML)': [100, 68, 100, 97, 75]
}

colors = {
    'Traditional\nCentralized': '#E74C3C',
    'DecenTrack\nPoW+ML': '#2ECC71',
    'BI-WDRS\n(Blockchain+ML)': '#9B59B6'
}

alphas = {
    'Traditional\nCentralized': 0.3,
    'DecenTrack\nPoW+ML': 0.7,
    'BI-WDRS\n(Blockchain+ML)': 0.6
}

# Number of variables
num_vars = len(categories)

# Compute angle for each axis
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # Complete the circle

# Create figure
fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(projection='polar'))

# Plot data for each system
for system_name, values in systems.items():
    values += values[:1]  # Complete the circle
    
    ax.plot(angles, values, 'o-', linewidth=2.5, 
            label=system_name, color=colors[system_name],
            markersize=8, markeredgewidth=2, markeredgecolor='black')
    
    ax.fill(angles, values, alpha=alphas[system_name], 
            color=colors[system_name])

# Customize axes
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')

# Set y-axis (radial)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
ax.set_rlabel_position(0)
ax.grid(True, linestyle='--', linewidth=1, alpha=0.6)

# Title and legend
plt.title('Multi-Metric System Performance Comparison\nDecenTrack vs BI-WDRS', 
          fontsize=16, fontweight='bold', pad=30)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
           fontsize=12, framealpha=0.95, edgecolor='black', fancybox=True)

# Add text box with insights
textstr = (
    'Key Insights:\n'
    '• DecenTrack excels in Throughput (97) & Response (100)\n'
    '• BI-WDRS leads in Accuracy (100) & Integrity (97)\n'
    '• Both outperform traditional systems across all metrics\n'
    '• DecenTrack score: 90.2 | BI-WDRS score: 88.0 (balanced vs specialized)'
)
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1)
ax.text(0.5, -0.15, textstr, transform=ax.transAxes,
        fontsize=11, bbox=props, ha='center', va='top',
        family='monospace', fontweight='bold')

plt.tight_layout()
plt.savefig('radar_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Radar chart saved as 'radar_performance_comparison.png'")
plt.show()

# Print normalized scores table
print("\n" + "="*80)
print("NORMALIZED SCORES (0-100 Scale)")
print("="*80)

df_scores = pd.DataFrame(systems).T
df_scores.columns = categories[:-1] + ['Throughput']
print(df_scores.to_string())
print("\nAverage Scores:")
for system_name, values in systems.items():
    avg = np.mean(values)
    print(f"  {system_name.strip()}: {avg:.1f}/100")