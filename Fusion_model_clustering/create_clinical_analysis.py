#!/usr/bin/env python3
"""Create clinical relevance analysis visualizations."""

import sys
from pathlib import Path

# Check for required packages
required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy', 
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'scipy': 'scipy'
}

missing_packages = []
for module, package in required_packages.items():
    try:
        __import__(module)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"ERROR: Missing required packages: {', '.join(missing_packages)}")
    print(f"Please install them with: pip3 install {' '.join(missing_packages)}")
    sys.exit(1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Paths
PROJECT_ROOT = Path(__file__).parent
clusters_df = pd.read_csv(PROJECT_ROOT / "analysis" / "clusters.csv")
# Try multiple possible locations for clinical data
clinical_paths = [
    PROJECT_ROOT / "data" / "clinical_data.csv",
    Path("/home/ubuntu/Fusion_model/Fusion_model_clustering/data/clinical_data.csv"),
    Path("/home/ubuntu/Fusion_model/MuFFLe/src/data/raw/clinical_data.csv"),
    Path("/home/ubuntu/Fusion_model/MuFFLe/Fusion_model_clustering/data/clinical_data.csv")
]
clinical_df = None
for path in clinical_paths:
    if path.exists():
        clinical_df = pd.read_csv(path)
        print(f"Loaded clinical data from: {path}")
        break
if clinical_df is None:
    raise FileNotFoundError(f"Could not find clinical_data.csv. Checked: {clinical_paths}")
output_dir = PROJECT_ROOT / "analysis" / "clinical_analysis"
output_dir.mkdir(exist_ok=True)

# Merge
merged = clusters_df.merge(clinical_df, left_on='patient_id', right_on='chimera_id_t3', how='inner')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

# 1. Progression rate by cluster
fig, ax = plt.subplots(figsize=(10, 6))
prog_by_cluster = merged.groupby('cluster')['progression'].agg(['sum', 'count', 'mean']).reset_index()
prog_by_cluster.columns = ['cluster', 'events', 'total', 'rate']
x_pos = np.arange(len(prog_by_cluster))
bars = ax.bar(x_pos, prog_by_cluster['rate']*100, color=['#E63946', '#457B9D', '#2A9D8F'])
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Progression Rate (%)', fontsize=12)
ax.set_title('Disease Progression Rate by Cluster', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Cluster {int(c)}' for c in prog_by_cluster['cluster']])
for i, (idx, row) in enumerate(prog_by_cluster.iterrows()):
    ax.text(i, row['rate']*100 + 2, f"{row['events']}/{row['total']}\n({row['rate']*100:.1f}%)", 
            ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(output_dir / 'progression_by_cluster.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. BRS distribution
fig, ax = plt.subplots(figsize=(10, 6))
brs_crosstab = pd.crosstab(merged['cluster'], merged['BRS'], normalize='index') * 100
brs_crosstab.plot(kind='bar', ax=ax, color=['#E63946', '#457B9D', '#2A9D8F'])
ax.set_title('BRS (Bladder Recurrence Score) Distribution by Cluster', fontsize=14, fontweight='bold')
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.legend(title='BRS Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels([f'Cluster {int(c)}' for c in brs_crosstab.index], rotation=0)
plt.tight_layout()
plt.savefig(output_dir / 'BRS_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Chi-square test for BRS
brs_ct = pd.crosstab(merged['cluster'], merged['BRS'])
chi2_brs, p_brs, dof_brs, _ = stats.chi2_contingency(brs_ct)

# 3. Age distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
merged.boxplot(column='age', by='cluster', ax=axes[0])
axes[0].set_title('Age Distribution by Cluster')
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Age (years)')

sns.violinplot(data=merged, x='cluster', y='age', ax=axes[1])
axes[1].set_title('Age Distribution by Cluster (Violin Plot)')
axes[1].set_xlabel('Cluster')
plt.suptitle('')
plt.tight_layout()
plt.savefig(output_dir / 'age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ANOVA for age
age_by_cluster = [merged[merged['cluster']==c]['age'].dropna() for c in sorted(merged['cluster'].unique())]
f_age, p_age = stats.f_oneway(*age_by_cluster)

# 4. EORTC risk distribution
fig, ax = plt.subplots(figsize=(10, 6))
eortc_crosstab = pd.crosstab(merged['cluster'], merged['EORTC'], normalize='index') * 100
eortc_crosstab.plot(kind='bar', ax=ax, color=['#E63946', '#457B9D'])
ax.set_title('EORTC Risk Category Distribution by Cluster', fontsize=14, fontweight='bold')
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.legend(title='EORTC Risk', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels([f'Cluster {int(c)}' for c in eortc_crosstab.index], rotation=0)
plt.tight_layout()
plt.savefig(output_dir / 'EORTC_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Summary figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Cluster sizes
ax1 = fig.add_subplot(gs[0, 0])
cluster_counts = merged['cluster'].value_counts().sort_index()
ax1.bar(cluster_counts.index.astype(str), cluster_counts.values, 
        color=['#E63946', '#457B9D', '#2A9D8F'])
ax1.set_title('Cluster Sizes', fontweight='bold')
ax1.set_xlabel('Cluster')
ax1.set_ylabel('Number of Patients')

# Progression rates
ax2 = fig.add_subplot(gs[0, 1])
prog_rate = merged.groupby('cluster')['progression'].mean() * 100
ax2.bar(prog_rate.index.astype(str), prog_rate.values, 
        color=['#E63946', '#457B9D', '#2A9D8F'])
ax2.set_title('Progression Rate by Cluster', fontweight='bold')
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Progression Rate (%)')

# Mean age
ax3 = fig.add_subplot(gs[0, 2])
mean_age = merged.groupby('cluster')['age'].mean()
ax3.bar(mean_age.index.astype(str), mean_age.values, 
        color=['#E63946', '#457B9D', '#2A9D8F'])
ax3.set_title('Mean Age by Cluster', fontweight='bold')
ax3.set_xlabel('Cluster')
ax3.set_ylabel('Mean Age (years)')

# Statistical summary
ax4 = fig.add_subplot(gs[1, :])
ax4.axis('off')
summary_text = f"""
CLINICAL RELEVANCE ANALYSIS SUMMARY
{'='*70}

Statistical Tests:
• BRS Distribution: Chi-square = {chi2_brs:.4f}, p = {p_brs:.4f} {'***' if p_brs < 0.001 else '**' if p_brs < 0.01 else '*' if p_brs < 0.05 else ''}
• Age Distribution: ANOVA F = {f_age:.4f}, p = {p_age:.4f} {'***' if p_age < 0.001 else '**' if p_age < 0.01 else '*' if p_age < 0.05 else ''}

Cluster Characteristics:
"""
for c in sorted(merged['cluster'].unique()):
    c_data = merged[merged['cluster'] == c]
    prog_rate_c = c_data['progression'].mean() * 100
    mean_age_c = c_data['age'].mean()
    brs_dist = c_data['BRS'].value_counts(normalize=True) * 100
    summary_text += f"""
Cluster {int(c)} (n={len(c_data)}):
  • Progression rate: {prog_rate_c:.1f}%
  • Mean age: {mean_age_c:.1f} years
  • BRS distribution: {', '.join([f'{k}: {v:.1f}%' for k, v in brs_dist.items()])}
"""

summary_text += "\n(* p<0.05, ** p<0.01, *** p<0.001)"
ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig(output_dir / 'clinical_relevance_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Analysis complete! Results saved to {output_dir}")
print(f"\nKey findings:")
print(f"- BRS distribution: p = {p_brs:.4f}")
print(f"- Age distribution: p = {p_age:.4f}")

