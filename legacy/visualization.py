"""
Data Visualization Module
Creates and saves EDA visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def create_visualizations(df, save_dir='plots'):
    """Create and save EDA visualizations."""
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Class distribution
    plt.figure(figsize=(8, 6))
    df['Outcome'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Outcome (0=No Diabetes, 1=Diabetes)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {save_dir}/class_distribution.png")
    
    # 2. Feature distributions
    feature_cols = [col for col in df.columns if col != 'Outcome']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(feature_cols[:4]):
        axes[idx].hist(df[col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col} Distribution', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {save_dir}/feature_distributions.png")
    
    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {save_dir}/correlation_heatmap.png")
    
    print(f"\n[OK] All visualizations saved to '{save_dir}/' directory")
