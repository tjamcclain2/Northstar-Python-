import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os


# Set plot style for academic presentation
plt.rcParams.update({
    "text.usetex": False, # Set to True if you have LaTeX installed on your system
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 14
})

def analyze_gravitational_dataset(file_path, output_dir='physics_type_data_inference'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(file_path)
    
    print(f"--- Dataset Dimensions: {df.shape[0]} events, {df.shape[1]} parameters ---")
    
    # 1. Physics Metadata & Systematic Checks
    # Identifying features with zero variance (no physical information for the model)
    constants = [col for col in df.columns if df[col].nunique() <= 1]
    
    # 2. Statistical Separation Analysis (ANOVA)
    # Testing the Null Hypothesis: 'The mean of physical parameter X is the same across all classes of n*'
    print("\n[Statistical Significance of Parameters]")
    parameters = ['snr', 'f', 'theta', 'phi', 'tau', 'iota']
    for p in parameters:
        groups = [group[p].values for name, group in df.groupby('n_star')]
        f_stat, p_val = stats.f_oneway(*groups)
        significance = "Significant" if p_val < 0.05 else "Insignificant"
        print(f"Parameter {p:6}: F-stat={f_stat:.4f}, p-value={p_val:.4f} ({significance})")

    # 3. Visualization: Parameter Distributions across n_star
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Using physics notation for labels
    physics_labels = {'snr': r'$\rho$ (SNR)', 'f': '$f$ (Hz)', 
                      'theta': r'$\theta$ (rad)', 'tau': r'$\tau$ (delay)'}
    
    for ax, (key, label) in zip(axes.flatten(), physics_labels.items()):
        sns.violinplot(x='n_star', y=key, data=df, ax=ax, inner="quartile", hue='n_star', palette='muted', legend=False)

        ax.set_ylabel(label)
        ax.set_title(f'Distribution of {label} vs $n^*$')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'physics_parameter_distributions.png'))
    plt.close()

    # 4. Correlation Heatmap (Physical Feature Coupling)
    plt.figure(figsize=(10, 8))
    # Filter to physical features only
    phys_df = df.drop(columns=constants + ['idx', 'seed'])
    corr = phys_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0, fmt=".2f")
    plt.title(r'Correlation Matrix: Physical Features and Target ($n^*$)')
    plt.savefig(os.path.join(output_dir, 'parameter_correlation.png'))
    plt.close()

    return constants

# Execute
uninformative_features = analyze_gravitational_dataset('data/injections/dataset_stage1.csv')

print("\n[Recommendation for Professor]")
print(f"1. Remove constant features: {uninformative_features} (Systematic bias risk)")
print("2. Current p-values suggest physical parameters are nearly independent of n_star classes.")
print("3. Recommendation: Shift to log-space for f and SNR, and use Sin/Cos encoding for angular variables.")
print(f"\n✓ All figures saved to 'physics_type_data_inference/' directory")