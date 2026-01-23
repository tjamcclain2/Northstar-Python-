import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = 'cs_type_dataset_inference'
os.makedirs(output_dir, exist_ok=True)

# 1. Load the dataset
df = pd.read_csv('data/injections/dataset_stage1.csv')

# 2. Identify 'Dead' Columns (Zero Variance)
# These are features that never change and provide zero information
constants = [col for col in df.columns if df[col].nunique() <= 1]

# 3. Correlation Analysis
# We drop constants and IDs to focus on physical features
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=constants + ['idx', 'seed'])
corr_matrix = df_numeric.corr()

# --- VISUALIZATION 1: Target Balance ---
plt.figure(figsize=(8, 5))
sns.countplot(x='n_star', data=df, hue='n_star', palette='viridis', legend=False)
plt.title('Target Class Distribution (n_star)')
plt.ylabel('Number of Samples')
plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
plt.close()

# --- VISUALIZATION 2: Feature Relationship Heatmap ---
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap (Physical Parameters vs n_star)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.close()

# --- VISUALIZATION 3: Feature Overlap (Boxplots) ---
# Highlighting if physical features differ between classes
features_to_check = ['snr', 'f', 'theta', 'tau']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(features_to_check):
    sns.boxplot(x='n_star', y=col, data=df, hue='n_star', palette='Set2', ax=axes[i], legend=False)
    axes[i].set_title(f'Distribution of {col} across n_star classes')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_boxplots.png'))
plt.close()

# 4. Summary Printout
print(f"--- Dataset Audit Summary ---")
print(f"Total Observations: {len(df)}")
print(f"Constant Features Detected: {constants}")
print(f"\nTop Correlations with Target (n_star):")
print(corr_matrix['n_star'].sort_values(ascending=False))
print(f"\nAll figures saved to '{output_dir}/' directory")