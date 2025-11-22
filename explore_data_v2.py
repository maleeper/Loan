import pandas as pd
import numpy as np
from pathlib import Path

# Load data
DATA_DIR = Path('data')
train_path = DATA_DIR / 'train.csv'
df = pd.read_csv(train_path)

print(f"Dataset Shape: {df.shape}")
print("-" * 30)

# Target distribution
print("\nTarget Distribution (loan_paid_back):")
print(df['loan_paid_back'].value_counts(normalize=True))
print("-" * 30)

# Missing values
print("\nMissing Values:")
missing = df.isnull().sum()
print(missing[missing > 0])
print("-" * 30)

# Numerical Columns Analysis
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'loan_paid_back' in numeric_cols:
    numeric_cols.remove('loan_paid_back')
if 'id' in numeric_cols:
    numeric_cols.remove('id')

print("\nNumerical Columns Statistics:")
print(df[numeric_cols].describe())

print("\nCorrelations with Target:")
correlations = df[numeric_cols + ['loan_paid_back']].corr()['loan_paid_back'].sort_values(ascending=False)
print(correlations)

print("\nPotential Outliers (Z-score > 3):")
for col in numeric_cols:
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    outliers_count = (z_scores > 3).sum()
    print(f"{col}: {outliers_count} outliers ({outliers_count/len(df)*100:.2f}%)")

print("-" * 30)

# Categorical Columns Analysis
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\nCategorical Columns Analysis:")
for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(f"Unique values: {df[col].nunique()}")
    print("Top 10 values:")
    print(df[col].value_counts().head(10))
    
    # Check relationship with target
    if df[col].nunique() < 50:
        print("Mean target by category:")
        print(df.groupby(col)['loan_paid_back'].mean().sort_values(ascending=False))
