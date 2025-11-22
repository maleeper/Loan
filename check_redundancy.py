import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')
train_path = DATA_DIR / 'train.csv'
df = pd.read_csv(train_path)

# Create my proposed feature
df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)

# Check correlation
corr = df['loan_to_income'].corr(df['debt_to_income_ratio'])
print(f"Correlation between loan_to_income and debt_to_income_ratio: {corr:.4f}")

# Check if they are identical (ignoring scale)
print("\nSample values:")
print(df[['loan_amount', 'annual_income', 'loan_to_income', 'debt_to_income_ratio']].head())
