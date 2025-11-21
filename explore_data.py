import pandas as pd
import os

data_dir = 'c:\\vs-code-projects\\Loan\\data'
train_path = os.path.join(data_dir, 'train.csv')
test_path = os.path.join(data_dir, 'test.csv')

print("Loading data...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

print("\nMissing values in Train:")
print(train_df.isnull().sum())

print("\nMissing values in Test:")
print(test_df.isnull().sum())

print("\nTarget Distribution:")
print(train_df['loan_paid_back'].value_counts(normalize=True))

print("\nColumn Types:")
print(train_df.dtypes)

print("\nUnique values in categorical columns (Train):")
categorical_cols = train_df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"{col}: {train_df[col].nunique()} unique values")
    print(train_df[col].value_counts().head())

print("\nUnique values in categorical columns (Test):")
for col in categorical_cols:
    if col in test_df.columns:
        print(f"{col}: {test_df[col].nunique()} unique values")
