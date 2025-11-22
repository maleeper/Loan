import sys
import os
sys.path.append(os.path.abspath("."))
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def main():
    DATA_DIR = Path('data')
    train_path = DATA_DIR / 'train.csv'
    
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    
    target_col = 'loan_paid_back'
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # Define column groups
    log_features = ['annual_income']
    numeric_features = ['debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
    ordinal_features = ['grade_subgrade']
    categorical_features = ['loan_purpose', 'gender', 'marital_status', 'education_level', 'employment_status']
    
    # Transformers
    log_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p, validate=False)),
        ('scaler', StandardScaler())
    ])
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Ordinal encoding for grades (A1..F5)
    # We assume alphabetical order correlates with credit quality (or at least is monotonic)
    grades = sorted(X['grade_subgrade'].unique())
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[grades], handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('log', log_transformer, log_features),
            ('num', numeric_transformer, numeric_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    print("Starting cross-validation...")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        # Use a subset for speed in this verification script
        X_sample = X.sample(n=20000, random_state=42)
        y_sample = y.loc[X_sample.index]
        
        scores = cross_val_score(pipeline, X_sample, y_sample, cv=skf, scoring='roc_auc', n_jobs=-1)
        print(f"{name} AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

if __name__ == "__main__":
    main()
