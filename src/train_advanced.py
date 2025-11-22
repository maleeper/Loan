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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def feature_engineering(df):
    df = df.copy()
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['monthly_debt'] = (df['annual_income'] / 12) * df['debt_to_income_ratio']
    df['interest_burden'] = df['loan_amount'] * (df['interest_rate'] / 100)
    return df

def main():
    DATA_DIR = Path('data')
    train_path = DATA_DIR / 'train.csv'
    
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    
    # Feature Engineering
    print("Applying feature engineering...")
    train_df = feature_engineering(train_df)
    
    target_col = 'loan_paid_back'
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # Define column groups
    log_features = ['annual_income']
    numeric_features = ['debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate',
                        'loan_to_income', 'monthly_debt', 'interest_burden']
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
    
    # Models
    xgb_clf = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric='auc')
    lgbm_clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    cat_clf = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_state=42, verbose=0, allow_writing_files=False)
    
    pipe_xgb = Pipeline([('preprocessor', preprocessor), ('model', xgb_clf)])
    pipe_lgbm = Pipeline([('preprocessor', preprocessor), ('model', lgbm_clf)])
    pipe_cat = Pipeline([('preprocessor', preprocessor), ('model', cat_clf)])
    
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', pipe_xgb),
            ('lgbm', pipe_lgbm),
            ('cat', pipe_cat)
        ],
        voting='soft'
    )
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    print("Evaluating Ensemble on subset (20k rows)...")
    X_sample = X.sample(n=20000, random_state=42)
    y_sample = y.loc[X_sample.index]
    
    scores = cross_val_score(voting_clf, X_sample, y_sample, cv=skf, scoring='roc_auc', n_jobs=-1)
    print(f"Ensemble AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

if __name__ == "__main__":
    main()
