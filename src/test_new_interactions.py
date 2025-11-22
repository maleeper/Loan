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
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def feature_engineering(df):
    df = df.copy()
    # Clipping
    for col in ['annual_income', 'debt_to_income_ratio']:
        limit = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=limit)
        
    # Previous features
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['monthly_debt'] = (df['annual_income'] / 12) * df['debt_to_income_ratio']
    df['interest_burden'] = df['loan_amount'] * (df['interest_rate'] / 100)
    
    # NEW FEATURES
    # Disposable Income: Annual Income * (1 - DTI)
    # This is a proxy for "Free Cash Flow"
    df['disposable_income'] = df['annual_income'] * (1 - df['debt_to_income_ratio'])
    
    # Credit-Income Interaction: Credit Score * log(Income)
    # Combines history and capacity
    df['credit_income_interaction'] = df['credit_score'] * np.log1p(df['annual_income'])
    
    return df

def get_preprocessor(X):
    log_features = ['annual_income']
    numeric_features = ['debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate',
                        'loan_to_income', 'monthly_debt', 'interest_burden',
                        'disposable_income', 'credit_income_interaction'] # Added new features here
    ordinal_features = ['grade_subgrade']
    categorical_features = ['loan_purpose', 'gender', 'marital_status', 'education_level', 'employment_status']
    
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
    return preprocessor

def main():
    DATA_DIR = Path('data')
    train_path = DATA_DIR / 'train.csv'
    
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    
    print("Feature Engineering (Clipping + New Interactions)...")
    train_df = feature_engineering(train_df)
    
    target_col = 'loan_paid_back'
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    preprocessor = get_preprocessor(X)
    
    # Using best params found in previous step (hardcoded for speed in this test)
    # Ideally we would retune, but let's check signal first
    best_xgb_params = {'n_estimators': 152, 'learning_rate': 0.045, 'max_depth': 5, 'subsample': 0.74, 'colsample_bytree': 0.92, 'reg_alpha': 8.43, 'reg_lambda': 6.62}
    best_lgbm_params = {'n_estimators': 126, 'learning_rate': 0.049, 'num_leaves': 39, 'feature_fraction': 0.76, 'bagging_fraction': 0.87, 'bagging_freq': 5}
    
    print("Building Stacking Classifier...")
    
    best_xgb = XGBClassifier(**best_xgb_params, random_state=42, n_jobs=-1, eval_metric='auc')
    best_lgbm = LGBMClassifier(**best_lgbm_params, random_state=42, n_jobs=-1, verbose=-1)
    cat_clf = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_state=42, verbose=0, allow_writing_files=False)
    
    pipe_xgb = Pipeline([('preprocessor', preprocessor), ('model', best_xgb)])
    pipe_lgbm = Pipeline([('preprocessor', preprocessor), ('model', best_lgbm)])
    pipe_cat = Pipeline([('preprocessor', preprocessor), ('model', cat_clf)])
    
    estimators = [
        ('xgb', pipe_xgb),
        ('lgbm', pipe_lgbm),
        ('cat', pipe_cat)
    ]
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3,
        n_jobs=-1,
        passthrough=False
    )
    
    print("Evaluating Stacking Classifier with New Interactions on subset (20k)...")
    X_eval = X.sample(n=20000, random_state=42)
    y_eval = y.loc[X_eval.index]
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(stacking_clf, X_eval, y_eval, cv=skf, scoring='roc_auc', n_jobs=-1)
    print(f"Stacking AUC (New Interactions): {scores.mean():.4f} (+/- {scores.std():.4f})")

if __name__ == "__main__":
    main()
