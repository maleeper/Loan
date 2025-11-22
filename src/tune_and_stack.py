import sys
import os
sys.path.append(os.path.abspath("."))
import pandas as pd
import numpy as np
from pathlib import Path
import optuna
from functools import partial

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
        
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['monthly_debt'] = (df['annual_income'] / 12) * df['debt_to_income_ratio']
    df['interest_burden'] = df['loan_amount'] * (df['interest_rate'] / 100)
    return df

def get_preprocessor(X):
    log_features = ['annual_income']
    numeric_features = ['debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate',
                        'loan_to_income', 'monthly_debt', 'interest_burden']
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

def objective_xgb(trial, X, y, preprocessor):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc'
    }
    
    model = XGBClassifier(**params)
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

def objective_lgbm(trial, X, y, preprocessor):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = LGBMClassifier(**params)
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

def main():
    DATA_DIR = Path('data')
    train_path = DATA_DIR / 'train.csv'
    
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    
    print("Feature Engineering & Clipping...")
    train_df = feature_engineering(train_df)
    
    target_col = 'loan_paid_back'
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # Use a subset for tuning to save time
    X_tune = X.sample(n=10000, random_state=42)
    y_tune = y.loc[X_tune.index]
    
    preprocessor = get_preprocessor(X)
    
    print("Tuning XGBoost...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(partial(objective_xgb, X=X_tune, y=y_tune, preprocessor=preprocessor), n_trials=10)
    print(f"Best XGB params: {study_xgb.best_params}")
    
    print("Tuning LightGBM...")
    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(partial(objective_lgbm, X=X_tune, y=y_tune, preprocessor=preprocessor), n_trials=10)
    print(f"Best LGBM params: {study_lgbm.best_params}")
    
    # CatBoost is usually good with defaults, but let's use standard params for now to save time/complexity
    # or just use the previous best params
    
    print("Building Stacking Classifier with Best Params...")
    
    best_xgb = XGBClassifier(**study_xgb.best_params, random_state=42, n_jobs=-1, eval_metric='auc')
    best_lgbm = LGBMClassifier(**study_lgbm.best_params, random_state=42, n_jobs=-1, verbose=-1)
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
    
    print("Evaluating Stacking Classifier on larger subset (20k)...")
    X_eval = X.sample(n=20000, random_state=42)
    y_eval = y.loc[X_eval.index]
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(stacking_clf, X_eval, y_eval, cv=skf, scoring='roc_auc', n_jobs=-1)
    print(f"Stacking AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

if __name__ == "__main__":
    main()
