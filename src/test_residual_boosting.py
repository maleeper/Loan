import sys
import os
sys.path.append(os.path.abspath("."))
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.special import expit, logit

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

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

# Preprocessor for Tree Models
def get_tree_preprocessor(X):
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

# Preprocessor for NN
def get_nn_preprocessor(X):
    log_features = ['annual_income']
    numeric_features = ['debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate',
                        'loan_to_income', 'monthly_debt', 'interest_burden']
    categorical_features = ['grade_subgrade', 'loan_purpose', 'gender', 'marital_status', 'education_level', 'employment_status']
    
    log_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p, validate=False)),
        ('scaler', StandardScaler())
    ])
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('log', log_transformer, log_features),
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def main():
    DATA_DIR = Path('data')
    train_path = DATA_DIR / 'train.csv'
    
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    
    print("Feature Engineering...")
    train_df = feature_engineering(train_df)
    
    target_col = 'loan_paid_back'
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # 1. Create Holdout Set (20%)
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Full Training Set: {X_train_full.shape}")
    print(f"Holdout Set: {X_holdout.shape}")
    
    # 2. Split Training into Stage 1 (50%) and Stage 2 (50%)
    # This means Stage 1 gets 40% of total, Stage 2 gets 40% of total
    X_stage1, X_stage2, y_stage1, y_stage2 = train_test_split(X_train_full, y_train_full, test_size=0.5, random_state=42, stratify=y_train_full)
    
    print(f"Stage 1 Set: {X_stage1.shape}")
    print(f"Stage 2 Set: {X_stage2.shape}")
    
    # Preprocessors
    tree_preprocessor = get_tree_preprocessor(X)
    
    # Fit preprocessor on Stage 1 (and apply to all)
    print("Preprocessing...")
    X_stage1_proc = tree_preprocessor.fit_transform(X_stage1)
    X_stage2_proc = tree_preprocessor.transform(X_stage2)
    X_holdout_proc = tree_preprocessor.transform(X_holdout)
    
    # --- STAGE 1: Train XGBoost ---
    print("Training Stage 1 (XGBoost)...")
    best_xgb_params = {'n_estimators': 152, 'learning_rate': 0.045, 'max_depth': 5, 'subsample': 0.74, 'colsample_bytree': 0.92, 'reg_alpha': 8.43, 'reg_lambda': 6.62}
    model_stage1 = XGBClassifier(**best_xgb_params, random_state=42, n_jobs=-1, eval_metric='auc')
    model_stage1.fit(X_stage1_proc, y_stage1)
    
    # Predict Logits for Stage 2
    # XGBoost predict_proba gives probabilities. We need logits (margin).
    # output_margin=True gives raw margin scores (logits)
    logits_stage2 = model_stage1.predict(X_stage2_proc, output_margin=True)
    logits_holdout = model_stage1.predict(X_holdout_proc, output_margin=True)
    
    auc_stage1 = roc_auc_score(y_holdout, expit(logits_holdout))
    print(f"Stage 1 AUC on Holdout: {auc_stage1:.5f}")
    
    # --- STAGE 2: Train LightGBM on Residuals ---
    print("Training Stage 2 (LightGBM with init_score)...")
    best_lgbm_params = {'n_estimators': 126, 'learning_rate': 0.049, 'num_leaves': 39, 'feature_fraction': 0.76, 'bagging_fraction': 0.87, 'bagging_freq': 5}
    model_stage2 = LGBMClassifier(**best_lgbm_params, random_state=42, n_jobs=-1, verbose=-1)
    
    # Train using init_score
    model_stage2.fit(X_stage2_proc, y_stage2, init_score=logits_stage2)
    
    # Evaluate Combined Model on Holdout
    # LightGBM predict(raw_score=True) gives the margin contribution of LightGBM
    # Final Logit = Stage 1 Logit + Stage 2 Logit
    # Note: sklearn API for LGBM doesn't easily support init_score in predict.
    # But we can just pass init_score to predict_proba? No, predict_proba doesn't take init_score.
    # We need to use the raw margin prediction and add it manually.
    
    lgbm_margin_holdout = model_stage2.predict(X_holdout_proc, raw_score=True)
    final_logits_holdout = logits_holdout + lgbm_margin_holdout
    final_probs_holdout = expit(final_logits_holdout)
    
    auc_residual = roc_auc_score(y_holdout, final_probs_holdout)
    print(f"Residual Boosting AUC on Holdout: {auc_residual:.5f}")
    
    # --- BASELINE: Train Stacking on Full Training (80%) ---
    print("Training Baseline Stacking on Full Training Set (80%)...")
    
    # We need to re-fit preprocessors on the full training set to be fair? 
    # Or just use the one fitted on Stage 1? To be fair, the Baseline should use all data.
    # Let's use a fresh pipeline for Baseline.
    
    X_train_full_proc = tree_preprocessor.fit_transform(X_train_full)
    X_holdout_proc_baseline = tree_preprocessor.transform(X_holdout)
    
    # Define Estimators (Same as our best model)
    best_xgb = XGBClassifier(**best_xgb_params, random_state=42, n_jobs=-1, eval_metric='auc')
    best_lgbm = LGBMClassifier(**best_lgbm_params, random_state=42, n_jobs=-1, verbose=-1)
    cat_clf = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_state=42, verbose=0, allow_writing_files=False)
    
    # Note: For speed in this test, I'm omitting MLP, but keeping the 3 tree models
    estimators = [
        ('xgb', best_xgb),
        ('lgbm', best_lgbm),
        ('cat', cat_clf)
    ]
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3,
        n_jobs=-1,
        passthrough=False
    )
    
    stacking_clf.fit(X_train_full_proc, y_train_full)
    
    y_pred_baseline = stacking_clf.predict_proba(X_holdout_proc_baseline)[:, 1]
    auc_baseline = roc_auc_score(y_holdout, y_pred_baseline)
    print(f"Baseline Stacking AUC on Holdout: {auc_baseline:.5f}")
    
    print("-" * 30)
    print(f"Residual Boosting: {auc_residual:.5f}")
    print(f"Baseline Stacking: {auc_baseline:.5f}")
    
    if auc_residual > auc_baseline:
        print("SUCCESS: Residual Boosting outperformed Baseline!")
    else:
        print("RESULT: Baseline outperformed Residual Boosting (likely due to data splitting).")

if __name__ == "__main__":
    main()
