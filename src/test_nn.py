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

# Preprocessor for Tree Models (Ordinal Encoding for Grades)
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

# Preprocessor for Neural Network (One-Hot Encoding for EVERYTHING, Standard Scaling)
def get_nn_preprocessor(X):
    log_features = ['annual_income']
    numeric_features = ['debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate',
                        'loan_to_income', 'monthly_debt', 'interest_burden']
    # For NN, we treat grade_subgrade as categorical to One-Hot Encode it
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
    
    print("Feature Engineering (Clipping)...")
    train_df = feature_engineering(train_df)
    
    target_col = 'loan_paid_back'
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    tree_preprocessor = get_tree_preprocessor(X)
    nn_preprocessor = get_nn_preprocessor(X)
    
    # Best params from before
    best_xgb_params = {'n_estimators': 152, 'learning_rate': 0.045, 'max_depth': 5, 'subsample': 0.74, 'colsample_bytree': 0.92, 'reg_alpha': 8.43, 'reg_lambda': 6.62}
    best_lgbm_params = {'n_estimators': 126, 'learning_rate': 0.049, 'num_leaves': 39, 'feature_fraction': 0.76, 'bagging_fraction': 0.87, 'bagging_freq': 5}
    
    print("Building Stacking Classifier with MLP...")
    
    best_xgb = XGBClassifier(**best_xgb_params, random_state=42, n_jobs=-1, eval_metric='auc')
    best_lgbm = LGBMClassifier(**best_lgbm_params, random_state=42, n_jobs=-1, verbose=-1)
    cat_clf = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_state=42, verbose=0, allow_writing_files=False)
    
    # MLP Classifier
    mlp_clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001, max_iter=200, random_state=42, early_stopping=True)
    
    pipe_xgb = Pipeline([('preprocessor', tree_preprocessor), ('model', best_xgb)])
    pipe_lgbm = Pipeline([('preprocessor', tree_preprocessor), ('model', best_lgbm)])
    pipe_cat = Pipeline([('preprocessor', tree_preprocessor), ('model', cat_clf)])
    pipe_mlp = Pipeline([('preprocessor', nn_preprocessor), ('model', mlp_clf)])
    
    estimators = [
        ('xgb', pipe_xgb),
        ('lgbm', pipe_lgbm),
        ('cat', pipe_cat),
        ('mlp', pipe_mlp)
    ]
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3,
        n_jobs=-1,
        passthrough=False
    )
    
    print("Evaluating Stacking Classifier with MLP on subset (20k)...")
    X_eval = X.sample(n=20000, random_state=42)
    y_eval = y.loc[X_eval.index]
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(stacking_clf, X_eval, y_eval, cv=skf, scoring='roc_auc', n_jobs=-1)
    print(f"Stacking AUC (with MLP): {scores.mean():.4f} (+/- {scores.std():.4f})")

if __name__ == "__main__":
    main()
