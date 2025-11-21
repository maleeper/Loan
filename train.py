import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Configuration
DATA_DIR = 'c:\\vs-code-projects\\Loan\\data'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
SUBMISSION_PATH = os.path.join(DATA_DIR, 'submission.csv')
N_FOLDS = 5
SEED = 42

def load_data():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

from sklearn.decomposition import PCA

# ... (imports)

def preprocess_data(train_df, test_df):
    print("Preprocessing data...")
    
    # Combine for consistent encoding
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # 1. Handle Grade/Subgrade
    subgrades = sorted(combined['grade_subgrade'].unique())
    subgrade_map = {val: i for i, val in enumerate(subgrades)}
    combined['grade_subgrade_encoded'] = combined['grade_subgrade'].map(subgrade_map)
    
    # 2. Log Transformation for skewed features
    skewed_features = ['annual_income', 'loan_amount', 'debt_to_income_ratio']
    for col in skewed_features:
        combined[f'log_{col}'] = np.log1p(combined[col])
        
    # 3. Feature Interactions
    # loan_to_income: Ratio of loan amount to annual income
    combined['loan_to_income'] = combined['loan_amount'] / (combined['annual_income'] + 1)
    # interest_burden: Estimate of total interest to be paid (simplified)
    combined['interest_burden'] = combined['loan_amount'] * combined['interest_rate']
    
    # 4. PCA for Dimensionality Reduction / Noise Removal
    # Select numerical columns for PCA
    numeric_cols = ['annual_income', 'debt_to_income_ratio', 'loan_amount', 'interest_rate', 
                    'log_annual_income', 'log_loan_amount', 'log_debt_to_income_ratio',
                    'loan_to_income', 'interest_burden']
    
    # Handle potential NaNs/Infs created by interactions (though +1 prevents div by zero usually)
    combined[numeric_cols] = combined[numeric_cols].fillna(0)
    combined[numeric_cols] = combined[numeric_cols].replace([np.inf, -np.inf], 0)

    # Standardize before PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined[numeric_cols])
    
    # Apply PCA, keeping 95% of variance
    pca = PCA(n_components=0.95, random_state=SEED)
    pca_data = pca.fit_transform(scaled_data)
    
    # Create PCA columns
    pca_cols = [f'pca_{i}' for i in range(pca_data.shape[1])]
    pca_df = pd.DataFrame(pca_data, columns=pca_cols)
    
    combined = pd.concat([combined, pca_df], axis=1)
    print(f"PCA reduced features to {len(pca_cols)} components explaining 95% variance.")

    # 5. Categorical Encoding
    categorical_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']
    
    for col in categorical_cols:
        le = LabelEncoder()
        combined[col] = combined[col].astype(str)
        combined[f'{col}_encoded'] = le.fit_transform(combined[col])
        
    # Features to use
    # We keep original numeric features + PCA features + encoded categoricals
    drop_cols = ['id', 'loan_paid_back', 'is_train', 'grade_subgrade'] + categorical_cols
    features = [c for c in combined.columns if c not in drop_cols]
    
    print(f"Features used: {features}")
    
    # Split back
    train_processed = combined[combined['is_train'] == 1].copy()
    test_processed = combined[combined['is_train'] == 0].copy()
    
    # Restore target
    train_processed['loan_paid_back'] = train_df['loan_paid_back']
    
    return train_processed, test_processed, features

def train_models(train_df, test_df, features):
    X = train_df[features]
    y = train_df['loan_paid_back']
    X_test = test_df[features]
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_preds_xgb = np.zeros(len(X))
    oof_preds_lgb = np.zeros(len(X))
    oof_preds_cat = np.zeros(len(X))
    
    test_preds_xgb = np.zeros(len(X_test))
    test_preds_lgb = np.zeros(len(X_test))
    test_preds_cat = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{N_FOLDS}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric='auc'
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
        oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        test_preds_xgb += xgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
        
        # LightGBM
        print("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=SEED,
            n_jobs=-1
        )
        # LightGBM early stopping is handled differently in newer versions or via callbacks
        # We'll use the standard fit with eval_set
        lgb_model.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        test_preds_lgb += lgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
        
        # CatBoost
        print("Training CatBoost...")
        cat_model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            eval_metric='AUC',
            random_seed=SEED,
            verbose=100,
            early_stopping_rounds=50
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        oof_preds_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        test_preds_cat += cat_model.predict_proba(X_test)[:, 1] / N_FOLDS
        
    # Evaluation
    auc_xgb = roc_auc_score(y, oof_preds_xgb)
    auc_lgb = roc_auc_score(y, oof_preds_lgb)
    auc_cat = roc_auc_score(y, oof_preds_cat)
    
    print(f"\nXGBoost AUC: {auc_xgb:.5f}")
    print(f"LightGBM AUC: {auc_lgb:.5f}")
    print(f"CatBoost AUC: {auc_cat:.5f}")
    
    # Ensemble (Simple Average)
    oof_ensemble = (oof_preds_xgb + oof_preds_lgb + oof_preds_cat) / 3
    auc_ensemble = roc_auc_score(y, oof_ensemble)
    print(f"Ensemble AUC: {auc_ensemble:.5f}")
    
    test_preds_ensemble = (test_preds_xgb + test_preds_lgb + test_preds_cat) / 3
    return test_preds_ensemble

def create_submission(test_df, preds):
    submission = pd.DataFrame({
        'id': test_df['id'],
        'loan_paid_back': preds
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

if __name__ == "__main__":
    train_df, test_df = load_data()
    train_processed, test_processed, features = preprocess_data(train_df, test_df)
    preds = train_models(train_processed, test_processed, features)
    create_submission(test_df, preds)
