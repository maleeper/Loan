from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def infer_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    features = df.drop(columns=[target_col])
    numeric_cols = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [c for c in features.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols

def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])
