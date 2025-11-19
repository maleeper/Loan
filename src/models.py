from typing import Literal
from sklearn.ensemble import RandomForestClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

ModelName = Literal["lgbm", "xgb", "rf", "catboost"]

def get_model(name: ModelName = "lgbm"):
    if name == "lgbm":
        if LGBMClassifier is None:
            raise ImportError("Install lightgbm")
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    if name == "xgb":
        if XGBClassifier is None:
            raise ImportError("Install xgboost")
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=6,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            n_jobs=-1,
            random_state=42,
        )
    if name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("Install catboost")
        return CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            verbose=False,
            random_state=42,
        )
    raise ValueError("Unknown model name")
