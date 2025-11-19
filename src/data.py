import pathlib
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "loan_default"
SUBMISSION_TARGET_COL = "loan_default"

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return train, test, sample_sub

def train_val_split(train: pd.DataFrame, target_col: str = TARGET_COL,
                    test_size: float = 0.2, random_state: int = 42):
    X = train.drop(columns=[target_col])
    y = train[target_col]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
