import argparse
import pathlib
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from src.data import load_raw_data, TARGET_COL, SUBMISSION_TARGET_COL
from src.features import infer_feature_types, make_preprocessor
from src.models import get_model
from src.utils import seed_everything, timer

def cross_validate(clf, X, y, n_splits):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        with timer(f"fold {fold}"):
            clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)
        print(f"Fold {fold} accuracy: {acc:.4f}")
    print("Mean CV accuracy:", sum(scores) / len(scores))
    return scores

def main(model_name="lgbm", n_splits=5):
    seed_everything(42)
    train, test, sample_sub = load_raw_data()
    X = train.drop(columns=[TARGET_COL])
    y = train[TARGET_COL]
    num_cols, cat_cols = infer_feature_types(train, TARGET_COL)
    pre = make_preprocessor(num_cols, cat_cols)
    model = get_model(model_name)
    clf = Pipeline([("preprocess", pre), ("model", model)])
    with timer("cross-validation"):
        cross_validate(clf, X, y, n_splits)
    clf.fit(X, y)
    preds = clf.predict_proba(test)[:, 1]
    submission = sample_sub.copy()
    submission[SUBMISSION_TARGET_COL] = preds
    out = pathlib.Path("submissions") / f"sub_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    submission.to_csv(out, index=False)
    print("Saved:", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lgbm")
    parser.add_argument("--splits", default=5, type=int)
    args = parser.parse_args()
    main(args.model, args.splits)
