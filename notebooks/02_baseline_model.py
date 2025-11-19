from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from src.data import load_raw_data, TARGET_COL
from src.features import infer_feature_types, make_preprocessor
from src.models import get_model

train, _, _ = load_raw_data()
X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
num_cols, cat_cols = infer_feature_types(train, TARGET_COL)
pre = make_preprocessor(num_cols, cat_cols)
model = get_model("lgbm")
clf = Pipeline([("preprocess", pre), ("model", model)])
clf.fit(X_train, y_train)
preds = clf.predict(X_val)
print("Accuracy:", accuracy_score(y_val, preds))
