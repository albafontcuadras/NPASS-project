import json, joblib, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Data
TRAIN = "spotify_songs_train2.csv"
TEST  = "spotify_songs_test2.csv"  

art = Path("artifacts"); art.mkdir(exist_ok=True)
target = "label"

train_df = pd.read_csv(TRAIN)
use_cols = train_df.drop(columns=[target]).select_dtypes(include=["number"]).columns.tolist()

# Model
X = train_df[use_cols].copy()
y = train_df[target].copy()

pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), use_cols)], verbose_feature_names_out=False)
model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
pipe = Pipeline([("preprocess", pre), ("model", model)])
pipe.fit(X, y)

joblib.dump(pipe, art / "model.joblib")

meta = {
  "model_name": "spotify_rf_v2",
  "created_at": datetime.utcnow().isoformat() + "Z",
  "framework": "scikit-learn",
  "sklearn_version": __import__("sklearn").__version__,
  "python_version": "{}.{}.{}".format(*__import__("sys").version_info[:3]),
  "target": target,
  "features": use_cols,
  "train_rows": int(len(train_df)),
}
try:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    test_df = pd.read_csv(TEST)
    y_true = test_df[target].values
    y_pred = pipe.predict(test_df[use_cols].copy())
    meta["metrics"] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro"))
    }
except Exception:
    pass

with open(art / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved artifacts to ./artifacts")
