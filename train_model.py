import pandas as pd, numpy as np, joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, recall_score,
                             precision_score, accuracy_score)

HORIZON = 24    # 24 readings x 5 min = 2 hours ahead
THRESH = 180    # high glucose limit (mg/dL)

ehr = pd.read_csv("ehr.csv")
cgm = pd.read_csv("cgm.csv", parse_dates=["time"])

def make_features(df):
    df = df.sort_values("time").copy()
    g = df["glucose"]
    df["g_lag3"] = g.shift(3)
    df["g_lag6"] = g.shift(6)
    df["g_lag12"] = g.shift(12)
    df["roc_15"] = g - df["g_lag3"]
    df["roc_30"] = g - df["g_lag6"]
    df["mean_1h"] = g.rolling(12).mean()
    df["std_1h"] = g.rolling(12).std()
    df["max_1h"] = g.rolling(12).max()
    hr = df["time"].dt.hour + df["time"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["future_max"] = g[::-1].rolling(HORIZON).max()[::-1].shift(-1)
    df = df.dropna()
    df["target"] = (df["future_max"] > THRESH).astype(int)
    return df.iloc[::6]

data = pd.concat([make_features(d) for _, d in cgm.groupby("patient_id")])
data = data.merge(ehr, on="patient_id")

FEATURES = ["glucose", "g_lag3", "g_lag6", "g_lag12", "roc_15", "roc_30",
            "mean_1h", "std_1h", "max_1h", "hour_sin", "hour_cos",
            "age", "bmi", "hba1c", "diabetes_years", "on_insulin"]

print("Rows:", len(data), "| High-glucose rate:", round(data["target"].mean(), 3))

split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(split.split(data, groups=data["patient_id"]))
X_tr, y_tr = data.iloc[tr][FEATURES], data.iloc[tr]["target"]
X_te, y_te = data.iloc[te][FEATURES], data.iloc[te]["target"]

model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                      eval_metric="logloss")
model.fit(X_tr, y_tr)

prob = model.predict_proba(X_te)[:, 1]
pred = (prob > 0.5).astype(int)
print("AUC      :", round(roc_auc_score(y_te, prob), 3))
print("Accuracy :", round(accuracy_score(y_te, pred), 3))
print("Recall   :", round(recall_score(y_te, pred), 3))
print("Precision:", round(precision_score(y_te, pred), 3))

print("\nTop features:")
imp = pd.Series(model.feature_importances_, index=FEATURES)
print(imp.sort_values(ascending=False).head(6).round(3))

joblib.dump({"model": model, "features": FEATURES}, "model.joblib")
print("\nSaved model.joblib")