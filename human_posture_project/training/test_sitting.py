import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../data/sitting/data.csv")

# =========================
# LABELS
# =========================
y = df["upperbody_label"]
subjects = df["subject"]

# =========================
# FEATURES (SAME AS TRAINING!)
# =========================
upper_body_keywords = [
    "nose", "eye", "ear",
    "shoulder", "elbow", "wrist",
    "hip"
]

feature_cols = [
    col for col in df.columns
    if any(k in col for k in upper_body_keywords)
]
X = df[feature_cols]

# =========================
# SUBJECT-WISE SPLIT
# =========================
unique_subjects = subjects.unique()
split_idx = int(0.7 * len(unique_subjects))

test_subjects = unique_subjects[split_idx:]
test_mask = subjects.isin(test_subjects)

X_test = X[test_mask]
y_test = y[test_mask]

# =========================
# LOAD MODEL
# =========================
model = joblib.load("../models/sitting_model.pkl")

# =========================
# TEST MODEL
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Sitting posture TEST accuracy: {accuracy:.2f}")
