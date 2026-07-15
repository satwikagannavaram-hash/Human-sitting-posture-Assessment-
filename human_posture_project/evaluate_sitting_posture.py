import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/sitting/data.csv")

# =========================
# TARGET
# =========================
y = df["upperbody_label"]
subjects = df["subject"]

# =========================
# USE ONLY UPPER BODY FEATURES
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
train_subjects = unique_subjects[:split_idx]
test_subjects = unique_subjects[split_idx:]

train_mask = subjects.isin(train_subjects)
test_mask = subjects.isin(test_subjects)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# =========================
# RANDOM FOREST MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=8,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n===== SITTING POSTURE MODEL METRICS =====")
print(f"Accuracy  : {accuracy*100:.1f}%")
print(f"Precision : {precision*100:.1f}%")
print(f"Recall    : {recall*100:.1f}%")
print(f"F1 Score  : {f1*100:.1f}%")

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "models/sitting_model.pkl")
print("\nModel saved successfully")