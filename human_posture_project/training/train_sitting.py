import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================
# LOAD DATASET
# =========================
print("📥 Loading dataset...")

df = pd.read_csv("../data/sitting/data.csv")

print("Dataset shape:", df.shape)

# =========================
# TARGET + SUBJECT SPLIT
# =========================
y = df["upperbody_label"]
subjects = df["subject"]

# =========================
# SELECT ONLY UPPER BODY FEATURES
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

print("Total features used:", len(feature_cols))
print("Total subjects:", subjects.nunique())

# =========================
# SUBJECT-WISE TRAIN TEST SPLIT (70-30)
# =========================
unique_subjects = subjects.unique()

split_idx = int(0.7 * len(unique_subjects))

train_subjects = unique_subjects[:split_idx]
test_subjects = unique_subjects[split_idx:]

train_mask = subjects.isin(train_subjects)
test_mask = subjects.isin(test_subjects)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

# =========================
# MODEL
# =========================
print("\n🚀 Training model...")

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
# PREDICTIONS
# =========================
y_pred = model.predict(X_test)

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nMODEL PERFORMANCE")
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================

# Map your labels to simple names
label_map = {
    "TLB": "Backward Bend",
    "TLF": "Forward Bend",
    "TLL": "Left Bend",
    "TLR": "Right Bend",
    "TUP": "Straight"
}

y_test_named = y_test.map(label_map)
y_pred_named = pd.Series(y_pred).map(label_map)

labels = [
    "Backward Bend",
    "Forward Bend",
    "Left Bend",
    "Right Bend",
    "Straight"
]

cm = confusion_matrix(y_test_named, y_pred_named, labels=labels)

# =========================
# PLOT COLORFUL CONFUSION MATRIX
# =========================
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",   # blue gradient like your print
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix – Sitting Posture Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# =========================

# SAVE MODEL
# =========================
joblib.dump(model, "../models/sitting_model.pkl")
print("\n✅ Model saved successfully")