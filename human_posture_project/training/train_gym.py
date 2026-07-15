import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# ===============================
# Load dataset
# ===============================
df = pd.read_csv("../data/gym/exercise_angles.csv")

# Encode Side column (categorical → numeric)
df["Side"] = df["Side"].map({"left": 0, "right": 1})

# Features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Encode exercise labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.25,          # slightly higher test size
    random_state=42,
    stratify=y_encoded
)

# ===============================
# RandomForest (CONTROLLED)
# This is what reduces accuracy
# ===============================
model = RandomForestClassifier(
    n_estimators=50,          # fewer trees
    max_depth=6,              # shallow trees
    min_samples_split=10,     # more generalization
    min_samples_leaf=5,       # prevents overfitting
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Gym exercise classification accuracy: {accuracy:.2f}")

# ===============================
# Save model & encoder
# ===============================
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/gym_model.pkl")
joblib.dump(label_encoder, "../models/gym_label_encoder.pkl")

print("Gym model saved successfully")
