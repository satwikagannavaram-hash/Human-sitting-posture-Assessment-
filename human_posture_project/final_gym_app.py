import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
from utils.gym_rules import check_gym_posture

# =====================
# LOAD MODEL + ENCODER
# =====================
model = joblib.load("models/gym_model.pkl")
encoder = joblib.load("models/gym_label_encoder.pkl")

FEATURE_ORDER = [
    "Side",
    "Shoulder_Angle", "Elbow_Angle", "Hip_Angle",
    "Knee_Angle", "Ankle_Angle",
    "Shoulder_Ground_Angle", "Elbow_Ground_Angle",
    "Hip_Ground_Angle", "Knee_Ground_Angle",
    "Ankle_Ground_Angle"
]

buffer = deque(maxlen=15)

# =====================
# MEDIAPIPE
# =====================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def angle(a, b, c):
    return abs(np.degrees(
        np.arctan2(c.y - b.y, c.x - b.x) -
        np.arctan2(a.y - b.y, a.x - b.x)
    ))

print("Running GYM Exercise Detection")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    label = "NO GYM EXERCISE"
    posture = "—"
    suggestions = []
    color = (0, 0, 255)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        # -------- FEATURES (MATCH TRAINING EXACTLY) --------
        side = 0  # front view assumed

        features = {
            "Side": side,
            "Shoulder_Angle": angle(lm[11], lm[13], lm[15]),
            "Elbow_Angle": angle(lm[13], lm[15], lm[17]),
            "Hip_Angle": angle(lm[11], lm[23], lm[25]),
            "Knee_Angle": angle(lm[23], lm[25], lm[27]),
            "Ankle_Angle": angle(lm[25], lm[27], lm[31]),
            "Shoulder_Ground_Angle": abs(lm[11].y - lm[12].y),
            "Elbow_Ground_Angle": abs(lm[13].y - lm[14].y),
            "Hip_Ground_Angle": abs(lm[23].y - lm[24].y),
            "Knee_Ground_Angle": abs(lm[25].y - lm[26].y),
            "Ankle_Ground_Angle": abs(lm[27].y - lm[28].y)
        }

        X = np.array([[features[k] for k in FEATURE_ORDER]])

        pred = model.predict(X)[0]
        buffer.append(pred)

        stable_pred = max(set(buffer), key=buffer.count)
        label = encoder.inverse_transform([stable_pred])[0]

        posture, suggestions = check_gym_posture(label, lm)

        color = (0, 255, 0) if posture == "CORRECT" else (0, 0, 255)
        draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # =====================
    # DISPLAY
    # =====================
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), color, -1)
    cv2.putText(frame, f"Exercise: {label}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Posture: {posture}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    y = 100
    for s in suggestions:
        cv2.putText(frame, f"- {s}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 20

    cv2.imshow("Gym Exercise Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
