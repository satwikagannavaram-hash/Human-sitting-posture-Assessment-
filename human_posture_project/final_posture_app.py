import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# =========================
# MediaPipe setup
# =========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# =========================
# Helper functions
# =========================
def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

# =========================
# Temporal smoothing buffer
# =========================
score_buffer = deque(maxlen=12)

# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    posture = "SITTING"
    raw_score = 100
    suggestions = []

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        nose = lm[mp_pose.PoseLandmark.NOSE]
        lsh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rsh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lear = lm[mp_pose.PoseLandmark.LEFT_EAR]
        rear = lm[mp_pose.PoseLandmark.RIGHT_EAR]
        lhip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        rhip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

        # =========================
        # 1️⃣ HEAD FORWARD CHECK (STRICT)
        # =========================
        head_x = (lear.x + rear.x) / 2
        shoulder_x = (lsh.x + rsh.x) / 2
        head_forward = abs(head_x - shoulder_x)

        if head_forward > 0.02:
            raw_score -= 25
            suggestions.append("Keep your head aligned with shoulders")

        # =========================
        # 2️⃣ SLOUCH CHECK (KEY FIX)
        # =========================
        shoulder_y = (lsh.y + rsh.y) / 2
        hip_y = (lhip.y + rhip.y) / 2
        vertical_ratio = abs(shoulder_y - hip_y)

        if vertical_ratio < 0.18:
            raw_score -= 35
            suggestions.append("Sit upright, avoid slouching")

        # =========================
        # 3️⃣ SHOULDER SYMMETRY (STRICT)
        # =========================
        if abs(lsh.y - rsh.y) > 0.025:
            raw_score -= 20
            suggestions.append("Level your shoulders")

        # =========================
        # 4️⃣ NECK ANGLE (CRITICAL)
        # =========================
        neck_angle = angle(
            [nose.x, nose.y],
            [(lsh.x + rsh.x) / 2, (lsh.y + rsh.y) / 2],
            [(lhip.x + rhip.x) / 2, (lhip.y + rhip.y) / 2]
        )

        if neck_angle < 160:
            raw_score -= 30
            suggestions.append("Do not bend forward or backward")

        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    else:
        raw_score = 0
        suggestions = ["No person detected"]

    # =========================
    # Temporal smoothing
    # =========================
    raw_score = max(0, min(100, raw_score))
    score_buffer.append(raw_score)
    score = int(np.mean(score_buffer))

    # =========================
    # FINAL STATUS (VERY STRICT)
    # =========================
    if score >= 85 and len(suggestions) == 0:
        status = "CORRECT"
        color = (0, 255, 0)
    elif score >= 65:
        status = "AVERAGE"
        color = (0, 215, 255)
    else:
        status = "INCORRECT"
        color = (0, 0, 255)

    # =========================
    # UI
    # =========================
    cv2.rectangle(frame, (0, 0), (w, 190), color, -1)
    cv2.putText(frame, f"POSTURE: {posture}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"STATUS: {status}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"SCORE: {score}/100", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    y = 160
    for s in suggestions[:3]:
        cv2.putText(frame, f"- {s}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25

    cv2.imshow("Sitting Posture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

