import mediapipe as mp
import math

mp_pose = mp.solutions.pose

def angle(a, b, c):
    return abs(
        math.degrees(
            math.atan2(c.y - b.y, c.x - b.x)
            - math.atan2(a.y - b.y, a.x - b.x)
        )
    )

def detect_posture_type(landmarks):
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    knee_visible = knee.visibility > 0.6 and ankle.visibility > 0.6

    # ---------------------------
    # CASE 1: Knees visible → reliable
    # ---------------------------
    if knee_visible:
        knee_angle = angle(hip, knee, ankle)

        if knee_angle < 150:
            return "SITTING"
        else:
            return "STANDING"

    # ---------------------------
    # CASE 2: Knees NOT visible
    # ---------------------------
    torso_height = abs(shoulder.y - hip.y)
    head_tilt = abs(nose.x - shoulder.x)

    # Chair sitting → torso compressed + slight head tilt
    if torso_height < 0.22:
        return "SITTING"

    # Upright torso but legs missing → ambiguous
    return "UNKNOWN"
