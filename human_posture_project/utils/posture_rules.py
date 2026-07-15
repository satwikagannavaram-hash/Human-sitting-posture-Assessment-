import math
import mediapipe as mp

PoseLandmark = mp.solutions.pose.PoseLandmark


def angle(a, b, c):
    return abs(
        math.degrees(
            math.atan2(c.y - b.y, c.x - b.x) -
            math.atan2(a.y - b.y, a.x - b.x)
        )
    )


def detect_posture_type(lm):
    """
    VERY ROBUST sitting vs standing detection
    """

    lsh = lm[PoseLandmark.LEFT_SHOULDER]
    rsh = lm[PoseLandmark.RIGHT_SHOULDER]
    lhip = lm[PoseLandmark.LEFT_HIP]
    rhip = lm[PoseLandmark.RIGHT_HIP]

    shoulder_y = (lsh.y + rsh.y) / 2
    hip_y = (lhip.y + rhip.y) / 2

    vertical_gap = hip_y - shoulder_y

    # 🔑 THIS THRESHOLD IS THE KEY FIX
    if vertical_gap < 0.18:
        return "SITTING"
    else:
        return "STANDING"


def check_sitting_posture(lm):
    score = 100
    issues = []

    nose = lm[PoseLandmark.NOSE]
    lsh = lm[PoseLandmark.LEFT_SHOULDER]
    rsh = lm[PoseLandmark.RIGHT_SHOULDER]
    lhip = lm[PoseLandmark.LEFT_HIP]

    neck_angle = angle(nose, lsh, lhip)

    if neck_angle < 150:
        score -= 30
        issues.append("Keep your neck straight")

    if abs(lsh.y - rsh.y) > 0.04:
        score -= 25
        issues.append("Align your shoulders")

    if score >= 80:
        status = "CORRECT"
    elif score >= 60:
        status = "AVERAGE"
    else:
        status = "INCORRECT"

    return status, max(score, 0), issues


def check_standing_posture(lm):
    score = 100
    issues = []

    nose = lm[PoseLandmark.NOSE]
    lsh = lm[PoseLandmark.LEFT_SHOULDER]
    lhip = lm[PoseLandmark.LEFT_HIP]

    spine_angle = angle(nose, lsh, lhip)

    if spine_angle < 165:
        score -= 40
        issues.append("Stand straight, avoid slouching")

    if score >= 80:
        status = "CORRECT"
    elif score >= 60:
        status = "AVERAGE"
    else:
        status = "INCORRECT"

    return status, max(score, 0), issues
