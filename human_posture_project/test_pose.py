import cv2
import mediapipe as mp
from utils.feature_extraction import extract_angles

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            angles = extract_angles(results.pose_landmarks.landmark)

            print(
                f"Left Elbow: {angles['left_elbow']:.1f}, "
                f"Left Knee: {angles['left_knee']:.1f}, "
                f"Neck: {angles['neck']:.1f}"
            )

        cv2.imshow("Pose + Angles", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
