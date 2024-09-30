import cv2
import mediapipe as mp
import numpy as np

# Constants
DANCER_HEIGHT_CM = 170  # Example dancer height in cm
PALM_SIZE_CM = 15  # 8 angulas ~ 15 cm
DISTANCE_THRESHOLD = 0.05  # Threshold to determine if wrist is on the hip (normalized units)

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False,
                    min_detection_confidence=0.5)

# Initialize drawing utils
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the 3D angle between three points
def calculate_3d_angle(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Function to calculate Euclidean distance between two landmarks
def calculate_distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

# Function to get height from pose landmarks (hip landmarks)
def get_hip_height(landmarks, frame_height):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    avg_hip_y = (left_hip.y + right_hip.y) / 2
    hip_height = avg_hip_y * frame_height
    return hip_height

# Function to check if wrist is placed on the hip
def is_wrist_on_hip(wrist, hip):
    distance = calculate_distance(wrist, hip)
    return distance < DISTANCE_THRESHOLD

# Function to check Aramandi position rules including height dip
def check_aramandi_rules(landmarks, frame_height, initial_hip_height):
    results = {}

    # Left and right wrist, elbow, shoulder, hip landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Rule 1: Wrist on hip or shoulder-to-elbow straight
    if is_wrist_on_hip(left_wrist, left_hip):
        # Check shoulder-to-shoulder alignment
        shoulder_alignment_angle = calculate_3d_angle(left_hip, left_shoulder, right_shoulder)
        results["Left Shoulder Alignment"] = shoulder_alignment_angle >= 170
    else:
        # Check if shoulder-to-elbow is straight (angle close to 180 degrees)
        left_elbow_angle = calculate_3d_angle(left_shoulder, left_elbow, left_wrist)
        results["Left Elbow Angle"] = left_elbow_angle >= 160  # Allow some flexibility

    if is_wrist_on_hip(right_wrist, right_hip):
        # Check shoulder-to-shoulder alignment
        shoulder_alignment_angle = calculate_3d_angle(right_hip, right_shoulder, left_shoulder)
        results["Right Shoulder Alignment"] = shoulder_alignment_angle >= 170
    else:
        # Check if shoulder-to-elbow is straight (angle close to 180 degrees)
        right_elbow_angle = calculate_3d_angle(right_shoulder, right_elbow, right_wrist)
        results["Right Elbow Angle"] = right_elbow_angle >= 160

    # Rule 2: Specific angle range for the elbow (e.g., between 30 and 90 degrees)
    left_elbow_angle_range = calculate_3d_angle(left_shoulder, left_elbow, left_wrist)
    results["Left Elbow Angle Range"] = 30 <= left_elbow_angle_range <= 90

    right_elbow_angle_range = calculate_3d_angle(right_shoulder, right_elbow, right_wrist)
    results["Right Elbow Angle Range"] = 30 <= right_elbow_angle_range <= 90

    # Rule 3: Spine straight (between hips and shoulders)
    spine_angle = calculate_3d_angle(left_hip, left_shoulder, right_hip)
    results["Spine Straightness"] = spine_angle >= 160

    # Rule 4: Both ankles should be rotated outward by 45 degrees
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    ankle_rotation_left = calculate_3d_angle(left_knee, left_ankle, left_ankle)
    ankle_rotation_right = calculate_3d_angle(right_knee, right_ankle, right_ankle)
    results["Ankle Rotation Left"] = ankle_rotation_left >= 45
    results["Ankle Rotation Right"] = ankle_rotation_right >= 45

    # Rule 5: Dip such that height decreases by palm size
    current_hip_height = get_hip_height(landmarks, frame_height)
    palm_size_pixels = (PALM_SIZE_CM / DANCER_HEIGHT_CM) * frame_height
    results["Height Dip"] = current_hip_height <= initial_hip_height - palm_size_pixels

    # Rule 6: Knees should project outward and form a straight line (angle between hips and knees)
    knee_alignment_angle = calculate_3d_angle(left_hip, left_knee, right_knee)
    results["Knee Alignment"] = knee_alignment_angle >= 170  # Assuming knees should form a near straight line

    return results

# Function to process video and check rules
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    initial_hip_height = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_world_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_world_landmarks.landmark

            frame_height = frame.shape[0]

            if initial_hip_height is None:
                initial_hip_height = get_hip_height(landmarks, frame_height)

            rule_results = check_aramandi_rules(landmarks, frame_height, initial_hip_height)

            # Display the rule results in top-left corner
            start_x, start_y = 20, 40  # Starting coordinates for text
            line_height = 30  # Vertical space between each line of text

            for i, (rule, passed) in enumerate(rule_results.items()):
                text = f"{rule}: {'Pass' if passed else 'Fail'}"
                position = (start_x, start_y + i * line_height)
                color = (0, 255, 0) if passed else (0, 0, 255)
                cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)

        # Display the frame
        cv2.imshow('Aramandi Position Analysis', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the analysis
process_video("../../../inputs/Aramandi.mp4")
