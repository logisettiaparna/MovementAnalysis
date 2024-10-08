import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False,
                    min_detection_confidence=0.5)

# Initialize drawing utils
mp_drawing = mp.solutions.drawing_utils


# Function to calculate the 3D angle between three points
def calculate_3d_angle(a, b, c):
    # Convert landmarks to numpy arrays
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    # Calculate the vectors
    ba = a - b
    bc = c - b

    # Calculate the cosine of the angle between the vectors
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Ensure the cosine value stays in the valid range to avoid floating-point errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate the angle in radians and then convert it to degrees
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


# Process a video
input_video_path = '../MovementAnalysisTestVideo.mp4'  # Input video path
output_video_path = '../DisplayFrame_Output_MovementAnalysisTestVideo.mp4'  # Output video path
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize VideoWriter to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize variables to track min and max angles-Future
# min_angles = {
#     "Left Elbow": float('inf'),
#     "Left Shoulder": float('inf'),
#     "Left Knee": float('inf'),
#     "Left Hip": float('inf'),
#     "Right Elbow": float('inf'),
#     "Right Shoulder": float('inf'),
#     "Right Knee": float('inf'),
#     "Right Hip": float('inf')
# }
#
# max_angles = {
#     "Left Elbow": float('-inf'),
#     "Left Shoulder": float('-inf'),
#     "Left Knee": float('-inf'),
#     "Left Hip": float('-inf'),
#     "Right Elbow": float('-inf'),
#     "Right Shoulder": float('-inf'),
#     "Right Knee": float('-inf'),
#     "Right Hip": float('-inf')
# }

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to RGB as MediaPipe expects RGB images
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to detect pose landmarks
    results = pose.process(image_rgb)

    # Draw pose landmarks on the frame
    if results.pose_world_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the world landmarks for 3D angle calculation
        landmarks = results.pose_world_landmarks.landmark

        # Get image size for scaling the landmarks to pixel coordinates
        height, width, _ = frame.shape

        # Define pairs of points to calculate angles for major joints

        # Left side
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

        # Right side
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Calculate angles using pose_world_landmarks in 3D space
        left_elbow_angle = calculate_3d_angle(left_shoulder, left_elbow, left_wrist)
        left_shoulder_angle = calculate_3d_angle(left_hip, left_shoulder, left_elbow)
        left_knee_angle = calculate_3d_angle(left_hip, left_knee, left_ankle)
        left_hip_angle = calculate_3d_angle(left_shoulder, left_hip, left_knee)

        right_elbow_angle = calculate_3d_angle(right_shoulder, right_elbow, right_wrist)
        right_shoulder_angle = calculate_3d_angle(right_hip, right_shoulder, right_elbow)
        right_knee_angle = calculate_3d_angle(right_hip, right_knee, right_ankle)
        right_hip_angle = calculate_3d_angle(right_shoulder, right_hip, right_knee)

        # Update min and max angles-Future
        # min_angles["Left Elbow"] = min(min_angles["Left Elbow"], left_elbow_angle)
        # min_angles["Left Shoulder"] = min(min_angles["Left Shoulder"], left_shoulder_angle)
        # min_angles["Left Knee"] = min(min_angles["Left Knee"], left_knee_angle)
        # min_angles["Left Hip"] = min(min_angles["Left Hip"], left_hip_angle)
        # min_angles["Right Elbow"] = min(min_angles["Right Elbow"], right_elbow_angle)
        # min_angles["Right Shoulder"] = min(min_angles["Right Shoulder"], right_shoulder_angle)
        # min_angles["Right Knee"] = min(min_angles["Right Knee"], right_knee_angle)
        # min_angles["Right Hip"] = min(min_angles["Right Hip"], right_hip_angle)
        #
        # max_angles["Left Elbow"] = max(max_angles["Left Elbow"], left_elbow_angle)
        # max_angles["Left Shoulder"] = max(max_angles["Left Shoulder"], left_shoulder_angle)
        # max_angles["Left Knee"] = max(max_angles["Left Knee"], left_knee_angle)
        # max_angles["Left Hip"] = max(max_angles["Left Hip"], left_hip_angle)
        # max_angles["Right Elbow"] = max(max_angles["Right Elbow"], right_elbow_angle)
        # max_angles["Right Shoulder"] = max(max_angles["Right Shoulder"], right_shoulder_angle)
        # max_angles["Right Knee"] = max(max_angles["Right Knee"], right_knee_angle)
        # max_angles["Right Hip"] = max(max_angles["Right Hip"], right_hip_angle)

        # Display angles in the top-left corner
        start_x, start_y = 20, 40  # Starting coordinates for the text
        line_height = 30  # Vertical space between each line of text

        # Create a list of the angles and their labels
        angles_text = [
            f"Left Elbow: {int(left_elbow_angle)}",
            f"Left Shoulder: {int(left_shoulder_angle)}",
            f"Left Knee: {int(left_knee_angle)}",
            f"Left Hip: {int(left_hip_angle)}",
            f"Right Elbow: {int(right_elbow_angle)}",
            f"Right Shoulder: {int(right_shoulder_angle)}",
            f"Right Knee: {int(right_knee_angle)}",
            f"Right Hip: {int(right_hip_angle)}"
        ]

        # Iterate through the angles and display each one in the top-left corner
        for i, text in enumerate(angles_text):
            position = (start_x, start_y + i * line_height)
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Write the processed frame to the output video
    out.write(frame)

    # Check if frame is valid before showing
    if frame.size > 0:
        cv2.imshow('Pose Estimation with Angles', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

## Process the last frame to display min and max angles
# if frame is not None:  # Check if the last frame is valid
#     for i, (angle_name, min_angle) in enumerate(min_angles.items()):
#         position = (frame_width - 250, frame_height - (40 + i * 30))  # Position for min angles
#         cv2.putText(frame, f"{angle_name} Min: {int(min_angle)}", position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
#
#     for i, (angle_name, max_angle) in enumerate(max_angles.items()):
#         position = (frame_width - 250, frame_height - (200 + i * 30))  # Position for max angles
#         cv2.putText(frame, f"{angle_name} Max: {int(max_angle)}", position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
#
#     # Write the last frame with min and max angles to the output video
#     out.write(frame)

# Add a delay to view the final frame
# cv2.imshow('Pose Estimation with Angles', frame)
# cv2.waitKey(5000)  # Wait for 5000 milliseconds (5 seconds)

# Release resources
cap.release()
out.release()  # Release the VideoWriter
cv2.destroyAllWindows()
