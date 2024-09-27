import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


# Load video
video_path = 'MovementAnalysisTestVideo.mp4'
cap = cv2.VideoCapture(video_path)

# Get video details
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter to save output video
output_video = cv2.VideoWriter('MovementAnalyzedVideoOutput.mp4',
                               cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Start pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make pose detection
        results = pose.process(image)

        # Convert image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks and angles on the frame
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get key points for the left arm (shoulder, elbow, wrist)
            try:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Convert normalized coordinates to actual pixel values
                shoulder = np.multiply(shoulder, [width, height]).astype(int)
                elbow = np.multiply(elbow, [width, height]).astype(int)
                wrist = np.multiply(wrist, [width, height]).astype(int)

                # Draw lines between shoulder, elbow, and wrist
                cv2.line(image, tuple(shoulder), tuple(elbow), (0, 255, 0), 2)
                cv2.line(image, tuple(elbow), tuple(wrist), (0, 255, 0), 2)

                # Calculate the angle at the elbow
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                # Display the angle near the elbow
                cv2.putText(image, str(int(elbow_angle)),
                            tuple(elbow), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # You can similarly calculate and draw angles for other joints (like knees, etc.)

            except:
                pass

            # Draw landmarks and pose connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Write the frame with drawings to the output video
        output_video.write(image)

        # Display the frame with annotations (optional)
        cv2.imshow('Video with Angles', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
output_video.release()
cv2.destroyAllWindows()
