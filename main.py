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

    #https://stackoverflow.com/questions/21483999/using-atan2-to-find-angle-between-two-vectors
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

                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

                foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

                ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]

                thumb = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y]

                # Convert normalized coordinates to actual pixel values
                shoulder = np.multiply(shoulder, [width, height]).astype(int)
                elbow = np.multiply(elbow, [width, height]).astype(int)
                wrist = np.multiply(wrist, [width, height]).astype(int)

                hip = np.multiply(hip, [width, height]).astype(int)
                knee = np.multiply(knee, [width, height]).astype(int)
                ankle = np.multiply(ankle, [width, height]).astype(int)
                heel = np.multiply(heel, [width, height]).astype(int)
                foot_index = np.multiply(foot_index, [width, height]).astype(int)
                ear = np.multiply(ear, [width, height]).astype(int)
                thumb = np.multiply(thumb, [width, height]).astype(int)

                # Draw lines between shoulder, elbow, and wrist
                cv2.line(image, tuple(shoulder), tuple(elbow), (0, 255, 0), 2)
                cv2.line(image, tuple(elbow), tuple(wrist), (0, 255, 0), 2)

                cv2.line(image, tuple(shoulder), tuple(hip), (0, 255, 0), 2)
                cv2.line(image, tuple(hip), tuple(knee), (0, 255, 0), 2)
                cv2.line(image, tuple(knee), tuple(ankle), (0, 255, 0), 2)

                cv2.line(image, tuple(ankle), tuple(heel), (0, 255, 0), 2)
                cv2.line(image, tuple(heel), tuple(foot_index), (0, 255, 0), 2)

                cv2.line(image, tuple(shoulder), tuple(ear), (0, 255, 0), 2)
                cv2.line(image, tuple(ear), tuple(thumb), (0, 255, 0), 2)


                # Calculate the angle at the elbow
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                hip_angle = calculate_angle(shoulder, hip, knee)
                knee_angle = calculate_angle(hip, knee, ankle)
                ankle_angle = calculate_angle(knee, ankle, foot_index)
                ear_angle = calculate_angle(shoulder, ear, thumb)

                # Display the angle near the elbow
                cv2.putText(image, str(int(elbow_angle)),
                            tuple(elbow), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(int(hip_angle)),
                            tuple(hip), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(int(knee_angle)),
                            tuple(knee), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(int(ankle_angle)),
                            tuple(ankle), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(int(ear_angle)),
                            tuple(ear), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Right Side
                # Get coordinates for the right arm (right shoulder, elbow, wrist)
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]

                right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

                right_thumb = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y]

                # Convert normalized coordinates to pixel values
                right_shoulder = np.multiply(right_shoulder, [width, height]).astype(int)
                right_elbow = np.multiply(right_elbow, [width, height]).astype(int)
                right_wrist = np.multiply(right_wrist, [width, height]).astype(int)

                right_hip = np.multiply(right_hip, [width, height]).astype(int)
                right_knee = np.multiply(right_knee, [width, height]).astype(int)
                right_ankle = np.multiply(right_ankle, [width, height]).astype(int)
                right_heel = np.multiply(right_heel, [width, height]).astype(int)
                right_foot_index = np.multiply(right_foot_index, [width, height]).astype(int)
                right_ear = np.multiply(right_ear, [width, height]).astype(int)
                right_thumb = np.multiply(right_thumb, [width, height]).astype(int)

                # Draw lines between right shoulder, elbow, and wrist
                cv2.line(image, tuple(right_shoulder), tuple(right_elbow), (0, 255, 0), 2)
                cv2.line(image, tuple(right_elbow), tuple(right_wrist), (0, 255, 0), 2)
                cv2.line(image, tuple(right_shoulder), tuple(right_hip), (0, 255, 0), 2)
                cv2.line(image, tuple(right_hip), tuple(right_knee), (0, 255, 0), 2)
                cv2.line(image, tuple(right_knee), tuple(right_ankle), (0, 255, 0), 2)
                cv2.line(image, tuple(right_ankle), tuple(right_heel), (0, 255, 0), 2)
                cv2.line(image, tuple(right_heel), tuple(right_foot_index), (0, 255, 0), 2)
                cv2.line(image, tuple(right_shoulder), tuple(right_ear), (0, 255, 0), 2)
                cv2.line(image, tuple(right_ear), tuple(right_thumb), (0, 255, 0), 2)

                # Calculate the right elbow angle
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot_index)
                right_ear_angle = calculate_angle(right_shoulder, right_ear, right_thumb)

                # Display the right elbow angle on the video
                cv2.putText(image, str(int(right_elbow_angle)),
                            tuple(right_elbow), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(int(right_hip_angle)),
                            tuple(right_hip), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(int(right_knee_angle)),
                            tuple(right_knee), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(int(right_ankle_angle)),
                            tuple(right_ankle), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(int(right_ear_angle)),
                            tuple(right_ear), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


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
