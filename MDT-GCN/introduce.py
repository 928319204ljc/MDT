import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Skeleton visualization configuration
VISUAL_KEYPOINTS = {
    "head": 0,
    "neck": 2,
    "shoulder_left": 11,
    "shoulder_right": 12,
    "elbow_left": 13,
    "elbow_right": 14,
    "hand_left": 15,
    "hand_right": 16,
    "hip_left": 23,
    "hip_right": 24,
    "knee_left": 25,
    "knee_right": 26,
    "foot_left": 27,
    "foot_right": 28,
    "abdomen": (23, 24),
    "stomach": (23, 24),
    "chest": (11, 12)
}

# Draw custom skeleton
def draw_custom_skeleton(frame, landmarks, width, height):
    stomach_offset = -15
    frame[:] = (192, 192, 192)  # Set background to gray

    for key, idx in VISUAL_KEYPOINTS.items():
        if isinstance(idx, int) and idx < len(landmarks):
            landmark = landmarks[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 15, (0, 255, 0), -1)  # Increased node size by 3 times

    if "abdomen" in VISUAL_KEYPOINTS:
        idx1, idx2 = VISUAL_KEYPOINTS["abdomen"]
        if idx1 < len(landmarks) and idx2 < len(landmarks):
            abdomen_x = int((landmarks[idx1].x + landmarks[idx2].x) / 2 * width)
            abdomen_y = int((landmarks[idx1].y + landmarks[idx2].y) / 2 * height)
            cv2.circle(frame, (abdomen_x, abdomen_y), 15, (0, 255, 0), -1)  # Increased node size by 3 times

    if "stomach" in VISUAL_KEYPOINTS:
        idx1, idx2 = VISUAL_KEYPOINTS["stomach"]
        if idx1 < len(landmarks) and idx2 < len(landmarks):
            stomach_x = int((landmarks[idx1].x + landmarks[idx2].x) / 2 * width)
            stomach_y = int((landmarks[idx1].y + landmarks[idx2].y) / 2 * height) + stomach_offset
            cv2.circle(frame, (stomach_x, stomach_y), 15, (0, 255, 0), -1)  # Increased node size by 3 times
            cv2.line(frame, (abdomen_x, abdomen_y), (stomach_x, stomach_y), (0, 255, 255), 12)  # Increased line thickness by 3 times

    if "chest" in VISUAL_KEYPOINTS:
        idx1, idx2 = VISUAL_KEYPOINTS["chest"]
        if idx1 < len(landmarks) and idx2 < len(landmarks):
            chest_x = int((landmarks[idx1].x + landmarks[idx2].x) / 2 * width)
            chest_y = int((landmarks[idx1].y + landmarks[idx2].y) / 2 * height)
            cv2.circle(frame, (chest_x, chest_y), 15, (0, 255, 0), -1)  # Increased node size by 3 times

    connections = [
        (VISUAL_KEYPOINTS["head"], VISUAL_KEYPOINTS["neck"]),
        (VISUAL_KEYPOINTS["neck"], VISUAL_KEYPOINTS["chest"]),
        (VISUAL_KEYPOINTS["chest"], VISUAL_KEYPOINTS["abdomen"]),
        (VISUAL_KEYPOINTS["shoulder_left"], VISUAL_KEYPOINTS["elbow_left"]),
        (VISUAL_KEYPOINTS["elbow_left"], VISUAL_KEYPOINTS["hand_left"]),
        (VISUAL_KEYPOINTS["shoulder_right"], VISUAL_KEYPOINTS["elbow_right"]),
        (VISUAL_KEYPOINTS["elbow_right"], VISUAL_KEYPOINTS["hand_right"]),
        (VISUAL_KEYPOINTS["hip_left"], VISUAL_KEYPOINTS["knee_left"]),
        (VISUAL_KEYPOINTS["knee_left"], VISUAL_KEYPOINTS["foot_left"]),
        (VISUAL_KEYPOINTS["hip_right"], VISUAL_KEYPOINTS["knee_right"]),
        (VISUAL_KEYPOINTS["knee_right"], VISUAL_KEYPOINTS["foot_right"]),
        (VISUAL_KEYPOINTS["shoulder_left"], VISUAL_KEYPOINTS["shoulder_right"]),
        (VISUAL_KEYPOINTS["hip_left"], VISUAL_KEYPOINTS["hip_right"]),
    ]

    for start, end in connections:
        if isinstance(start, tuple):
            idx1, idx2 = start
            start_x = int((landmarks[idx1].x + landmarks[idx2].x) / 2 * width)
            start_y = int((landmarks[idx1].y + landmarks[idx2].y) / 2 * height)
        else:
            start_x, start_y = int(landmarks[start].x * width), int(landmarks[start].y * height)

        if isinstance(end, tuple):
            idx1, idx2 = end
            end_x = int((landmarks[idx1].x + landmarks[idx2].x) / 2 * width)
            end_y = int((landmarks[idx1].y + landmarks[idx2].y) / 2 * height)
        else:
            end_x, end_y = int(landmarks[end].x * width), int(landmarks[end].y * height)

        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 12)  # Increased line thickness by 3 times

    return frame

# Load the video file
video_path = r  # Replace with your MP4 file path
cap = cv2.VideoCapture(video_path)

frames = []
poses = []
frame_indices = [1, 5, 10, 15, 20, 25]  # Frames to extract
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count not in frame_indices:
        continue

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect pose
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        poses.append(result.pose_landmarks)
        frames.append(frame)

    if len(frames) == len(frame_indices):
        break

cap.release()

# Plot the frames and pose landmarks
fig, axs = plt.subplots(2, len(frames), figsize=(20, 10))

for i, (frame, landmarks) in enumerate(zip(frames, poses)):
    # Draw the original frame
    axs[0, i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axs[0, i].axis('off')
    axs[0, i].set_title(f'T={frame_indices[i]}')

    # Draw the custom pose landmarks
    height, width, _ = frame.shape
    frame_with_skeleton = draw_custom_skeleton(frame.copy(), landmarks.landmark, width, height)
    axs[1, i].imshow(cv2.cvtColor(frame_with_skeleton, cv2.COLOR_BGR2RGB))
    axs[1, i].axis('off')

plt.tight_layout()
plt.show()