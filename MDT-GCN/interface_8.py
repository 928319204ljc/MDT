import cv2
import torch
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, Label, Frame, Scrollbar, Canvas
from tkinter import ttk
from PIL import Image, ImageTk
from model.agcn_diff_combine_score_fagg import Model
from graph.EGait import Graph
import threading
import time
import os


# 加载模型和权重
def load_model(weights_path, device='gpu'):
    model = Model(num_class=4, num_point=16, num_constraints=31, graph="graph.EGait.Graph", graph_args=dict())
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.train()
    return model


# 用于模型输入的骨骼数据提取
def extract_features_from_video(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)

    joint_features = []
    movement_features = []
    prev_joints = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            joints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark[:16]], dtype=np.float32).T
            joint_features.append(joints)

            if prev_joints is not None:
                velocity = joints - prev_joints
                acceleration = velocity - (prev_joints - joints if prev_joints is not None else 0)
                angle_velocity = np.arctan2(velocity[1], velocity[0])
                angle_acceleration = np.arctan2(acceleration[1], acceleration[0])
                movement_feature = np.concatenate(
                    (velocity, acceleration, angle_velocity[None], angle_acceleration[None]), axis=0)
                movement_features.append(movement_feature)
            prev_joints = joints

    cap.release()

    joint_features = np.array(joint_features)
    if joint_features.shape[0] < 48:
        pad_size = 48 - joint_features.shape[0]
        joint_features = np.pad(joint_features, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
    joint_features = joint_features[:48].transpose(1, 0, 2)[np.newaxis, :, :, :, np.newaxis]

    movement_features = np.array(movement_features)
    if movement_features.shape[0] < 48:
        pad_size = 48 - movement_features.shape[0]
        movement_features = np.pad(movement_features, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
    movement_features = movement_features[:48].transpose(1, 0, 2)[np.newaxis, :, :, :, np.newaxis]

    return torch.tensor(joint_features).float(), torch.tensor(movement_features).float()


# 骨骼节点可视化配置
VISUAL_KEYPOINTS = {
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


# 绘制骨骼结构函数
def draw_custom_skeleton(frame, landmarks, width, height):
    stomach_offset = -15

    for key, idx in VISUAL_KEYPOINTS.items():
        if isinstance(idx, int) and idx < len(landmarks):
            landmark = landmarks[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    if "abdomen" in VISUAL_KEYPOINTS:
        idx1, idx2 = VISUAL_KEYPOINTS["abdomen"]
        if idx1 < len(landmarks) and idx2 < len(landmarks):
            abdomen_x = int((landmarks[idx1].x + landmarks[idx2].x) / 2 * width)
            abdomen_y = int((landmarks[idx1].y + landmarks[idx2].y) / 2 * height)
            cv2.circle(frame, (abdomen_x, abdomen_y), 5, (0, 255, 0), -1)

    if "stomach" in VISUAL_KEYPOINTS:
        idx1, idx2 = VISUAL_KEYPOINTS["stomach"]
        if idx1 < len(landmarks) and idx2 < len(landmarks):
            stomach_x = int((landmarks[idx1].x + landmarks[idx2].x) / 2 * width)
            stomach_y = int((landmarks[idx1].y + landmarks[idx2].y) / 2 * height) + stomach_offset
            cv2.circle(frame, (stomach_x, stomach_y), 5, (0, 255, 0), -1)
            cv2.line(frame, (abdomen_x, abdomen_y), (stomach_x, stomach_y), (0, 255, 255), 2)

    if "chest" in VISUAL_KEYPOINTS:
        idx1, idx2 = VISUAL_KEYPOINTS["chest"]
        if idx1 < len(landmarks) and idx2 < len(landmarks):
            chest_x = int((landmarks[idx1].x + landmarks[idx2].x) / 2 * width)
            chest_y = int((landmarks[idx1].y + landmarks[idx2].y) / 2 * height)
            cv2.circle(frame, (chest_x, chest_y), 5, (0, 255, 0), -1)

    connections = [
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

        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)

    return frame


# 骨架帧提取函数
def extract_skeleton_frame(frame, pose, width, height):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        frame = draw_custom_skeleton(frame, landmarks, width, height)
    return frame


# 实时情绪识别界面
class RealTimeEmotionRecognitionApp:
    def __init__(self, root, model, device):
        self.root = root
        self.root.title("实时步态情绪识别")
        self.root.geometry("480x360")
        self.root.configure(bg="white")

        # Add scrollbar
        self.canvas = Canvas(root, bg="white")
        self.scrollbar = Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas, bg="white")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="n")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.model = model
        self.device = device

        self.video_label = Label(self.scrollable_frame, bg="white")
        self.video_label.pack(expand=True, pady=10)

        self.running = False
        self.capture = None
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

        self.start_button = ttk.Button(self.scrollable_frame, text="开始实时检测", command=self.start_recognition)
        self.start_button.pack(pady=10)
        self.stop_button = ttk.Button(self.scrollable_frame, text="停止检测", command=self.stop_recognition)
        self.stop_button.pack(pady=10)

        self.result_label = ttk.Label(self.scrollable_frame, text="", style="TLabel")
        self.result_label.pack(pady=10)

    def start_recognition(self):
        if not self.running:
            self.running = True
            self.capture = cv2.VideoCapture(0)
            threading.Thread(target=self.update_frame, daemon=True).start()

    def stop_recognition(self):
        if self.running:
            self.running = False
            if self.capture is not None:
                self.capture.release()
            self.video_label.config(image='')

    def update_frame(self):
        frame_buffer = []
        frame_count = 0
        while self.running and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            if results.pose_landmarks:
                frame = draw_custom_skeleton(frame, results.pose_landmarks.landmark, frame.shape[1], frame.shape[0])
                frame_buffer.append(frame_rgb)

                # 每60帧保存一次视频片段并进行情绪识别
                if len(frame_buffer) == 60:
                    video_path = self.save_temp_video(frame_buffer)
                    joint_features, movement_features = extract_features_from_video(video_path)
                    emotion = self.recognize_emotion(joint_features, movement_features)
                    self.result_label.config(text=emotion)
                    frame_buffer = []
                    os.remove(video_path)

            frame_count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        if self.capture is not None:
            self.capture.release()

    def save_temp_video(self, frames):
        height, width, layers = frames[0].shape
        video_path = "temp_video.avi"
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        return video_path

    def recognize_emotion(self, joint_features, movement_features):
        emotions = ['Happy', 'Sad', 'Angry', 'Neutral']
        joint_features = joint_features.to(self.device)
        movement_features = movement_features.to(self.device)

        with torch.no_grad():
            output_p, _, output_m = self.model(joint_features, movement_features)
            output = (output_p + output_m) / 2
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
            result_text = "检测结果:\n" + "\n".join(
                [f"{emotion}: {prob * 100:.2f}%" for emotion, prob in zip(emotions, probabilities)])
            return result_text


# GUI 界面
class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("步态情绪识别")
        self.root.geometry("1000x500")
        self.root.configure(bg="white")

        # Add scrollbar
        self.canvas = Canvas(root, bg="white")
        self.scrollbar = Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas, bg="white")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="n")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        main_frame = Frame(self.scrollable_frame, bg="white")
        main_frame.pack(expand=True)

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=10, background="white", foreground="black")
        style.configure("TLabel", font=("Arial", 14), background="white", foreground="black")

        self.input_video_label = ttk.Label(main_frame, text="输入视频", style="TLabel", background="white")
        self.input_video_label.grid(row=0, column=0, padx=10, pady=10)

        self.skeleton_video_label = ttk.Label(main_frame, text="骨架视频", style="TLabel", background="white")
        self.skeleton_video_label.grid(row=0, column=1, padx=10, pady=10)

        self.select_button = ttk.Button(main_frame, text="选择视频文件", command=self.select_video)
        self.select_button.grid(row=1, column=0, columnspan=2, pady=20)

        self.result_label = ttk.Label(main_frame, text="", style="TLabel")
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        weights_path = r"C:\Users\ljc\OneDrive\桌面\毕业论文准备\ewalk_best_model.pth"
        self.model = load_model(weights_path, self.device)

    def select_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if video_path:
            self.result_label.config(text="正在检测，请稍候...")
            self.root.update()

            self.show_videos(video_path)
            emotion = recognize_emotion_in_video(video_path, self.model, self.device)
            self.result_label.config(text=f"检测结果: {emotion}")

    def show_videos(self, video_path):
        cap = cv2.VideoCapture(video_path)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        display_height = 400
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        display_width = int(width * (display_height / height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (display_width, display_height))

            input_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_imgtk = ImageTk.PhotoImage(image=input_img)
            self.input_video_label.imgtk = input_imgtk
            self.input_video_label.config(image=input_imgtk)

            skeleton_frame = frame.copy()
            skeleton_frame = extract_skeleton_frame(skeleton_frame, pose, display_width, display_height)
            skeleton_img = Image.fromarray(cv2.cvtColor(skeleton_frame, cv2.COLOR_BGR2RGB))
            skeleton_imgtk = ImageTk.PhotoImage(image=skeleton_img)
            self.skeleton_video_label.imgtk = skeleton_imgtk
            self.skeleton_video_label.config(image=skeleton_imgtk)

            self.root.update()

        cap.release()


# 识别情绪
def recognize_emotion_in_video(video_path, model, device='gpu'):
    emotions = ['Happy', 'Sad', 'Angry', 'Neutral']
    joint_features, movement_features = extract_features_from_video(video_path)

    joint_features = joint_features.to(device)
    movement_features = movement_features.to(device)

    with torch.no_grad():
        output_p, _, output_m = model(joint_features, movement_features)
        output = (output_p + output_m) / 2
        probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        result_text = "检测结果:\n" + "\n".join(
            [f"{emotion}: {prob * 100:.2f}%" for emotion, prob in zip(emotions, probabilities)])
        return result_text



# 主程序运行
if __name__ == "__main__":
    root = tk.Tk()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights_path = r"C:\Users\ljc\OneDrive\桌面\毕业论文准备\ewalk_best_model.pth"
    model = load_model(weights_path, device)

    # 初始化实时检测应用程序
    real_time_app = RealTimeEmotionRecognitionApp(root, model, device)

    # 初始化视频检测应用程序
    app = EmotionRecognitionApp(root)

    root.mainloop()
