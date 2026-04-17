"""
MediaPipe Pose Detection Module
Handles pose detection and landmark extraction
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque


class PoseDetector:
    def __init__(self, smoothing_window=5, static_image_mode=False):
        """
        Initialize MediaPipe Pose detector

        Args:
            smoothing_window: Number of frames for moving average smoothing
            static_image_mode: Whether detector should treat each frame as static.
        """
        self.mp_pose = mp.solutions.pose
        # Use stream-friendly defaults; caller can opt into static mode for photos.
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Smoothing buffers for key landmarks
        self.smoothing_window = smoothing_window
        self.landmark_history = {
            'left_shoulder': deque(maxlen=smoothing_window),
            'right_shoulder': deque(maxlen=smoothing_window),
            'left_elbow': deque(maxlen=smoothing_window),
            'right_elbow': deque(maxlen=smoothing_window),
            'left_wrist': deque(maxlen=smoothing_window),
            'right_wrist': deque(maxlen=smoothing_window),
            'left_hip': deque(maxlen=smoothing_window),
            'right_hip': deque(maxlen=smoothing_window),
            'left_knee': deque(maxlen=smoothing_window),
            'right_knee': deque(maxlen=smoothing_window),
            'left_ankle': deque(maxlen=smoothing_window),
            'right_ankle': deque(maxlen=smoothing_window),
            'nose': deque(maxlen=smoothing_window),
        }
    
    def detect(self, frame):
        """
        Detect pose in frame and return landmarks
        
        Args:
            frame: BGR image frame
            
        Returns:
            landmarks: Dictionary of landmark positions (x, y, z, visibility)
            confidence: Detection confidence score
            mp_landmarks: Raw MediaPipe landmarks object
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        landmarks = {}
        confidence = 0.0
        
        if results.pose_landmarks:
            # Calculate average visibility as confidence
            visibilities = [lm.visibility for lm in results.pose_landmarks.landmark]
            confidence = np.mean(visibilities)
            
            # Extract and smooth landmarks
            mp_landmarks = results.pose_landmarks.landmark
            
            # Key landmark indices
            landmark_map = {
                'nose': 0,
                'left_shoulder': 11,
                'right_shoulder': 12,
                'left_elbow': 13,
                'right_elbow': 14,
                'left_wrist': 15,
                'right_wrist': 16,
                'left_hip': 23,
                'right_hip': 24,
                'left_knee': 25,
                'right_knee': 26,
                'left_ankle': 27,
                'right_ankle': 28,
            }
            
            # Get frame dimensions
            h, w = frame.shape[:2]
            
            # Extract and smooth landmarks
            for name, idx in landmark_map.items():
                lm = mp_landmarks[idx]
                x, y, z = lm.x * w, lm.y * h, lm.z * w
                visibility = lm.visibility
                
                # Add to history for smoothing
                self.landmark_history[name].append((x, y, z, visibility))
                
                # Calculate smoothed position
                if len(self.landmark_history[name]) > 0:
                    history = list(self.landmark_history[name])
                    avg_x = np.mean([p[0] for p in history])
                    avg_y = np.mean([p[1] for p in history])
                    avg_z = np.mean([p[2] for p in history])
                    avg_vis = np.mean([p[3] for p in history])
                    
                    landmarks[name] = {
                        'x': avg_x,
                        'y': avg_y,
                        'z': avg_z,
                        'visibility': avg_vis
                    }
                else:
                    landmarks[name] = {
                        'x': x,
                        'y': y,
                        'z': z,
                        'visibility': visibility
                    }
        
        return landmarks, confidence, results.pose_landmarks
    
    def draw_landmarks(self, frame, mp_landmarks):
        """
        Draw pose landmarks on frame
        
        Args:
            frame: BGR image frame
            mp_landmarks: MediaPipe landmarks object
            
        Returns:
            frame: Frame with drawn landmarks
        """
        if mp_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                mp_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        return frame
    
    def reset_smoothing(self):
        """Reset smoothing buffers"""
        for key in self.landmark_history:
            self.landmark_history[key].clear()

