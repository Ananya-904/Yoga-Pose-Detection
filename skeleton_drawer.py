"""
Skeleton Drawing Module
Draws pose landmarks and connections on frames
"""

import cv2
import numpy as np



class SkeletonDrawer:
    def __init__(self):
        """Initialize skeleton drawer with color scheme"""
        # Color scheme (BGR format for OpenCV)
        self.colors = {
            'landmark': (0, 255, 0),      # Green
            'connection': (255, 0, 0),    # Blue
            'angle_text': (0, 255, 255),  # Yellow
            'feedback': (0, 165, 255),    # Orange
        }
        
        # Pose connections (landmark pairs to draw lines between)
        self.connections = [
            # Torso
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            # Left arm
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            # Right arm
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            # Left leg
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            # Right leg
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            # Spine
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
        ]
    
    def get_point(self, landmarks, key):
        """Extract point coordinates from landmarks"""
        if key in landmarks and landmarks[key]['visibility'] > 0.5:
            return (int(landmarks[key]['x']), int(landmarks[key]['y']))
        return None
    
    def draw_skeleton(self, frame, landmarks):
        """
        Draw skeleton connections on frame
        
        Args:
            frame: BGR image frame
            landmarks: Dictionary of landmark positions
            
        Returns:
            frame: Frame with skeleton drawn
        """
        # Draw connections
        for start_key, end_key in self.connections:
            start_point = self.get_point(landmarks, start_key)
            end_point = self.get_point(landmarks, end_key)
            
            if start_point and end_point:
                cv2.line(frame, start_point, end_point, 
                        self.colors['connection'], 2)
        
        # Draw landmarks
        for key, landmark in landmarks.items():
            if landmark['visibility'] > 0.5:
                x, y = int(landmark['x']), int(landmark['y'])
                cv2.circle(frame, (x, y), 5, self.colors['landmark'], -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)
        
        return frame
    
    def draw_angles(self, frame, angles, landmarks):
        """
        Draw angle values on frame near joints
        
        Args:
            frame: BGR image frame
            angles: Dictionary of calculated angles
            landmarks: Dictionary of landmark positions
            
        Returns:
            frame: Frame with angles drawn
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        
        # Draw knee angles
        if angles.get('knee'):
            knee_angles = angles['knee']
            if 'left' in knee_angles and knee_angles['left']:
                left_knee = self.get_point(landmarks, 'left_knee')
                if left_knee:
                    text = f"L: {knee_angles['left']:.0f}°"
                    cv2.putText(frame, text, 
                              (left_knee[0] + 10, left_knee[1]),
                              font, font_scale, self.colors['angle_text'], thickness)
            
            if 'right' in knee_angles and knee_angles['right']:
                right_knee = self.get_point(landmarks, 'right_knee')
                if right_knee:
                    text = f"R: {knee_angles['right']:.0f}°"
                    cv2.putText(frame, text,
                              (right_knee[0] + 10, right_knee[1]),
                              font, font_scale, self.colors['angle_text'], thickness)
        
        # Draw elbow angles
        if angles.get('elbow'):
            elbow_angles = angles['elbow']
            if 'left' in elbow_angles and elbow_angles['left']:
                left_elbow = self.get_point(landmarks, 'left_elbow')
                if left_elbow:
                    text = f"L: {elbow_angles['left']:.0f}°"
                    cv2.putText(frame, text,
                              (left_elbow[0] + 10, left_elbow[1]),
                              font, font_scale, self.colors['angle_text'], thickness)
            
            if 'right' in elbow_angles and elbow_angles['right']:
                right_elbow = self.get_point(landmarks, 'right_elbow')
                if right_elbow:
                    text = f"R: {elbow_angles['right']:.0f}°"
                    cv2.putText(frame, text,
                              (right_elbow[0] + 10, right_elbow[1]),
                              font, font_scale, self.colors['angle_text'], thickness)
        
        # Draw spine angle
        if angles.get('spine') and angles['spine'] is not None:
            left_hip = self.get_point(landmarks, 'left_hip')
            right_hip = self.get_point(landmarks, 'right_hip')
            if left_hip and right_hip:
                hip_mid = ((left_hip[0] + right_hip[0]) // 2,
                          (left_hip[1] + right_hip[1]) // 2)
                text = f"Spine: {angles['spine']:.0f}°"
                cv2.putText(frame, text,
                          (hip_mid[0] - 40, hip_mid[1] - 20),
                          font, font_scale, self.colors['angle_text'], thickness)
        
        return frame
    
    def draw_confidence(self, frame, confidence):
        """
        Draw confidence score on frame
        
        Args:
            frame: BGR image frame
            confidence: Confidence score (0-1)
            
        Returns:
            frame: Frame with confidence drawn
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Convert to percentage
        conf_percent = confidence * 100
        
        # Color based on confidence
        if conf_percent > 70:
            color = (0, 255, 0)  # Green
        elif conf_percent > 50:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        text = f"Confidence: {conf_percent:.1f}%"
        cv2.putText(frame, text, (10, 30),
                   font, font_scale, color, thickness)
        
        return frame
    
    def draw_pose_name(self, frame, pose_name):
        """
        Draw detected pose name on frame
        
        Args:
            frame: BGR image frame
            pose_name: Name of detected pose
            
        Returns:
            frame: Frame with pose name drawn
        """
        if not pose_name:
            return frame
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)
        
        text = f"Pose: {pose_name}"
        cv2.putText(frame, text, (10, 60),
                   font, font_scale, color, thickness)
        
        return frame

