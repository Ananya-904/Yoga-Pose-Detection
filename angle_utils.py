"""
Joint Angle Calculation Utilities
Calculates angles for knee, elbow, shoulder, hip, and spine alignment
"""
# It does 4 main things:
# Extract key body points from MediaPipe landmarks
# Calculate angle between three points (main math logic)
# Calculate joint-specific angles (knee, elbow, shoulder, hip)
# Calculate spine alignment (body leaning left, right, forward)

import numpy as np
import math


def calculate_angle(point1, point2, point3):
    """
    Calculate angle between three points (point2 is the vertex)
    
    Args:
        point1: (x, y) coordinates of first point
        point2: (x, y) coordinates of vertex point
        point3: (x, y) coordinates of third point
        
    Returns:
        angle: Angle in degrees (0-180)
    """
    # Convert to numpy arrays
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    
    # Calculate vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


def get_point(landmarks, key):
    """
    Extract (x, y) coordinates from landmarks dictionary
    
    Args:
        landmarks: Dictionary of landmark positions
        key: Landmark key name
        
    Returns:
        (x, y) tuple or None if not found
         Purpose:
purpose:
Extract landmark coordinates safely.
Why needed?
MediaPipe may fail to detect some landmarks or visibility may be low.
    """
    if key in landmarks and landmarks[key]['visibility'] > 0.5:
        return (landmarks[key]['x'], landmarks[key]['y'])
    return None


def calculate_knee_angles(landmarks):
    """
    Calculate left and right knee angles
    
    Args:
        landmarks: Dictionary of landmark positions
        
    Returns:
        dict: {'left': angle, 'right': angle} or None if not detectable
    """
    angles = {}
    
    # Left knee: hip -> knee -> ankle
    left_hip = get_point(landmarks, 'left_hip')
    left_knee = get_point(landmarks, 'left_knee')
    left_ankle = get_point(landmarks, 'left_ankle')
    
    if left_hip and left_knee and left_ankle:
        angles['left'] = calculate_angle(left_hip, left_knee, left_ankle)
    
    # Right knee: hip -> knee -> ankle
    right_hip = get_point(landmarks, 'right_hip')
    right_knee = get_point(landmarks, 'right_knee')
    right_ankle = get_point(landmarks, 'right_ankle')
    
    if right_hip and right_knee and right_ankle:
        angles['right'] = calculate_angle(right_hip, right_knee, right_ankle)
    
    return angles if angles else None


def calculate_elbow_angles(landmarks):
    """
    Calculate left and right elbow angles
    
    Args:
        landmarks: Dictionary of landmark positions
        
    Returns:
        dict: {'left': angle, 'right': angle} or None if not detectable
    """
    angles = {}
    
    # Left elbow: shoulder -> elbow -> wrist
    left_shoulder = get_point(landmarks, 'left_shoulder')
    left_elbow = get_point(landmarks, 'left_elbow')
    left_wrist = get_point(landmarks, 'left_wrist')
    
    if left_shoulder and left_elbow and left_wrist:
        angles['left'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    # Right elbow: shoulder -> elbow -> wrist
    right_shoulder = get_point(landmarks, 'right_shoulder')
    right_elbow = get_point(landmarks, 'right_elbow')
    right_wrist = get_point(landmarks, 'right_wrist')
    
    if right_shoulder and right_elbow and right_wrist:
        angles['right'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    return angles if angles else None


def calculate_shoulder_angles(landmarks):
    """
    Calculate left and right shoulder angles (arm elevation)
    
    Args:
        landmarks: Dictionary of landmark positions
        
    Returns:
        dict: {'left': angle, 'right': angle} or None if not detectable
    """
    angles = {}
    
    # Left shoulder: hip -> shoulder -> elbow
    left_hip = get_point(landmarks, 'left_hip')
    left_shoulder = get_point(landmarks, 'left_shoulder')
    left_elbow = get_point(landmarks, 'left_elbow')
    
    if left_hip and left_shoulder and left_elbow:
        angles['left'] = calculate_angle(left_hip, left_shoulder, left_elbow)
    
    # Right shoulder: hip -> shoulder -> elbow
    right_hip = get_point(landmarks, 'right_hip')
    right_shoulder = get_point(landmarks, 'right_shoulder')
    right_elbow = get_point(landmarks, 'right_elbow')
    
    if right_hip and right_shoulder and right_elbow:
        angles['right'] = calculate_angle(right_hip, right_shoulder, right_elbow)
    
    return angles if angles else None


def calculate_hip_angles(landmarks):
    """
    Calculate left and right hip angles (leg elevation)
    
    Args:
        landmarks: Dictionary of landmark positions
        
    Returns:
        dict: {'left': angle, 'right': angle} or None if not detectable
    """
    angles = {}
    
    # Left hip: shoulder -> hip -> knee
    left_shoulder = get_point(landmarks, 'left_shoulder')
    left_hip = get_point(landmarks, 'left_hip')
    left_knee = get_point(landmarks, 'left_knee')
    
    if left_shoulder and left_hip and left_knee:
        angles['left'] = calculate_angle(left_shoulder, left_hip, left_knee)
    
    # Right hip: shoulder -> hip -> knee
    right_shoulder = get_point(landmarks, 'right_shoulder')
    right_hip = get_point(landmarks, 'right_hip')
    right_knee = get_point(landmarks, 'right_knee')
    
    if right_shoulder and right_hip and right_knee:
        angles['right'] = calculate_angle(right_shoulder, right_hip, right_knee)
    
    return angles if angles else None


def calculate_spine_alignment(landmarks):
    """
    Calculate spine alignment (deviation from vertical)
    
    Args:
        landmarks: Dictionary of landmark positions
        
    Returns:
        float: Spine angle in degrees (0 = vertical, >0 = leaning)
    """
    left_shoulder = get_point(landmarks, 'left_shoulder')
    right_shoulder = get_point(landmarks, 'right_shoulder')
    left_hip = get_point(landmarks, 'left_hip')
    right_hip = get_point(landmarks, 'right_hip')
    
    if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
        return None
    
    # Calculate midpoints
    shoulder_mid = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    )
    hip_mid = (
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2
    )
    
    # Calculate angle from vertical
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    
    # Vertical line points (same x, different y)
    vertical_point = (hip_mid[0], hip_mid[1] - 100)
    
    angle = calculate_angle(vertical_point, hip_mid, shoulder_mid)
    
    # Adjust for left/right leaning
    if dx > 0:
        angle = 180 - angle
    
    return angle


def calculate_all_angles(landmarks):
    """
    Calculate all joint angles at once
    
    Args:
        landmarks: Dictionary of landmark positions
        
    Returns:
        dict: Dictionary containing all calculated angles
    """
    return {
        'knee': calculate_knee_angles(landmarks),
        'elbow': calculate_elbow_angles(landmarks),
        'shoulder': calculate_shoulder_angles(landmarks),
        'hip': calculate_hip_angles(landmarks),
        'spine': calculate_spine_alignment(landmarks)
    }

