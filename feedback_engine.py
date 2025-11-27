"""
Feedback Engine Module
Generates real-time correction feedback based on pose analysis
purpose:
Check the current body angles
Compare with ideal yoga pose angle ranges
Tell the user what to fix
"""

from pose_rules import get_pose_rules, check_angle_in_range, get_angle_deviation, POSE_RULES


class FeedbackEngine:
    def __init__(self):
        """Initialize feedback engine
        Starts with empty feedback"""
        self.feedback_messages = []
    
    def analyze_pose(self, landmarks, angles, current_pose):
        """
        Analyze pose and generate feedback messages
        
        Args:
            landmarks: Dictionary of landmark positions
            angles: Dictionary of calculated angles
            current_pose: Name of current yoga pose
            
        Returns:
            list: List of feedback messages
        """
        self.feedback_messages = []
        
        if not current_pose or current_pose == "Unknown":
            self.feedback_messages.append("Please assume a yoga pose")
            return self.feedback_messages
        
        pose_rules = get_pose_rules(current_pose)
        if not pose_rules:
            return self.feedback_messages
        
        # Check knee angles
        if angles.get('knee') and pose_rules.get('knee'):
            self._check_knee_angles(angles['knee'], pose_rules['knee'])
        
        # Check elbow angles
        if angles.get('elbow') and pose_rules.get('elbow'):
            self._check_elbow_angles(angles['elbow'], pose_rules['elbow'])
        
        # Check shoulder angles
        if angles.get('shoulder') and pose_rules.get('shoulder'):
            self._check_shoulder_angles(angles['shoulder'], pose_rules['shoulder'])
        
        # Check hip angles
        if angles.get('hip') and pose_rules.get('hip'):
            self._check_hip_angles(angles['hip'], pose_rules['hip'])
        
        # Check spine alignment
        if angles.get('spine') is not None and pose_rules.get('spine'):
            self._check_spine_alignment(angles['spine'], pose_rules['spine'])
        
        # If no issues found
        if not self.feedback_messages:
            self.feedback_messages.append("Great form! Keep it up!")
        
        return self.feedback_messages
    
    def _check_knee_angles(self, knee_angles, target_ranges):
        """Check and provide feedback on knee angles"""
        for side in ['left', 'right']:
            if side in knee_angles and knee_angles[side] is not None:
                angle = knee_angles[side]
                if side in target_ranges:
                    target_range = target_ranges[side]
                    
                    if not check_angle_in_range(angle, target_range):
                        deviation = get_angle_deviation(angle, target_range)
                        
                        if angle < target_range[0]:
                            # Knee too bent
                            if target_range[0] > 170:  # Should be straight
                                self.feedback_messages.append(
                                    f"Straighten your {side} knee"
                                )
                            else:
                                self.feedback_messages.append(
                                    f"Bend your {side} knee more"
                                )
                        else:
                            # Knee too straight
                            self.feedback_messages.append(
                                f"Bend your {side} knee slightly"
                            )
    
    def _check_elbow_angles(self, elbow_angles, target_ranges):
        """Check and provide feedback on elbow angles"""
        for side in ['left', 'right']:
            if side in elbow_angles and elbow_angles[side] is not None:
                angle = elbow_angles[side]
                if side in target_ranges:
                    target_range = target_ranges[side]
                    
                    if not check_angle_in_range(angle, target_range):
                        if angle < target_range[0]:
                            self.feedback_messages.append(
                                f"Straighten your {side} arm"
                            )
                        else:
                            self.feedback_messages.append(
                                f"Bend your {side} arm slightly"
                            )
    
    def _check_shoulder_angles(self, shoulder_angles, target_ranges):
        """Check and provide feedback on shoulder/arm elevation"""
        for side in ['left', 'right']:
            if side in shoulder_angles and shoulder_angles[side] is not None:
                angle = shoulder_angles[side]
                if side in target_ranges:
                    target_range = target_ranges[side]
                    
                    if not check_angle_in_range(angle, target_range):
                        if angle < target_range[0]:
                            self.feedback_messages.append(
                                f"Lift your {side} arm higher"
                            )
                        else:
                            self.feedback_messages.append(
                                f"Lower your {side} arm slightly"
                            )
    
    def _check_hip_angles(self, hip_angles, target_ranges):
        """Check and provide feedback on hip angles"""
        for side in ['left', 'right']:
            if side in hip_angles and hip_angles[side] is not None:
                angle = hip_angles[side]
                if side in target_ranges:
                    target_range = target_ranges[side]
                    
                    if not check_angle_in_range(angle, target_range):
                        if angle < target_range[0]:
                            self.feedback_messages.append(
                                f"Straighten your {side} leg"
                            )
                        else:
                            self.feedback_messages.append(
                                f"Bend your {side} leg more"
                            )
    
    def _check_spine_alignment(self, spine_angle, target_range):
        """Check and provide feedback on spine alignment"""
        if not check_angle_in_range(spine_angle, target_range):
            deviation = get_angle_deviation(spine_angle, target_range)
            
            if spine_angle < target_range[0]:
                self.feedback_messages.append("Lean forward slightly")
            elif spine_angle > target_range[1]:
                if target_range[1] < 100:  # Should be vertical
                    self.feedback_messages.append("Keep your spine straight")
                else:
                    self.feedback_messages.append("Lean back slightly")
            else:
                self.feedback_messages.append("Keep your spine straight")


def recognize_pose(angles, landmarks):
    """
    Recognize yoga pose based on angles and landmarks
    
    Args:
        angles: Dictionary of calculated angles
        landmarks: Dictionary of landmark positions
        
    Returns:
        str: Name of recognized pose or "Unknown"
        After feedback, the code also tries to identify which pose the user is performing.
    """
    if not angles or not landmarks:
        return "Unknown"
    
    # Simple pose recognition based on key angle patterns
    knee_angles = angles.get('knee', {})
    hip_angles = angles.get('hip', {})
    shoulder_angles = angles.get('shoulder', {})
    spine_angle = angles.get('spine')
    
    # Check if all key points are visible
    required_landmarks = ['left_shoulder', 'right_shoulder', 'left_hip', 
                         'right_hip', 'left_knee', 'right_knee']
    if not all(key in landmarks for key in required_landmarks):
        return "Unknown"
    
    # Mountain Pose: All joints nearly straight, spine vertical
    if (knee_angles.get('left') and knee_angles.get('right') and
        knee_angles['left'] > 170 and knee_angles['right'] > 170 and
        spine_angle and 85 <= spine_angle <= 95):
        return "Mountain Pose"
    
    # Tree Pose: One leg bent to side
    left_knee = knee_angles.get('left', 180)
    right_knee = knee_angles.get('right', 180)
    left_hip = hip_angles.get('left', 180)
    right_hip = hip_angles.get('right', 180)
    
    # Check if one leg is significantly more bent
    if (abs(left_knee - right_knee) > 60 or abs(left_hip - right_hip) > 60):
        if (left_knee < 120 or right_knee < 120):
            return "Tree Pose"
    
    # Warrior I: Front leg bent, back leg straight
    if (knee_angles.get('left') and knee_angles.get('right')):
        if ((knee_angles['left'] < 130 and knee_angles['right'] > 170) or
            (knee_angles['right'] < 130 and knee_angles['left'] > 170)):
            return "Warrior I"
    
    # Warrior II: Similar to Warrior I but with arms extended
    if (knee_angles.get('left') and knee_angles.get('right') and
        shoulder_angles.get('left') and shoulder_angles.get('right')):
        if ((knee_angles['left'] < 130 and knee_angles['right'] > 170) or
            (knee_angles['right'] < 130 and knee_angles['left'] > 170)):
            if (shoulder_angles['left'] < 150 or shoulder_angles['right'] < 150):
                return "Warrior II"
    
    # Triangle Pose: Both legs straight, spine leaning
    if (knee_angles.get('left') and knee_angles.get('right') and
        knee_angles['left'] > 170 and knee_angles['right'] > 170 and
        spine_angle and spine_angle < 90):
        return "Triangle Pose"
    
    # Fallback: choose pose with smallest average angle deviation
    best_pose = None
    best_score = float("inf")

    for pose_name, pose_rules in POSE_RULES.items():
        total_deviation = 0.0
        measurements = 0

        for joint, rule in pose_rules.items():
            current_angles = angles.get(joint)
            if current_angles is None:
                continue

            if isinstance(rule, dict):
                # joint has left/right entries
                for side, target_range in rule.items():
                    if isinstance(current_angles, dict):
                        value = current_angles.get(side)
                    else:
                        value = None
                    if value is None:
                        continue
                    deviation = get_angle_deviation(value, target_range)
                    if deviation is not None:
                        total_deviation += deviation
                        measurements += 1
            else:
                # single value (e.g., spine)
                if isinstance(current_angles, dict):
                    continue
                value = current_angles
                if value is None:
                    continue
                deviation = get_angle_deviation(value, rule)
                if deviation is not None:
                    total_deviation += deviation
                    measurements += 1

        if measurements == 0:
            continue

        avg_deviation = total_deviation / measurements

        if avg_deviation < best_score:
            best_score = avg_deviation
            best_pose = pose_name

    if best_pose is not None:
        # Accept even if deviation is large; caller can still refine feedback
        return best_pose

    return "Unknown"

