"""
Feedback Engine Module
Generates real-time correction feedback based on pose analysis.

It now supports two complementary mechanisms:
- Rule-based feedback and fallback pose recognition using ideal angle ranges.
- Optional ML-based pose classification if a trained model file is present.
"""

import os
from typing import Dict, Optional

import numpy as np
from joblib import load

from pose_rules import (
    get_pose_rules,
    check_angle_in_range,
    get_angle_deviation,
    POSE_RULES,
)


# Optional ML classifier (RandomForest) trained via `ml_model.py`
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_classifier.joblib")
_POSE_MODEL = None


def _load_pose_model():
    global _POSE_MODEL
    if _POSE_MODEL is not None:
        return _POSE_MODEL
    if not os.path.exists(_MODEL_PATH):
        return None
    try:
        _POSE_MODEL = load(_MODEL_PATH)
        return _POSE_MODEL
    except Exception:
        # If loading fails, fall back silently to rule-based logic
        _POSE_MODEL = None
        return None


def _angles_to_vector_for_model(angles: Dict) -> Optional[np.ndarray]:
    """
    Convert angles dict into feature vector using the order stored with the model.
    Returns None if the model is missing its feature metadata.
    """
    model_bundle = _load_pose_model()
    if not model_bundle:
        return None

    feature_order = model_bundle.get("feature_order")
    if not feature_order:
        return None

    vec = []
    # Two supported formats for backward compatibility:
    # 1) List[Tuple[joint, side]] from older image-based trainer
    # 2) List[str] of feature names like "knee_left", "spine" from CSV trainer
    first_item = feature_order[0]

    if isinstance(first_item, tuple):
        # Old format: (joint, side)
        for joint, side in feature_order:
            joint_dict = angles.get(joint) or {}
            value = None
            if isinstance(joint_dict, dict):
                value = joint_dict.get(side)
            if value is None:
                value = 0.0
            vec.append(float(value))

        spine_angle = angles.get("spine")
        if spine_angle is None:
            spine_angle = 0.0
        vec.append(float(spine_angle))
    else:
        # New format: feature names
        for name in feature_order:
            if name.lower() == "spine":
                value = angles.get("spine")
            else:
                # Expect "<joint>_<side>", e.g. "knee_left"
                parts = name.split("_", 1)
                if len(parts) == 2:
                    joint, side = parts
                    joint_dict = angles.get(joint) or {}
                    value = joint_dict.get(side) if isinstance(joint_dict, dict) else None
                else:
                    value = None

            if value is None:
                value = 0.0
            vec.append(float(value))

    return np.asarray([vec], dtype=np.float32)


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
    Recognize yoga pose based on angles and landmarks.

    Strategy:
    1. If a trained ML model (RandomForest) is available, use it first.
    2. Fall back to existing rule-based heuristics if the model is missing
       or cannot make a confident prediction.
    """
    if not angles or not landmarks:
        return "Unknown"

    # Try ML model first (if present)
    model_bundle = _load_pose_model()
    if model_bundle is not None:
        vec = _angles_to_vector_for_model(angles)
        if vec is not None:
            scaler = model_bundle.get("scaler")
            clf = model_bundle.get("model")
            classes = model_bundle.get("classes")
            if scaler is not None and clf is not None and classes:
                vec_scaled = scaler.transform(vec)
                pred_idx = clf.predict(vec_scaled)[0]
                try:
                    return classes[int(pred_idx)]
                except (IndexError, ValueError):
                    pass  # fall back to rules

    # --- Rule-based recognition using POSE_RULES ---
    # Choose the pose whose ideal angle rules deviate the least from current angles.
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

