"""
Yoga Pose Rules and Ideal Angle Ranges
Defines target angles for different yoga poses
"""

# Ideal angle ranges for each yoga pose (in degrees)
POSE_RULES = {
    'Mountain Pose': {
        'knee': {'left': (175, 180), 'right': (175, 180)},
        'elbow': {'left': (170, 180), 'right': (170, 180)},
        'shoulder': {'left': (170, 180), 'right': (170, 180)},
        'hip': {'left': (170, 180), 'right': (170, 180)},
        'spine': (85, 95),  # Nearly vertical
    },
    'Tree Pose': {
        'knee': {
            'left': (175, 180),  # Standing leg straight
            'right': (45, 90)    # Bent leg (knee to side)
        },
        'elbow': {'left': (170, 180), 'right': (170, 180)},
        'shoulder': {'left': (170, 180), 'right': (170, 180)},
        'hip': {
            'left': (170, 180),  # Standing leg
            'right': (90, 135)   # Bent leg
        },
        'spine': (85, 95),
    },
    'Warrior I': {
        'knee': {
            'left': (90, 120),   # Front leg bent
            'right': (175, 180)  # Back leg straight
        },
        'elbow': {'left': (170, 180), 'right': (170, 180)},
        'shoulder': {'left': (170, 180), 'right': (170, 180)},
        'hip': {
            'left': (90, 135),   # Front leg
            'right': (170, 180)  # Back leg
        },
        'spine': (85, 95),
    },
    'Warrior II': {
        'knee': {
            'left': (90, 120),   # Front leg bent
            'right': (175, 180)  # Back leg straight
        },
        'elbow': {'left': (170, 180), 'right': (170, 180)},
        'shoulder': {
            'left': (90, 180),   # Arms extended horizontally
            'right': (90, 180)
        },
        'hip': {
            'left': (90, 135),   # Front leg
            'right': (170, 180)  # Back leg
        },
        'spine': (85, 95),
    },
    'Triangle Pose': {
        'knee': {
            'left': (175, 180),  # Front leg straight
            'right': (175, 180)  # Back leg straight
        },
        'elbow': {'left': (170, 180), 'right': (170, 180)},
        'shoulder': {
            'left': (90, 180),   # Arms extended
            'right': (90, 180)
        },
        'hip': {
            'left': (170, 180),
            'right': (170, 180)
        },
        'spine': (60, 90),  # Leaning to side
    },
    'Downward Dog': {
        'knee': {
            'left': (150, 180),  # Slightly bent
            'right': (150, 180)
        },
        'elbow': {'left': (170, 180), 'right': (170, 180)},
        'shoulder': {'left': (160, 180), 'right': (160, 180)},
        'hip': {
            'left': (90, 135),   # Hips up
            'right': (90, 135)
        },
        'spine': (100, 120),  # Forward lean
    },
}


def get_pose_rules(pose_name):
    """
    Get ideal angle ranges for a specific pose
    
    Args:
        pose_name: Name of the yoga pose
        
    Returns:
        dict: Ideal angle ranges or None if pose not found
    """
    return POSE_RULES.get(pose_name)


def get_all_pose_names():
    """
    Get list of all supported pose names
    
    Returns:
        list: List of pose names
    """
    return list(POSE_RULES.keys())


def check_angle_in_range(angle, target_range):
    """
    Check if angle is within target range
    
    Args:
        angle: Current angle in degrees
        target_range: Tuple of (min, max) angle range
        
    Returns:
        bool: True if angle is within range
    """
    if angle is None or target_range is None:
        return False
    return target_range[0] <= angle <= target_range[1]


def get_angle_deviation(angle, target_range):
    """
    Calculate how far angle is from target range
    
    Args:
        angle: Current angle in degrees
        target_range: Tuple of (min, max) angle range
        
    Returns:
        float: Deviation in degrees (0 if within range, positive if outside)
    """
    if angle is None or target_range is None:
        return None
    
    if target_range[0] <= angle <= target_range[1]:
        return 0.0
    
    if angle < target_range[0]:
        return target_range[0] - angle
    else:
        return angle - target_range[1]

