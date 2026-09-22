#!/usr/bin/env python3
"""
Simple Pose Shifting Utilities for Phase 2 Implementation.

Simplified version of pose_shift_utils.py with basic random offset approach
based on motion planner inaccuracy analysis results:
- Position error: ~2cm → use ±3cm shifts
- Orientation error: ~28.6° → use ±45° shifts

Key Functions:
- generate_shifted_pose(): Creates a single shifted pose with simple random offsets
- find_family_pose(): Finds closest family pose from initial states
"""

import numpy as np
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from scipy.spatial.transform import Rotation as R

# Repository root (…/lilo-vla), resolved relative to this file so nothing depends on the CWD.
_LILO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ATOMIC_DEMOS_PATH = str(
    _LILO_ROOT / "datasets" / "hdf5_datasets" / "atomic_local_demos" / "full_atomic_skills"
)

# Default configuration parameters based on motion planner analysis
DEFAULT_POSITION_SHIFT_RANGE = 0.03  # ±3cm for each axis
DEFAULT_ORIENTATION_SHIFT_RANGE = np.radians(45)  # ±45° for each axis


def collect_step_data(action: np.ndarray, obs: Dict, reward: float, done: bool, env) -> Dict:
    """
    Collect data from a single environment step according to hdf5_structure.md specifications.
    """
    step_data = {
        'actions': action,                                     # Action taken this step
        'dones': done,                                         # Episode termination flag
        'rewards': reward,                                     # Reward from step
        'states': env.sim.get_state().flatten(),              # Full simulation state (84-dim)
        'robot_states': obs['robot0_proprio-state'][:9],      # First 9 elements of proprio state
        'obs': {
            'joint_states': obs['robot0_joint_pos'],          # 7-DOF joint positions
            'gripper_states': obs['robot0_gripper_qpos'],     # 2-element gripper positions
            'ee_pos': obs['robot0_eef_pos'],                  # 3D end-effector position
            'ee_ori': R.from_quat(obs['robot0_eef_quat']).as_euler('xyz'),  # Convert quat to euler
            'ee_states': np.concatenate([                      # Combined pose (6-dim)
                obs['robot0_eef_pos'], 
                R.from_quat(obs['robot0_eef_quat']).as_euler('xyz')
            ]),
            'agentview_rgb': obs['agentview_image'],          # Agent viewpoint RGB image
            'eye_in_hand_rgb': obs['robot0_eye_in_hand_image'] # Wrist camera RGB image
        }
    }
    return step_data


# NOTE: extract_ee_pose_from_initial_state() function removed
# EE pose should be obtained directly from environment observations using:
# obs = env.env._get_observations()
# ee_pos = obs['robot0_eef_pos']
# ee_quat = obs['robot0_eef_quat']
# This is more reliable than trying to extract from initial state arrays.


def generate_shifted_pose(target_pose: Tuple[np.ndarray, np.ndarray],
                         position_shift_range: float = DEFAULT_POSITION_SHIFT_RANGE,
                         orientation_shift_range: float = DEFAULT_ORIENTATION_SHIFT_RANGE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single shifted pose with simple random offsets.

    Simplified approach based on motion planner inaccuracy analysis:
    - Position: Add random offset ±position_shift_range to each axis
    - Orientation: Add random offset ±orientation_shift_range to each axis in axis-angle space

    Args:
        target_pose: (position, quaternion) - original target pose
        position_shift_range: Random offset range for position (default: ±3cm)
        orientation_shift_range: Random offset range for orientation (default: ±45°)

    Returns:
        Tuple of (shifted_position, shifted_quaternion)
    """
    target_position, target_quaternion = target_pose

    # Simple position shift: add random offset to each axis
    position_offsets = np.random.uniform(-position_shift_range, position_shift_range, 3)
    shifted_position = target_position + position_offsets

    # Simple orientation shift: modify all 3 axes while ensuring total difference <= orientation_shift_range
    target_rotation = R.from_quat(target_quaternion)
    target_axis_angle = target_rotation.as_rotvec()

    # Generate random offsets for all 3 axes
    orientation_offsets = np.random.uniform(-orientation_shift_range, orientation_shift_range, 3)

    # Check if the magnitude exceeds the limit and scale down if needed
    offset_magnitude = np.linalg.norm(orientation_offsets)
    if offset_magnitude > orientation_shift_range:
        orientation_offsets = orientation_offsets * (orientation_shift_range / offset_magnitude)

    shifted_axis_angle = target_axis_angle + orientation_offsets
    shifted_rotation = R.from_rotvec(shifted_axis_angle)
    shifted_quaternion = shifted_rotation.as_quat()

    return shifted_position, shifted_quaternion


def compute_pose_distance(pos1: np.ndarray, quat1: np.ndarray,
                         pos2: np.ndarray, quat2: np.ndarray,
                         position_weight: float = 1.0,
                         orientation_weight: float = 0.1) -> float:
    """
    Compute combined 3D Euclidean + orientation distance between poses.

    Args:
        pos1, quat1: First pose
        pos2, quat2: Second pose
        position_weight: Weight for position distance
        orientation_weight: Weight for orientation distance

    Returns:
        Combined distance metric
    """
    # Euclidean distance for position
    pos_distance = np.linalg.norm(pos1 - pos2)

    # Quaternion distance (angle between rotations)
    rot1 = R.from_quat(quat1)
    rot2 = R.from_quat(quat2)
    relative_rot = rot1.inv() * rot2
    angle_distance = np.abs(relative_rot.as_rotvec()).sum()  # Sum of axis-angle components

    return position_weight * pos_distance + orientation_weight * angle_distance


def find_family_pose(shifted_pose: Tuple[np.ndarray, np.ndarray],
                    current_skill_initial_states: List[np.ndarray],
                    original_pose: Tuple[np.ndarray, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find family pose - simplified version that just returns the original pose.
    This is appropriate since the evaluation is designed to compare shifted vs original poses.

    Args:
        shifted_pose: (position, quaternion) - the shifted target pose (not used)
        current_skill_initial_states: List of initial states for current skill (not used)
        original_pose: (position, quaternion) - the original pose to return

    Returns:
        (family_position, family_quaternion) - original pose
    """
    if original_pose is not None:
        original_position, original_quaternion = original_pose
        print(f"✅ Using original pose as family pose")
        print(f"   📍 Family pose: pos=[{original_position[0]:.3f}, {original_position[1]:.3f}, {original_position[2]:.3f}]")
        return original_position, original_quaternion

    # If no original pose provided, this is an error in the calling code
    raise ValueError("original_pose must be provided. EE poses should be obtained from environment observations, not initial states.")


def load_initial_states_for_skill(skill_name: str,
                                 atomic_demos_path: str = _DEFAULT_ATOMIC_DEMOS_PATH) -> List[np.ndarray]:
    """
    Load initial states for a specific skill.

    Args:
        skill_name: Name of the skill (e.g., "KITCHEN_SCENE1_open_the_top_drawer")
        atomic_demos_path: Path to atomic demos directory

    Returns:
        List of initial state arrays for the skill
    """
    # Only use plain .init files (not _original.init or _augmented.init)
    plain_init_path = f"{atomic_demos_path}/{skill_name}.init"

    try:
        if os.path.exists(plain_init_path):
            init_file_path = plain_init_path
        else:
            raise FileNotFoundError(f"No init file found for {skill_name}: tried {plain_init_path}")

        with open(init_file_path, 'rb') as f:
            initial_states = pickle.load(f)

        print(f"📁 Loaded {len(initial_states)} initial states for skill: {skill_name} from {os.path.basename(init_file_path)}")
        return initial_states

    except FileNotFoundError:
        raise FileNotFoundError(f"Initial states file not found for skill: {skill_name}")
    except Exception as e:
        raise RuntimeError(f"Error loading initial states: {e}")


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing simple pose shifting utilities...")

    # Note: Full testing requires environment setup and observations
    # Basic pose shifting can be tested with dummy poses
    print("📋 Testing pose shifting with dummy target pose...")

    # Create a dummy target pose
    target_position = np.array([0.0, 0.0, 1.0])
    target_quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
    target_pose = (target_position, target_quaternion)

    # Test with different shift parameters
    shift_configs = [
        (0.02, np.radians(30)),  # ±2cm, ±30°
        (0.05, np.radians(45)),  # ±5cm, ±45°
        (0.1, np.radians(60)),   # ±10cm, ±60°
    ]

    for i, (pos_range, ori_range) in enumerate(shift_configs):
        print(f"\n📋 Test {i+1}: pos_range=±{pos_range*100:.0f}cm, ori_range=±{np.degrees(ori_range):.0f}°")

        shifted_pose = generate_shifted_pose(target_pose, pos_range, ori_range)
        shifted_pos, shifted_quat = shifted_pose

        pos_diff = np.linalg.norm(shifted_pos - target_position)

        print(f"   Target:  pos=[{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
        print(f"   Shifted: pos=[{shifted_pos[0]:.3f}, {shifted_pos[1]:.3f}, {shifted_pos[2]:.3f}] | Δ={pos_diff*100:.1f}cm")

        # Test family pose (should return original pose)
        family_pose = find_family_pose(shifted_pose, [], target_pose)
        family_pos, family_quat = family_pose
        print(f"   Family:  pos=[{family_pos[0]:.3f}, {family_pos[1]:.3f}, {family_pos[2]:.3f}]")

    print(f"\n✅ Simple pose shifting utilities test completed!")
    print("💡 For full testing with real initial states, run within evaluation script with proper environment setup")