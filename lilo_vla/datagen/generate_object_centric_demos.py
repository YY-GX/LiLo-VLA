#!/usr/bin/env python3
"""
Generate object-centric augmented demos using above-pose strategy.

This script creates augmented demonstrations by:
1. Starting robot from an "above object" pose
2. Using MPlib motion planning to move to a demo trajectory waypoint
3. Replaying the rest of the original demo

This creates diverse training data with different approach trajectories while
maintaining successful task completion.

Example usage:
  python -m lilo_vla.datagen.generate_object_centric_demos --debug_mode
  python -m lilo_vla.datagen.generate_object_centric_demos --debug_mode --trigger_distance 0.03 --above_height 0.05
  python -m lilo_vla.datagen.generate_object_centric_demos --debug_mode --use_init_states
"""

import sys
import os
import h5py
import numpy as np
import pickle
import argparse
import json
import time
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import re
import imageio
import cv2
from PIL import Image

# ==============================================================================
# Project Setup
# ==============================================================================

# Package bootstrap (writes the LIBERO config and applies the compatibility
# patches). MUST be imported before `libero`.
import lilo_vla  # noqa: F401

# Repository root (the `lilo-vla/` directory) -- used to resolve bundled
# configs and to place generated outputs.
_LILO_ROOT = Path(__file__).resolve().parents[2]

# LIBERO environment
from libero.libero.envs import OffScreenRenderEnv

# Pose calculation utilities (object poses, above-pose calculation, special handling)
from lilo_vla.reaching.approach_pose import (
    get_object_pose,
    get_special_object_handling,
    calculate_above_pose,
    REST_QUAT
)

# Motion planning (MPlib collision-aware planning)
from lilo_vla.reaching.motion_planning.mplib_planner import MPlibMotionPlanner

# Segmentation utilities (wrist camera masks for target/grasped objects)
from lilo_vla.interaction.masking import (
    create_wrist_segmentation_mask,
    create_wrist_segmentation_mask_with_grasped
)

# LIBERO utilities
from lilo_vla.envs.libero_env import quat2axisangle

# ==============================================================================
# Configuration
# ==============================================================================

# Path to skill config
SKILL_CONFIG_PATH = str(_LILO_ROOT / "configs" / "skill_config.json")

# ==============================================================================
# Debug Configuration
# ==============================================================================

# DEBUG_BDDL_FILES: BDDL files to process in debug mode (--debug_mode flag)
# Comment out unwanted files to focus on specific skills during development
DEBUG_BDDL_FILES = [
    # "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_place.bddl",
    # "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet_place.bddl"
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate_pick.bddl",
    # "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate_pick.bddl",
    # "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate_place.bddl",
    # "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate_place.bddl",
    # "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate_place.bddl",
    # "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_place.bddl",
    # "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_place.bddl",
    # "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet.bddl",
    # "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_place.bddl",        
    # "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet_place.bddl",
    # "KITCHEN_SCENE3_turn_on_the_stove.bddl",
    # "KITCHEN_SCENE8_turn_off_the_stove.bddl"
]

# MAKE_UP_BDDL_SKILLS: Skills to regenerate when using --make_up_mode flag
# Use this to re-generate specific skills without processing entire dataset

# Restored for the release: the full 27-skill set that produced the published
# dataset (the working copy had all but one entry commented out).
MAKE_UP_BDDL_SKILLS = [
    "place_tomato_sauce_in_basket",
    "close_the_bottom_drawer_of_the_cabinet",
    "open_the_microwave",
    "pick_alphabet_soup",
    "pick_black_bowl",
    "pick_butter",
    "pick_chocolate_pudding",
    "pick_cream_cheese",
    "pick_frying_pan",
    "pick_moka_pot",
    "pick_tomato_sauce",
    "pick_white_mug",
    "pick_wine_bottle",
    "pick_yellow_and_white_mug",
    "place_alphabet_soup_in_basket",
    "place_black_bowl_on_the_plate",
    "place_butter_in_basket",
    "place_chocolate_pudding_to_right_of_plate",
    "place_cream_cheese_in_basket",
    "place_frying_pan_on_the_stove",
    "place_moka_pot_on_the_stove",
    "place_white_mug_on_the_plate",
    "place_wine_bottle_in_bottom_drawer_of_the_cabinet",
    "place_yellow_and_white_mug_on_right_plate",
    "place_black_bowl_on_the_black_bowl",
    "turn_on_the_stove",
    "place_black_bowl_in_bottom_drawer_of_the_cabinet",
]
# MAKE_UP_BDDL_SKILLS = [
#     "place_chocolate_pudding_to_right_of_plate",
#     "place_tomato_sauce_in_basket",
#     "place_white_mug_on_the_plate",
#     "close_the_bottom_drawer_of_the_cabinet",
#     "place_black_bowl_in_bottom_drawer_of_the_cabinet",
#     "place_wine_bottle_in_bottom_drawer_of_the_cabinet"
#     # "turn_on_the_stove",
#     # "place_alphabet_soup_in_basket",
#     # "place_butter_in_basket"
#     # "turn_off_the_stove",  # this has many demos, i list here cuz no seg for it, so need to regenerate
#     # "place_frying_pan_on_the_stove",  # this has many demos, i list here cuz no seg for it, so need to regenerate
#     # "place_moka_pot_on_the_stove"  # this has many demos, i list here cuz no seg for it, so need to regenerate
#     # "place_alphabet_soup_in_basket",
#     # "place_ketchup_in_top_drawer_of_the_cabinet",
#     # "place_white_bowl_on_the_plate"
# ]


# ==============================================================================
# Configuration Loading
# ==============================================================================

def load_skill_config():
    """Load skill config from JSON."""
    with open(SKILL_CONFIG_PATH, 'r') as f:
        return json.load(f)


# ==============================================================================
# Skill Information and Object Detection
# ==============================================================================

def infer_skill_info_from_bddl(bddl_filename: str) -> Optional[Tuple[str, list, str, Optional[list]]]:
    """
    Infer skill name, target object candidates, skill type, and grasped object from BDDL file using skill_config.json.

    Args:
        bddl_filename: BDDL filename (e.g., "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.bddl")

    Returns:
        (skill_name, target_object_list, skill_type, grasped_object_list) tuple if found, None otherwise
        target_object_list: List of possible target objects to try (from skill_config.json)
        grasped_object_list: List of possible grasped objects (for place skills) or None
    """
    # Construct full path to BDDL file
    bddl_base_path = str(_LILO_ROOT / "configs" / "bddl" / "atomic_skills")
    bddl_file_path = os.path.join(bddl_base_path, bddl_filename)

    if not os.path.exists(bddl_file_path):
        return None

    # Get skill_name and skill_type from skill_config
    skill_config = load_skill_config()

    skill_name = None
    skill_type = None
    target_object_list = None
    grasped_object_list = None
    for sname, skill_info in skill_config.items():
        if bddl_filename in skill_info['bddl_files']:
            skill_name = sname
            skill_type = skill_info['skill_type']
            target_object_list = skill_info['target_object']  # Now a list
            grasped_object_list = skill_info.get('grasped_object_name', None)  # May not exist for non-place skills
            break

    # If not found in config, parse BDDL directly as fallback
    if skill_type is None:
        with open(bddl_file_path, 'r') as f:
            content = f.read()

        # Extract obj_of_interest from BDDL
        obj_interest_match = re.search(r'\(:obj_of_interest\s+(.*?)\)', content, re.DOTALL)
        if not obj_interest_match:
            return None

        # Get first object listed
        obj_of_interest = obj_interest_match.group(1).strip().split()[0]
        target_object_list = [f"{obj_of_interest}_main"]

        # Infer skill_type from language
        language_match = re.search(r'\(:language\s+(.*?)\)', content)
        if language_match:
            language = language_match.group(1).strip()
            if language.startswith('pick'):
                skill_type = 'pick'
            elif language.startswith('place'):
                skill_type = 'place'
            else:
                skill_type = 'other'
            skill_name = language.replace(' ', '_')
        else:
            skill_type = 'other'
            skill_name = bddl_filename.replace('.bddl', '')

    return skill_name, target_object_list, skill_type, grasped_object_list


def find_grasped_object_in_env(env, grasped_object_list: Optional[List[str]]) -> Optional[str]:
    """
    Find which grasped object actually exists in the environment.
    
    Args:
        env: LIBERO environment
        grasped_object_list: List of possible grasped object names (from skill_config.json)
    
    Returns:
        Name of the grasped object that exists in the environment, or None if none found
    """
    if grasped_object_list is None:
        return None
    
    # Check each candidate object name
    for obj_name in grasped_object_list:
        try:
            # Try to get object pose - if it exists, this will succeed
            pos, quat = get_object_pose(env, obj_name)
            if pos is not None:
                return obj_name
        except:
            continue
    
    return None


# ==============================================================================
# Initialization State Loading
# ==============================================================================

def load_init_state(skill_name: str, init_dir: str, demo_idx: int) -> Optional[np.ndarray]:
    """
    Load initial simulation state from .init file.

    Args:
        skill_name: Name of the skill (e.g., "place_tomato_sauce_in_basket")
        init_dir: Directory containing .init files
        demo_idx: Index of demo to load state for

    Returns:
        Initial state as numpy array, or None if file not found
    """
    if skill_name == "place_butter_in_basket":
        skill_name = "place_butter_in_tray"
    init_path = Path(init_dir) / f"{skill_name}.init"
    if not init_path.exists():
        return None

    try:
        with open(init_path, 'rb') as f:
            init_states = pickle.load(f)  # Load list of states

        # Select state based on demo_idx (with wraparound)
        if isinstance(init_states, list) and len(init_states) > 0:
            init_state = init_states[demo_idx % len(init_states)]
            return init_state
        else:
            print(f"  ⚠️  Warning: Unexpected init_states format in {init_path}")
            return None
    except Exception as e:
        print(f"  ⚠️  Warning: Failed to load init state from {init_path}: {e}")
        return None


# ==============================================================================
# Trigger Detection (When to Start Above-Pose Strategy)
# ==============================================================================

def detect_trigger_timestep(demo_data: Dict, skill_type: str, env) -> int:
    """
    Detect trigger timestep based on skill type.

    For pick/other: Detects when end-effector makes contact with target object
    For place: Uses last timestep of demo

    Args:
        demo_data: Demo data with actions
        skill_type: "pick", "place", or "other"
        env: LIBERO environment for contact detection

    Returns:
        Trigger timestep index
    """
    actions = demo_data['actions']

    if skill_type == "place":
        # For place: return last timestep of the demo
        print(f"Place skill: using last timestep {len(actions) - 1} as trigger")
        return len(actions) - 1

    # Pick/other: detect contact with object
    print("Detecting contact trigger for pick/other skill...")
    EE_GEOM_NAMES = [
        "robot0_eef", "robot0_gripper0_finger0", "robot0_gripper0_finger1",
        "gripper0_hand_collision", "gripper0_finger0_collision", "gripper0_finger1_collision"
    ]

    # Use proper replay method for accurate contact detection
    env.reset()
    for t, action in enumerate(actions):
        # Check contacts BEFORE taking action
        for j in range(env.sim.data.ncon):
            contact = env.sim.data.contact[j]
            g1 = env.sim.model.geom_id2name(contact.geom1)
            g2 = env.sim.model.geom_id2name(contact.geom2)
            if g1 in EE_GEOM_NAMES or g2 in EE_GEOM_NAMES:
                print(f"Found EE contact at timestep {t}")
                return t

        # Take the action
        obs, reward, done, info = env.step(action)
        if done:
            break

    print("Warning: No contact trigger found, using last timestep")
    return len(actions) - 1


def detect_pick_timestep_for_place(demo_data: Dict) -> int:
    """
    Detect pick timestep for place skills by detecting gripper closing.

    Place skill demos are actually pick+place sequences. This function finds
    when gripper closes to identify the transition point between pick and place.
    Returns timestep + 20 to ensure object is fully grasped before proceeding.

    Args:
        demo_data: Demo data with actions

    Returns:
        Pick timestep + 20 (timestep to replay until for place skills)
    """
    actions = demo_data['actions']
    gripper_commands = actions[:, -1]  # Last dimension is gripper
    
    # Find gripper closing: negative -> positive transition
    for t in range(1, len(gripper_commands)):
        if gripper_commands[t - 1] < 0 and gripper_commands[t] > 0:
            pick_timestep_plus_20 = min(t + 20, len(actions) - 1)
            print(f"Found gripper closing at timestep {t} (place skill), replay until {pick_timestep_plus_20}")
            return pick_timestep_plus_20
    
    # Fallback: if no gripper closing found, use last timestep
    print(f"Warning: No gripper closing found for place skill, using last timestep")
    return len(actions) - 1


# ==============================================================================
# Pose Utilities
# ==============================================================================

def calculate_pose_distance(pose1: Tuple[np.ndarray, np.ndarray],
                            pose2: Tuple[np.ndarray, np.ndarray],
                            w_pos: float = 1.0,
                            w_ori: float = 0.5) -> float:
    """
    Calculate weighted distance between two poses.

    Args:
        pose1: (pos, quat) tuple, quat in [w, x, y, z] format
        pose2: (pos, quat) tuple, quat in [w, x, y, z] format
        w_pos: Weight for position distance
        w_ori: Weight for orientation distance

    Returns:
        Weighted distance
    """
    pos1, quat1 = pose1
    pos2, quat2 = pose2

    # Position distance
    pos_dist = np.linalg.norm(pos1 - pos2)

    # Orientation distance (geodesic distance between quaternions)
    # Both inputs are assumed to be in [w, x, y, z] format
    # Convert to [x, y, z, w] for scipy
    quat1_xyzw = np.array([quat1[1], quat1[2], quat1[3], quat1[0]])
    quat2_xyzw = np.array([quat2[1], quat2[2], quat2[3], quat2[0]])
    
    r1 = R.from_quat(quat1_xyzw)
    r2 = R.from_quat(quat2_xyzw)
    ori_dist = (r1.inv() * r2).magnitude()

    return w_pos * pos_dist + w_ori * ori_dist


# ==============================================================================
# Observation and Video Processing
# ==============================================================================

def combine_camera_views(obs: dict, env=None, target_object: str = None, include_segmentation: bool = False,
                         skill_type: str = None, grasped_object_name: str = None) -> np.ndarray:
    """
    Combine agentview and wrist camera into side-by-side frame.

    Creates visualization: [agentview | wrist_rgb | wrist_segmentation (optional)]
    All images are rotated 180° for correct viewing orientation.

    Args:
        obs: Observation dict containing 'agentview_image' and 'robot0_eye_in_hand_image'
        env: Environment (required if include_segmentation=True)
        target_object: Target object name (required if include_segmentation=True)
        include_segmentation: If True, also include wrist segmentation mask (3 images total)
        skill_type: Skill type ("pick", "place", or "other") - required for place segmentation
        grasped_object_name: Grasped object name (required for place skills)

    Returns:
        Combined image with agentview on left, wrist on right (both rotated 180°)
        If include_segmentation=True: agentview | wrist_rgb | wrist_segmentation
        If wrist camera is not available, returns only agentview (rotated 180°)
    """
    # Get both camera views
    agentview = obs['agentview_image'].copy()

    # Check if wrist camera is available
    if 'robot0_eye_in_hand_image' in obs:
        wrist = obs['robot0_eye_in_hand_image'].copy()
    else:
        # Fallback: use agentview for both sides if wrist camera not available
        print("  ⚠️  Warning: robot0_eye_in_hand_image not found, using agentview for both sides")
        wrist = agentview.copy()

    # Ensure uint8 format
    if agentview.dtype != np.uint8:
        agentview = (agentview * 255).astype(np.uint8)
    if wrist.dtype != np.uint8:
        wrist = (wrist * 255).astype(np.uint8)

    # Rotate both 180 degrees
    agentview = cv2.rotate(agentview, cv2.ROTATE_180)
    wrist = cv2.rotate(wrist, cv2.ROTATE_180)

    # Create segmentation mask if requested
    if include_segmentation:
        try:
            # Check if segmentation already exists in obs (from motion planner)
            if 'robot0_eye_in_hand_segmentation' in obs:
                seg_mask = obs['robot0_eye_in_hand_segmentation'].copy()
            elif env is not None and target_object is not None:
                # Create segmentation mask on-the-fly (grayscale uint8)
                # For place skills, include grasped object in mask
                if skill_type == 'place' and grasped_object_name is not None:
                    seg_mask = create_wrist_segmentation_mask_with_grasped(env, target_object, grasped_object_name)
                else:
                    seg_mask = create_wrist_segmentation_mask(env, target_object)
            else:
                # No segmentation available
                seg_mask = np.zeros((256, 256), dtype=np.uint8)

            # Rotate 180 degrees
            seg_mask = cv2.rotate(seg_mask, cv2.ROTATE_180)
            # Convert grayscale to RGB for concatenation
            seg_mask_rgb = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
            # Concatenate all three: agentview | wrist_rgb | wrist_segmentation
            combined = np.concatenate([agentview, wrist, seg_mask_rgb], axis=1)
        except Exception as e:
            print(f"  ⚠️  Warning: Failed to create segmentation for video: {e}")
            # Fallback to just agentview + wrist
            combined = np.concatenate([agentview, wrist], axis=1)
    else:
        # Concatenate side-by-side (agentview left, wrist right)
        combined = np.concatenate([agentview, wrist], axis=1)

    return combined


# ==============================================================================
# Target Timestep Finding (Find Starting Point for Replay)
# ==============================================================================

def find_target_timestep_from_trigger(demo_data: Dict, trigger_timestep: int, min_distance: float) -> int:
    """
    Find the timestep that is at least min_distance away from trigger.

    Searches backward from trigger to find where to inject above-pose strategy.
    This ensures the robot has sufficient distance to execute the approach.
    
    Args:
        demo_data: Demo data with obs
        trigger_timestep: Trigger timestep index
        min_distance: Minimum distance in meters (e.g., 0.15 for 15cm)
        
    Returns:
        Target timestep index (at least min_distance away from trigger)
    """
    ee_positions = demo_data['obs']['ee_pos']
    trigger_ee_pos = ee_positions[trigger_timestep]
    
    # Search backward from trigger_timestep
    for t in range(trigger_timestep - 1, -1, -1):
        current_ee_pos = ee_positions[t]
        distance = np.linalg.norm(current_ee_pos - trigger_ee_pos)
        
        if distance >= min_distance:
            print(f"Found target timestep {t} at distance {distance:.3f}m from trigger timestep {trigger_timestep}")
            return t
    
    # Fallback: if no timestep is min_distance away, return first timestep
    print(f"Warning: No timestep found {min_distance*100:.1f}cm away from trigger, using timestep 0")
    return 0



def collect_step_data(action: np.ndarray, obs: Dict, reward: float, done: bool, env,
                     target_object: str = None, state: np.ndarray = None,
                     skill_type: str = None, grasped_object_name: str = None) -> Dict:
    """
    Collect data from a single environment step according to hdf5_structure.md specifications.

    Args:
        action: Action taken this step
        obs: Observation after taking action
        reward: Reward received
        done: Episode termination flag
        env: Environment (used as fallback if state not provided)
        target_object: Target object name for segmentation mask (optional)
        state: Full simulation state (84-dim). If None, will get from env.sim.get_state()
        skill_type: Skill type ("pick", "place", or "other") - required for place segmentation
        grasped_object_name: Grasped object name (required for place skills)
    """
    # Get or create wrist segmentation mask
    wrist_segmentation = None
    if 'robot0_eye_in_hand_segmentation' in obs:
        # Use pre-computed segmentation from obs (e.g., from motion planner)
        wrist_segmentation = obs['robot0_eye_in_hand_segmentation']
    elif target_object is not None:
        # Create segmentation on-the-fly if not already in obs
        # For place skills, include grasped object in mask
        try:
            if skill_type == 'place' and grasped_object_name is not None:
                wrist_segmentation = create_wrist_segmentation_mask_with_grasped(env, target_object, grasped_object_name)
            else:
                wrist_segmentation = create_wrist_segmentation_mask(env, target_object)
        except Exception as e:
            print(f"  ⚠️  Warning: Failed to create wrist segmentation: {e}")
            # Fallback to zeros
            wrist_segmentation = np.zeros((256, 256), dtype=np.uint8)

    # Get object pose for relative EE calculation
    # Try to get from obs first (added by execute_cartesian_trajectory), otherwise get from env
    if 'target_object_pos' in obs and 'target_object_quat' in obs and target_object is not None:
        object_pos = obs['target_object_pos']
        object_quat_wxyz = obs['target_object_quat']  # [w,x,y,z] from MuJoCo
    elif target_object is not None:
        try:
            object_pos, object_quat_wxyz = get_object_pose(env.sim, target_object)
            if object_pos is None or object_quat_wxyz is None:
                # Fallback if object not found
                object_pos = np.zeros(3)
                object_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # Identity [w,x,y,z]
        except:
            # Fallback to zeros/identity if object pose can't be retrieved
            object_pos = np.zeros(3)
            object_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # Identity [w,x,y,z]
    else:
        # No target object specified, use zeros/identity (ee will be absolute)
        object_pos = np.zeros(3)
        object_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # Identity [w,x,y,z]

    # Calculate relative EE position (relative to object)
    absolute_ee_pos = obs['robot0_eef_pos']
    relative_ee_pos = absolute_ee_pos - object_pos

    # Calculate relative EE orientation (relative to object)
    # obs['robot0_eef_quat'] is [x,y,z,w] (LIBERO format)
    # object_quat_wxyz is [w,x,y,z] (MuJoCo format)
    ee_quat_xyzw = obs['robot0_eef_quat']  # [x,y,z,w]

    # Convert object quat from [w,x,y,z] to [x,y,z,w] for scipy
    object_quat_xyzw = np.array([
        object_quat_wxyz[1], object_quat_wxyz[2],
        object_quat_wxyz[3], object_quat_wxyz[0]
    ])  # [x,y,z,w]

    # Calculate relative orientation: relative_rot = object_rot_inv * ee_rot
    object_rot = R.from_quat(object_quat_xyzw)  # scipy expects [x,y,z,w]
    ee_rot = R.from_quat(ee_quat_xyzw)          # scipy expects [x,y,z,w]
    relative_rot = object_rot.inv() * ee_rot
    relative_quat_xyzw = relative_rot.as_quat()  # [x,y,z,w]

    # Convert to axis-angle (quat2axisangle expects [x,y,z,w])
    absolute_ee_ori = quat2axisangle(ee_quat_xyzw)
    relative_ee_ori = quat2axisangle(relative_quat_xyzw)

    step_data = {
        'actions': action,                                     # Action taken this step
        'dones': done,                                         # Episode termination flag
        'rewards': reward,                                     # Reward from step
        'states': state if state is not None else env.sim.get_state().flatten(),  # Full simulation state (84-dim)
        'robot_states': obs['robot0_proprio-state'][:9],      # First 9 elements of proprio state
        'obs': {
            'joint_states': obs['robot0_joint_pos'],          # 7-DOF joint positions
            'gripper_states': obs['robot0_gripper_qpos'],     # 2-element gripper positions
            'absolute_ee_pos': absolute_ee_pos,               # 3D absolute end-effector position
            'ee_pos': relative_ee_pos,                        # 3D relative end-effector position (relative to object)
            'absolute_ee_ori': absolute_ee_ori,               # 3D absolute end-effector orientation (axis-angle)
            'ee_ori': relative_ee_ori,                        # 3D relative end-effector orientation (axis-angle, relative to object)
            'ee_states': np.concatenate([                      # Combined pose (6-dim, relative position + relative orientation)
                relative_ee_pos,
                relative_ee_ori
            ]),
            'agentview_rgb': obs['agentview_image'],          # Agent viewpoint RGB image
            'eye_in_hand_rgb': obs['robot0_eye_in_hand_image'] # Wrist camera RGB image
        }
    }

    # Add wrist segmentation if available
    if wrist_segmentation is not None:
        step_data['obs']['eye_in_hand_segmentation'] = wrist_segmentation

    return step_data


# ==============================================================================
# Demo Generation Core Logic
# ==============================================================================

def generate_single_augmented_demo(demo_data: Dict,
                                   env,
                                   skill_name: str,
                                   target_object: str,
                                   bddl_file: str,
                                   trigger_timestep: int,
                                   iteration: int,
                                   skill_type: str,
                                   grasped_object_list: Optional[List[str]],
                                   source_hdf5: str,
                                   demo_idx: int,
                                   args) -> Tuple[bool, List[Dict], Dict]:
    """
    Generate one augmented demo using above-pose strategy.

    Pipeline:
    1. Calculate above-pose for target object (with optional random shift)
    2. Move robot to above-pose using MPlib motion planning
    3. Find nearest demo waypoint and move to it
    4. Replay remaining demo trajectory

    Args:
        demo_data: Original demo data
        env: LIBERO environment
        skill_name: Skill name (e.g., "place_tomato_sauce_in_basket")
        target_object: Target object name
        bddl_file: Path to BDDL file (for special object handling)
        trigger_timestep: Trigger timestep
        iteration: Iteration number (0 = no shift)
        skill_type: Skill type ("pick", "place", "other")
        grasped_object_list: List of possible grasped object names (for place skills)
        source_hdf5: Path to the original raw demo HDF5 file
        args: Command line arguments

    Returns:
        (success, step_data_list, metadata) tuple
    """
    
    actions = demo_data['actions']

    # Step 1: Reset environment first (critical for proper initialization)
    obs_reset = env.reset()

    # Stabilize environment: run 5 dummy actions to let objects settle due to gravity
    for _ in range(5):
        obs_reset, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))  # Gripper open action

    # Step 1.5: Load and set init state if --use_init_states is enabled
    grasped_object_name = None
    if args.use_init_states:
        init_state = load_init_state(skill_name, args.init_states_dir, demo_idx)
        if init_state is not None:
            print(f"  🔄 Loading init state from {skill_name}.init (demo {demo_idx})")

            # For place skills: disable gravity, set state, close gripper, enable gravity
            if skill_type == "place":
                # Save old gravity
                old_gravity = env.sim.model.opt.gravity.copy()
                # Disable gravity
                env.sim.model.opt.gravity[:] = 0
                env.sim.forward()

                # Set init state
                env.set_init_state(init_state)

                # Close gripper (5 steps with action [0,0,0,0,0,0,1.0])
                for _ in range(5):
                    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

                # Re-enable gravity
                env.sim.model.opt.gravity[:] = old_gravity
                env.sim.forward()
                obs_reset = env.env._get_observations()

                # Find grasped object (already in gripper from init state)
                if grasped_object_list is not None:
                    grasped_object_name = find_grasped_object_in_env(env, grasped_object_list)
                    if grasped_object_name:
                        print(f"  ✓ Object grasped: {grasped_object_name}")
                    else:
                        print(f"  ⚠️⚠️⚠️ Warning: Could not find grasped object from list {grasped_object_list}")
            else:
                # For non-place skills: just set state normally
                env.set_init_state(init_state)
                obs_reset = env.env._get_observations()
                print(f"  ✓ Init state loaded for {skill_type} skill")
        else:
            print(f"  ⚠️  Warning: Init file not found for {skill_name}, using default reset")

    # Step 2: Set initial state from first step of demo (after reset)
    # COMMENTED OUT: Setting init state causes pipeline failures (KeyError when predicate evaluation references missing objects)
    # if 'states' in demo_data and len(demo_data['states']) > 0:
    #     first_state = demo_data['states'][0]  # Get first state from demo (shape: (84,))
    #     # Ensure it's a numpy array
    #     if not isinstance(first_state, np.ndarray):
    #         first_state = np.array(first_state)
    #     # Set state to sim (after reset, so environment is fully initialized)
    #     # Note: set_init_state may trigger check_success() which can fail if object states aren't initialized
    #     # Wrap in try-except to handle KeyError when predicate evaluation references missing objects
    #     try:
    #         env.set_init_state(first_state)
    #     except KeyError as e:
    #         print(f"  ⚠️  Warning: Could not set init state (KeyError: {e}), using default reset state")
    #         # Continue with default reset state - the state might still be set in the sim
    #         # but check_success failed due to missing object state references
    #     except Exception as e:
    #         print(f"  ⚠️  Warning: Could not set init state ({type(e).__name__}: {e}), using default reset state")
    #         # Continue with default reset state
    
    step_data_list = []
    video_frames = []  # Collect video frames based on mode
    video_collect_started = False  # Track when to start collecting (for normal mode: after moving to above pose)

    # Collect reset observation (for debug mode or normal mode with video saving)
    # Note: Using reset observation directly since we're not setting init state from demo
    if args.debug_mode or args.save_videos_normal:
        video_frames.append(combine_camera_views(obs_reset, env, target_object, include_segmentation=args.debug_mode,
                                                 skill_type=skill_type, grasped_object_name=grasped_object_name))

    # Step 2 (original): For place skill, replay until pick timestep (gripper closing + 20)
    # NOTE: Skip this step if using init states (object already grasped from .init file)
    if skill_type == "place" and not args.use_init_states:
        pick_timestep = detect_pick_timestep_for_place(demo_data)
        # Replay until object is grasped
        for t in range(min(pick_timestep, len(actions))):
            obs, reward, done, info = env.step(actions[t])
            # Collect observations for video (debug mode or normal mode with video saving: all frames)
            if args.debug_mode or args.save_videos_normal:
                video_frames.append(combine_camera_views(obs, env, target_object, include_segmentation=args.debug_mode,
                                                         skill_type=skill_type, grasped_object_name=None))

        # Find grasped object after replay
        if grasped_object_list is not None:
            grasped_object_name = find_grasped_object_in_env(env, grasped_object_list)
            if grasped_object_name is None:
                print(f"  ⚠️  Warning: Could not find grasped object from list {grasped_object_list}")

    # Step 3: Get object pose from current environment state
    obj_pos, _, above_height, bbox_z = get_special_object_handling(
        env, target_object, bddl_file, args.above_height, skill_type=skill_type,
        grasped_object_name=grasped_object_name, save_debug_files=args.debug_mode
    )

    if obj_pos is None:
        print(f"  Demo: Failed to get object pose")
        # For place skills, return complete video frames even on early failure
        metadata = {'source_hdf5': source_hdf5}
        if skill_type == "place" and args.debug_mode:
            metadata['video_frames_complete'] = list(video_frames)
        return False, [], metadata

    # Step 4: Calculate above pose
    shift = (iteration > 0)

    # For augmented demos (iteration > 0), randomly choose shift type (50/50)
    z_only_positive = False
    if shift:
        z_only_positive = (random.random() < 0.5)
        if z_only_positive:
            print(f"   Using z-only positive shift mode (z shift range: 0 to {args.z_range * 3:.4f}m)")
        else:
            print(f"   Using standard shift mode (xy: ±{args.xy_range}m, z: ±{args.z_range}m, ori: ±{args.ori_range}°)")

    above_pos_libero, above_quat_libero, shift_info = calculate_above_pose(
        obj_pos,
        above_height=above_height,
        shift=shift,
        xy_range=args.xy_range,
        z_range=args.z_range,
        ori_range=args.ori_range,
        bbox_z=bbox_z,
        z_only_positive=z_only_positive
    )

    # Step 5: Move to above pose

    # Initialize MPlib planner
    motion_planner = MPlibMotionPlanner(env, collision_aware=True, velocity_factor=args.velocity_factor, time_step=args.time_step, safety_margin=args.safety_margin, verbose=False)

    # Set pointcloud output directory
    pointcloud_output_dir = str(_LILO_ROOT / "outputs" / "pointclouds" / "mplib_debug" / "complex")

    # Move to above pose
    print(f"   Grasped object name: {grasped_object_name}")
    success_above, [obs_above, actions_above, rewards_above, dones_above, states_above] = motion_planner.move_to_pose(
        above_pos_libero,
        above_quat_libero,
        position_threshold=0.02,
        orientation_threshold=0.524,
        skill_type='pick' if skill_type != 'place' else 'place',
        grasped_object_name=grasped_object_name,
        target_object=target_object,
        save_pointcloud=args.debug_mode,
        pointcloud_output_dir=pointcloud_output_dir
    )

    if not success_above:
        print(f"  Demo: Failed to move to above pose")
        # Return video frames even on early failure
        metadata = {'source_hdf5': source_hdf5}
        if args.debug_mode or len(video_frames) > 0:
            metadata['video_frames'] = video_frames
        return False, [], metadata

    # Start collecting video frames after moving to above pose (for normal mode without video saving)
    # When save_videos_normal is enabled, collect all frames from reset (like debug mode)
    if not args.save_videos_normal:
        video_collect_started = True

    # Collect video frames from moving to above pose (but NOT step_data - demo starts FROM above pose)
    for obs in obs_above:
        if args.debug_mode or args.save_videos_normal or video_collect_started:
            video_frames.append(combine_camera_views(obs, env, target_object, include_segmentation=args.debug_mode,
                                                     skill_type=skill_type, grasped_object_name=grasped_object_name))
    
    # Get current observation AFTER reaching above pose - this is where demo collection starts
    # The last observation from obs_above is the state at above pose
    obs_at_above = obs_above[-1] if len(obs_above) > 0 else env.env._get_observations()

    # Add observation at above pose to video frames if not already included
    if args.debug_mode or args.save_videos_normal or video_collect_started:
        if len(obs_above) == 0 or obs_at_above['agentview_image'] is not obs_above[-1]['agentview_image']:
            video_frames.append(combine_camera_views(obs_at_above, env, target_object, include_segmentation=args.debug_mode))
    


    # Step 6: Determine target timestep (at least trigger_distance away from trigger for all skill types)
    # Special handling for specific drawer placement skills in makeup mode
    trigger_distance = args.trigger_distance
    if args.make_up_mode and skill_name in ["place_black_bowl_in_bottom_drawer_of_the_cabinet",
                                              "place_wine_bottle_in_bottom_drawer_of_the_cabinet"]:
        trigger_distance = 0.1
        print(f"   Using trigger_distance=0.1 for {skill_name} (makeup mode)")
    target_t = find_target_timestep_from_trigger(demo_data, trigger_timestep, trigger_distance)

    # Step 7: Move to target pose (convert axis-angle to quaternion)
    target_pos = demo_data['obs']['ee_pos'][target_t]
    target_quat_axis_angle = demo_data['obs']['ee_ori'][target_t]
    # Convert axis-angle to quaternion (scipy returns [x, y, z, w])
    target_quat_xyzw = R.from_rotvec(target_quat_axis_angle).as_quat()
    # Convert to [w, x, y, z] format for move_to_pose
    target_quat = np.array([target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]])

    print(f"   Grasped object name: {grasped_object_name}")
    success_target, [obs_target, actions_target, rewards_target, dones_target, states_target] = motion_planner.move_to_pose(
        target_pos,
        target_quat,
        position_threshold=0.02,
        orientation_threshold=0.524,
        skill_type='pick' if skill_type != 'place' else 'place',
        grasped_object_name=grasped_object_name,
        target_object=target_object,
        save_pointcloud=args.debug_mode,
        pointcloud_output_dir=pointcloud_output_dir
    )

    # Capture init state from first state of moving from above region to target
    # This is the state at the start of the demo (robot at above pose with correct object configuration)
    init_state_at_above = states_target[0] if len(states_target) > 0 else env.sim.get_state().flatten()

    if not success_target:
        print(f"  Demo: Failed to move to target pose")
        # Return video frames even on early failure
        metadata = {'source_hdf5': source_hdf5}
        if args.debug_mode or len(video_frames) > 0:
            metadata['video_frames'] = video_frames
        return False, [], metadata

    # Collect observations from moving to target
    for i, obs in enumerate(obs_target):
        if args.debug_mode or args.save_videos_normal or video_collect_started:
            video_frames.append(combine_camera_views(obs, env, target_object, include_segmentation=args.debug_mode,
                                                     skill_type=skill_type, grasped_object_name=grasped_object_name))
        # Collect step_data: action taken FROM above pose (for i=0) or from previous state
        action = actions_target[i] if i < len(actions_target) else np.zeros(7)
        reward = rewards_target[i] if i < len(rewards_target) else 0.0
        done = dones_target[i] if i < len(dones_target) else False
        state = states_target[i] if i < len(states_target) else None
        # obs is the observation after taking action (obs_{i+1} where obs_0 is at above pose)
        step_data_list.append(collect_step_data(action, obs, reward, done, env, target_object, state,
                                                skill_type, grasped_object_name))

    # Step 8: Replay rest of demo
    for t in range(target_t, len(actions)):
        obs, reward, done, info = env.step(actions[t])
        state = env.sim.get_state().flatten()  # Capture state after step
        step_data_list.append(collect_step_data(actions[t], obs, reward, done, env, target_object, state,
                                                skill_type, grasped_object_name))
        # Collect observations for video
        if args.debug_mode or args.save_videos_normal or video_collect_started:
            video_frames.append(combine_camera_views(obs, env, target_object, include_segmentation=args.debug_mode,
                                                     skill_type=skill_type, grasped_object_name=grasped_object_name))

        if done:
            success = True
            # Concatenate above_pos and above_quat into single array: [x, y, z, x, y, z, w]
            above_pose_combined = np.concatenate([above_pos_libero, above_quat_libero]).tolist()
            metadata = {
                'above_pose': above_pose_combined,
                'target_timestep': target_t,
                'trigger_timestep': trigger_timestep,
                'iteration': iteration,
                'shifted': shift,
                'shift_info': shift_info,
                'init_state_at_above': init_state_at_above  # Correct initial state at above pose
            }
            # Include video frames in metadata
            if args.debug_mode or len(video_frames) > 0:
                metadata['video_frames'] = video_frames
            metadata['source_hdf5'] = source_hdf5
            return True, step_data_list, metadata

    # Failed to complete (step 8: replay didn't complete)
    print(f"  Demo: Failed to complete")
    # Return collected data even if it failed, so video can be saved
    metadata = {'source_hdf5': source_hdf5}
    # Include video frames in metadata
    if args.debug_mode or len(video_frames) > 0:
        metadata['video_frames'] = video_frames
    return False, step_data_list, metadata


def process_single_demo(demo_data: Dict,
                       demo_idx: int,
                       skill_name: str,
                       target_object: str,
                       skill_type: str,
                       bddl_file: str,
                       grasped_object_list: Optional[List[str]],
                       env,
                       args,
                       source_hdf5: str) -> Tuple[List, List, List, int, float, List]:
    """
    Process one original demo to generate N augmented demos.

    Generates args.num_augmentations variations of the demo, each with
    different random above-pose shifts (if enabled).

    Returns:
        (successful_demos, failed_demos, successful_init_states, failure_count, time_used, failure_metadatas) tuple
        failure_count: Total number of failures (including those with empty step_data)
        time_used: Time taken to generate all augmented demos for this demo (in seconds)
        failure_metadatas: List of metadata dicts for failures (for video saving)
    """
    # Start timing
    start_time = time.time()
    
    # Detect trigger
    trigger_timestep = detect_trigger_timestep(demo_data, skill_type, env)
    print(f"  Demo {demo_idx}: trigger={trigger_timestep}")

    successful_demos = []
    failed_demos = []
    successful_init_states = []
    failure_count = 0  # Count all failures (including those with empty step_data)
    failure_metadatas = []  # Track metadata for failures (for video saving)

    # Generate augmented demos
    for iter_idx in range(args.num_augmentations):
        success, step_data, metadata = generate_single_augmented_demo(
            demo_data, env, skill_name, target_object, bddl_file, trigger_timestep, iter_idx, skill_type, grasped_object_list, source_hdf5, demo_idx, args
        )

        if success:
            # Convert to demo format
            demo = {
                'actions': np.array([s['actions'] for s in step_data]),
                'dones': np.array([s['dones'] for s in step_data], dtype=np.uint8),
                'rewards': np.array([s['rewards'] for s in step_data], dtype=np.uint8),
                'robot_states': np.array([s['robot_states'] for s in step_data]),
                'states': np.array([s['states'] for s in step_data]),
                'obs': {}
            }

            # Convert observations
            obs_keys = step_data[0]['obs'].keys()
            for key in obs_keys:
                demo['obs'][key] = np.array([s['obs'][key] for s in step_data])

            demo['metadata'] = metadata
            demo['extra_info'] = {
                'shifted': metadata.get('shifted', False),
                'shift_params': metadata.get('shift_info', None)
            }
            successful_demos.append(demo)

            # Save initial state AT above pose (before moving to target)
            # Only save for iteration 0 (no shift) to avoid saving duplicate init states
            if iter_idx == 0:
                init_state = metadata['init_state_at_above']
                successful_init_states.append(init_state)
            
            # Save video (only in debug mode, for all iterations)
            if args.debug_mode:
                base_video_dir = str(_LILO_ROOT / 'outputs' / 'videos' / 'demo_gen')
                if hasattr(args, 'debug_timestamp') and args.debug_timestamp:
                    video_dir = os.path.join(base_video_dir, args.debug_timestamp)
                else:
                    video_dir = base_video_dir
                os.makedirs(video_dir, exist_ok=True)
                
                # Use video_frames from metadata if available, otherwise use demo observations
                if 'video_frames' in metadata:
                    # Save video from collected frames
                    save_demo_video_complete(metadata['video_frames'], skill_name, demo_idx, iter_idx, video_dir, 
                                           is_failure=False, debug_mode=args.debug_mode)
                else:
                    # Fallback: save from demo observations
                    save_demo_video(demo, skill_name, demo_idx, iter_idx, video_dir, 
                                  is_failure=False, debug_mode=args.debug_mode)
            
            print(f"    Iter {iter_idx}: SUCCESS")
        else:
            failure_count += 1  # Count every failure
            
            # Save video for failures (only in debug mode, for all iterations)
            if args.debug_mode:
                base_video_dir = str(_LILO_ROOT / 'outputs' / 'videos' / 'demo_gen')
                if hasattr(args, 'debug_timestamp') and args.debug_timestamp:
                    video_dir = os.path.join(base_video_dir, args.debug_timestamp)
                else:
                    video_dir = base_video_dir
                os.makedirs(video_dir, exist_ok=True)
                
                # Use video_frames from metadata if available
                if 'video_frames' in metadata:
                    save_demo_video_complete(metadata['video_frames'], skill_name, demo_idx, iter_idx, video_dir, 
                                           is_failure=True, debug_mode=args.debug_mode)
                elif len(step_data) > 0:
                    # Fallback: save from step_data if available
                    demo = {
                        'actions': np.array([s['actions'] for s in step_data]),
                        'dones': np.array([s['dones'] for s in step_data], dtype=np.uint8),
                        'rewards': np.array([s['rewards'] for s in step_data], dtype=np.uint8),
                        'robot_states': np.array([s['robot_states'] for s in step_data]),
                        'states': np.array([s['states'] for s in step_data]),
                        'obs': {}
                    }
                    obs_keys = step_data[0]['obs'].keys()
                    for key in obs_keys:
                        demo['obs'][key] = np.array([s['obs'][key] for s in step_data])
                    
                    save_demo_video(demo, skill_name, demo_idx, iter_idx, video_dir, 
                                  is_failure=True, debug_mode=args.debug_mode)
            
            # Still save failed demo if we have data
            if len(step_data) > 0:
                demo = {
                    'actions': np.array([s['actions'] for s in step_data]),
                    'dones': np.array([s['dones'] for s in step_data], dtype=np.uint8),
                    'rewards': np.array([s['rewards'] for s in step_data], dtype=np.uint8),
                    'robot_states': np.array([s['robot_states'] for s in step_data]),
                    'states': np.array([s['states'] for s in step_data]),
                    'obs': {}
                }
                obs_keys = step_data[0]['obs'].keys()
                for key in obs_keys:
                    demo['obs'][key] = np.array([s['obs'][key] for s in step_data])
                demo['metadata'] = {'iteration': iter_idx, 'failed': True, 'source_hdf5': source_hdf5}
                failed_demos.append(demo)
                failure_metadatas.append(metadata)  # Store metadata for video saving
                
                # Video saving is now handled above (for all iterations in debug mode)
            print(f"    Iter {iter_idx}: FAILED")

    # Calculate time used
    end_time = time.time()
    time_used = end_time - start_time
    
    # Print timing info in debug mode
    if args.debug_mode:
        print(f"  Demo {demo_idx}: Time used = {time_used:.2f} seconds")

    return successful_demos, failed_demos, successful_init_states, failure_count, time_used, failure_metadatas


# ==============================================================================
# Data I/O (HDF5 and Pickle Operations)
# ==============================================================================

def save_demos_hdf5(demos: List[Dict], output_file: str):
    """Save demos to HDF5 file."""
    with h5py.File(output_file, 'w') as h5file:
        data_group = h5file.create_group('data')

        for demo_idx, demo_data in enumerate(demos):
            demo_group = data_group.create_group(f'demo_{demo_idx}')

            demo_group.create_dataset('actions', data=demo_data['actions'])
            demo_group.create_dataset('dones', data=demo_data['dones'])
            demo_group.create_dataset('rewards', data=demo_data['rewards'])
            demo_group.create_dataset('robot_states', data=demo_data['robot_states'])
            demo_group.create_dataset('states', data=demo_data['states'])

            obs_group = demo_group.create_group('obs')
            for obs_key, obs_data in demo_data['obs'].items():
                obs_group.create_dataset(obs_key, data=obs_data)

            # Save metadata
            if 'metadata' in demo_data:
                meta_group = demo_group.create_group('metadata')
                for key, val in demo_data['metadata'].items():
                    # Skip video_frames - it's only for video generation, not demo data
                    if key == 'video_frames':
                        continue
                    if isinstance(val, (list, tuple)):
                        # Check if it's a list of numpy arrays (like video frames)
                        if len(val) > 0 and isinstance(val[0], np.ndarray):
                            # Skip lists of arrays - they're too large and not needed in HDF5
                            continue
                        meta_group.create_dataset(key, data=np.array(val))
                    elif isinstance(val, dict):
                        # Convert dict to JSON string for HDF5 storage
                        meta_group.attrs[key] = json.dumps(val)
                    elif isinstance(val, np.ndarray):
                        meta_group.create_dataset(key, data=val)
                    elif val is None:
                        # Skip None values
                        continue
                    else:
                        meta_group.attrs[key] = val


def save_init_states(init_states: List[np.ndarray], output_file: str):
    """Save initial states as .init file."""
    with open(output_file, 'wb') as f:
        pickle.dump(init_states, f)


# ==============================================================================
# Progress Tracking
# ==============================================================================

def load_progress_stats(output_dir: str) -> Dict:
    """Load progress stats from JSON file if it exists."""
    progress_file = os.path.join(output_dir, "progress_stats.json")
    if not os.path.exists(progress_file):
        return {}
    
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        # Convert back to internal format (completed_skills dict)
        completed_skills = {}
        if 'completed_skills' in progress_data:
            for skill_name, stats in progress_data['completed_skills'].items():
                completed_skills[skill_name] = {
                    'num_success': stats['num_success'],
                    'num_failed': stats['num_failed']
                }
        return completed_skills
    except Exception as e:
        print(f"⚠️  Warning: Could not load progress stats from {progress_file}: {e}")
        return {}


def save_progress_stats(output_dir: str, completed_skills: Dict, all_skill_names: List[str]):
    """Save progress stats to JSON file."""
    progress_file = os.path.join(output_dir, "progress_stats.json")
    completed_skill_names = list(completed_skills.keys())
    not_completed_skill_names = [s for s in all_skill_names if s not in completed_skill_names]
    
    completed_with_stats = {}
    for skill_name, stats in completed_skills.items():
        total_attempts = stats['num_success'] + stats['num_failed']
        success_rate = stats['num_success'] / total_attempts if total_attempts > 0 else 0.0
        completed_with_stats[skill_name] = {
            'num_success': stats['num_success'],
            'num_failed': stats['num_failed'],
            'total_attempts': total_attempts,
            'success_rate': success_rate
        }
    
    progress_data = {
        'completed_skills': completed_with_stats,
        'not_completed_skills': not_completed_skill_names,
        'total_skills': len(all_skill_names),
        'completed_count': len(completed_skill_names),
        'not_completed_count': len(not_completed_skill_names),
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)


# ==============================================================================
# Video Saving
# ==============================================================================

def save_demo_video_complete(frames: List[np.ndarray], skill_name: str, demo_idx: int, iter_idx: int, video_dir: str, 
                            is_failure: bool = False, debug_mode: bool = False):
    """
    Save video from complete trajectory frames.
    
    Args:
        frames: List of combined camera view frames (already rotated and combined from combine_camera_views)
        skill_name: Skill name for filename
        demo_idx: Demo index
        iter_idx: Iteration index
        video_dir: Directory to save video
        is_failure: If True, add "_failure" suffix, otherwise "_success"
        debug_mode: If True, add "_debug" prefix to suffix
    """
    if len(frames) == 0:
        return
    
    # Convert frames list to numpy array
    frames_array = np.array(frames)
    
    # Ensure frames are in correct format (uint8, 0-255)
    if frames_array.dtype != np.uint8:
        if frames_array.max() <= 1.0:
            frames_array = (frames_array * 255).astype(np.uint8)
        else:
            frames_array = frames_array.astype(np.uint8)
    
    # Frames are already rotated and combined by combine_camera_views, no need to rotate again
    
    # Create video filename
    if debug_mode:
        suffix = "_debug_failure" if is_failure else "_debug_success"
    else:
        suffix = "_failure" if is_failure else "_success"
    video_filename = f"{skill_name}_demo{demo_idx}_iter{iter_idx}{suffix}.mp4"
    video_path = os.path.join(video_dir, video_filename)
    
    # Save video using imageio
    try:
        imageio.mimwrite(video_path, frames_array, fps=30, quality=8)
        print(f"      💾 Saved complete video: {video_filename}")
    except Exception as e:
        print(f"      ⚠️  Failed to save video {video_filename}: {e}")


def save_demo_video(demo: Dict, skill_name: str, demo_idx: int, iter_idx: int, video_dir: str, 
                   is_failure: bool = False, debug_mode: bool = False):
    """
    Save video from demo observations (agentview_rgb frames).
    
    Args:
        demo: Demo dict with 'obs' containing 'agentview_rgb'
        skill_name: Skill name for filename
        demo_idx: Demo index
        iter_idx: Iteration index
        video_dir: Directory to save video
        is_failure: If True, add "_failure" suffix, otherwise "_success"
        debug_mode: If True, add "_debug" prefix to suffix
    """
    if 'agentview_rgb' not in demo['obs']:
        return
    
    frames = demo['obs']['agentview_rgb']
    if frames.shape[0] == 0:
        return
    
    # Ensure frames are in correct format (uint8, 0-255)
    # Frames should be (T, H, W, 3)
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    
    # Rotate frames by 180 degrees
    rotated_frames = []
    for frame in frames:
        # Convert to PIL Image, rotate, convert back to numpy
        img = Image.fromarray(frame)
        img_rotated = img.rotate(180)
        rotated_frames.append(np.array(img_rotated))
    frames = np.array(rotated_frames)
    
    # Create video filename
    if debug_mode:
        suffix = "_debug_failure" if is_failure else "_debug_success"
    else:
        suffix = "_failure" if is_failure else "_success"
    video_filename = f"{skill_name}_demo{demo_idx}_iter{iter_idx}{suffix}.mp4"
    video_path = os.path.join(video_dir, video_filename)
    
    # Save video using imageio
    try:
        # Use ffmpeg codec if available, otherwise use default
        imageio.mimwrite(video_path, frames, fps=30, quality=8)
        print(f"      💾 Saved video: {video_filename}")
    except Exception as e:
        print(f"      ⚠️  Failed to save video {video_filename}: {e}")


def load_hdf5_demo_data(demo_file: str) -> Dict:
    """Load demo data from HDF5 file."""
    def recursively_extract(group):
        result = {}
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                result[key] = item[()]
            elif isinstance(item, h5py.Group):
                result[key] = recursively_extract(item)
        return result

    with h5py.File(demo_file, 'r') as f:
        return recursively_extract(f)


def save_first_obs_image_from_hdf5(hdf5_file: str, skill_name: str, image_dir: str):
    """
    Save the first observation image from the first demo in an HDF5 file.
    
    Args:
        hdf5_file: Path to HDF5 file
        skill_name: Name of the skill (for filename)
        image_dir: Directory to save the image
    """
    try:
        # Load HDF5 data
        demo_data = load_hdf5_demo_data(hdf5_file)
        
        # Get first demo
        data_group = demo_data.get('data', {})
        demo_keys = [k for k in data_group.keys() if k.startswith('demo_')]
        if len(demo_keys) == 0:
            print(f"  ⚠️  Warning: No demos found in {hdf5_file}")
            return
        
        first_demo_key = sorted(demo_keys)[0]
        first_demo = data_group[first_demo_key]
        
        # Get first observation image (agentview_rgb)
        obs_group = first_demo.get('obs', {})
        if 'agentview_rgb' not in obs_group:
            print(f"  ⚠️  Warning: No agentview_rgb found in {hdf5_file}")
            return
        
        agentview_rgb = obs_group['agentview_rgb']
        if agentview_rgb.shape[0] == 0:
            print(f"  ⚠️  Warning: Empty agentview_rgb in {hdf5_file}")
            return
        
        # Get first image (should be shape (H, W, C))
        first_image = agentview_rgb[0]
        
        # Convert to PIL Image if needed
        if isinstance(first_image, np.ndarray):
            # Ensure image is in correct format (H, W, C) with uint8 dtype
            if first_image.dtype != np.uint8:
                # Normalize to [0, 255] if needed
                if first_image.max() <= 1.0:
                    first_image = (first_image * 255).astype(np.uint8)
                else:
                    first_image = first_image.astype(np.uint8)
            
            # Ensure image is (H, W, C) format with RGB channels
            if len(first_image.shape) == 3 and first_image.shape[2] == 3:
                # Create PIL Image (RGB format)
                img = Image.fromarray(first_image, 'RGB')
            elif len(first_image.shape) == 2:
                # Grayscale image, convert to RGB
                img = Image.fromarray(first_image, 'L').convert('RGB')
            else:
                print(f"  ⚠️  Warning: Unexpected image shape {first_image.shape} in {hdf5_file}")
                return
        else:
            print(f"  ⚠️  Warning: Unexpected image type {type(first_image)} in {hdf5_file}")
            return
        
        # Create output directory
        os.makedirs(image_dir, exist_ok=True)
        
        # Save image
        image_filename = f"{skill_name}_first_obs.png"
        image_path = os.path.join(image_dir, image_filename)
        img.save(image_path)
        print(f"  💾 Saved first observation image: {image_path}")
        
    except Exception as e:
        print(f"  ⚠️  Warning: Failed to save first observation image from {hdf5_file}: {e}")


def process_skill(skill_name: str,
                 hdf5_file_path: str,
                 bddl_file_path: str,
                 target_object_list: list,
                 skill_type: str,
                 grasped_object_list: Optional[List[str]],
                 output_dir: str,
                 args) -> Dict:
    """
    Process all demos for one skill.

    Loads all demos from HDF5, processes each to create augmented variations,
    and saves results to output directory.

    Args:
        target_object_list: List of possible target objects to try

    Returns:
        Statistics dict
    """
    print(f"\n{'='*80}")
    print(f"Processing skill: {skill_name}")
    print(f"  BDDL: {bddl_file_path}")
    print(f"  HDF5: {hdf5_file_path}")
    print(f"  Object candidates: {target_object_list}")
    print(f"  Type: {skill_type}")

    # Load demos
    all_demo_data = load_hdf5_demo_data(hdf5_file_path)
    demo_keys = [k for k in all_demo_data['data'].keys() if k.startswith('demo_')]
    print(f"  Found {len(demo_keys)} original demos")

    # Limit demos for debugging
    if args.debug_num_demos > 0:
        demo_keys = demo_keys[:args.debug_num_demos]
        print(f"  Limited to {len(demo_keys)} demos (debug mode)")

    # Initialize environment
    env_args = {
        'bddl_file_name': bddl_file_path,
        'camera_heights': 256,
        'camera_widths': 256,
        'horizon': 10000  # Set very large horizon to prevent early termination
    }
    env = OffScreenRenderEnv(**env_args)
    env.reset()

    # Try each target object in the list until one is found
    target_object = None
    for candidate in target_object_list:
        obj_pos, _ = get_object_pose(env, candidate)
        if obj_pos is not None:
            target_object = candidate
            print(f"  ✅ Found target object: {target_object}")
            break

    if target_object is None:
        print(f"  ❌ None of the target object candidates {target_object_list} found in environment")
        env.close()
        return {
            'skill_name': skill_name,
            'num_success': 0,
            'num_failed': 0,
            'num_init_states': 0
        }

    # Process each demo
    all_successful_demos = []
    all_failed_demos = []
    all_init_states = []
    total_failure_count = 0  # Count all failures (including empty step_data ones)
    demo_times = []  # Track time for each demo (only in debug mode)
    
    # Track videos for normal mode (1 success + 1 failure per skill)
    success_video_saved = False
    failure_video_saved = False
    success_video_demo = None
    success_video_metadata = None
    failure_video_metadata = None

    for demo_idx, demo_key in enumerate(demo_keys):
        demo_data = all_demo_data['data'][demo_key]

        success_demos, fail_demos, init_states, failure_count, time_used, failure_metadatas_batch = process_single_demo(
            demo_data, demo_idx, skill_name, target_object, skill_type, bddl_file_path, grasped_object_list, env, args, hdf5_file_path
        )
        
        # Collect timing data in debug mode
        if args.debug_mode:
            demo_times.append(time_used)

        # Track videos for normal mode (1 success + 1 failure per skill)
        if args.save_videos_normal and not success_video_saved and len(success_demos) > 0:
            success_video_demo = success_demos[0]
            success_video_metadata = success_demos[0].get('metadata', {})
            success_video_saved = True
        
        if args.save_videos_normal and not failure_video_saved and len(failure_metadatas_batch) > 0:
            failure_video_metadata = failure_metadatas_batch[0]
            failure_video_saved = True

        all_successful_demos.extend(success_demos)
        all_failed_demos.extend(fail_demos)
        all_init_states.extend(init_states)
        total_failure_count += failure_count

    env.close()

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    if all_successful_demos:
        success_file = os.path.join(output_dir, f"{skill_name}.hdf5")
        save_demos_hdf5(all_successful_demos, success_file)
        print(f"✅ Saved {len(all_successful_demos)} successful demos: {success_file}")
        
        # Save first observation image in debug mode
        if args.debug_mode:
            base_image_dir = str(_LILO_ROOT / 'outputs' / 'images' / 'demo_gen')
            if hasattr(args, 'debug_timestamp') and args.debug_timestamp:
                image_dir = os.path.join(base_image_dir, args.debug_timestamp)
            else:
                image_dir = base_image_dir
            os.makedirs(image_dir, exist_ok=True)
            save_first_obs_image_from_hdf5(success_file, skill_name, image_dir)

    if all_failed_demos:
        failure_file = os.path.join(output_dir, f"{skill_name}_failure.hdf5")
        save_demos_hdf5(all_failed_demos, failure_file)
        print(f"⚠️  Saved {len(all_failed_demos)} failed demos: {failure_file}")

    if all_init_states:
        init_file = os.path.join(output_dir, f"{skill_name}.init")
        save_init_states(all_init_states, init_file)
        print(f"💾 Saved {len(all_init_states)} init states: {init_file}")
    
    # Save init states for failures (extract from demo data for reference)
    # Note: Since we don't set init state from demo, these states are from the demo data
    # and may not exactly match the environment state, but are saved for reference
    failed_init_states = []
    for fail_demo in all_failed_demos:
        if 'states' in fail_demo and len(fail_demo['states']) > 0:
            failed_init_states.append(fail_demo['states'][0])
    
    if failed_init_states:
        failure_init_file = os.path.join(output_dir, f"{skill_name}_failure.init")
        save_init_states(failed_init_states, failure_init_file)
        print(f"💾 Saved {len(failed_init_states)} failure init states: {failure_init_file}")
    
    # Save videos for normal mode (1 success + 1 failure per skill)
    if args.save_videos_normal:
        # Use different video directory if output_path contains "fewer"
        if 'fewer' in args.output_path.lower():
            video_dir = str(_LILO_ROOT / 'outputs' / 'videos' / 'demo_gen' / 'fewer')
        else:
            video_dir = str(_LILO_ROOT / 'outputs' / 'videos' / 'demo_gen')
        os.makedirs(video_dir, exist_ok=True)
        
        # Save success video (starting from above region, align with saved demo)
        if success_video_saved and success_video_demo is not None:
            # Extract frames from demo starting from above region (first frame is after reaching above pose)
            # Note: demo structure uses 'agentview_rgb' key, not 'agentview_image'
            if 'obs' in success_video_demo and 'agentview_rgb' in success_video_demo['obs']:
                frames = success_video_demo['obs']['agentview_rgb']
                save_demo_video_complete(frames, skill_name, 0, 0, video_dir, 
                                       is_failure=False, debug_mode=False)
            else:
                print(f"  ⚠️  Warning: Could not extract success video frames for {skill_name} (missing obs/agentview_rgb)")
        
        # Save failure video (from reset, like debug mode)
        if failure_video_saved and failure_video_metadata is not None:
            if 'video_frames' in failure_video_metadata:
                save_demo_video_complete(failure_video_metadata['video_frames'], skill_name, 0, 0, video_dir, 
                                       is_failure=True, debug_mode=False)

    return {
        'skill_name': skill_name,
        'num_success': len(all_successful_demos),
        'num_failed': total_failure_count,  # Use total failure count (includes empty step_data failures)
        'num_init_states': len(all_init_states),
        'demo_times': demo_times if args.debug_mode else []  # Return timing data in debug mode
    }


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """
    Main entry point for demo generation.

    Modes:
    - Normal mode: Process all skills from skill_config.json
    - Debug mode (--debug_mode): Process only DEBUG_BDDL_FILES with limited demos
    - Make-up mode (--make_up_mode): Regenerate specific skills from MAKE_UP_BDDL_SKILLS
    - Multi-BDDL mode (--multi_bddl_only): Process only skills with >3 BDDL variants

    Resume capability: Use --resume to skip already completed skills
    """
    parser = argparse.ArgumentParser(description='Generate augmented demos using above-pose strategy')

    # Paths
    parser.add_argument('--dataset_path', type=str,
                       default="datasets/hdf5_datasets/libero_90_no_noops/",
                       help='Path to raw demo HDF5 files (files named {BDDL_NAME}_demo.hdf5)')
    parser.add_argument('--output_path', type=str,
                       default='datasets/hdf5_datasets/atomic_above',
                       help='Output directory for augmented demos')

    # Hyperparameters
    parser.add_argument('--num_augmentations', type=int, default=5,
                       help='Number of augmented demos per original demo')
    parser.add_argument('--trigger_distance', type=float, default=0.03,
                       help='Minimum distance from trigger timestep to find target timestep in meters (default: 0.15m = 15cm)')
    parser.add_argument('--above_height', type=float, default=0.1,
                       help='Height above object (meters)')
    parser.add_argument('--xy_range', type=float, default=0.05,
                       help='XY shift range (meters, default: 0.05m = 5cm)')
    parser.add_argument('--z_range', type=float, default=0.05,
                       help='Z shift range (meters, default: 0.05m = 5cm)')
    parser.add_argument('--ori_range', type=float, default=15.0,
                       help='Orientation shift range in degrees (default: 15.0, so shift from -15 to +15)')
    parser.add_argument('--velocity_factor', type=float, default=1.0,
                       help='MPLib execution speed factor (0.0-1.0, default: 1.0)')
    parser.add_argument('--time_step', type=float, default=0.025,
                       help='MPLib trajectory planning time step (default: 0.05). Smaller values = more waypoints = slower execution.')
    parser.add_argument('--safety_margin', type=float, default=1.1,
                       help='Safety margin multiplier for bounding box size (default: 1.1 = 10%% expansion)')    

    # Debug options
    parser.add_argument('--debug_mode', action='store_true',
                       help='Debug mode: only process 3 specific skills with 2 iterations and 3 demos each')
    parser.add_argument('--debug_num_demos', type=int, default=0,
                       help='Limit number of demos per skill (0 = all)')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--save_videos_normal', action='store_true',
                       help='Save videos in normal mode (1 success + 1 failure per skill)')
    parser.add_argument('--resume', action='store_true', default=False,
                       help='Resume from previous progress (loads progress_stats.json and skips completed skills)')
    parser.add_argument('--multi_bddl_only', action='store_true',
                       help='Only generate demos for skills with >3 BDDL files (to address heterogeneous demos)')
    parser.add_argument('--make_up_mode', action='store_true',
                       help='Make-up mode: only generate demos for skills in MAKE_UP_BDDL_SKILLS list')
    parser.add_argument('--use_init_states', action='store_true',
                       help='Use .init files to set initial states (skips pick replay for place skills, ensures object already grasped)')
    parser.add_argument('--init_states_dir', type=str,
                       default="datasets/hdf5_datasets/atomic_above_fewer/all",
                       help='Directory containing .init files for each skill')

    args = parser.parse_args()
    
    # Set random seed if provided
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    # Override settings in debug mode
    if args.debug_mode:
        args.num_augmentations = 1
        args.debug_num_demos = 1
        print(f"\n🐛 DEBUG MODE ENABLED")
        print(f"{'='*80}")
        print(f"  Processing only {len(DEBUG_BDDL_FILES)} skills")
        print(f"  {args.num_augmentations} iterations per demo")
        print(f"  {args.debug_num_demos} demos per skill")
        print(f"  Total videos: {len(DEBUG_BDDL_FILES)} × {args.debug_num_demos} × {args.num_augmentations} = {len(DEBUG_BDDL_FILES) * args.debug_num_demos * args.num_augmentations}")

    print(f"\n🚀 Above-Pose Augmented Demo Generation")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_path}")
    print(f"Augmentations per demo: {args.num_augmentations}")
    print(f"Above height: {args.above_height}m")
    print(f"XY range: ±{args.xy_range}m, Z range: ±{args.z_range}m")
    print(f"Velocity factor: {args.velocity_factor}")
    print(f"Time step: {args.time_step}")
    print(f"Safety margin: {args.safety_margin}")

    # Load skill config
    skill_config = load_skill_config()

    # Determine which BDDL files to process
    if args.debug_mode:
        bddl_files_to_process = DEBUG_BDDL_FILES
    elif args.make_up_mode:
        # Filter to only process skills in MAKE_UP_BDDL_SKILLS
        bddl_files_to_process = []
        make_up_skills = []
        for skill_name, skill_info in skill_config.items():
            if skill_name in MAKE_UP_BDDL_SKILLS:
                bddl_files_to_process.extend(skill_info['bddl_files'])
                make_up_skills.append(f"{skill_name} ({len(skill_info['bddl_files'])} files)")
        print(f"\n💄 Make-up mode: Processing only skills in MAKE_UP_BDDL_SKILLS")
        print(f"   Skills: {', '.join(make_up_skills)}")
        if len(make_up_skills) == 0:
            print(f"   ⚠️  Warning: No matching skills found in MAKE_UP_BDDL_SKILLS")
            print(f"   Available skills: {', '.join(sorted(skill_config.keys()))}")
    elif args.multi_bddl_only:
        # Filter skills with >3 BDDL files (heterogeneous demos issue)
        bddl_files_to_process = []
        multi_bddl_skills = []
        for skill_name, skill_info in skill_config.items():
            if len(skill_info['bddl_files']) > 3:
                bddl_files_to_process.extend(skill_info['bddl_files'])
                multi_bddl_skills.append(f"{skill_name} ({len(skill_info['bddl_files'])} files)")
        print(f"\n🎯 Multi-BDDL mode: Processing only skills with >3 BDDL files")
        print(f"   Skills: {', '.join(multi_bddl_skills)}")
    else:
        # Get all BDDL files from skill config
        bddl_files_to_process = []
        for skill_info in skill_config.values():
            bddl_files_to_process.extend(skill_info['bddl_files'])
        bddl_files_to_process = list(set(bddl_files_to_process))  # Remove duplicates

    print(f"\nProcessing {len(bddl_files_to_process)} BDDL files...")

    # Generate timestamp for debug mode (for organizing videos and images in subfolders)
    if args.debug_mode:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.debug_timestamp = timestamp
        print(f"📁 Debug mode: Using timestamp {timestamp} for videos and images")
    else:
        args.debug_timestamp = None

    # Get all skill names for progress tracking
    all_skill_names = sorted(list(skill_config.keys())) if not args.debug_mode else []
    os.makedirs(args.output_path, exist_ok=True)

    # Disable resume for multi_bddl_only mode (to allow overwriting existing demos)
    if args.multi_bddl_only and args.resume:
        print(f"\n⚠️  Note: --resume disabled in --multi_bddl_only mode (will overwrite existing demos)")
        args.resume = False

    # Load existing progress to resume from where we left off (if resume is enabled)
    if args.resume:
        completed_skills = load_progress_stats(args.output_path)
        if completed_skills:
            print(f"\n📊 Resuming from previous run:")
            print(f"   Found {len(completed_skills)} completed skills")
            for skill_name in sorted(completed_skills.keys()):
                stats = completed_skills[skill_name]
                total = stats['num_success'] + stats['num_failed']
                print(f"   - {skill_name}: {stats['num_success']}/{total} success")
        else:
            print(f"\n🆕 Starting fresh run (no previous progress found)")
    else:
        completed_skills = {}
        print(f"\n🆕 Starting fresh run (--resume not enabled)")

    # Process each BDDL file
    all_stats = []
    bddl_base_path = str(_LILO_ROOT / "configs" / "bddl" / "atomic_skills")

    for bddl_filename in bddl_files_to_process:
        # Infer skill info from BDDL filename
        skill_info = infer_skill_info_from_bddl(bddl_filename)
        if skill_info is None:
            print(f"⚠️  Skipping {bddl_filename}: could not infer skill info")
            continue

        skill_name, target_object_list, skill_type, grasped_object_list = skill_info
        
        # Skip if this skill is already completed
        if skill_name in completed_skills:
            print(f"\n⏭️  Skipping {skill_name} (already completed)")
            print(f"   Previous stats: {completed_skills[skill_name]['num_success']} success, {completed_skills[skill_name]['num_failed']} failed")
            continue
        
        bddl_path = os.path.join(bddl_base_path, bddl_filename)

        # Find HDF5 file: remove _pick/_place suffix, then add "_demo.hdf5"
        # e.g., KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove_pick.bddl
        #    -> KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove_demo.hdf5
        # e.g., KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.bddl
        #    -> KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5
        base_name = bddl_filename.replace('.bddl', '')
        # Remove _pick or _place suffix if present
        if base_name.endswith('_pick'):
            base_name = base_name[:-5]  # Remove '_pick'
        elif base_name.endswith('_place'):
            base_name = base_name[:-6]  # Remove '_place'
        hdf5_file = os.path.join(args.dataset_path, f"{base_name}_demo.hdf5")

        if not os.path.exists(hdf5_file):
            print(f"⚠️  Skipping {bddl_filename}: HDF5 file not found at {hdf5_file}")
            continue

        if not os.path.exists(bddl_path):
            print(f"⚠️  Skipping {bddl_filename}: BDDL file not found at {bddl_path}")
            continue

        # Process skill
        stats = process_skill(
            skill_name,
            hdf5_file,
            bddl_path,
            target_object_list,
            skill_type,
            grasped_object_list,
            args.output_path,
            args
        )
        all_stats.append(stats)
        
        # Track completed skill and save progress
        completed_skills[skill_name] = {
            'num_success': stats['num_success'],
            'num_failed': stats['num_failed']
        }
        if all_skill_names:
            save_progress_stats(args.output_path, completed_skills, all_skill_names)

    # Print summary
    print(f"\n{'='*80}")
    print(f"🎉 SUMMARY")
    print(f"{'='*80}")
    print(f"Processed {len(all_stats)} skills:")
    total_success = sum(s['num_success'] for s in all_stats)
    total_failed = sum(s['num_failed'] for s in all_stats)
    for stats in all_stats:
        print(f"  {stats['skill_name']}: {stats['num_success']} success, {stats['num_failed']} failed")
    print(f"\nTotal: {total_success} successful demos, {total_failed} failed demos")
    
    # Calculate and print average time per demo in debug mode
    if args.debug_mode:
        all_demo_times = []
        for stats in all_stats:
            if 'demo_times' in stats and len(stats['demo_times']) > 0:
                all_demo_times.extend(stats['demo_times'])
        
        if len(all_demo_times) > 0:
            avg_time = np.mean(all_demo_times)
            min_time = np.min(all_demo_times)
            max_time = np.max(all_demo_times)
            total_time = np.sum(all_demo_times)
            print(f"\n{'='*80}")
            print(f"⏱️  TIMING STATISTICS (Debug Mode)")
            print(f"{'='*80}")
            print(f"Total demos processed: {len(all_demo_times)}")
            print(f"Average time per demo: {avg_time:.2f} seconds")
            print(f"Min time per demo: {min_time:.2f} seconds")
            print(f"Max time per demo: {max_time:.2f} seconds")
            print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"{'='*80}")
    
    # Final progress save
    if all_skill_names:
        save_progress_stats(args.output_path, completed_skills, all_skill_names)


if __name__ == "__main__":
    main()
