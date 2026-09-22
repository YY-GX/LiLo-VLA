#!/usr/bin/env python3
"""
Local pose calculation tools for LIBERO environment.

Provides utilities for calculating object poses and "above-object" end-effector poses
for manipulation tasks, with special handling for cabinets, drawers, and other objects.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R

from lilo_vla.reaching.motion_planning.mplib_planner_core import get_object_bounding_box


# ==============================================================================
# Constants
# ==============================================================================

# Default gripper orientation (pointing down, wxyz format)
REST_QUAT = np.array([-6.99529601e-06, 9.99596605e-01, 2.46212834e-04, -2.84001205e-02])


# ==============================================================================
# Low-level Helpers
# ==============================================================================

def get_object_pose(env_or_sim, obj_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get object position and quaternion from simulation.

    Args:
        env_or_sim: LIBERO environment or mujoco sim object
        obj_name: Object name (e.g., 'black_bowl_1_main')

    Returns:
        (position, quaternion_xyzw) or (None, None) if object not found
    """
    # Handle both env and sim inputs
    sim = env_or_sim.sim if hasattr(env_or_sim, 'sim') else env_or_sim

    # Find object body ID
    try:
        body_id = sim.model.body_name2id(obj_name)
    except:
        return None, None

    # Get position from xpos
    pos = sim.data.body_xpos[body_id].copy()

    # Get quaternion from xquat (already in xyzw format)
    quat = sim.data.body_xquat[body_id].copy()

    return pos, quat


def get_vertical_extent(bbox_info: Dict) -> float:
    """
    Calculate vertical (Z-axis) half-extent of object from bounding box.

    Transforms AABB corners from object's local frame to world frame
    and computes distance from center to top surface.

    Args:
        bbox_info: Dict with 'aabb' (half-extents) and 'pose' (4x4 transform)

    Returns:
        Z half-extent in world frame (meters)
    """
    aabb = np.array(bbox_info["aabb"])
    pose = bbox_info["pose"]

    # Generate 8 corners of AABB (all combinations of ±aabb)
    corners = np.array([
        [sx * aabb[0], sy * aabb[1], sz * aabb[2]]
        for sx in [-1, 1] for sy in [-1, 1] for sz in [-1, 1]
    ])

    # Transform corners to world frame
    corners_world = (pose[:3, :3] @ corners.T).T + pose[:3, 3]

    # Return half-extent from center to top
    return corners_world[:, 2].max() - pose[2, 3]


# ==============================================================================
# Main Functions
# ==============================================================================

def get_special_object_handling(
    env,
    target_object: str,
    bddl_file: str,
    default_above_height: float,
    skill_type: str = "pick",
    grasped_object_name: Optional[str] = None,
    save_debug_files: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Calculate object pose and above-height with special handling for specific objects.

    Handles special cases:
    - Wooden cabinets: Opens drawers, shifts position away from cabinet center
    - Place skills: Adjusts height for grasped object
    - Specific BDDL tasks: Custom height/position adjustments

    Args:
        env: LIBERO environment
        target_object: Target object name (e.g., 'wooden_cabinet_1_cabinet_bottom')
        bddl_file: BDDL filename for task context
        default_above_height: Default height above object (meters)
        skill_type: 'pick', 'place', or 'other'
        grasped_object_name: Object being grasped (for place skills)
        save_debug_files: Save segmentation debug files

    Returns:
        (obj_pos, obj_quat, above_height, bbox_z) tuple
        - obj_pos: Object position [x, y, z] in world frame
        - obj_quat: Object quaternion (currently None)
        - above_height: Height to position EE above object
        - bbox_z: Object's Z half-extent
    """
    # -------------------------------------------------------------------------
    # 1. Get base object bbox and position
    # -------------------------------------------------------------------------
    bbox_info = get_object_bounding_box(env, target_object, save_debug_files=save_debug_files)
    obj_pos = bbox_info["pose"][:3, 3].copy()  # Copy to allow modification
    bbox_z = get_vertical_extent(bbox_info)
    above_height = default_above_height

    # -------------------------------------------------------------------------
    # 2. Adjust above_height for grasped object (place skills only)
    # -------------------------------------------------------------------------
    if skill_type == "place" and grasped_object_name is not None:
        try:
            grasped_bbox_info = get_object_bounding_box(env, grasped_object_name, save_debug_files=save_debug_files)
            grasped_bbox_z = get_vertical_extent(grasped_bbox_info)
            # Add 2.1x the grasped object height (empirical safety margin)
            above_height += (grasped_bbox_z * 2.1)
            print(f"   📏 Added grasped object ({grasped_object_name}) bbox_z ({grasped_bbox_z:.3f}m) to above_height")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not get bbox_z for grasped object {grasped_object_name}: {e}")
            # Fallback: add 5cm if bbox fails
            above_height += 0.05

    # -------------------------------------------------------------------------
    # 3. Apply special handling rules by object/task type
    # -------------------------------------------------------------------------
    # Cabinet drawers (opening operations)
    if "wooden_cabinet" in target_object and "open" in bddl_file:
        if "bottom" in bddl_file:
            above_height = 0.25
            bbox_z = 0.0
        elif "middle" in bddl_file:
            above_height = 0.18
            bbox_z = 0.0
        elif "top" in bddl_file:
            above_height = 0.10
            bbox_z = 0.0


    # Cabinet top surface placement
    elif "on_top_of_the_cabinet_place" in bddl_file:
        above_height = 0.25
        bbox_z = 0.0

    # Generic cabinet operations (non-open): shift away from cabinet center
    elif "wooden_cabinet" in target_object:
        cabinet_id = env.sim.model.body_name2id("wooden_cabinet_1_main")
        cabinet_pos = env.sim.data.body_xpos[cabinet_id]
        # Shift 5 away from cabinet center (left or right)
        obj_pos[1] += 0.05 * np.sign(obj_pos[1] - cabinet_pos[1])

    if "place" in bddl_file and "white_mug" in target_object:
        above_height += 0.05
    if "place" in bddl_file and "yellow_and_white_mug" in target_object:
        above_height += 0.05

    # Task-specific adjustments - comment out for now
    # if "place_ketchup_in_top_drawer_of_the_cabinet" in bddl_file:
    #     above_height += 0.07
    # if "turn_on_the_stove" in bddl_file or "turn_off_the_stove" in bddl_file:
    #     above_height += 0.02
    # if "put_the_chocolate_pudding_to_the_right_of_the_plate" in bddl_file:
    #     above_height += 0.10
        # obj_pos[0] -= 0.15

    # -------------------------------------------------------------------------
    # 4. Center object in wrist camera view (5cm X-axis adjustment)
    # -------------------------------------------------------------------------
    obj_pos[0] -= 0.05

    print(f"   📏 after special handling: obj_pos: {obj_pos}, above_height: {above_height}, bbox_z: {bbox_z}")

    return obj_pos, None, above_height, bbox_z


def calculate_above_pose(
    obj_pos: np.ndarray,
    above_height: float = 0.10,
    shift: bool = False,
    xy_range: float = 0.04,
    z_range: float = 0.02,
    ori_range: float = 30.0,
    bbox_z: float = 0.0,
    z_only_positive: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Calculate end-effector pose above object with optional random shifts.

    Computes target EE position above object's top surface, optionally adding
    random XYZ position shifts and Z-axis orientation shifts for data augmentation.

    Args:
        obj_pos: Object position [x, y, z] in world frame
        above_height: Height above object's top surface (meters)
        shift: Whether to apply random shifts (for data augmentation)
        xy_range: XY shift range ±meters (e.g., 0.04 = ±4cm)
        z_range: Z shift range ±meters (e.g., 0.02 = ±2cm)
        ori_range: Orientation shift range ±degrees (e.g., 30.0 = ±30°)
        bbox_z: Object's Z half-extent (meters)
        z_only_positive: If True, only apply positive Z shift (no XY/ori shifts)

    Returns:
        (above_pos, above_quat, shift_info) tuple
        - above_pos: Target EE position [x, y, z]
        - above_quat: Target EE orientation (wxyz format)
        - shift_info: Dict with shift amounts (None if shift=False)
    """
    # -------------------------------------------------------------------------
    # 1. Calculate base above position (object top + above_height)
    # -------------------------------------------------------------------------
    above_pos = np.array([
        obj_pos[0],
        obj_pos[1],
        obj_pos[2] + bbox_z + above_height
    ])
    above_quat = REST_QUAT.copy()
    shift_info = None

    # -------------------------------------------------------------------------
    # 2. Apply random shifts (if enabled)
    # -------------------------------------------------------------------------
    if shift:
        if z_only_positive:
            # Mode A: Positive Z shift only (no XY/orientation shifts)
            xy_shift = np.array([0.0, 0.0])
            z_shift = np.random.uniform(0, z_range * 3)
            above_pos[2] += z_shift
            ori_angle_deg = 0.0

            shift_info = {
                'xy_shift': xy_shift.tolist(),
                'z_shift': float(z_shift),
                'ori_shift': float(ori_angle_deg),
                'z_only_positive': True
            }

        else:
            # Mode B: Full random shifts (XYZ position + Z-axis rotation)
            # Position shifts
            xy_shift = np.random.uniform(-xy_range, xy_range, size=2)
            z_shift = np.random.uniform(-z_range, z_range)
            above_pos[0] += xy_shift[0]
            above_pos[1] += xy_shift[1]
            above_pos[2] += z_shift

            # Orientation shift: random rotation around Z-axis (gripper rotation)
            ori_angle_deg = np.random.uniform(-ori_range, ori_range)
            ori_angle_rad = np.deg2rad(ori_angle_deg)

            # Convert REST_QUAT (wxyz) to scipy format (xyzw)
            rest_quat_xyzw = np.array([REST_QUAT[1], REST_QUAT[2], REST_QUAT[3], REST_QUAT[0]])
            rest_rot = R.from_quat(rest_quat_xyzw)

            # Apply Z-axis rotation
            z_axis_rot = R.from_rotvec([0, 0, ori_angle_rad])
            shifted_rot = z_axis_rot * rest_rot

            # Convert back to wxyz format
            shifted_quat_xyzw = shifted_rot.as_quat()
            above_quat = np.array([
                shifted_quat_xyzw[3],  # w
                shifted_quat_xyzw[0],  # x
                shifted_quat_xyzw[1],  # y
                shifted_quat_xyzw[2]   # z
            ])

            shift_info = {
                'xy_shift': xy_shift.tolist(),
                'z_shift': float(z_shift),
                'ori_shift': float(ori_angle_deg),
                'z_only_positive': False
            }

    return above_pos, above_quat, shift_info
