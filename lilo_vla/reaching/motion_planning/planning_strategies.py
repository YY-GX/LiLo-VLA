"""
Planning strategies for MPlib motion planning with goal pose perturbation fallback.

This module provides:
1. Three-strategy planning flow (OLD CODE → Adaptive → Fixed waypoints)
2. Pipeline with perturbation fallback (try original pose, then perturb up to N times)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from mplib import Pose

# Import subgoal planning utilities
from lilo_vla.reaching.motion_planning.mplib_planner_core import (
    plan_with_fixed_waypoints,
    plan_with_adaptive_subdivision,
)


def perturb_pose(pose_4x4: np.ndarray,
                 perturbation_idx: int = 0,
                 use_fixed_perturbations: bool = False,
                 xy_range: float = 0.04,
                 z_range: float = 0.02,
                 ori_range: float = 15.0) -> tuple:
    """
    Apply perturbation to SE(3) pose (fixed pattern or random).

    Args:
        pose_4x4: Target pose as 4x4 transformation matrix
        perturbation_idx: Perturbation index (0-based)
        use_fixed_perturbations: If True, try fixed perturbations before random (default: False)
        xy_range: XY shift range in meters (default: 0.04m = 4cm)
        z_range: Z shift range in meters (default: 0.02m = 2cm)
        ori_range: Orientation shift range in degrees (default: 15.0°)

    Returns:
        (perturbed_pose_4x4, shift_info) tuple where:
        - perturbed_pose_4x4: Perturbed 4x4 transformation matrix
        - shift_info: Dict with 'xy_shift', 'z_shift', 'ori_shift', 'type' for logging
    """
    # Define fixed perturbations to try before random sampling
    # These are common failure cases: z-axis lift (clear collisions), x-axis shift with/without rotation
    FIXED_PERTURBATIONS = [
        {'x': 0.0, 'y': 0.0, 'z': 0.02, 'ori': 0.0},    # +Z shift only (lift 2cm to clear collisions)
        {'x': 0.04, 'y': 0.0, 'z': 0.0, 'ori': 0.0},    # +X shift only
        {'x': -0.04, 'y': 0.0, 'z': 0.0, 'ori': 0.0},   # -X shift only
        {'x': 0.04, 'y': 0.0, 'z': 0.0, 'ori': 10.0},   # +X shift + rotate +10°
        {'x': 0.04, 'y': 0.0, 'z': 0.0, 'ori': -10.0},  # +X shift + rotate -10°
    ]

    # Extract position and orientation
    position = pose_4x4[:3, 3].copy()
    rotation_matrix = pose_4x4[:3, :3].copy()

    # Determine if using fixed or random perturbation
    if use_fixed_perturbations and perturbation_idx < len(FIXED_PERTURBATIONS):
        # Use fixed perturbation
        fixed_pert = FIXED_PERTURBATIONS[perturbation_idx]
        xy_shift = np.array([fixed_pert['x'], fixed_pert['y']])
        z_shift = fixed_pert['z']
        ori_angle_deg = fixed_pert['ori']
        perturbation_type = f"fixed_{perturbation_idx + 1}"
    else:
        # Use random perturbation
        xy_shift = np.random.uniform(-xy_range, xy_range, size=2)
        z_shift = np.random.uniform(-z_range, z_range)
        ori_angle_deg = np.random.uniform(-ori_range, ori_range)
        perturbation_type = "random"

    # Apply position shift
    position[0] += xy_shift[0]
    position[1] += xy_shift[1]
    position[2] += z_shift

    # Apply orientation shift (rotation around z-axis)
    ori_angle_rad = np.deg2rad(ori_angle_deg)

    # Create rotation around z-axis
    z_axis_rot = R.from_rotvec([0, 0, ori_angle_rad])

    # Apply rotation: new_rot = z_axis_rot * original_rot
    original_rot = R.from_matrix(rotation_matrix)
    perturbed_rot = z_axis_rot * original_rot
    perturbed_rotation_matrix = perturbed_rot.as_matrix()

    # Reconstruct 4x4 pose matrix
    perturbed_pose = np.eye(4)
    perturbed_pose[:3, :3] = perturbed_rotation_matrix
    perturbed_pose[:3, 3] = position

    # Create shift info for logging
    shift_info = {
        'xy_shift': xy_shift.tolist(),
        'z_shift': float(z_shift),
        'ori_shift': float(ori_angle_deg),
        'type': perturbation_type
    }

    return perturbed_pose, shift_info


def plan_with_three_strategies(planner, q0, target_pose,
                                time_step=0.01,
                                mv_link_to_ctrl=None,
                                planning_time=10.0,
                                verbose=False):
    """
    Execute three-strategy planning flow in sequence.

    Tries strategies in order until one succeeds:
    1. OLD CODE: Simple retry with increased planning budget (3 attempts)
    2. Strategy B: Adaptive subdivision (max_depth=3)
    3. Strategy A: Fixed waypoints (num_waypoints=2)

    Args:
        planner: MPLibPlanner instance
        q0: Initial joint configuration (7D or 9D)
        target_pose: Target pose in controller frame (4x4 matrix)
        time_step: Time step for planning (default: 0.01)
        mv_link_to_ctrl: Transformation from controller to move_group link
        planning_time: Planning time budget for RRT (default: 10.0s)
        verbose: Verbose logging (default: False)

    Returns:
        Planning result dict with keys:
        - "status": "Success" or "Failed"
        - "score": Success score (0.0 or 1.0)
        - "position": Joint trajectory (N x 7)
        - "cartesian": Cartesian trajectory (N x 4 x 4)
    """
    if len(q0) == 7:
        q0_full = np.concatenate([q0, np.zeros(2)], axis=0)
    else:
        q0_full = q0

    arm_dim = 7

    # Convert target pose from controller to move_group link
    mv_link_pose = target_pose @ mv_link_to_ctrl
    mv_link_pose_mplib = Pose(mv_link_pose)

    # ========================================
    # STRATEGY 1: OLD CODE - Simple retry with increased planning budget
    # ========================================
    if verbose:
        print(f"   Attempting OLD CODE: Simple retry with increased planning budget...")

    res = None
    for attempt in range(3):
        res = planner.planner.plan_pose(
            mv_link_pose_mplib, q0_full,
            time_step=time_step,
            wrt_world=True,
            planning_time=planning_time
        )

        if res["status"] == "Success":
            res["score"] = 1.0
            res["cartesian"] = planner.convert_joint_to_ctrl_poses(res["position"])
            if verbose:
                print(f"   ✅ Planning succeeded with OLD CODE (simple retry, attempt {attempt + 1}/3)")
            res["position"] = np.array(res["position"])[:, :arm_dim]
            return res
        else:
            if verbose:
                print(f"   MPLib planning attempt {attempt + 1}/3 failed: {res['status']}")

    if verbose:
        print(f"   ❌ OLD CODE failed after 3 attempts")

    # ========================================
    # STRATEGY 2: Adaptive subdivision
    # ========================================
    if verbose:
        print(f"   Attempting Strategy B: Adaptive subdivision...")

    res = plan_with_adaptive_subdivision(
        planner, q0_full, target_pose,
        max_depth=3,
        time_step=time_step,
        mv_link_to_ctrl=mv_link_to_ctrl,
        planning_time=planning_time,
        verbose=verbose
    )

    if res["status"] == "Success":
        if verbose:
            print(f"   ✅ Planning succeeded with STRATEGY B (adaptive subdivision)")
        return res
    else:
        if verbose:
            print(f"   ❌ Strategy B failed")

    # ========================================
    # STRATEGY 3: Fixed waypoints
    # ========================================
    if verbose:
        print(f"   Attempting Strategy A: Fixed waypoints...")

    res = plan_with_fixed_waypoints(
        planner, q0_full, target_pose,
        num_waypoints=2,
        time_step=time_step,
        mv_link_to_ctrl=mv_link_to_ctrl,
        planning_time=planning_time,
        verbose=verbose
    )

    if res["status"] == "Success":
        if verbose:
            print(f"   ✅ Planning succeeded with STRATEGY A (fixed waypoints)")
        return res
    else:
        if verbose:
            print(f"   ❌ All three strategies failed (OLD CODE, Strategy B, Strategy A)")
        return {
            "status": "Failed",
            "score": 0.0,
            "position": [q0_full[:arm_dim]],
            "cartesian": []
        }


def plan_pose_with_perturbation_fallback(planner, q0, target_pose,
                                         max_perturbations=20,
                                         use_fixed_perturbations=False,
                                         xy_range=0.04,
                                         z_range=0.02,
                                         ori_range=15.0,
                                         time_step=0.01,
                                         mv_link_to_ctrl=None,
                                         planning_time=10.0,
                                         verbose=False):
    """
    Pipeline: Try three-strategy planning, then fall back to goal pose perturbation.

    Algorithm:
    1. Try plan_with_three_strategies() on original target pose
    2. If failed, loop up to max_perturbations times:
       a. Perturb goal pose (fixed pattern first if enabled, then random)
       b. Try plan_with_three_strategies() on perturbed pose
       c. If success → return
    3. If all attempts exhausted → return failure

    Args:
        planner: MPLibPlanner instance
        q0: Initial joint configuration (7D or 9D)
        target_pose: Target pose in controller frame (4x4 matrix)
        max_perturbations: Max number of perturbation attempts (default: 20)
        use_fixed_perturbations: If True, try 5 fixed perturbations before random (default: False)
        xy_range: XY perturbation range in meters (default: 0.04m = 4cm)
        z_range: Z perturbation range in meters (default: 0.02m = 2cm)
        ori_range: Orientation perturbation range in degrees (default: 15.0°)
        time_step: Time step for planning (default: 0.01)
        mv_link_to_ctrl: Transformation from controller to move_group link
        planning_time: Planning time budget for RRT (default: 10.0s)
        verbose: Verbose logging (default: False)

    Returns:
        Planning result dict with keys:
        - "status": "Success" or "Failed"
        - "score": Success score (0.0 or 1.0)
        - "position": Joint trajectory (N x 7)
        - "cartesian": Cartesian trajectory (N x 4 x 4)
    """
    if len(q0) == 7:
        q0_full = np.concatenate([q0, np.zeros(2)], axis=0)
    else:
        q0_full = q0

    arm_dim = 7

    # ========================================
    # Step 1: Try original target pose with three strategies
    # ========================================
    if verbose:
        print(f"   Planning to original target pose...")

    res = plan_with_three_strategies(
        planner, q0_full, target_pose,
        time_step=time_step,
        mv_link_to_ctrl=mv_link_to_ctrl,
        planning_time=planning_time,
        verbose=verbose
    )

    if res["status"] == "Success":
        return res

    # ========================================
    # Step 2: Original pose failed, try perturbation fallback
    # ========================================
    if verbose:
        print(f"   ❌ Original pose failed, attempting Strategy C: Goal pose perturbation...")
        if use_fixed_perturbations:
            print(f"      Strategy: Try 5 fixed perturbations (Z+2cm, X±4cm, X+rot), then random (XY=±{xy_range*100:.1f}cm, Z=±{z_range*100:.1f}cm, Ori=±{ori_range:.1f}°)")
        else:
            print(f"      Strategy: Random perturbations only (XY=±{xy_range*100:.1f}cm, Z=±{z_range*100:.1f}cm, Ori=±{ori_range:.1f}°)")

    for perturbation_idx in range(max_perturbations):
        # Perturb goal pose (fixed pattern or random)
        perturbed_pose, shift_info = perturb_pose(
            target_pose,
            perturbation_idx=perturbation_idx,
            use_fixed_perturbations=use_fixed_perturbations,
            xy_range=xy_range,
            z_range=z_range,
            ori_range=ori_range
        )

        if verbose:
            print(f"   📏 Perturbation {perturbation_idx + 1}/{max_perturbations} ({shift_info['type']}): "
                  f"xy=[{shift_info['xy_shift'][0]:.3f}, {shift_info['xy_shift'][1]:.3f}], "
                  f"z={shift_info['z_shift']:.3f}, "
                  f"ori={shift_info['ori_shift']:.1f}°")

        # Try three strategies on perturbed pose
        res = plan_with_three_strategies(
            planner, q0_full, perturbed_pose,
            time_step=time_step,
            mv_link_to_ctrl=mv_link_to_ctrl,
            planning_time=planning_time,
            verbose=False  # Suppress per-strategy logs for perturbations
        )

        if res["status"] == "Success":
            if verbose:
                print(f"   ✅ Perturbation {perturbation_idx + 1} ({shift_info['type']}) succeeded!")
                print(f"   ✅ Planning succeeded with STRATEGY C (goal perturbation)")
            return res
        else:
            if verbose:
                print(f"   ❌ Perturbation {perturbation_idx + 1} ({shift_info['type']}) failed")

    # ========================================
    # Step 3: All attempts exhausted
    # ========================================
    if verbose:
        print(f"   ❌ All strategies failed including {max_perturbations} perturbation attempts")

    return {
        "status": "Failed",
        "score": 0.0,
        "position": [q0_full[:arm_dim]],
        "cartesian": []
    }
