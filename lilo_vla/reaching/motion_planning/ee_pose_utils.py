#!/usr/bin/env python3
"""
Utility script to compare and convert EE poses from different sources.

Two methods to get EE pose:
1. get_controller_robot_pose() - from gripper0_grip_site
2. LIBERO obs - from env._get_observations()
"""

import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

import libero.libero.envs as libero_envs
from libero.libero import benchmark, get_libero_path
from lilo_vla.reaching.motion_planning.action_utils import get_controller_robot_pose
import robosuite.utils.transform_utils as T


def get_ee_pose_method1(env):
    """
    Method 1: Get EE pose using get_controller_robot_pose (gripper0_grip_site).

    Returns:
        pos: [x, y, z] position
        quat: [x, y, z, w] quaternion (scipy format)
        rot_mat: 3x3 rotation matrix
    """
    pos, rot_mat = get_controller_robot_pose(env, "right")
    quat = T.mat2quat(rot_mat)  # [x, y, z, w]
    return pos, quat, rot_mat


def get_ee_pose_method2(env):
    """
    Method 2: Get EE pose from LIBERO observations.

    Returns:
        pos: [x, y, z] position
        quat: [x, y, z, w] quaternion (scipy format)
    """
    obs = env.env._get_observations()
    pos = obs['robot0_eef_pos'].copy()
    quat = obs['robot0_eef_quat'].copy()  # Already [x, y, z, w]
    return pos, quat


def libero_to_controller_pose(libero_pos, libero_rot_mat):
    """
    Convert LIBERO obs pose to controller pose (Method 2 → Method 1).

    Args:
        libero_pos: [x, y, z] position from LIBERO obs
        libero_rot_mat: 3x3 rotation matrix from LIBERO obs

    Returns:
        ctrl_pos: [x, y, z] position for controller (same as input)
        ctrl_rot_mat: 3x3 rotation matrix for controller
    """
    # Position is the same
    ctrl_pos = libero_pos.copy()

    # Apply transformation: R_controller = R_libero @ T^(-1)
    # Where T is the constant transformation from controller to LIBERO
    T_ctrl_to_libero = get_transformation_matrix()
    ctrl_rot_mat = libero_rot_mat @ T_ctrl_to_libero.T  # T^(-1) = T.T for rotation matrices

    # print(f"   📍 Controller: pos={ctrl_pos}")
    # print(f"   📍 Controller: quat={R.from_matrix(ctrl_rot_mat).as_quat()}")
    # print(f"   📍 Controller: axis={np.degrees(R.from_matrix(ctrl_rot_mat).as_rotvec())}°")

    return ctrl_pos, ctrl_rot_mat


def controller_to_libero_pose(ctrl_pos, ctrl_rot_mat):
    """
    Convert controller pose to LIBERO obs pose (Method 1 → Method 2).

    Args:
        ctrl_pos: [x, y, z] position from controller
        ctrl_rot_mat: 3x3 rotation matrix from controller

    Returns:
        libero_pos: [x, y, z] position for LIBERO obs (same as input)
        libero_rot_mat: 3x3 rotation matrix for LIBERO obs
    """
    # Position is the same
    libero_pos = ctrl_pos.copy()

    # Apply transformation: R_libero = R_controller @ T
    # Where T is the constant transformation from controller to LIBERO
    T_ctrl_to_libero = get_transformation_matrix()
    libero_rot_mat = ctrl_rot_mat @ T_ctrl_to_libero

    return libero_pos, libero_rot_mat


def get_transformation_matrix():
    """
    Get the constant transformation matrix from controller frame to LIBERO obs frame.

    This transformation is computed empirically from actual robot poses:
    - Controller (gripper0_grip_site): R_ctrl
    - LIBERO obs (robot0_eef_quat): R_libero

    Relationship: R_libero = R_controller @ T

    Returns:
        T: 3x3 rotation matrix representing the transformation
    """
    # Empirically measured rotation matrices at reset:
    R_ctrl = np.array([
        [-4.91829687e-04,  9.98386745e-01, -5.67773315e-02],
        [ 9.99999879e-01,  4.92624356e-04,  0.00000000e+00],
        [ 2.79698964e-05, -5.67773246e-02, -9.98386866e-01]
    ])

    R_libero = np.array([
        [ 9.98386745e-01,  4.91829687e-04, -5.67773315e-02],
        [ 4.92624356e-04, -9.99999879e-01, -8.14845695e-19],
        [-5.67773246e-02, -2.79698964e-05, -9.98386866e-01]
    ])

    # Solve for T: R_libero = R_ctrl @ T
    # Therefore: T = R_ctrl.T @ R_libero
    T = R_ctrl.T @ R_libero

    return T


def compare_ee_poses(env):
    """
    Compare EE poses from both methods and print detailed comparison.
    """
    print("="*70)
    print("EE POSE COMPARISON")
    print("="*70)

    # Method 1: get_controller_robot_pose
    m1_pos, m1_quat, m1_rot_mat = get_ee_pose_method1(env)
    m1_rot = R.from_quat(m1_quat)
    m1_axis = m1_rot.as_rotvec()
    m1_axis_deg = np.degrees(m1_axis)

    print("\n📍 Method 1: get_controller_robot_pose() [gripper0_grip_site]")
    print(f"   Position [x,y,z]:         {m1_pos}")
    print(f"   Quaternion [x,y,z,w]:     {m1_quat}")
    print(f"   Axis-angle [x,y,z] (deg): {m1_axis_deg}")
    print(f"   Rotation matrix:\n{m1_rot_mat}")

    # Method 2: LIBERO obs
    m2_pos, m2_quat = get_ee_pose_method2(env)
    m2_rot = R.from_quat(m2_quat)  # Already [x,y,z,w], use directly
    m2_axis = m2_rot.as_rotvec()
    m2_axis_deg = np.degrees(m2_axis)
    m2_rot_mat = m2_rot.as_matrix()

    print("\n📍 Method 2: LIBERO env._get_observations()")
    print(f"   Position [x,y,z]:         {m2_pos}")
    print(f"   Quaternion [x,y,z,w]:     {m2_quat}")
    print(f"   Axis-angle [x,y,z] (deg): {m2_axis_deg}")
    print(f"   Rotation matrix:\n{m2_rot_mat}")

    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    pos_diff = np.linalg.norm(m1_pos - m2_pos)
    print(f"\n📏 Position difference: {pos_diff:.10f}m ({pos_diff*100:.6f}cm)")
    print(f"   Positions match: {np.allclose(m1_pos, m2_pos)}")

    # Compare orientations
    ori_diff_rad = (m1_rot * m2_rot.inv()).magnitude()
    ori_diff_deg = np.degrees(ori_diff_rad)
    print(f"\n📐 Orientation difference: {ori_diff_deg:.4f}° ({ori_diff_rad:.6f}rad)")
    print(f"   Orientations match: {np.allclose(m1_quat, m2_quat)}")

    if not np.allclose(m1_quat, m2_quat):
        print(f"\n⚠️  ORIENTATIONS ARE DIFFERENT!")
        print(f"   This confirms that LIBERO obs orientation ≠ controller orientation")

        # Try to find the transformation
        # R_libero = R_transform @ R_controller  OR  R_libero = R_controller @ R_transform
        transform1 = m2_rot * m1_rot.inv()  # m2 = transform1 @ m1
        transform2 = m1_rot.inv() * m2_rot  # m2 = m1 @ transform2

        print(f"\n   Transformation 1 (m2 = T1 * m1):")
        print(f"      Axis-angle: {np.degrees(transform1.as_rotvec())}")
        print(f"      Matrix:\n{transform1.as_matrix()}")

        print(f"\n   Transformation 2 (m2 = m1 * T2):")
        print(f"      Axis-angle: {np.degrees(transform2.as_rotvec())}")
        print(f"      Matrix:\n{transform2.as_matrix()}")

    print("\n" + "="*70)


def main():
    """Test the utility functions."""
    # Create LIBERO environment
    benchmark_dict = benchmark.get_benchmark_dict()
    bm = benchmark_dict["long_horizon_tasks_v0"]()
    task = bm.get_task(1)

    bddl_path = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)

    env_args = {
        "bddl_file_name": bddl_path,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "ignore_done": True,
        "use_camera_obs": False,
        "control_freq": 20,
    }

    env = libero_envs.OffScreenRenderEnv(**env_args)
    env.reset()
    print("✅ Environment created and reset\n")

    # Compare poses
    compare_ee_poses(env)

    # Test conversion functions
    print("\n" + "="*70)
    print("TESTING CONVERSION FUNCTIONS")
    print("="*70)

    # Get original poses
    ctrl_pos, _, ctrl_rot_mat = get_ee_pose_method1(env)
    libero_pos, libero_quat = get_ee_pose_method2(env)
    libero_rot_mat = R.from_quat(libero_quat).as_matrix()  # Already [x,y,z,w]

    print("\n🔄 Test 1: Controller → LIBERO → Controller")
    libero_pos_conv, libero_rot_conv = controller_to_libero_pose(ctrl_pos, ctrl_rot_mat)
    ctrl_pos_back, ctrl_rot_back = libero_to_controller_pose(libero_pos_conv, libero_rot_conv)

    print(f"   Original ctrl rot:\n{ctrl_rot_mat}")
    print(f"   Converted to libero:\n{libero_rot_conv}")
    print(f"   Converted back to ctrl:\n{ctrl_rot_back}")
    print(f"   ✅ Roundtrip successful: {np.allclose(ctrl_rot_mat, ctrl_rot_back, atol=1e-6)}")

    print("\n🔄 Test 2: LIBERO → Controller → LIBERO")
    ctrl_pos_conv, ctrl_rot_conv = libero_to_controller_pose(libero_pos, libero_rot_mat)
    libero_pos_back, libero_rot_back = controller_to_libero_pose(ctrl_pos_conv, ctrl_rot_conv)

    print(f"   Original libero rot:\n{libero_rot_mat}")
    print(f"   Converted to ctrl:\n{ctrl_rot_conv}")
    print(f"   Converted back to libero:\n{libero_rot_back}")
    print(f"   ✅ Roundtrip successful: {np.allclose(libero_rot_mat, libero_rot_back, atol=1e-6)}")

    print("\n🔄 Test 3: Verify conversion matches actual data")
    libero_pos_predicted, libero_rot_predicted = controller_to_libero_pose(ctrl_pos, ctrl_rot_mat)
    print(f"   Actual libero rot:\n{libero_rot_mat}")
    print(f"   Predicted libero rot:\n{libero_rot_predicted}")
    print(f"   ✅ Conversion accurate: {np.allclose(libero_rot_mat, libero_rot_predicted, atol=1e-6)}")

    print("\n" + "="*70)

    env.close()


if __name__ == "__main__":
    main()
