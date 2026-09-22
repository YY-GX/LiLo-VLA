import numpy as np
from tracikpy import TracIKSolver
import os
from scipy.spatial.transform import Rotation as R

# Default URDF path (relative to project root)
DEFAULT_URDF_PATH = os.path.join('externals', 'tracikpy', 'data', 'franka_panda.urdf')
DEFAULT_BASE_LINK = 'panda_link0'
DEFAULT_TIP_LINK = 'panda_hand'
# True neutral pose for Franka Panda in LIBERO/robosuite
DEFAULT_NEUTRAL_QPOS = np.array([0, -0.161037389, 0.00, -2.44459747, 0.00, 2.22675220, np.pi / 4])

__all__ = [
    'solve_ik',
    'forward_kinematics',
    'parse_pose',
    'parse_joints',
    'pose6d_to_matrix',
    'DEFAULT_URDF_PATH',
    'DEFAULT_BASE_LINK',
    'DEFAULT_TIP_LINK',
    'DEFAULT_NEUTRAL_QPOS',
]

def pose6d_to_matrix(pose6d):
    """
    Convert 6D pose (xyz + axis-angle) to 4x4 homogeneous matrix.
    Args:
        pose6d (np.ndarray): (6,) array, first 3 are xyz, last 3 are axis-angle.
    Returns:
        np.ndarray: (4,4) pose matrix.
    """
    pos = pose6d[:3]
    rotvec = pose6d[3:]
    rot = R.from_rotvec(rotvec).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = pos
    return mat

def solve_ik(desired_pose, current_joints=None, urdf_path=DEFAULT_URDF_PATH, base_link=DEFAULT_BASE_LINK, tip_link=DEFAULT_TIP_LINK):
    """
    Solve IK for the Franka Panda using TracIKPy.
    Args:
        desired_pose (np.ndarray): 4x4 desired end-effector pose (homogeneous matrix) in base link frame.
            If you have a 6D pose (xyz + axis-angle), use pose6d_to_matrix first.
        current_joints (np.ndarray or None): Initial guess for joint angles (7,). If None, uses robot's neutral pose.
        urdf_path (str): Path to the URDF file.
        base_link (str): Name of the base link.
        tip_link (str): Name of the tip (end-effector) link.
    Returns:
        np.ndarray or None: Solution joint angles, or None if no solution found.
    """
    ik_solver = TracIKSolver(urdf_path, base_link, tip_link)
    if current_joints is None:
        current_joints = DEFAULT_NEUTRAL_QPOS
    qout = ik_solver.ik(desired_pose, qinit=current_joints)
    return qout

def forward_kinematics(joints, urdf_path=DEFAULT_URDF_PATH, base_link=DEFAULT_BASE_LINK, tip_link=DEFAULT_TIP_LINK):
    """
    Compute forward kinematics for the Franka Panda using TracIKPy.
    Args:
        joints (np.ndarray): Joint angles (7,).
        urdf_path (str): Path to the URDF file.
        base_link (str): Name of the base link.
        tip_link (str): Name of the tip (end-effector) link.
    Returns:
        tuple: (position, quaternion) where position is (3,) and quaternion is (4,) [x,y,z,w],
               or (None, None) if forward kinematics fails.
    """
    try:
        fk_solver = TracIKSolver(urdf_path, base_link, tip_link)
        # TracIKSolver has a method to get the forward kinematics  
        # The fk method returns a 4x4 homogeneous matrix
        pose_matrix = fk_solver.fk(joints)
        
        if pose_matrix is not None:
            # Extract position and rotation from 4x4 matrix
            position = pose_matrix[:3, 3]
            rotation_matrix = pose_matrix[:3, :3]
            
            # Convert rotation matrix to quaternion
            rotation = R.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()  # [x, y, z, w]
            
            return position, quaternion
        else:
            return None, None
            
    except Exception as e:
        print(f"Forward kinematics failed: {e}")
        return None, None

def parse_pose(pose_str):
    """
    Parse a 4x4 pose from a comma-separated string of 16 values.
    Args:
        pose_str (str): Comma-separated string of 16 values (row-major order).
    Returns:
        np.ndarray: 4x4 pose matrix.
    """
    vals = [float(x) for x in pose_str.strip().split(',')]
    if len(vals) != 16:
        raise ValueError('Pose must have 16 comma-separated values (row-major 4x4 matrix).')
    return np.array(vals).reshape((4, 4))

def parse_joints(joint_str, n_joints=7):
    """
    Parse joint angles from a comma-separated string.
    Args:
        joint_str (str): Comma-separated string of joint values.
        n_joints (int): Number of joints (default 7 for Panda).
    Returns:
        np.ndarray: Joint angles.
    """
    vals = [float(x) for x in joint_str.strip().split(',')]
    if len(vals) != n_joints:
        raise ValueError(f'Expected {n_joints} joint values.')
    return np.array(vals)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TracIKPy utility: solve IK for Franka Panda.')
    parser.add_argument('--pose', type=str, required=True, help='Desired 4x4 pose as 16 comma-separated values (row-major).')
    parser.add_argument('--joints', type=str, default=None, help='Current joint angles as 7 comma-separated values.')
    parser.add_argument('--urdf', type=str, default=DEFAULT_URDF_PATH, help='Path to URDF file.')
    parser.add_argument('--base', type=str, default=DEFAULT_BASE_LINK, help='Base link name.')
    parser.add_argument('--tip', type=str, default=DEFAULT_TIP_LINK, help='Tip link name.')
    args = parser.parse_args()

    pose = parse_pose(args.pose)
    joints = parse_joints(args.joints) if args.joints is not None else DEFAULT_NEUTRAL_QPOS
    qout = solve_ik(pose, current_joints=joints, urdf_path=args.urdf, base_link=args.base, tip_link=args.tip)
    if qout is not None:
        print('IK solution:', ','.join([f'{x:.6f}' for x in qout]))
    else:
        print('No IK solution found for the given pose and initial joints.') 