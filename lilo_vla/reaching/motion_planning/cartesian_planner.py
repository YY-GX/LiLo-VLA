#!/usr/bin/env python3
"""
Motion Planner utilities for Phase 2 Implementation.

Extracted from execute_phase2_long_horizon_pipeline.py and enhanced
to fix early stopping issues that only consider position error.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R, Slerp

import sys
import os

# Note: tracik_tools import moved to _move_to_pose_ik() method (lazy import)


class MotionPlanner:
    """Motion planning and IK functionality."""

    def __init__(self, env, method: str = "cartesian_linear", num_steps: int = 400, pos_gain: float = 5.0, ori_gain: float = 5.0, pos_success_threshold: float = 0.025, ori_success_threshold: float = 0.524, verbose: bool = False):
        """
        Initialize motion planner.

        Args:
            env: LIBERO environment with robot
            method: Motion planning method ('ik_setjoint' or 'cartesian_linear')
            num_steps: Number of interpolation steps for cartesian_linear method
            pos_gain: Position gain for cartesian_linear method
            ori_gain: Orientation gain for cartesian_linear method
            pos_success_threshold: Looser position threshold for final success check (default: 0.025m = 2.5cm)
            ori_success_threshold: Looser orientation threshold for final success check (default: 0.524 rad ≈ 30°)
            verbose: Enable verbose logging
        """
        self.env = env
        self.method = method
        self.num_steps = num_steps
        self.pos_gain = pos_gain
        self.ori_gain = ori_gain
        self.pos_success_threshold = pos_success_threshold
        self.ori_success_threshold = ori_success_threshold
        self.verbose = verbose
        print(f"🔧 MotionPlanner initialized with method: {method}, steps: {num_steps}, gains: pos={pos_gain}, ori={ori_gain}, success_thresholds: pos={pos_success_threshold}m, ori={np.degrees(ori_success_threshold):.1f}°")

    def _get_gripper_action(self, skill_type: str) -> float:
        """
        Get gripper action based on skill type.
        
        Args:
            skill_type: Skill type ("pick", "place", "atomic")
            
        Returns:
            Gripper action value (-1 for open, 1 for closed)
        """
        if skill_type == "place":
            return 1.0  # Hold object during place skills
        else:
            return -1.0  # Open gripper for pick and atomic skills

    def inverse_kinematics(self, target_ee_pos: np.ndarray, target_ee_quat: np.ndarray, silent: bool = False, skill_type: str = "atomic") -> Optional[np.ndarray]:
        """
        Calculate joint positions to reach target end-effector pose using TracIK.

        Args:
            target_ee_pos: Target end-effector position [x, y, z]
            target_ee_quat: Target end-effector quaternion [x, y, z, w]
            silent: Whether to suppress debug logging

        Returns:
            Joint positions or None if IK fails
        """
        # Lazy import: only import when IK is actually needed
        from lilo_vla.utils.tracik_tools import solve_ik, pose6d_to_matrix, DEFAULT_NEUTRAL_QPOS

        try:

            # Get current joint positions as starting point
            current_qpos = self.env.sim.data.qpos.copy()
            robot_joints = current_qpos[:7]  # Assuming 7-DOF robot

            # Convert quaternion [x,y,z,w] to axis-angle (rotation vector)
            target_rot = R.from_quat(target_ee_quat)
            target_rotvec = target_rot.as_rotvec()

            # Create 6D pose (xyz + axis-angle)
            pose6d = np.concatenate([target_ee_pos, target_rotvec])

            # Apply coordinate transform from MuJoCo world coords to robot base link coords
            # Translation: MuJoCo = FK + [-0.665, 0.0, 0.816]
            # Therefore: FK = MuJoCo - [-0.665, 0.0, 0.816] = MuJoCo + [0.665, 0.0, -0.816]
            # Orientation: Flip X and Z axes as requested
            coord_transform_translation = np.array([0.665, 0.0, -0.816])
            transformed_pos = target_ee_pos + coord_transform_translation

            # Flip X and Z axes for rotation
            transformed_rotvec = target_rotvec.copy()
            # transformed_rotvec[0] = -transformed_rotvec[0]  # Flip X-axis rotation
            # transformed_rotvec[2] = -transformed_rotvec[2]  # Flip Z-axis rotation
            transformed_rotvec[0] = transformed_rotvec[0]  # Flip X-axis rotation
            transformed_rotvec[2] = transformed_rotvec[2]  # Flip Z-axis rotation

            pose6d_robot_coords = np.concatenate([transformed_pos, transformed_rotvec])

            # Convert 6D pose to 4x4 homogeneous matrix using pose6d_to_matrix
            target_pose_matrix = pose6d_to_matrix(pose6d_robot_coords)

            # 5. EE pose in robot coordinate (after applying offset and flip x and z axis)
            target_rot = R.from_quat(target_ee_quat)
            target_axis = target_rot.as_rotvec()
            if not silent:
                print(f"5️⃣ EE pose in robot coordinate: pos=[{transformed_pos[0]:.4f}, {transformed_pos[1]:.4f}, {transformed_pos[2]:.4f}], axis=[{transformed_rotvec[0]:.4f}, {transformed_rotvec[1]:.4f}, {transformed_rotvec[2]:.4f}]")

            # Solve IK using TracIK
            target_joints = solve_ik(target_pose_matrix, current_joints=robot_joints)

            if target_joints is not None:
                # Get EE pose after IK + set joint for logging
                original_qpos = self.env.sim.data.qpos.copy()
                temp_qpos = original_qpos.copy()
                temp_qpos[:7] = target_joints
                self.env.sim.data.qpos[:] = temp_qpos
                self.env.sim.forward()

                # 6. EE pose in world coordinate after IK + set joint
                final_pos, final_quat = self._get_ee_pose_from_simulation(skill_type)
                if final_pos is not None and not silent:
                    final_rot = R.from_quat(final_quat)
                    final_axis = final_rot.as_rotvec()
                    print(f"6️⃣ EE pose in world coordinate (after IK): pos=[{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}], axis=[{final_axis[0]:.4f}, {final_axis[1]:.4f}, {final_axis[2]:.4f}]")

                # Restore original joint positions
                self.env.sim.data.qpos[:] = original_qpos
                self.env.sim.forward()

                return target_joints
            else:
                # Try with neutral pose as initial guess
                target_joints = solve_ik(target_pose_matrix, current_joints=DEFAULT_NEUTRAL_QPOS)

                if target_joints is not None:
                    return target_joints
                else:
                    return None

        except Exception as e:
            if self.verbose:
                print(f"   ❌ IK failed: {e}")
                import traceback
                traceback.print_exc()
            return None

    def move_to_joints(self, target_joints: np.ndarray, skill_type: str = "atomic") -> bool:
        """
        Move robot to target joint positions.

        Args:
            target_joints: Target joint positions

        Returns:
            True if movement successful, False otherwise
        """
        try:
            print(f"🚀 Moving robot to target pose...")

            # Set joint positions directly
            current_qpos = self.env.sim.data.qpos.copy()
            current_qpos[:7] = target_joints  # Set robot joint positions

            self.env.sim.data.qpos[:] = current_qpos
            self.env.sim.forward()  # Forward kinematics to update poses

            # CRITICAL: Step environment to update observations for VLA
            gripper_action = self._get_gripper_action(skill_type)
            dummy_action = np.array([0, 0, 0, 0, 0, 0, gripper_action])  # 6-DOF + gripper action based on skill type
            obs, _, _, _ = self.env.step(dummy_action)

            # Record frame during motion planning (if video recording is enabled)
            if hasattr(self.env, 'pipeline') and hasattr(self.env.pipeline, '_record_frame'):
                self.env.pipeline._record_frame(obs)

            print(f"   ✅ Robot moved to target pose")
            return True, obs

        except Exception as e:
            if self.verbose:
                print(f"   ❌ Robot movement failed: {e}")
            return False, None

    def _get_ee_pose_from_simulation(self, skill_type: str = "atomic"):
        """
        Get ground truth end-effector pose directly from MuJoCo simulation.

        Returns:
            Tuple of (position, quaternion) or (None, None) if not found
        """
        try:
            try:
                # Try step with dummy action to get fresh observation
                gripper_action = self._get_gripper_action(skill_type)
                dummy_action = np.array([0, 0, 0, 0, 0, 0, gripper_action])  # 6-DOF + gripper based on skill
                obs, _, _, _ = self.env.step(dummy_action)
                if 'robot0_eef_pos' in obs and 'robot0_eef_quat' in obs:
                    position = obs['robot0_eef_pos'].copy()
                    quaternion = obs['robot0_eef_quat'].copy()
                    return position, quaternion
            except Exception as e:
                if self.verbose:
                    print(f"Debug: Failed to get EE pose from step: {e}")

            return None, None

        except Exception as e:
            if self.verbose:
                print(f"Error getting EE pose from simulation: {e}")
            return None, None

    def _validate_ik_solution(self, current_joints: np.ndarray, desired_pos: np.ndarray, desired_quat: np.ndarray, calculated_joints: np.ndarray):
        """
        Validate IK solution by comparing desired pose with achieved pose from MuJoCo simulation.
        This method is now simplified for visualization mode.
        """
        pass  # Removed detailed validation logging as requested

    def move_to_pose(self, target_ee_pos: np.ndarray, target_ee_quat: np.ndarray, num_steps: int = None, position_threshold: float = 0.02, orientation_threshold: float = 0.5, collect_data: bool = False, skill_type: str = "atomic") -> tuple:
        """
        Move robot to target end-effector pose using the configured motion planning method.

        Args:
            target_ee_pos: Target end-effector position [x, y, z]
            target_ee_quat: Target end-effector quaternion [x, y, z, w]
            num_steps: Number of interpolation steps for cartesian_linear method (default: 400)
            position_threshold: Position convergence threshold in meters (default: 2cm)
            orientation_threshold: Orientation convergence threshold in radians (default: 0.5 rad ≈ 28.6°)
            collect_data: If True, collect and return all intermediate steps during motion planning
            skill_type: Skill type ("pick", "place", "atomic") - affects gripper action during motion planning

        Returns:
            Tuple of (success: bool, obs: dict) or (success: bool, collected_steps: List[Dict]) if collect_data=True
        """
        if num_steps is None:
            num_steps = self.num_steps

        if self.method == "ik_setjoint":
            return self._move_to_pose_ik_setjoint(target_ee_pos, target_ee_quat, collect_data, skill_type)
        elif self.method == "cartesian_linear":
            return self._move_to_pose_cartesian_linear(target_ee_pos, target_ee_quat, num_steps, position_threshold, orientation_threshold, collect_data, skill_type)
        else:
            raise ValueError(f"Unknown motion planner method: {self.method}")

    def _move_to_pose_ik_setjoint(self, target_ee_pos: np.ndarray, target_ee_quat: np.ndarray, collect_data: bool = False, skill_type: str = "atomic") -> tuple:
        """
        Move to pose using IK + setjoint method (original implementation).

        Args:
            target_ee_pos: Target end-effector position [x, y, z]
            target_ee_quat: Target end-effector quaternion [x, y, z, w]
            collect_data: If True, collect and return all intermediate steps

        Returns:
            Tuple of (success: bool, obs: dict) or (success: bool, collected_steps: List[Dict])
        """
        # Use existing IK + setjoint implementation
        target_joints = self.inverse_kinematics(target_ee_pos, target_ee_quat, silent=not self.verbose, skill_type=skill_type)
        if target_joints is None:
            if self.verbose:
                print(f"   ❌ IK failed for target pose")
            return False, None

        if collect_data:
            # For IK method, we can't easily collect intermediate steps
            # Just do the movement and collect the final step
            success, obs = self.move_to_joints(target_joints, skill_type)
            if success and obs is not None:
                # Import collect_step_data function
                from lilo_vla.utils.pose_shift_utils import collect_step_data
                # Create a dummy action for the IK movement
                dummy_action = np.zeros(7)
                step_data = collect_step_data(dummy_action, obs, 0.0, False, self.env)
                return success, [step_data]
            return success, []
        else:
            return self.move_to_joints(target_joints, skill_type)

    def _move_to_pose_cartesian_linear(self, target_ee_pos: np.ndarray, target_ee_quat: np.ndarray, num_steps: int = 400, position_threshold: float = 0.02, orientation_threshold: float = 0.5, collect_data: bool = False, skill_type: str = "atomic") -> tuple:
        """
        Move to pose using Cartesian Linear Interpolation.

        Uses linear interpolation for position and SLERP for orientation,
        then converts each waypoint to OSC delta control via env.step().

        FIXED: Early stopping now considers both position AND orientation errors.

        Args:
            target_ee_pos: Target end-effector position [x, y, z]
            target_ee_quat: Target end-effector quaternion [x, y, z, w]
            num_steps: Number of interpolation steps
            position_threshold: Position convergence threshold in meters (default: 2cm)
            orientation_threshold: Orientation convergence threshold in radians (default: 0.5 rad)
            collect_data: If True, collect and return all intermediate steps

        Returns:
            Tuple of (success: bool, obs: dict) or (success: bool, collected_steps: List[Dict])
        """
        print(f"🎯 Cartesian Linear Interpolation to target pose ({num_steps} steps)")

        try:
            # Get current end-effector pose
            current_pos, current_quat = self._get_ee_pose_from_simulation(skill_type)
            if current_pos is None or current_quat is None:
                if self.verbose:
                    print(f"   ❌ Failed to get current EE pose")
                return False, None

            # Convert quaternions to axis-angle for printing
            current_rot = R.from_quat(current_quat)
            current_axis = current_rot.as_rotvec()
            current_axis_deg = np.degrees(current_axis)

            target_rot = R.from_quat(target_ee_quat)
            target_axis = target_rot.as_rotvec()
            target_axis_deg = np.degrees(target_axis)

            print(f"   📍 Current EE: pos=[{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}], axis=[{current_axis_deg[0]:.1f}°, {current_axis_deg[1]:.1f}°, {current_axis_deg[2]:.1f}°]")
            print(f"   🎯 Target EE:  pos=[{target_ee_pos[0]:.4f}, {target_ee_pos[1]:.4f}, {target_ee_pos[2]:.4f}], axis=[{target_axis_deg[0]:.1f}°, {target_axis_deg[1]:.1f}°, {target_axis_deg[2]:.1f}°]")

            # Calculate and print initial distance to target
            initial_pos_error = np.linalg.norm(current_pos - target_ee_pos)
            current_rot = R.from_quat(current_quat)
            target_rot = R.from_quat(target_ee_quat)
            relative_rot = target_rot * current_rot.inv()
            initial_orient_error = relative_rot.magnitude()
            print(f"   📏 Initial distance to target: pos_error={initial_pos_error:.4f}m, orient_error={np.degrees(initial_orient_error):.1f}°")

            # Generate interpolated waypoints
            waypoints_pos = []
            waypoints_quat = []

            # Linear interpolation for position
            for i in range(num_steps + 1):
                t = i / num_steps
                interp_pos = current_pos + t * (target_ee_pos - current_pos)
                waypoints_pos.append(interp_pos)

            # SLERP for orientation interpolation
            # Convert quaternions to scipy Rotation objects
            current_rot = R.from_quat(current_quat)  # [x, y, z, w]
            target_rot = R.from_quat(target_ee_quat)  # [x, y, z, w]

            # Create SLERP interpolator
            key_times = [0, 1]
            key_rotations = R.from_quat([current_quat, target_ee_quat])
            slerp = Slerp(key_times, key_rotations)

            # Generate interpolated orientations
            for i in range(num_steps + 1):
                t = i / num_steps
                interp_rot = slerp(t)
                interp_quat = interp_rot.as_quat()  # [x, y, z, w]
                waypoints_quat.append(interp_quat)

            # Execute waypoints using OSC delta control
            obs = None
            collected_steps = []
            # Use the provided thresholds (passed as parameters)

            for i, (waypoint_pos, waypoint_quat) in enumerate(zip(waypoints_pos, waypoints_quat)):
                # Skip the first waypoint (current pose)
                if i == 0:
                    continue

                # Get current pose again for delta calculation
                curr_pos, curr_quat = self._get_ee_pose_from_simulation(skill_type)
                if curr_pos is None or curr_quat is None:
                    if self.verbose:
                        print(f"   ❌ Failed to get current EE pose at step {i}")
                    return (False, obs) if not collect_data else (False, collected_steps)

                # FIXED: Check for convergence considering BOTH position AND orientation errors
                position_error = np.linalg.norm(curr_pos - target_ee_pos)

                # Calculate orientation error (angular distance)
                curr_rot = R.from_quat(curr_quat)
                target_rot = R.from_quat(target_ee_quat)
                relative_rot = target_rot * curr_rot.inv()
                orientation_error = relative_rot.magnitude()  # Angular distance in radians

                if position_error < position_threshold and orientation_error < orientation_threshold:
                    print(f"   🎯 Converged at step {i}/{num_steps}: pos_error={position_error:.4f}m < {position_threshold:.4f}m, "
                          f"orient_error={np.degrees(orientation_error):.1f}° < {np.degrees(orientation_threshold):.1f}°")
                    obs, _, _, _ = self.env.step(np.zeros(7))  # Final dummy step to get fresh observation

                    # Record frame during motion planning (if video recording is enabled)
                    if hasattr(self.env, 'pipeline') and hasattr(self.env.pipeline, '_record_frame'):
                        self.env.pipeline._record_frame(obs)

                    # Collect final step if data collection is enabled
                    if collect_data and obs is not None:
                        from lilo_vla.utils.pose_shift_utils import collect_step_data
                        step_data = collect_step_data(np.zeros(7), obs, 0.0, False, self.env)
                        collected_steps.append(step_data)

                    break

                # Calculate position delta
                pos_delta = waypoint_pos - curr_pos

                # Calculate orientation delta (axis-angle)
                curr_rot = R.from_quat(curr_quat)
                waypoint_rot = R.from_quat(waypoint_quat)
                # Relative rotation: R_target = R_delta * R_current => R_delta = R_target * R_current^-1
                delta_rot = waypoint_rot * curr_rot.inv()
                delta_axis_angle = delta_rot.as_rotvec()

                # Scale down the deltas for smooth motion (OSC gain)
                pos_gain = self.pos_gain  # Position gain from instance variable
                ori_gain = self.ori_gain  # Orientation gain from instance variable

                pos_delta_scaled = pos_delta * pos_gain
                ori_delta_scaled = delta_axis_angle * ori_gain

                # Create OSC delta action: [dx, dy, dz, drx, dry, drz, gripper]
                # Set gripper action based on skill type
                gripper_action = self._get_gripper_action(skill_type)
                delta_action = np.concatenate([pos_delta_scaled, ori_delta_scaled, [gripper_action]])

                # Execute action and check for episode termination
                try:
                    obs, reward, done, info = self.env.step(delta_action)

                    # Record frame during motion planning (if video recording is enabled)
                    # Access pipeline through the environment's parent pipeline
                    if hasattr(self.env, 'pipeline') and hasattr(self.env.pipeline, '_record_frame'):
                        self.env.pipeline._record_frame(obs)

                    # Collect step data if enabled
                    if collect_data and obs is not None:
                        from lilo_vla.utils.pose_shift_utils import collect_step_data
                        step_data = collect_step_data(delta_action, obs, reward, done, self.env)
                        collected_steps.append(step_data)

                    # Check if episode terminated during motion planning
                    if done:
                        if self.verbose:
                            print(f"   ⚠️ Environment episode terminated at step {i}/{num_steps}")
                            print(f"   🔄 Resetting environment to continue...")
                        # Reset environment and break out of the interpolation
                        obs = self.env.reset()
                        return (False, obs) if not collect_data else (False, collected_steps)

                except Exception as e:
                    if self.verbose:
                        print(f"   ❌ Motion planning failed at step {i}/{num_steps}: {e}")
                    return (False, obs) if not collect_data else (False, collected_steps)

                # Debug logging every 10 steps
                if i % 10 == 0 or i == num_steps:
                    curr_pos_check, curr_quat_check = self._get_ee_pose_from_simulation(skill_type)
                    if curr_pos_check is not None and curr_quat_check is not None:
                        curr_rot_check = R.from_quat(curr_quat_check)
                        curr_axis_check = curr_rot_check.as_rotvec()
                        curr_axis_deg_check = np.degrees(curr_axis_check)

                        # Calculate distance to target
                        step_pos_error = np.linalg.norm(curr_pos_check - target_ee_pos)
                        target_rot_check = R.from_quat(target_ee_quat)
                        relative_rot_check = target_rot_check * curr_rot_check.inv()
                        step_orient_error = relative_rot_check.magnitude()

                        print(f"   Step {i:2d}/{num_steps}: pos=[{curr_pos_check[0]:.4f}, {curr_pos_check[1]:.4f}, {curr_pos_check[2]:.4f}], axis=[{curr_axis_deg_check[0]:.1f}°, {curr_axis_deg_check[1]:.1f}°, {curr_axis_deg_check[2]:.1f}°], "
                              f"dist_to_target: {step_pos_error:.4f}m, {np.degrees(step_orient_error):.1f}°")

            # Final pose check - determine success based on threshold
            final_pos, final_quat = self._get_ee_pose_from_simulation(skill_type)
            if final_pos is not None:
                pos_error = np.linalg.norm(final_pos - target_ee_pos)

                # Calculate final orientation error
                final_rot = R.from_quat(final_quat)
                target_rot = R.from_quat(target_ee_quat)
                relative_rot = target_rot * final_rot.inv()
                orient_error = relative_rot.magnitude()

                print(f"   📏 Final EE: pos=[{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}], "
                      f"pos_error={pos_error:.4f}m, orient_error={np.degrees(orient_error):.1f}°")

                # Check if within success threshold
                # Use configurable success thresholds (typically looser than convergence thresholds)
                # This allows convergence to stop iteration early while still accepting slightly looser final poses

                if pos_error < self.pos_success_threshold and orient_error < self.ori_success_threshold:
                    print(f"   ✅ Motion planning succeeded: pos_error={pos_error:.4f}m < {self.pos_success_threshold:.4f}m, "
                          f"orient_error={np.degrees(orient_error):.1f}° < {np.degrees(self.ori_success_threshold):.1f}°")
                    return (True, obs) if not collect_data else (True, collected_steps)
                else:
                    print(f"   ❌ Motion planning failed: pos_error={pos_error:.4f}m >= {self.pos_success_threshold:.4f}m or "
                          f"orient_error={np.degrees(orient_error):.1f}° >= {np.degrees(self.ori_success_threshold):.1f}°")
                    return (False, obs) if not collect_data else (False, collected_steps)

            # Could not get final pose
            print(f"   ❌ Motion planning failed: could not get final pose")
            return (False, obs) if not collect_data else (False, collected_steps)

        except Exception as e:
            if self.verbose:
                print(f"   ❌ Cartesian Linear Interpolation failed: {e}")
                import traceback
                traceback.print_exc()
            return (False, None) if not collect_data else (False, [])