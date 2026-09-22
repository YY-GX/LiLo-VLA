"""
MPLib Motion Planner API for execute_long_horizon_pipeline_recovery.py
"""

import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from PIL import Image

from lilo_vla.reaching.motion_planning.mplib_core import MPLibPlanner
from lilo_vla.reaching.motion_planning.mplib_planner_core import (
    extract_object_pointcloud,
    extract_scene_pointcloud,
    get_object_bounding_box,
    compute_grasp_pose_relative_to_move_group,
    plan_to_pose_simple,
    plan_to_pose_collision_aware,
    execute_cartesian_trajectory
)
from lilo_vla.reaching.motion_planning.action_utils import get_controller_robot_pose
from lilo_vla.reaching.motion_planning.ee_pose_utils import libero_to_controller_pose
import robosuite.utils.transform_utils as T

# Debug-dump root.  The pre-migration code wrote these into the research repo's
# `scripts/phase3/pipeline/outputs/`; in the release tree they go to <lilo-vla>/outputs/.
_LILO_ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger(__name__)


def _generate_aabb_visualization_points(aabb, grasp_pose, move_group_pose, num_points_per_edge=100):
    """
    Generate points representing an AABB (axis-aligned bounding box) at a given pose.

    Args:
        aabb: [half_w, half_h, half_d] - half-extents (distance from center to edge)
        grasp_pose: (4x4) pose of AABB relative to move_group link frame
        move_group_pose: (4x4) pose of move_group link in world frame
        num_points_per_edge: Number of points to sample along each edge (default: 10)

    Returns:
        aabb_points: (N, 3) array of points representing the AABB box edges
    """
    # Transform grasp_pose from move_group frame to world frame
    aabb_world_pose = move_group_pose @ grasp_pose

    # Extract position and rotation
    aabb_pos = aabb_world_pose[:3, 3]
    aabb_rot = aabb_world_pose[:3, :3]
    
    # AABB is already half-extents (distance from center to edge)
    half_w, half_h, half_d = aabb[0], aabb[1], aabb[2]
    
    # Generate 8 corners of the AABB in local frame
    corners_local = np.array([
        [-half_w, -half_h, -half_d],  # 0: min corner
        [ half_w, -half_h, -half_d],  # 1
        [ half_w,  half_h, -half_d],  # 2
        [-half_w,  half_h, -half_d],  # 3
        [-half_w, -half_h,  half_d],  # 4
        [ half_w, -half_h,  half_d],  # 5
        [ half_w,  half_h,  half_d],  # 6
        [-half_w,  half_h,  half_d],  # 7: max corner
    ])
    
    # Transform corners to world frame
    corners_world = (aabb_rot @ corners_local.T).T + aabb_pos
    
    # Generate points along edges of the box
    edge_points = []
    
    # Define edges (12 edges of a box)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]
    
    for start_idx, end_idx in edges:
        start = corners_world[start_idx]
        end = corners_world[end_idx]
        # Sample points along this edge
        for i in range(num_points_per_edge + 1):
            t = i / num_points_per_edge
            point = start + t * (end - start)
            edge_points.append(point)
    
    return np.array(edge_points)


def _generate_sphere_points(center, radius=0.005, num_points=10):
    """
    Generate points on a sphere surface.
    
    Args:
        center: (3,) center position of the sphere
        radius: Radius of the sphere (default: 2cm)
        num_points: Number of points to generate (default: 100)
    
    Returns:
        sphere_points: (N, 3) array of points on the sphere surface
    """
    # Generate points uniformly distributed on sphere surface using spherical coordinates
    sphere_points = []
    for i in range(num_points):
        # Uniform distribution on sphere using golden angle spiral
        theta = 2 * np.pi * i / ((1 + np.sqrt(5)) / 2)  # Golden angle
        phi = np.arccos(1 - 2 * i / num_points) if num_points > 1 else 0
        
        # Spherical to Cartesian
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        sphere_points.append(center + np.array([x, y, z]))
    
    return np.array(sphere_points)


def _save_pointcloud_ply(scene_pts, output_path, aabb_points=None, target_sphere_points=None,
                         object_center_sphere_points=None, move_group_sphere_points=None, verbose=False):
    """
    Save pointcloud to PLY file with colors.
    Scene points are white/gray.
    AABB points (if provided) are red.
    Target sphere points (if provided) are green.
    Object center sphere (if provided) is orange.
    Move group sphere (if provided) is blue.
    
    Args:
        scene_pts: (N, 3) array of scene points
        output_path: Path to save PLY file
        aabb_points: (M, 3) optional array of AABB visualization points
        target_sphere_points: (K, 3) optional array of target pose sphere points
        object_center_sphere_points: (L, 3) optional array of object center sphere points (orange)
        move_group_sphere_points: (P, 3) optional array of move group sphere points (blue)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Combine all points: scene, AABB, target sphere, object center, move group
    point_list = [scene_pts]
    color_list = [np.full((len(scene_pts), 3), [200, 200, 200], dtype=np.uint8)]  # Gray
    
    if aabb_points is not None and len(aabb_points) > 0:
        point_list.append(aabb_points)
        color_list.append(np.full((len(aabb_points), 3), [255, 0, 0], dtype=np.uint8))  # Red
    
    if target_sphere_points is not None and len(target_sphere_points) > 0:
        point_list.append(target_sphere_points)
        color_list.append(np.full((len(target_sphere_points), 3), [0, 255, 0], dtype=np.uint8))  # Green
    
    if object_center_sphere_points is not None and len(object_center_sphere_points) > 0:
        point_list.append(object_center_sphere_points)
        color_list.append(np.full((len(object_center_sphere_points), 3), [255, 165, 0], dtype=np.uint8))  # Orange
    
    if move_group_sphere_points is not None and len(move_group_sphere_points) > 0:
        point_list.append(move_group_sphere_points)
        color_list.append(np.full((len(move_group_sphere_points), 3), [0, 0, 255], dtype=np.uint8))  # Blue
    
    all_points = np.vstack(point_list) if len(point_list) > 1 else point_list[0]
    all_colors = np.vstack(color_list) if len(color_list) > 1 else color_list[0]
    total_points = len(all_points)
    
    # Write PLY file
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {total_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertices with colors
        for point, color in zip(all_points, all_colors):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
    
    info_parts = [f"{len(scene_pts)} scene points"]
    if aabb_points is not None and len(aabb_points) > 0:
        info_parts.append(f"{len(aabb_points)} AABB points")
    if target_sphere_points is not None and len(target_sphere_points) > 0:
        info_parts.append(f"{len(target_sphere_points)} target sphere points")
    if object_center_sphere_points is not None and len(object_center_sphere_points) > 0:
        info_parts.append(f"{len(object_center_sphere_points)} object center sphere points")
    if move_group_sphere_points is not None and len(move_group_sphere_points) > 0:
        info_parts.append(f"{len(move_group_sphere_points)} move group sphere points")
    info_str = ", ".join(info_parts)
    if verbose:
        print(f"   💾 Saved pointcloud: {output_path} ({info_str})")


class MPlibMotionPlanner:
    """MPLib-based motion planner with collision avoidance support."""

    def __init__(self, env, collision_aware=True, use_agentview_only=False,
                 velocity_factor=0.8, verbose=False, safety_margin=1.2,
                 erode_kernel_size=3, outlier_std_ratio=0.0, time_step=0.05):
        """
        Initialize MPLib motion planner.

        Args:
            env: LIBERO environment
            collision_aware: Enable collision avoidance (default: True)
            use_agentview_only: Use only agentview camera for scene extraction
            velocity_factor: Execution speed (0.0-1.0, default: 0.8 optimized balance)
            verbose: Enable debug logging
            safety_margin: AABB size multiplier for attached object (default: 1.1 = 10% margin)
            erode_kernel_size: Kernel size for mask erosion to remove boundary pixels with unreliable depth (default: 3)
            outlier_std_ratio: Std ratio for statistical 3D outlier removal (default: 2.0)
            time_step: Time step for trajectory planning (default: 0.05). Smaller values = more waypoints = slower execution.
        """
        self.env = env
        self.collision_aware = collision_aware
        self.use_agentview_only = use_agentview_only
        self.velocity_factor = velocity_factor
        self.verbose = verbose
        self.safety_margin = safety_margin
        self.erode_kernel_size = erode_kernel_size
        self.outlier_std_ratio = outlier_std_ratio
        self.time_step = time_step

        # Initialize MPLibPlanner
        robot_model = env.robots[0].robot_model
        base_pos = env.sim.data.get_body_xpos(robot_model.root_body)
        # check whether base_pos is [-0.56, 0.0, 0.912]
        # actual" [-0.66   0.     0.912]
        if self.verbose:
            print(f"🤖 Robot base_pos: {base_pos}")
        base_pose = np.eye(4)
        base_pose[:3, 3] = np.array(base_pos)
        self.planner = MPLibPlanner(base_pose, move_group="panda_hand", scene_resolution=0.01, time_step=self.time_step)

        if self.verbose:
            camera_str = "agentview only" if use_agentview_only else "multi-camera"
            logger.info(f"MPlibMotionPlanner initialized: collision_aware={collision_aware}, {camera_str}")

    def move_to_pose(self, target_pos, target_quat, position_threshold=0.02,
                     orientation_threshold=0.524, skill_type="atomic",
                     grasped_object_name=None, target_object=None, is_libero_pose=True,
                     save_pointcloud=False, pointcloud_output_dir=None):
        """
        Plan and execute motion to target pose.

        Args:
            target_pos: (3,) target position [x, y, z] in LIBERO frame (if is_libero_pose=True) or controller frame (if False)
            target_quat: (4,) target quaternion [w, x, y, z] format
            position_threshold: Success threshold for position (meters)
            orientation_threshold: Success threshold for orientation (radians)
            skill_type: "pick", "place", or "atomic" (determines gripper state)
            grasped_object_name: Object name if grasping (for collision avoidance)
            target_object: Target object name for segmentation masks (optional)
            is_libero_pose: If True, transform pose from LIBERO to controller frame. If False, use pose directly.
            save_pointcloud: If True, save scene pointcloud to PLY file with timestamp (default: False)

        Returns:
            success: bool - True if reached target within thresholds
            [observations, actions, rewards, dones, states]: list of lists containing:
                - observations: list[dict] - All observations during trajectory execution
                - actions: list[np.ndarray] - All actions taken
                - rewards: list[float] - All rewards received
                - dones: list[bool] - All done flags
                - states: list[np.ndarray] - All full simulation states (84-dim)
        """
        # Convert quaternion format: target_quat is [w,x,y,z] from pose_calculator, convert to [x,y,z,w] for scipy
        target_quat_xyzw = [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
        libero_rot = R.from_quat(target_quat_xyzw)
        libero_rot_mat = libero_rot.as_matrix()  # Now using correct [x,y,z,w] format
        
        # Print input target pose (before transformation)
        if self.verbose:
            libero_axis_deg = np.degrees(libero_rot.as_rotvec())
            print(f"🎯 Input target pose (LIBERO): pos=[{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}], axis_deg=[{libero_axis_deg[0]:.1f}°, {libero_axis_deg[1]:.1f}°, {libero_axis_deg[2]:.1f}°]")
        
        # Transform from LIBERO to controller pose only if needed
        if is_libero_pose:
            ctrl_pos, ctrl_rot_mat = libero_to_controller_pose(target_pos, libero_rot_mat)
        else:
            # Use input pose directly (already in controller frame)
            ctrl_pos = target_pos
            ctrl_rot_mat = libero_rot_mat

        # Build 4x4 target pose in controller frame
        target_pose = np.eye(4)
        target_pose[:3, :3] = ctrl_rot_mat
        target_pose[:3, 3] = ctrl_pos

        # Get current joint positions
        current_qpos = np.array(self.env.sim.data.qpos[:7])

        # Determine gripper action
        gripper_action = 1.0 if skill_type == "place" else -1.0

        # Plan motion
        if self.collision_aware:
            # Determine exclusions and attached object
            exclude_list = []
            attached_obj = None

            if skill_type == "place" and grasped_object_name is not None:
                exclude_list = [grasped_object_name]

                # Get controller pose FIRST (gripper0_grip_site) in world frame
                gripper_pos, gripper_rot = get_controller_robot_pose(self.env, "right")
                controller_pose = np.eye(4)
                controller_pose[:3, :3] = gripper_rot
                controller_pose[:3, 3] = gripper_pos

                # Get object bbox (should be at gripper since already grasped)
                bbox = get_object_bounding_box(self.env, grasped_object_name,
                                              safety_margin=self.safety_margin,
                                              erode_kernel_size=self.erode_kernel_size,
                                              outlier_std_ratio=self.outlier_std_ratio,
                                              verbose=self.verbose)
                object_pos = bbox["pose"][:3, 3]
                
                # Compute move_group pose in world frame
                move_group_pose = controller_pose @ self.planner.mv_link_to_ctrl

                # Transform to move_group link frame and compute relative grasp_pose
                # mv_link_to_ctrl transforms FROM controller TO move_group link (see mplib_core.py line 83)
                grasp_pose = compute_grasp_pose_relative_to_move_group(
                    bbox["pose"], controller_pose, self.planner.mv_link_to_ctrl
                )

                attached_obj = {
                    "aabb": bbox["aabb"],
                    "grasp_pose": grasp_pose,
                    "move_group_pose": move_group_pose,  # Store for visualization
                    "bbox": bbox  # Store bbox for visualization (contains object pose)
                }

            # Extract scene
            camera_names = ["agentview"] if self.use_agentview_only else ["agentview", "birdview", "sideview"]
            scene_pts = extract_scene_pointcloud(self.env, camera_names=camera_names,
                                                 exclude_object_names=exclude_list,
                                                 save_segmentation=False)

            # Transform scene points from world frame to robot base frame
            # Pointclouds are extracted in world frame (origin at [0,0,0])
            # MPlib expects pointclouds in robot base frame (origin at [-0.66, 0, 0.912])
            # COMMENTED OUT: MPLib actually expects points in world frame, not base frame
            # if len(scene_pts) > 0:
            #     print(f"   Scene pts BEFORE transform: min={scene_pts.min(axis=0)}, max={scene_pts.max(axis=0)}")
            #     base_pose_inv = np.linalg.inv(self.planner.base_pose)
            #     scene_pts = (base_pose_inv[:3, :3] @ scene_pts.T).T + base_pose_inv[:3, 3]
            #     print(f"   Scene pts AFTER transform: min={scene_pts.min(axis=0)}, max={scene_pts.max(axis=0)}")

            if self.verbose:
                logger.info(f"Scene points: {len(scene_pts)}, attached_obj: {attached_obj is not None}")

            # Plan with collision avoidance
            result = plan_to_pose_collision_aware(self.planner, current_qpos, target_pose,
                                                  scene_pts=scene_pts, attached_obj=attached_obj, verbose=self.verbose)
        
            # Save pointcloud if requested
            if save_pointcloud:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if pointcloud_output_dir is None:
                    output_dir = str(_LILO_ROOT / "outputs" / "pointclouds" / "mplib_debug")
                else:
                    output_dir = pointcloud_output_dir
                os.makedirs(output_dir, exist_ok=True)

                # Generate visualization points
                aabb_points = None
                if attached_obj is not None:
                    # CRITICAL FIX: Recompute move_group_pose at CURRENT robot state
                    # The stored move_group_pose is from when planning started, but robot has moved!
                    # Need to get CURRENT controller pose to visualize AABB at correct location
                    current_gripper_pos, current_gripper_rot = get_controller_robot_pose(self.env, "right")
                    current_controller_pose = np.eye(4)
                    current_controller_pose[:3, :3] = current_gripper_rot
                    current_controller_pose[:3, 3] = current_gripper_pos
                    current_move_group_pose = current_controller_pose @ self.planner.mv_link_to_ctrl

                    # Generate AABB visualization points using CURRENT move_group pose
                    aabb_points = _generate_aabb_visualization_points(
                        attached_obj["aabb"],
                        attached_obj["grasp_pose"],
                        current_move_group_pose,  # Use CURRENT pose, not stored one!
                        num_points_per_edge=100
                    )

                # Generate target sphere points at target pose position
                target_pos = target_pose[:3, 3]  # Target position in controller frame (world frame)
                target_sphere_points = _generate_sphere_points(target_pos, radius=0.02, num_points=100)

                # Save TWO pointclouds for comparison:
                # 1. WITH attached object (for visualization - to verify bbox correctness)
                # 2. WITHOUT attached object (used for planning)
                if attached_obj is not None:
                    # Extract scene WITH attached object for visualization (no filtering)
                    # This includes robot arm, bowl, and all scene geometry
                    scene_pts_with_attached = extract_scene_pointcloud(
                        self.env, camera_names=camera_names, exclude_object_names=[], no_filtering=True
                    )

                    # Generate orange sphere at object center (bbox["pose"] position)
                    bbox = attached_obj.get("bbox")
                    if bbox is not None:
                        object_center_pos = bbox["pose"][:3, 3]
                        object_center_sphere_points = _generate_sphere_points(object_center_pos, radius=0.02, num_points=100)
                    else:
                        object_center_sphere_points = None

                    # Generate blue sphere at move_group pose
                    move_group_pos = current_move_group_pose[:3, 3]
                    move_group_sphere_points = _generate_sphere_points(move_group_pos, radius=0.02, num_points=100)

                    # Save pointcloud WITH attached object (includes orange and blue spheres)
                    output_path_with = os.path.join(output_dir, f"scene_pointcloud_with_attached_{timestamp}.ply")
                    _save_pointcloud_ply(scene_pts_with_attached, output_path_with,
                                        aabb_points=aabb_points, target_sphere_points=target_sphere_points,
                                        object_center_sphere_points=object_center_sphere_points,
                                        move_group_sphere_points=move_group_sphere_points, verbose=self.verbose)

                    # Save pointcloud WITHOUT attached object (used for planning)
                    output_path_exclude = os.path.join(output_dir, f"scene_pointcloud_exclude_attached_{timestamp}.ply")
                    _save_pointcloud_ply(scene_pts, output_path_exclude,
                                        aabb_points=aabb_points, target_sphere_points=target_sphere_points, verbose=self.verbose)
                else:
                    # No attached object, just save one pointcloud
                    output_path = os.path.join(output_dir, f"scene_pointcloud_{timestamp}.ply")
                    _save_pointcloud_ply(scene_pts, output_path,
                                        aabb_points=aabb_points, target_sphere_points=target_sphere_points, verbose=self.verbose)
                
                # Save camera images
                camera_names = ["agentview"] if self.use_agentview_only else ["agentview", "birdview", "sideview"]
                for camera_name in camera_names:
                    try:
                        render_result = self.env.sim.render(width=512, height=512, camera_name=camera_name, depth=False)
                        if isinstance(render_result, tuple):
                            rgb_image = render_result[0]
                        else:
                            rgb_image = render_result
                        
                        if rgb_image is not None:
                            rgb_image = np.array(rgb_image)
                            if rgb_image.dtype != np.uint8:
                                rgb_image = (rgb_image * 255).astype(np.uint8)
                            if len(rgb_image.shape) == 2:
                                rgb_image = np.stack([rgb_image, rgb_image, rgb_image], axis=-1)
                            elif len(rgb_image.shape) == 3 and rgb_image.shape[2] == 1:
                                rgb_image = np.repeat(rgb_image, 3, axis=2)
                            
                            # Save image
                            image_path = os.path.join(output_dir, f"{camera_name}_{timestamp}.png")
                            Image.fromarray(rgb_image).save(image_path)
                            if self.verbose:
                                print(f"   💾 Saved {camera_name} image: {image_path}")
                    except Exception as e:
                        continue
        else:
            # Plan without collision avoidance
            result = plan_to_pose_simple(self.planner, current_qpos, target_pose, verbose=self.verbose)

        # Check planning success
        if not result["success"]:
            print(">> MPLib Planning: FAILED")
            if self.verbose:
                logger.warning("Planning failed")
            current_obs = self.env.env._get_observations()
            current_state = self.env.sim.get_state().flatten()
            # Return consistent format: success, [observations, actions, rewards, dones, states]
            return False, [[current_obs], [], [], [], [current_state]]
        else:
            # Always print planning success (one line) - exception to verbose rule
            print(">> MPLib Planning: SUCCESS")

        # Execute trajectory
        num_waypoints = len(result["cart_traj"])
        gripper_actions = [gripper_action] * num_waypoints
        exec_result = execute_cartesian_trajectory(self.env, result["cart_traj"], gripper_actions,
                                                   arm_name="right", velocity_factor=self.velocity_factor,
                                                   target_object=target_object, verbose=self.verbose)

        # CRITICAL: Detach object after placing to clean up MPLib state
        if self.collision_aware and skill_type == "place" and grasped_object_name is not None:
            self.planner.detach_obj(also_remove=True)
            if self.verbose:
                logger.info(f"Detached object '{grasped_object_name}' after placement")

        # Check execution success
        pos_err = exec_result["position_error"]
        ori_err = exec_result["rotation_error"]
        success = (pos_err < position_threshold) and (ori_err < orientation_threshold)

        # Always print execution result (one line) - exception to verbose rule
        print(f">> MPLib Execution: SUCCESS={success}; pos_err={pos_err:.4f}m, ori_err={np.degrees(ori_err):.1f}°")
        if self.verbose:
            logger.info(f"Execution: pos_err={pos_err:.4f}m, ori_err={np.degrees(ori_err):.1f}°, success={success}")

        # Return success status and list of [observations, actions, rewards, dones, states]
        return success, [exec_result["observations"], exec_result["actions"], exec_result["rewards"], exec_result["dones"], exec_result["states"]]
