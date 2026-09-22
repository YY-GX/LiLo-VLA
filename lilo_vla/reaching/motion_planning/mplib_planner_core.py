"""
Core planning utilities for MPLib motion planning.
Extracted from collision_aware_planner.py and simple_planner.py.
"""

import numpy as np
import time
import cv2
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
from mplib import Pose
from scipy.spatial.transform import Rotation as R, Slerp
from lilo_vla.reaching.motion_planning.action_utils import pose_traj_to_action, get_controller_robot_pose
from lilo_vla.reaching.motion_planning.pointcloud_tools import extract_pointcloud_from_camera_adapt3r
from lilo_vla.interaction.masking import create_wrist_segmentation_mask

# Import get_object_pose from contact detector
from lilo_vla.perception.oracle import get_object_pose

# Debug-dump root.  The pre-migration code wrote these into the research repo's
# `scripts/phase3/pipeline/outputs/`; in the release tree they go to <lilo-vla>/outputs/.
_LILO_ROOT = Path(__file__).resolve().parents[3]


# ============================================
# SECTION 1: Scene Management
# ============================================

def extract_object_pointcloud(env, object_name, camera_names=["agentview", "birdview", "sideview"],
                              resolution=512, voxel_size=0.01,
                              erode_kernel_size=3, outlier_std_ratio=2.0, save_debug_files=False, verbose=False):
    """
    Extract pointcloud for a specific object using MuJoCo segmentation with noise filtering.

    Key insight: Even with clean segmentation masks, boundary pixels often have wrong depth values
    due to depth sensor artifacts, mixed pixels, and occlusion boundaries. This causes 3D points
    to be projected to wrong locations.

    Filtering strategy:
    1. Erode segmentation mask BEFORE 3D reconstruction (removes boundary pixels with unreliable depth)
    2. Statistical outlier removal on 3D pointcloud (catches remaining isolated noise points)

    Args:
        env: LIBERO environment instance
        object_name: Name of the object body in MuJoCo (or body_id as int)
        camera_names: List of camera names to use
        resolution: Image resolution for rendering
        voxel_size: Voxel grid size for downsampling
        erode_kernel_size: Kernel size for mask erosion to remove unreliable boundary pixels (default: 3)
                          Larger = more aggressive erosion (safer but loses more object surface)
                          Set to 0 to disable erosion
        outlier_std_ratio: Standard deviation ratio for 3D outlier removal (default: 2.0)
                          Points beyond mean ± (std * ratio) are removed
                          Set to 0 to disable outlier removal

    Returns:
        obj_pts: (N, 3) pointcloud containing only the target object's points
    """
    sim = env.sim

    # Get body ID and all geom IDs for this object
    if isinstance(object_name, int):
        body_id = object_name
    else:
        try:
            body_id = sim.model.body_name2id(object_name)
        except:
            if verbose:
                print(f"   ⚠️  WARNING: Object '{object_name}' not found in simulation")
            return np.zeros((0, 3))

    # Collect all geom IDs belonging to this body
    object_geom_ids = set()
    for geom_id in range(sim.model.ngeom):
        if sim.model.geom_bodyid[geom_id] == body_id:
            object_geom_ids.add(geom_id)

    if len(object_geom_ids) == 0:
        if verbose:
            print(f"   ⚠️  WARNING: No geometries found for object '{object_name}'")
        return np.zeros((0, 3))

    # Extract points from all cameras
    all_points = []

    for camera_name in camera_names:
        try:
            # Render depth and segmentation
            result = sim.render(width=resolution, height=resolution, camera_name=camera_name, depth=True)
            rgb, depth = result if isinstance(result, tuple) else (None, result)
            depth = np.array(depth)
            seg_img = sim.render(width=resolution, height=resolution, camera_name=camera_name, segmentation=True)

            # Prepare RGB image
            if rgb is not None:
                rgb_image = np.array(rgb)
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                if len(rgb_image.shape) == 2:
                    rgb_image = np.stack([rgb_image, rgb_image, rgb_image], axis=-1)
            else:
                rgb_image = np.zeros((resolution, resolution, 3), dtype=np.uint8)

            # Extract full pointcloud using Adapt3R's method
            points_world = extract_pointcloud_from_camera_adapt3r(
                sim, camera_name, rgb_image, depth, resolution, resolution
            )
            points_world_flat = points_world.reshape(-1, 3)

            # Create mask for target object only
            object_mask = np.zeros((resolution, resolution), dtype=bool)
            for geom_id in object_geom_ids:
                object_mask |= (seg_img[:, :, 1] == geom_id)

            # Apply erosion to remove boundary pixels with unreliable depth (if enabled)
            if erode_kernel_size > 0:
                kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
                object_mask_uint8 = object_mask.astype(np.uint8)
                # Erosion: shrink the mask inward, removing boundary pixels
                # This removes pixels at object edges where depth values are unreliable
                # (due to mixed pixels, depth sensor artifacts, occlusion boundaries)
                object_mask_uint8 = cv2.erode(object_mask_uint8, kernel, iterations=1)
                object_mask = object_mask_uint8.astype(bool)

            # Save segmentation image for this object (after morphological filtering) - only in debug mode
            if save_debug_files:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = str(_LILO_ROOT / "outputs" / "pointclouds" / "mplib_debug")
                os.makedirs(output_dir, exist_ok=True)
                seg_output_path = os.path.join(output_dir, f"object_segmentation_{object_name}_{camera_name}_{timestamp}.png")

                # Convert mask to uint8 image (0-255) for visualization
                seg_image = (object_mask.astype(np.float32) * 255).astype(np.uint8)
                Image.fromarray(seg_image).save(seg_output_path)

            # Flatten mask and filter points
            object_mask_flat = object_mask.flatten()
            object_points = points_world_flat[object_mask_flat]

            if len(object_points) > 0:
                all_points.append(object_points)

        except Exception as e:
            if verbose:
                print(f"   ⚠️  WARNING: Failed to extract from camera '{camera_name}': {e}")
            continue

    # Merge and voxelize
    if len(all_points) == 0:
        return np.zeros((0, 3))

    merged = np.concatenate(all_points, axis=0)

    # Statistical outlier removal (if enabled)
    if outlier_std_ratio > 0 and len(merged) > 10:
        # Calculate mean distance from centroid
        centroid = merged.mean(axis=0)
        distances = np.linalg.norm(merged - centroid, axis=1)

        # Remove points beyond mean ± (std * ratio)
        mean_dist = distances.mean()
        std_dist = distances.std()
        if verbose:
            print(f"   🧹 Mean distance: {mean_dist:.4f}m, Std distance: {std_dist:.4f}m")
        threshold = mean_dist + std_dist * outlier_std_ratio

        inlier_mask = distances <= threshold
        num_outliers = (~inlier_mask).sum()

        if num_outliers > 0:
            merged = merged[inlier_mask]
            if verbose:
                print(f"   🧹 Removed {num_outliers} outlier points (distance > {threshold:.4f}m)")

    # Voxelization to remove duplicates
    voxel_indices = np.floor(merged / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    obj_pts = merged[unique_indices]

    return obj_pts


def extract_scene_pointcloud(env, camera_names=["agentview"], exclude_object_names=None,
                             resolution=512, return_full=False, no_filtering=False,
                             dilation_kernel_size=20, voxel_size=0.01, save_segmentation=False):
    """
    Extract environment pointcloud with robot masking using 2D mask dilation.
    
    Uses elegant 2D morphological dilation on segmentation masks instead of complex 3D filtering.

    Args:
        env: LIBERO environment instance
        camera_names: List of camera names to use (default: ["agentview"])
        exclude_object_names: List of object names to exclude from pointcloud
        resolution: Image resolution for rendering (default: 512)
        return_full: If True, also return full pointcloud before masking
        no_filtering: If True, return raw pointcloud without any robot/object filtering (for visualization)
        dilation_kernel_size: Kernel size for morphological dilation (default: 5 pixels)
                              Small (3-5) = ~2-5mm expansion, Medium (7-10) = ~5-10mm, Large (15-20) = ~10-20mm
        voxel_size: Voxel grid size for downsampling (default: 0.01m = 10mm)
        save_segmentation: If True, save segmentation images before/after dilation for debugging

    Returns:
        scene_pts: (N, 3) masked pointcloud with robot and excluded objects removed
        full_pts: (optional) full pointcloud if return_full=True
    """
    if exclude_object_names is None:
        exclude_object_names = []

    sim = env.sim

    # Step 1: Collect robot and excluded object geometry IDs
    robot_geom_ids = set()
    excluded_geom_ids = set()

    # Collect robot geoms
    try:
        robot = env.robots[0]
        robot_model = robot.robot_model
        robot_body_id = sim.model.body_name2id(robot_model.root_body)

        # Find all bodies belonging to robot
        robot_keywords = ['panda', 'robot', 'gripper', 'finger', 'hand']
        robot_body_ids = {robot_body_id}
        for body_id in range(sim.model.nbody):
            body_name = sim.model.body_id2name(body_id)
            if body_name and any(kw in body_name.lower() for kw in robot_keywords):
                robot_body_ids.add(body_id)

        # Get all geoms for these bodies
        for geom_id in range(sim.model.ngeom):
            if sim.model.geom_bodyid[geom_id] in robot_body_ids:
                robot_geom_ids.add(geom_id)
    except:
        pass

    # Collect excluded object geoms
    for obj_name in exclude_object_names:
        try:
            body_id = sim.model.body_name2id(obj_name)
            for geom_id in range(sim.model.ngeom):
                if sim.model.geom_bodyid[geom_id] == body_id:
                    excluded_geom_ids.add(geom_id)
        except:
            pass

    # Step 2: Extract pointclouds from cameras
    all_points_masked = []
    all_points_full = []

    for camera_name in camera_names:
        try:
            # Render depth and segmentation
            result = sim.render(width=resolution, height=resolution, camera_name=camera_name, depth=True)
            rgb, depth = result if isinstance(result, tuple) else (None, result)
            depth = np.array(depth)
            seg_img = sim.render(width=resolution, height=resolution, camera_name=camera_name, segmentation=True)

            # Prepare RGB image
            if rgb is not None:
                rgb_image = np.array(rgb)
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                if len(rgb_image.shape) == 2:
                    rgb_image = np.stack([rgb_image, rgb_image, rgb_image], axis=-1)
            else:
                rgb_image = np.zeros((resolution, resolution, 3), dtype=np.uint8)

            # Extract pointcloud using Adapt3R's method
            points_world = extract_pointcloud_from_camera_adapt3r(
                sim, camera_name, rgb_image, depth, resolution, resolution
            )

            points_world_flat = points_world.reshape(-1, 3)

            if return_full:
                all_points_full.append(points_world_flat)

            # Step 3: Create segmentation mask (before dilation)
            robot_excluded_mask = np.zeros((resolution, resolution), dtype=bool)
            for geom_id in robot_geom_ids:
                robot_excluded_mask |= (seg_img[:, :, 1] == geom_id)
            for geom_id in excluded_geom_ids:
                robot_excluded_mask |= (seg_img[:, :, 1] == geom_id)

            # Step 4: Apply 2D morphological dilation to expand robot/excluded object masks
            if not no_filtering and dilation_kernel_size > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
                dilated_mask = cv2.dilate(robot_excluded_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            else:
                dilated_mask = robot_excluded_mask

            # Save segmentation images if requested
            if save_segmentation:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = str(_LILO_ROOT / "outputs" / "pointclouds" / "mplib_debug" / "complex")
                os.makedirs(output_dir, exist_ok=True)
                
                # Save before dilation (normalize to 0-255 for visualization)
                seg_before = (robot_excluded_mask.astype(np.float32) * 255).astype(np.uint8)
                Image.fromarray(seg_before).save(os.path.join(output_dir, f"seg_before_dilation_{camera_name}_{timestamp}.png"))
                
                # Save after dilation
                seg_after = (dilated_mask.astype(np.float32) * 255).astype(np.uint8)
                Image.fromarray(seg_after).save(os.path.join(output_dir, f"seg_after_dilation_{camera_name}_{timestamp}.png"))

            # Step 5: Apply mask to pointcloud
            if no_filtering:
                # For visualization: include all points (robot + objects + scene)
                mask = seg_img[:, :, 1] > 0  # Only mask out invalid segmentation
            else:
                # For planning: exclude dilated robot/excluded object regions
                mask = ~dilated_mask & (seg_img[:, :, 1] > 0)

            if mask.sum() > 0:
                points_masked = points_world[mask]
                all_points_masked.append(points_masked)

        except Exception as e:
            continue

    # Step 6: Process masked points
    if len(all_points_masked) == 0:
        scene_pts = np.zeros((0, 3))
    else:
        merged_masked = np.concatenate(all_points_masked, axis=0)

        # Step 7: Voxelization to remove duplicates and downsample
        # Smaller voxel_size = more detail, more points, slower planning
        # Larger voxel_size = less detail, fewer points, faster planning
        voxel_indices = np.floor(merged_masked / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        scene_pts = merged_masked[unique_indices]

    # Return results
    if return_full:
        if len(all_points_full) == 0:
            full_pts = np.zeros((0, 3))
        else:
            merged_full = np.concatenate(all_points_full, axis=0)
            # Use same voxel_size for consistency
            voxel_indices_full = np.floor(merged_full / voxel_size).astype(np.int32)
            _, unique_indices_full = np.unique(voxel_indices_full, axis=0, return_index=True)
            full_pts = merged_full[unique_indices_full]
        return scene_pts, full_pts
    else:
        return scene_pts


def extract_scene_pointcloud_old(env, camera_names=["agentview"], exclude_object_names=None,
                             resolution=512, return_full=False):
    """
    Extract environment pointcloud with robot masking using Adapt3R's method.
    Uses Open3D for proper depth denormalization and camera transformation.
    
    Args:
        env: LIBERO environment instance
        camera_names: List of camera names to use (e.g., ["agentview", "birdview", "sideview"])
        exclude_object_names: List of object names to exclude from pointcloud
        resolution: Image resolution for rendering
        return_full: If True, also return full pointcloud (before masking)
    
    Returns:
        scene_pts: Masked pointcloud (robot and excluded objects removed)
        full_pts: (optional) Full pointcloud if return_full=True
    """
    # ============================================
    # Parameters & Constants
    # ============================================
    if exclude_object_names is None:
        exclude_object_names = []
    
    # Geometric filtering parameters
    ROBOT_FILTER_PARAMS = {
        'geom_safety_margin': 0.02,  # 2cm safety margin for robot geoms
        'body_threshold': 0.02,      # 8cm threshold for robot bodies (links are 5-10cm radius)
    }
    
    EXCLUDED_OBJECT_FILTER_PARAMS = {
        'geom_safety_margin': 0.02,  # 2cm safety margin for excluded object geoms
        'body_threshold': 0.02,       # 10cm threshold for excluded objects (bowls/pots/etc)
    }
    
    # Voxelization parameters
    VOXEL_SIZE = 0.01  # 1cm voxel size for deduplication
    
    # Robot identification keywords
    ROBOT_KEYWORDS = ['panda', 'robot', 'gripper', 'finger', 'hand', 'link', 'joint']
    
    sim = env.sim

    # ============================================
    # Helper: Collect Robot Geometry
    # ============================================
    def collect_robot_geometry():
        """Collect robot geom IDs and positions for masking and geometric filtering."""
        geom_ids = set()
        geom_positions = []  # List of (position, max_size) tuples
        body_positions = []
        
        try:
            robot = env.robots[0]
            robot_model = robot.robot_model

            # Method 1: Get from contact_geoms
            if hasattr(robot, 'contact_geoms'):
                for geom_name in robot.contact_geoms:
                    try:
                        geom_id = sim.model.geom_name2id(geom_name)
                        geom_ids.add(geom_id)
                        geom_pos = sim.data.geom_xpos[geom_id]
                        geom_size = sim.model.geom_size[geom_id]
                        geom_positions.append((geom_pos, np.max(geom_size)))
                    except:
                        pass

            # Method 2: Find robot bodies by keywords
            robot_body_id = sim.model.body_name2id(robot_model.root_body)
            robot_body_ids = set([robot_body_id])
            body_positions.append(sim.data.body_xpos[robot_body_id])
            
            for body_id in range(sim.model.nbody):
                body_name = sim.model.body_id2name(body_id)
                if body_name and any(kw in body_name.lower() for kw in ROBOT_KEYWORDS):
                    robot_body_ids.add(body_id)
                    body_positions.append(sim.data.body_xpos[body_id])

            # Method 3: Get all geoms belonging to robot bodies
            for geom_id in range(sim.model.ngeom):
                if sim.model.geom_bodyid[geom_id] in robot_body_ids:
                    geom_ids.add(geom_id)
                    geom_pos = sim.data.geom_xpos[geom_id]
                    geom_size = sim.model.geom_size[geom_id]
                    geom_positions.append((geom_pos, np.max(geom_size)))
            
            # Method 4: Fallback - check all geoms by name
            for geom_id in range(sim.model.ngeom):
                if geom_id not in geom_ids:
                    geom_name = sim.model.geom_id2name(geom_id)
                    if geom_name and any(kw in geom_name.lower() for kw in ROBOT_KEYWORDS[:6]):  # Exclude 'joint'
                        geom_ids.add(geom_id)
                        geom_pos = sim.data.geom_xpos[geom_id]
                        geom_size = sim.model.geom_size[geom_id]
                        geom_positions.append((geom_pos, np.max(geom_size)))
        except Exception as e:
            # Last resort: check all geoms by name
            for i in range(sim.model.ngeom):
                geom_name = sim.model.geom_id2name(i)
                if geom_name and any(kw in geom_name.lower() for kw in ROBOT_KEYWORDS[:6]):
                    geom_ids.add(i)
                    try:
                        geom_pos = sim.data.geom_xpos[i]
                        geom_size = sim.model.geom_size[i]
                        geom_positions.append((geom_pos, np.max(geom_size)))
                    except:
                        pass
        
        return geom_ids, geom_positions, body_positions

    # ============================================
    # Helper: Collect Excluded Object Geometry
    # ============================================
    def collect_excluded_object_geometry(obj_names):
        """Collect excluded object geom IDs and positions for masking and geometric filtering."""
        geom_ids = set()
        geom_positions = []
        body_positions = []
        
        for obj_name in obj_names:
            try:
                body_id = sim.model.body_name2id(obj_name)
                body_positions.append(sim.data.body_xpos[body_id])
                # Get all geoms belonging to this body
                body_geom_ids = [i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] == body_id]
                geom_ids.update(body_geom_ids)
                # Store geom positions for geometric filtering
                for geom_id in body_geom_ids:
                    try:
                        geom_pos = sim.data.geom_xpos[geom_id]
                        geom_size = sim.model.geom_size[geom_id]
                        geom_positions.append((geom_pos, np.max(geom_size)))
                    except:
                        pass
            except:
                # Fallback: search by name
                for i in range(sim.model.ngeom):
                    geom_name = sim.model.geom_id2name(i)
                    if geom_name and obj_name in geom_name:
                        geom_ids.add(i)
                        try:
                            geom_pos = sim.data.geom_xpos[i]
                            geom_size = sim.model.geom_size[i]
                            geom_positions.append((geom_pos, np.max(geom_size)))
                        except:
                            pass
        
        return geom_ids, geom_positions, body_positions

    # ============================================
    # Helper: Apply Geometric Filtering
    # ============================================
    def apply_geometric_filter(points, geom_positions, body_positions, filter_params, filter_name):
        """
        Apply geometric filtering to remove points near specified geometry.
        
        Args:
            points: Pointcloud array (N, 3)
            geom_positions: List of (position, max_size) tuples for geoms
            body_positions: List of body positions
            filter_params: Dict with 'geom_safety_margin' and 'body_threshold'
            filter_name: Name for logging (e.g., 'robot', 'excluded objects')
        
        Returns:
            filtered_points: Filtered pointcloud
        """
        if len(geom_positions) == 0 and len(body_positions) == 0:
            return points
        
        keep_mask = np.ones(len(points), dtype=bool)
        
        # Filter by geom positions (remove points clearly inside geoms)
        geom_safety_margin = filter_params['geom_safety_margin']
        for geom_pos, geom_radius in geom_positions:
            distances = np.linalg.norm(points - geom_pos, axis=1)
            keep_mask &= (distances > geom_radius + geom_safety_margin)
        
        # Filter by body positions (remove points very close to bodies)
        body_threshold = filter_params['body_threshold']
        for body_pos in body_positions:
            distances = np.linalg.norm(points - body_pos, axis=1)
            keep_mask &= (distances > body_threshold)
        
        points_before = len(points)
        filtered_points = points[keep_mask]
        points_after = len(filtered_points)

        return filtered_points

    # ============================================
    # Main Execution
    # ============================================
    
    # Collect geometry for robot and excluded objects
    robot_geom_ids, robot_geom_positions, robot_body_positions = collect_robot_geometry()
    excluded_geom_ids, excluded_geom_positions, excluded_body_positions = collect_excluded_object_geometry(exclude_object_names)

    # Extract pointclouds from cameras
    all_points_masked = []
    all_points_full = []

    for camera_name in camera_names:
        try:
            # Render images
            result = sim.render(width=resolution, height=resolution, camera_name=camera_name, depth=True)
            rgb, depth = result if isinstance(result, tuple) else (None, result)
            depth = np.array(depth)
            seg_img = sim.render(width=resolution, height=resolution, camera_name=camera_name, segmentation=True)
            
            # Handle RGB image (convert to uint8 if needed)
            if rgb is not None:
                rgb_image = np.array(rgb)
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                if len(rgb_image.shape) == 2:
                    rgb_image = np.stack([rgb_image, rgb_image, rgb_image], axis=-1)
                elif rgb_image.shape[2] == 1:
                    rgb_image = np.repeat(rgb_image, 3, axis=2)
            else:
                rgb_image = np.zeros((resolution, resolution, 3), dtype=np.uint8)

            # Extract pointcloud using Adapt3R's method
            points_world_reshaped = extract_pointcloud_from_camera_adapt3r(
                sim, camera_name, rgb_image, depth, resolution, resolution
            )
            
            points_world_all = points_world_reshaped.reshape(-1, 3)

            if return_full:
                all_points_full.append(points_world_all)

            # Create segmentation mask for robot and excluded objects
            mask = np.ones((resolution, resolution), dtype=bool)
            
            # Mask out robot geoms
            for geom_id in robot_geom_ids:
                mask &= (seg_img[:, :, 1] != geom_id)
            
            # Mask out excluded object geoms
            for geom_id in excluded_geom_ids:
                mask &= (seg_img[:, :, 1] != geom_id)
            
            # Mask out invalid segmentation (0 or out of range)
            valid_seg_mask = (seg_img[:, :, 1] > 0) & (seg_img[:, :, 1] < sim.model.ngeom)
            mask &= valid_seg_mask

            if mask.sum() > 0:
                points_world_masked = points_world_reshaped[mask]
                all_points_masked.append(points_world_masked)
        except Exception as e:
            # Skip cameras that fail (e.g., camera doesn't exist in this scene)
            continue

    # Process masked points
    if len(all_points_masked) == 0:
        scene_pts = np.zeros((0, 3))
    else:
        merged_masked = np.concatenate(all_points_masked, axis=0)
        
        # Apply geometric filtering for robot and excluded objects
        merged_masked = apply_geometric_filter(
            merged_masked, robot_geom_positions, robot_body_positions,
            ROBOT_FILTER_PARAMS, 'robot'
        )
        
        merged_masked = apply_geometric_filter(
            merged_masked, excluded_geom_positions, excluded_body_positions,
            EXCLUDED_OBJECT_FILTER_PARAMS, 'excluded objects'
        )
        
        # Voxelization to remove duplicates
        voxel_indices = np.floor(merged_masked / VOXEL_SIZE).astype(np.int32)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        scene_pts = merged_masked[unique_indices]

    # Return results
    if return_full:
        if len(all_points_full) == 0:
            full_pts = np.zeros((0, 3))
        else:
            merged_full = np.concatenate(all_points_full, axis=0)
            voxel_indices_full = np.floor(merged_full / VOXEL_SIZE).astype(np.int32)
            _, unique_indices_full = np.unique(voxel_indices_full, axis=0, return_index=True)
            full_pts = merged_full[unique_indices_full]
        return scene_pts, full_pts
    else:
        return scene_pts


def get_object_bounding_box(env, object_name, safety_margin=1.1, use_pointcloud=True,
                           erode_kernel_size=3, outlier_std_ratio=2.0, save_debug_files=False, verbose=False):
    """
    Get axis-aligned bounding box of object using pointcloud-based approach.

    Algorithm (pointcloud method):
    1. Extract object's pointcloud from cameras with noise filtering
    2. Calculate pointcloud center in world frame
    3. Transform points to canonical frame (using MuJoCo orientation, centered at pointcloud center)
    4. Compute asymmetric AABB: mins/maxs → half-extents and center offset
    5. Calculate AABB center pose (pointcloud center + offset transformed to world frame)
    6. Return AABB half-extents and AABB center pose

    Args:
        env: Environment containing the simulation
        object_name: Name of object body in MuJoCo
        safety_margin: Multiplier for AABB size (default 1.1 = 10% margin)
        use_pointcloud: If True, use camera pointcloud; if False, use MuJoCo geometry (legacy)
        erode_kernel_size: Kernel size for mask erosion to remove boundary pixels (default: 3)
                          Removes pixels with unreliable depth at object edges. Set to 0 to disable.
        outlier_std_ratio: Standard deviation ratio for 3D outlier removal (default: 2.0)
                          Removes points beyond mean ± (std * ratio). Set to 0 to disable.

    Returns:
        dict with keys:
            "aabb": Asymmetric half-extents [x, y, z] with safety margin applied
            "center_pos": Pointcloud center position (or MuJoCo body_xpos if no pointcloud)
            "pose": AABB center pose in world frame (4x4 matrix)
                    - Position is at AABB center (accounts for object asymmetry)
                    - Orientation is from MuJoCo (body_xmat)
    """
    sim = env.sim

    try:
        body_id = sim.model.body_name2id(object_name)
        body_pos = sim.data.body_xpos[body_id]
        body_rot = sim.data.body_xmat[body_id].reshape(3, 3)

        # Object pose in world frame
        obj_pose = np.eye(4)
        obj_pose[:3, :3] = body_rot
        obj_pose[:3, 3] = body_pos

        if use_pointcloud:
            # ========================================
            # POINTCLOUD-BASED AABB (NEW METHOD)
            # ========================================

            # Extract pointcloud for ONLY this specific object using segmentation
            camera_names = ["agentview", "birdview", "sideview"]
            obj_pts = extract_object_pointcloud(env, object_name, camera_names=camera_names,
                                               erode_kernel_size=erode_kernel_size,
                                               outlier_std_ratio=outlier_std_ratio,
                                               save_debug_files=save_debug_files, verbose=verbose)

            if len(obj_pts) == 0:
                # if verbose:
                print(f"   ⚠️  WARNING: No pointcloud extracted for '{object_name}', using defaults")
                aabb = [0.05, 0.05, 0.05]
            else:
                # Step 1: Calculate pointcloud center (use this as reference, not MuJoCo body_xpos)
                pts_center_world = obj_pts.mean(axis=0)

                # Step 2: Transform points to canonical frame relative to pointcloud center
                # Use MuJoCo orientation but center at pointcloud centroid
                canonical_pts = (body_rot.T @ (obj_pts - pts_center_world).T).T

                # Save canonical points to PLY file for debugging - only in debug mode
                if True:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = str(_LILO_ROOT / "outputs" / "pointclouds" / "mplib_debug")
                    os.makedirs(output_dir, exist_ok=True)
                    canonical_ply_path = os.path.join(output_dir, f"canonical_pts_{object_name}_{timestamp}.ply")
                    
                    with open(canonical_ply_path, 'w') as f:
                        f.write("ply\n")
                        f.write("format ascii 1.0\n")
                        f.write(f"element vertex {len(canonical_pts)}\n")
                        f.write("property float x\n")
                        f.write("property float y\n")
                        f.write("property float z\n")
                        f.write("property uchar red\n")
                        f.write("property uchar green\n")
                        f.write("property uchar blue\n")
                        f.write("end_header\n")
                        
                        # Write points in orange color for visibility
                        for pt in canonical_pts:
                            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} 255 165 0\n")
                    
                    if verbose:
                        print(f"   💾 Saved canonical points: {canonical_ply_path} ({len(canonical_pts)} points)")

                # Step 3: Calculate asymmetric AABB
                mins = canonical_pts.min(axis=0)
                maxs = canonical_pts.max(axis=0)

                # AABB half-extents (with safety margin)
                aabb_half_extents = (maxs - mins) / 2.0 * safety_margin

                # AABB center offset from pointcloud center (in canonical frame)
                # For symmetric objects: offset ≈ [0, 0, 0]
                # For asymmetric objects (e.g., bowl): offset accounts for asymmetry
                aabb_center_offset_canonical = (maxs + mins) / 2.0

                aabb = aabb_half_extents.tolist()  # These are half-extents

                # Step 4: Calculate AABB center pose in world frame
                # Transform offset from canonical frame to world frame
                aabb_center_offset_world = body_rot @ aabb_center_offset_canonical
                aabb_center_world = pts_center_world + aabb_center_offset_world

                # Create pose centered at AABB center (not pointcloud center or MuJoCo body)
                obj_pose[:3, :3] = body_rot
                obj_pose[:3, 3] = aabb_center_world

                if verbose:
                    print(f"   📊 Pointcloud AABB (asymmetric): {aabb}")
                    print(f"   📊 Safety margin: {safety_margin}")
                    print(f"   📊 Num points: {len(obj_pts)}")
                    print(f"   📊 Canonical mins: {mins}, maxs: {maxs}")
                    print(f"   📊 AABB center offset (canonical): {aabb_center_offset_canonical}")
                    print(f"   📊 Pointcloud center: {pts_center_world}")
                    print(f"   📊 AABB center (returned pose): {aabb_center_world}")

        else:
            # ========================================
            # GEOMETRY-BASED AABB (LEGACY METHOD)
            # ========================================
            geom_ids = [i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] == body_id]

            if len(geom_ids) == 0:
                aabb = [0.05, 0.05, 0.05]
            else:
                world_pts = []
                for geom_id in geom_ids:
                    geom_pos = sim.data.geom_xpos[geom_id]
                    geom_size = sim.model.geom_size[geom_id]

                    # Generate 8 corners of geometry bounding box
                    for dx, dy, dz in [(-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
                                       (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)]:
                        world_pts.append(geom_pos + np.array([dx, dy, dz]) * geom_size)

                world_pts = np.array(world_pts)

                # Transform to canonical frame
                canonical_pts = (body_rot.T @ (world_pts - body_pos).T).T

                # ASYMMETRIC approach (may cause issues with touch_links)
                mins = canonical_pts.min(axis=0)
                maxs = canonical_pts.max(axis=0)
                aabb_half_extents = (maxs - mins) / 2.0
                aabb = (aabb_half_extents * safety_margin).tolist()

                if verbose:
                    print(f"   📊 Geometry AABB (asymmetric): {aabb}")

        # Return object's MuJoCo pose directly (no offset)
        return {"aabb": aabb, "center_pos": body_pos, "pose": obj_pose}

    except Exception as e:
        if verbose:
            print(f"   ⚠️  ERROR in get_object_bounding_box for '{object_name}': {e}")
            import traceback
            traceback.print_exc()
        return {"aabb": [0.05, 0.05, 0.05], "center_pos": np.zeros(3), "pose": np.eye(4)}


def compute_grasp_pose_relative_to_move_group(object_pose, controller_pose, mv_link_to_ctrl):
    """
    Compute object pose relative to move_group link frame.
    
    Args:
        object_pose: Object pose in world frame (4x4 matrix)
        controller_pose: Controller (gripper0_grip_site) pose in world frame (4x4 matrix)
        mv_link_to_ctrl: Transformation matrix from controller TO move_group link frame (4x4 matrix)
                        Note: Despite the name, this transforms FROM controller TO move_group link
                        (see mplib_core.py line 83: mv_link_pose = tar_ctrl_pose @ self.mv_link_to_ctrl)
    
    Returns:
        grasp_pose: Object pose relative to move_group link frame (4x4 matrix)
    """
    # Transform controller pose to move_group link pose in world frame
    # mv_link_to_ctrl transforms FROM controller TO move_group link
    move_group_pose = controller_pose @ mv_link_to_ctrl
    
    # Compute object pose relative to move_group link frame
    return np.linalg.inv(move_group_pose) @ object_pose


# ============================================
# SECTION 2: Planning Functions
# ============================================

def plan_to_pose_simple(planner, current_qpos, target_pose, verbose=False):
    """Plan to target pose WITHOUT collision avoidance."""
    start_time = time.time()
    plan_res = planner.plan_to_pose(current_qpos, target_pose, verbose=verbose)
    plan_time = time.time() - start_time

    return {
        "success": plan_res["status"] == "Success",
        "score": plan_res["score"],
        "joint_traj": plan_res["position"],
        "cart_traj": plan_res["cartesian"],
        "plan_time": plan_time
    }


def plan_to_pose_collision_aware(planner, current_qpos, target_pose, scene_pts=None, attached_obj=None, verbose=False):
    """Plan to target pose WITH collision avoidance."""
    planner.clear_scene()
    # scene_pts = None

    if scene_pts is not None and len(scene_pts) > 0:
        planner.update_scene(scene_pts, "scene")

    if attached_obj is not None:
        planner.attach_obj(grasp_pose=attached_obj["grasp_pose"], aabb=attached_obj["aabb"])

    start_time = time.time()
    plan_res = planner.plan_to_pose(current_qpos, target_pose, verbose=verbose)
    plan_time = time.time() - start_time

    return {
        "success": plan_res["status"] == "Success",
        "score": plan_res["score"],
        "joint_traj": plan_res["position"],
        "cart_traj": plan_res["cartesian"],
        "plan_time": plan_time,
        "collision_free": plan_res["status"] == "Success"
    }


# ============================================
# SECTION 3: Execution Functions
# ============================================

def execute_cartesian_trajectory(env, cart_traj, gripper_actions, arm_name="right", velocity_factor=0.9,
                                target_object=None, verbose=False):
    """Execute Cartesian trajectory using OSC controller.

    Args:
        env: LIBERO environment
        cart_traj: Cartesian trajectory (list of 4x4 poses)
        gripper_actions: List of gripper actions
        arm_name: Arm name (default: "right")
        velocity_factor: Velocity scaling factor (default: 0.9)
        target_object: Target object name for segmentation (optional)
        verbose: Verbose output (default: False)
    """
    observations = []
    actions = []
    rewards = []
    dones = []
    infos = []
    states = []

    # Get initial and target positions
    init_pos, _ = get_controller_robot_pose(env, arm_name)
    target_final_pos = cart_traj[-1][:3, 3]
    total_dist = np.linalg.norm(target_final_pos - init_pos)

    if verbose:
        print(f"   📍 Executing trajectory with {len(cart_traj)} waypoints")
        print(f"      Initial pos: {init_pos}")
        print(f"      Target  pos: {target_final_pos}")
        print(f"      Total dist:  {total_dist:.4f}m")

    for i, target_pose in enumerate(cart_traj):
        target_pos = target_pose[:3, 3]
        target_rot = target_pose[:3, :3]
        gripper_action = gripper_actions[i] if i < len(gripper_actions) else gripper_actions[-1]

        res = pose_traj_to_action(env, target_pos=target_pos, target_rot=target_rot,
                                  arm_name=arm_name, velocity_factor=velocity_factor)
        # Combine arm action (6D) with gripper action (1D) to get 7D action
        action_arm = res["action"]  # 6D numpy array: [delta_pos (3), delta_rot (3)]
        action = np.concatenate([action_arm, np.array([gripper_action])])  # 7D numpy array
        obs, reward, done, info = env.step(action.tolist())  # env.step expects list

        # Add segmentation mask and object pose to observation if target_object is provided
        # State is saved/restored inside create_wrist_segmentation_mask to prevent corruption
        if target_object is not None:
            try:
                seg_mask = create_wrist_segmentation_mask(env, target_object)
                obs['robot0_eye_in_hand_segmentation'] = seg_mask
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Warning: Failed to add segmentation: {e}")
                obs['robot0_eye_in_hand_segmentation'] = np.zeros((256, 256), dtype=np.uint8)

            # Add object pose to observation
            try:
                obj_pos, obj_quat = get_object_pose(env.sim, target_object)
                obs['target_object_pos'] = obj_pos  # 3D position
                obs['target_object_quat'] = obj_quat  # Quaternion [w, x, y, z]
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Warning: Failed to add object pose: {e}")
                # Fallback to zeros if object pose can't be retrieved
                obs['target_object_pos'] = np.zeros(3)
                obs['target_object_quat'] = np.array([1.0, 0.0, 0.0, 0.0])

        observations.append(obs)
        actions.append(action)  # Store as numpy array (7D)
        rewards.append(reward)
        dones.append(done)
        infos.append(info)
        states.append(env.sim.get_state().flatten())  # Capture full simulation state

    # Get final pose
    final_pos, final_rot = get_controller_robot_pose(env, arm_name)
    final_ee_pose = np.eye(4)
    final_ee_pose[:3, :3] = final_rot
    final_ee_pose[:3, 3] = final_pos

    # Compute errors
    target_final_pose = cart_traj[-1]
    position_error = np.linalg.norm(final_pos - target_final_pose[:3, 3])
    delta_rot = final_rot @ target_final_pose[:3, :3].T
    trace = np.trace(delta_rot)
    rotation_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))

    return {
        "success": position_error < 0.05,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "infos": infos,
        "states": states,
        "final_ee_pose": final_ee_pose,
        "position_error": position_error,
        "rotation_error": rotation_error
    }


# ============================================
# SECTION 4: Subgoal Planning Utilities
# ============================================

def interpolate_poses(pose1, pose2, alpha):
    """Interpolate between two SE(3) poses using linear position interpolation and SLERP."""
    pos1 = pose1[:3, 3]
    pos2 = pose2[:3, 3]
    interp_pos = (1 - alpha) * pos1 + alpha * pos2

    key_times = [0, 1]
    key_rots = R.from_matrix([pose1[:3, :3], pose2[:3, :3]])
    slerp = Slerp(key_times, key_rots)
    interp_rot = slerp([alpha])[0].as_matrix()

    interp_pose = np.eye(4)
    interp_pose[:3, :3] = interp_rot
    interp_pose[:3, 3] = interp_pos
    return interp_pose


def get_current_ee_pose(planner, q):
    """Return controller-frame EE pose for a given joint configuration."""
    if len(q) == 7:
        q = np.concatenate([q, np.zeros(2)])

    planner.set_qpos(q)
    mv_link_pose = planner.get_link_pose(wrt_world=True)
    return mv_link_pose @ planner.ctrl_to_mv_link


def plan_with_fixed_waypoints(planner, q0, target_pose, num_waypoints=2,
                               time_step=0.05, mv_link_to_ctrl=None, planning_time=5.0, verbose=False):
    """Plan by inserting evenly spaced Cartesian waypoints and solving IK per segment."""
    if len(q0) == 7:
        q0_full = np.concatenate([q0, np.zeros(2)])
    else:
        q0_full = q0

    arm_dim = 7
    current_pose = get_current_ee_pose(planner, q0_full)

    waypoint_poses = []
    for i in range(1, num_waypoints + 2):
        alpha = i / (num_waypoints + 1)
        waypoint_poses.append(interpolate_poses(current_pose, target_pose, alpha))

    if mv_link_to_ctrl is not None:
        waypoint_mv_poses = [wp @ mv_link_to_ctrl for wp in waypoint_poses]
    else:
        waypoint_mv_poses = waypoint_poses

    joint_waypoints = [q0_full]
    for i, mv_pose in enumerate(waypoint_mv_poses):
        ik_status, q_target = planner.planner.IK(
            planner.planner._transform_goal_to_wrt_base(Pose(mv_pose)),
            start_qpos=joint_waypoints[-1],
            n_init_qpos=10,
            return_closest=True
        )

        if ik_status != "Success":
            if verbose:
                print(f"   IK failed for waypoint {i + 1}/{len(waypoint_mv_poses)}")
            return {
                "status": "Failed",
                "score": 0.,
                "position": [q0_full[:arm_dim]],
                "cartesian": []
            }

        joint_waypoints.append(q_target)

    all_positions = []
    for i in range(len(joint_waypoints) - 1):
        q_start = joint_waypoints[i]
        mv_goal_pose = Pose(waypoint_mv_poses[i])

        res = planner.planner.plan_pose(
            mv_goal_pose, q_start,
            time_step=time_step,
            wrt_world=True,
            planning_time=planning_time
        )

        if res["status"] != "Success":
            if verbose:
                print(f"   RRT failed for segment {i + 1}/{len(joint_waypoints) - 1}")
            return {
                "status": "Failed",
                "score": 0.,
                "position": [q0_full[:arm_dim]],
                "cartesian": []
            }

        if i == 0:
            all_positions.extend(res["position"])
        else:
            all_positions.extend(res["position"][1:])

    all_positions = np.array(all_positions)
    all_cartesian = planner.convert_joint_to_ctrl_poses(all_positions)

    return {
        "status": "Success",
        "score": 1.0,
        "position": all_positions[:, :arm_dim],
        "cartesian": all_cartesian
    }


def plan_with_adaptive_subdivision(planner, q0, target_pose, max_depth=3,
                                   time_step=0.05, mv_link_to_ctrl=None, planning_time=5.0, verbose=False):
    """Recursively subdivide motion if direct planning fails."""
    if len(q0) == 7:
        q0_full = np.concatenate([q0, np.zeros(2)])
    else:
        q0_full = q0

    arm_dim = 7

    if mv_link_to_ctrl is not None:
        target_mv_pose = target_pose @ mv_link_to_ctrl
    else:
        target_mv_pose = target_pose

    def _recursive_plan(q_start, goal_mv_pose, goal_ee_pose, depth):
        res = planner.planner.plan_pose(
            Pose(goal_mv_pose), q_start,
            time_step=time_step,
            wrt_world=True,
            planning_time=planning_time
        )

        if res["status"] == "Success":
            return True, res["position"]

        if depth < max_depth:
            if verbose:
                print(f"   Direct planning failed at depth {depth}, subdividing...")
            start_ee_pose = get_current_ee_pose(planner, q_start)
            mid_ee_pose = interpolate_poses(start_ee_pose, goal_ee_pose, 0.5)
            mid_mv_pose = mid_ee_pose @ mv_link_to_ctrl if mv_link_to_ctrl is not None else mid_ee_pose

            ik_status, q_mid = planner.planner.IK(
                planner.planner._transform_goal_to_wrt_base(Pose(mid_mv_pose)),
                start_qpos=q_start,
                n_init_qpos=10,
                return_closest=True
            )

            if ik_status != "Success":
                if verbose:
                    print(f"   IK failed for midpoint at depth {depth}")
                return False, None

            success1, traj1 = _recursive_plan(q_start, mid_mv_pose, mid_ee_pose, depth + 1)
            if not success1:
                return False, None

            success2, traj2 = _recursive_plan(q_mid, goal_mv_pose, goal_ee_pose, depth + 1)
            if not success2:
                return False, None

            combined_traj = list(traj1) + list(traj2)[1:]
            return True, combined_traj

        if verbose:
            print(f"   Max depth {max_depth} reached, planning failed")
        return False, None

    success, joint_traj = _recursive_plan(q0_full, target_mv_pose, target_pose, depth=0)

    if not success:
        return {
            "status": "Failed",
            "score": 0.,
            "position": [q0_full[:arm_dim]],
            "cartesian": []
        }

    joint_traj_array = np.array(joint_traj)
    cartesian_traj = planner.convert_joint_to_ctrl_poses(joint_traj_array)

    return {
        "status": "Success",
        "score": 1.0,
        "position": joint_traj_array[:, :arm_dim],
        "cartesian": cartesian_traj
    }
