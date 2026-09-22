"""
Pointcloud extraction utilities adapted from Adapt3R.
These functions provide correct depth denormalization and camera transformations
for LIBERO environments.
"""

import numpy as np
import open3d as o3d
import robosuite.utils.transform_utils as T


def depth_im_to_meters(depth_im, sim):
    """
    Convert normalized depth image to actual depth in meters using MuJoCo parameters.
    Adapted from Adapt3R's implementation.
    
    Args:
        depth_im: Normalized depth image (0-1) from MuJoCo rendering
        sim: MuJoCo simulation object
        
    Returns:
        depth_im_m: Depth image in meters
    """
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    depth_im_m = near / (1 - np.array(depth_im) * (1 - near / far))
    return depth_im_m


def get_camera_intrinsic_matrix(camera_name, sim, img_height, img_width):
    """
    Obtains camera intrinsic matrix.
    Adapted from Adapt3R's implementation.
    
    Args:
        camera_name: Name of camera
        sim: MuJoCo simulation object
        img_height: Height of camera images in pixels
        img_width: Width of camera images in pixels
        
    Returns:
        K: 3x3 camera intrinsic matrix
    """
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * img_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, img_width / 2], [0, f, img_height / 2], [0, 0, 1]])
    return K


def get_camera_extrinsic_matrix(camera_name, sim):
    """
    Returns a 4x4 homogeneous matrix corresponding to the camera pose in the
    world frame. MuJoCo has a weird convention for how it sets up the
    camera body axis, so we also apply a correction so that the x and y
    axis are along the camera view and the z axis points along the viewpoint.
    Adapted from Adapt3R's implementation.
    
    Args:
        camera_name: Name of camera
        sim: MuJoCo simulation object
        
    Returns:
        R: 4x4 camera extrinsic matrix (camera to world transformation)
    """
    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[cam_id]
    camera_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    R = T.make_pose(camera_pos, camera_rot)

    # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
    camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    R = R @ camera_axis_correction
    return R


def cammat2o3d(cam_mat, width, height):
    """
    Generates Open3D camera intrinsic matrix object from numpy camera intrinsic matrix.
    Adapted from Adapt3R's utils.
    
    Args:
        cam_mat: 3x3 numpy array representing camera intrinsic matrix
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        o3d_cam_mat: Open3D PinholeCameraIntrinsic object
    """
    cx = cam_mat[0, 2]
    fx = cam_mat[0, 0]
    cy = cam_mat[1, 2]
    fy = cam_mat[1, 1]
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def extract_pointcloud_from_camera_adapt3r(sim, camera_name, rgb_image, depth_image, 
                                           img_height, img_width):
    """
    Extract pointcloud from a single camera using Adapt3R's method.
    This uses Open3D for proper backprojection and transformation.
    
    Args:
        sim: MuJoCo simulation object
        camera_name: Name of camera (e.g., 'agentview')
        rgb_image: RGB image array (H, W, 3)
        depth_image: Normalized depth image (H, W) from MuJoCo rendering
        img_height: Image height in pixels
        img_width: Image width in pixels
        
    Returns:
        points_world: (H, W, 3) numpy array of 3D points in world coordinates, 
                     matching image layout for masking purposes
    """
    # Convert depth to meters using Adapt3R's method
    depths_m = depth_im_to_meters(depth_image, sim)
    
    # Get camera parameters
    intrinsics = get_camera_intrinsic_matrix(camera_name, sim, img_height, img_width)
    extrinsics = get_camera_extrinsic_matrix(camera_name, sim)
    
    # Create Open3D RGBD image (following Adapt3R approach)
    # Note: Open3D expects images flipped vertically ([::-1])
    rgb_im = o3d.geometry.Image(np.ascontiguousarray(rgb_image[::-1]))
    depth_im = o3d.geometry.Image(np.clip(np.ascontiguousarray(depths_m[::-1]), 0, 4))
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_im,
        depth_im,
        convert_rgb_to_intensity=False,
        depth_trunc=5,
        depth_scale=1
    )
    
    # Convert camera intrinsic matrix to Open3D format
    o3d_cam_mat = cammat2o3d(intrinsics, img_width, img_height)
    
    # Create pointcloud from RGBD image (Open3D handles backprojection automatically)
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_cam_mat)
    
    # Transform to world coordinates using extrinsic matrix
    transformed_cloud = cloud.transform(extrinsics)
    
    # Extract points as numpy array
    # Open3D returns points in row-major order matching the image
    # Reshape to (H, W, 3) to match image layout for masking
    points_world = np.asarray(transformed_cloud.points)
    points_world = points_world.reshape(img_height, img_width, 3)
    
    # Flip vertically to match original image orientation (Open3D flips images)
    points_world = points_world[::-1]
    
    return points_world

