"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from lilo_vla.interaction.backends.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256, horizon=None):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "camera_depths": True}  # yy: I add camera_depths
    
    # Set custom horizon if provided (needed for pose shifting with motion planning)
    if horizon is not None:
        env_args["horizon"] = horizon
        
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img

# yy: I add the following 2 functions
def get_libero_image_depth(obs):
    """Extracts third-person depth image, rotates it 180°, and duplicates channel to (256, 256, 3)."""
    img = obs["agentview_depth"]  # Expected shape: (256, 256, 1)
    img = img[::-1, ::-1]  # Rotate 180° (vertical + horizontal flip)

    # Duplicate the depth channel to 3 channels
    img_3ch = np.repeat(img, 3, axis=2)  # Shape: (256, 256, 3)

    return img_3ch

def get_libero_wrist_image_depth(obs):
    """Extracts wrist camera depth image, rotates it 180°, and duplicates channel to (256, 256, 3)."""
    img = obs["robot0_eye_in_hand_depth"]  # Expected shape: (256, 256, 1)
    img = img[::-1, ::-1]  # Rotate 180° (vertical + horizontal flip)
    # Duplicate the single channel 3 times along the last axis
    img_3ch = np.repeat(img, 3, axis=2)  # Shape: (256, 256, 3)
    return img_3ch

def get_libero_env_agentview_zoomout(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment with agentview camera zoomed out, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": True,
        "render_camera": "agentview",
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    # Set camera parameters directly in the MuJoCo model (zoom out)
    sim = env.sim
    model = sim.model
    # Find the agentview camera index
    cam_id = None
    for i in range(model.ncam):
        name = model.camera_id2name(i)
        if name == "agentview":
            cam_id = i
            break
    if cam_id is None:
        cam_id = 0  # fallback to first camera
    model.cam_pos[cam_id] = [2.0, 0.0, 2.0]  # Further and higher
    model.cam_fovy[cam_id] = 70  # Wider field of view
    # Optionally, set lookat or orientation if needed
    return env, task_description

def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
