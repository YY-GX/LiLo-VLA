#!/usr/bin/env python3
"""
LiLo-VLA long-horizon pipeline (above-pose / object-centric strategy).

Streamlined Long-Horizon Pipeline - Minimal logging, maximum clarity.

Executes long-horizon tasks using VLA with atomic skill decomposition.
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Optional

# Suppress warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

from lilo_vla.reaching.motion_planning.cartesian_planner import MotionPlanner as StandardMotionPlanner
from lilo_vla.reaching.motion_planning.mplib_planner import MPlibMotionPlanner
from lilo_vla.reaching.motion_planning.mplib_planner_core import get_object_bounding_box
from lilo_vla.reaching.approach_pose import (
    get_object_pose,
    get_special_object_handling,
    calculate_above_pose
)
# NOTE: `create_wrist_segmentation_mask_with_grasped` is deliberately NOT imported
# here. The source file never imported it either, so the (unreachable, --mask_as_agentview
# only) call below raises NameError into its own try/except exactly as it did before the
# migration. Do not "fix" this.
from lilo_vla.interaction.masking import create_wrist_segmentation_mask, apply_distractor_masking
from lilo_vla.interaction.random_erasing import apply_random_erasing_to_mask

from scipy.spatial.transform import Rotation as R

# Package root (lilo-vla/lilo_vla) and project root (lilo-vla/).
# This file lives at lilo-vla/lilo_vla/core/pipeline.py
_PKG = Path(__file__).resolve().parent.parent
_LILO_ROOT = _PKG.parent


def load_module(path: str, name: str):
    """Load script module dynamically."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def move_ee_up(env, motion_planner, next_skill_type: str = 'other', grasped_object: str = None, max_retries: int = 5) -> bool:
    """Move end-effector up 5cm in Z-axis using motion planner with retry logic."""
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Get current observation
            obs = env.env._get_observations()
            current_ee_pos = obs['robot0_eef_pos'].copy()
            current_ee_quat = obs['robot0_eef_quat'].copy()

            # Move up 5cm in Z-axis
            target_ee_pos = current_ee_pos.copy()
            target_ee_pos[2] += 0.10  # 5cm up
            # Convert from [x, y, z, w] (xyzw - robosuite format) to [w, x, y, z] (wxyz - for move_to_pose)
            target_ee_quat = np.array([current_ee_quat[3], current_ee_quat[0], current_ee_quat[1], current_ee_quat[2]])

            if retry_count == 0:
                print(f"   📈 Moving EE up 5cm: {current_ee_pos[2]:.3f} → {target_ee_pos[2]:.3f}")
            else:
                print(f"   📈 Retry {retry_count + 1}/{max_retries}: Moving EE up 5cm")

            move_kwargs = {
                'position_threshold': 0.02,  # 1cm threshold
                'orientation_threshold': 0.348,  # 20 degrees
                'skill_type': next_skill_type
            }

            if isinstance(motion_planner, MPlibMotionPlanner):
                # Pass grasped object if provided (e.g., after pick)
                move_kwargs['grasped_object_name'] = grasped_object
            else:
                move_kwargs['num_steps'] = 200  # Limit steps for quick upward movement

            # Use motion planner to move up with limited steps
            move_success, result = motion_planner.move_to_pose(
                target_ee_pos, target_ee_quat,
                **move_kwargs
            )
            observations = result[0]

            # Record all frames if pipeline has recording capability
            if hasattr(env, 'pipeline') and hasattr(env.pipeline, '_record_frame'):
                for obs in observations:
                    env.pipeline._record_frame(obs)

            if move_success:
                if retry_count > 0:
                    print(f"   ✅ Successfully moved EE up 5cm on attempt {retry_count + 1}/{max_retries}")
                else:
                    print(f"   ✅ Successfully moved EE up 5cm")
                return True
            else:
                print(f"   ❌ Failed to move EE up 5cm (attempt {retry_count + 1}/{max_retries})")
                retry_count += 1

        except Exception as e:
            print(f"   ❌ Error moving EE up (attempt {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1

    # All retries exhausted
    print(f"   ❌ Failed to move EE up after {max_retries} attempts")
    return False


def post_actions(skill: str, env, motion_planner, skill_succeeded: bool = False, next_skill: str = None, grasped_object: str = None, max_move_up_retries: int = 5) -> bool:
    """Execute gripper actions and move EE up."""
    skill_lower = skill.lower()
    if 'pick' in skill_lower:
        gripper_action = 1.0  # Close gripper
    elif 'place' in skill_lower or 'stack' in skill_lower:
        gripper_action = -1.0  # Open gripper
    else:
        gripper_action = 0.0  # No gripper action

    action = np.array([0, 0, 0, 0, 0, 0, gripper_action])

    print(f"   🤏 Executing post-actions: {'close' if gripper_action > 0 else 'open' if gripper_action < 0 else 'no'} gripper")
    for _ in range(5):
        obs, _, _, _ = env.step(action)
        # Record frames during post-actions if pipeline has recording capability
        if hasattr(env, 'pipeline') and hasattr(env.pipeline, '_record_frame'):
            env.pipeline._record_frame(obs)

    # Move EE up 5cm with next skill type
    if next_skill:
        next_skill_lower = next_skill.lower()
        if 'pick' in next_skill_lower:
            next_skill_type = 'pick'
        elif 'place' in next_skill_lower or 'stack' in next_skill_lower:
            next_skill_type = 'place'
        else:
            next_skill_type = 'other'

        # Pass grasped object if we just completed a pick
        held_object = grasped_object if 'pick' in skill_lower else None
        move_ee_up(env, motion_planner, next_skill_type, grasped_object=held_object, max_retries=max_move_up_retries)

    return True


class LongHorizonPipeline:
    def __init__(self, vla_checkpoint: str, task_name: str, wrist_only: bool = False,
                 visualize_poses: bool = False, ee_offset: float = 0.0,
                 motion_planner_method: str = "cartesian_linear", continue_on_failure: bool = False,
                 mp_steps: int = 400, mp_pos_gain: float = 5.0, mp_ori_gain: float = 5.0,
                 mp_pos_threshold: float = 0.02, mp_ori_threshold: float = 0.524,
                 mp_pos_success_threshold: float = 0.025, mp_ori_success_threshold: float = 0.524,
                 motion_planner_type: str = "standard", collision_aware: bool = True,
                 mp_velocity_factor: float = 0.9, obj_ee_pairs_dir: str = None,
                 max_vla_retries: int = 3, max_move_up_retries: int = 5,
                mplib_joint_vel_limits: float = 2.0, mplib_joint_acc_limits: float = 4.0,
                mplib_waypoint_target_ratio: int = 50, mplib_planning_time: float = 1.0,
                 mplib_safety_margin: float = 1.1, mplib_erode_kernel_size: int = 3,
                 mplib_outlier_std_ratio: float = 0.0, mplib_save_pointcloud: bool = False,
                 mplib_verbose: bool = False, vla_horizon: int = 200,
                 verbose_vla_failure: bool = False, above_height: float = 0.03,
                 benchmark_name: str = "long_horizon_tasks_v1", use_relative_pose: bool = True,
                 apply_background_erasing: bool = False, apply_distractor_masking: bool = False,
                 mask_as_agentview: bool = False, recovery_mode: str = 'full',
                 unnorm_key: Optional[str] = None):

        self.task_name = task_name
        self.wrist_only = wrist_only
        self.visualize_poses = visualize_poses
        self.ee_offset = ee_offset
        self.motion_planner_method = motion_planner_method
        self.motion_planner_type = motion_planner_type
        self.collision_aware = collision_aware
        self.continue_on_failure = continue_on_failure
        self.mp_steps = mp_steps
        self.mp_pos_gain = mp_pos_gain
        self.mp_ori_gain = mp_ori_gain
        self.mp_pos_threshold = mp_pos_threshold
        self.mp_ori_threshold = mp_ori_threshold
        self.mp_pos_success_threshold = mp_pos_success_threshold
        self.mp_ori_success_threshold = mp_ori_success_threshold
        self.mp_velocity_factor = mp_velocity_factor

        # MPlib-specific parameters
        self.mplib_joint_vel_limits = mplib_joint_vel_limits
        self.mplib_joint_acc_limits = mplib_joint_acc_limits
        self.mplib_waypoint_target_ratio = mplib_waypoint_target_ratio
        self.mplib_planning_time = mplib_planning_time
        self.mplib_safety_margin = mplib_safety_margin
        self.mplib_erode_kernel_size = mplib_erode_kernel_size
        self.mplib_outlier_std_ratio = mplib_outlier_std_ratio
        self.mplib_save_pointcloud = mplib_save_pointcloud
        self.mplib_verbose = mplib_verbose
        self.obj_ee_pairs_dir = obj_ee_pairs_dir
        self.max_vla_retries = max_vla_retries
        self.recovery_mode = recovery_mode
        self.unnorm_key = unnorm_key
        self.max_move_up_retries = max_move_up_retries
        self.vla_horizon = vla_horizon
        self.verbose_vla_failure = verbose_vla_failure
        self.above_height = above_height
        self.benchmark_name = benchmark_name
        self.use_relative_pose = use_relative_pose
        self.apply_background_erasing = apply_background_erasing
        self.apply_distractor_masking = apply_distractor_masking
        self.mask_as_agentview = mask_as_agentview

        self.env = None
        self.motion_planner = None
        self.vla = None
        self.skill_sequence = []
        self.current_masked_wrist_img = None  # Store masked wrist image for video recording

        # Video/image recording setup
        self.video_writer = None
        self.video_filename = None
        self.images_dir = None
        self.video_dir = None  # Store base video directory for reuse across trials
        self.trial_num = 1  # Track trial number for video naming
        self._setup_recording()

        # MPlib pointcloud debug directory setup
        self.mplib_pointcloud_dir = None
        if self.mplib_save_pointcloud:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_clean = self.task_name.replace(" ", "_").lower()
            pointcloud_base = _LILO_ROOT / "outputs" / "pointclouds" / "long_eval_debug"
            self.mplib_pointcloud_dir = pointcloud_base / f"{task_clean}_{timestamp}"
            self.mplib_pointcloud_dir.mkdir(parents=True, exist_ok=True)
            print(f"📊 MPlib debug pointclouds will be saved to: {self.mplib_pointcloud_dir}")

        # Load components
        # NOTE: these are loaded via load_module (not plain imports) on purpose -- each call
        # creates a private module instance, and skill_checker holds mutable module-level
        # caches whose identity matters.
        self.task_planner = load_module(str(_PKG / "core" / "task_planner.py"), "task_planner")
        self.skill_checker = load_module(str(_PKG / "core" / "skill_checker.py"), "skill_checker")

        # Load task config for target object lookup
        task_config_path = str(_LILO_ROOT / "configs" / "tasks_and_skills.json")
        with open(task_config_path, 'r') as f:
            self.task_config = json.load(f)
        self.skill_mappings = self.task_config.get("skill_mappings", {})

        # Load skill config for BDDL filename lookup
        skill_config_path = str(_LILO_ROOT / "configs" / "skill_config.json")
        with open(skill_config_path, 'r') as f:
            self.skill_config = json.load(f)

        if not visualize_poses:
            from dataclasses import dataclass
            from typing import Union
            from lilo_vla.interaction.backends.openvla_oft import get_processor
            from lilo_vla.interaction.backends.robot_utils import get_model

            @dataclass
            class VLAConfig:
                model_family: str = "openvla"
                pretrained_checkpoint: Union[str, Path] = ""
                use_l1_regression: bool = True
                use_diffusion: bool = False
                num_diffusion_steps: int = 50
                use_film: bool = False
                num_images_in_input: int = 2
                use_proprio: bool = True
                center_crop: bool = True
                num_open_loop_steps: int = 8
                unnorm_key: Union[str, Path] = ""
                load_in_8bit: bool = False
                load_in_4bit: bool = False
                task_suite_name: str = "atomic_skills"
                wrist_only: bool = False
                agent_only: bool = False
                pro_only: bool = False
                is_oss: bool = False
                is_depth: bool = False

            if vla_checkpoint is None:
                raise ValueError(
                    "vla_checkpoint is required. unnorm_key is read from the checkpoint's "
                    "dataset_statistics.json; pass --unnorm_key if that file is missing."
                )

            cfg = VLAConfig()
            cfg.pretrained_checkpoint = vla_checkpoint
            cfg.num_images_in_input = 1 if (wrist_only and not self.mask_as_agentview) else 2
            cfg.wrist_only = wrist_only

            self.vla = get_model(cfg)
            self.processor = get_processor(cfg)

            # Load action head if needed
            from lilo_vla.interaction.backends.openvla_oft import get_action_head, get_proprio_projector, get_noisy_action_projector
            self.action_head = None
            self.proprio_projector = None
            self.noisy_action_projector = None

            if cfg.use_l1_regression or cfg.use_diffusion:
                self.action_head = get_action_head(cfg, self.vla.llm_dim)

            if cfg.use_proprio:
                self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, proprio_dim=8)

            if cfg.use_diffusion:
                self.noisy_action_projector = get_noisy_action_projector(cfg, self.vla.llm_dim)

            # Auto-detect unnorm_key from checkpoint path
            # --- Resolve unnorm_key -------------------------------------------
            # The action head emits NORMALISED actions; unnorm_key selects the
            # statistics used to map them back to metric units. Picking the wrong
            # key does not fail -- it silently rescales every action.
            #
            # Order of preference:
            #   1. explicit cfg.unnorm_key (CLI --unnorm_key)
            #   2. dataset_statistics.json in the checkpoint -- AUTHORITATIVE. It
            #      holds exactly the dataset(s) this checkpoint was fine-tuned on.
            #   3. the checkpoint DIRECTORY NAME (openvla-7b+<dataset>+b16+...)
            #   4. the path segment after `runs/`
            # Step 2 exists because a checkpoint fetched from the Hugging Face Hub
            # lands in `models--<org>--<repo>/snapshots/<sha>`, where 3 and 4 both
            # fail and the old fallback picked `bridge_orig` -- a key that IS present
            # in the base model's norm_stats, so nothing complained.
            explicit_key = self.unnorm_key or None
            cfg.unnorm_key = explicit_key or None
            source = "explicit --unnorm_key"

            if not cfg.unnorm_key:
                stats_path = Path(vla_checkpoint) / "dataset_statistics.json"
                if stats_path.is_file():
                    try:
                        with open(stats_path) as f:
                            finetune_stats = json.load(f)
                        if len(finetune_stats) == 1:
                            cfg.unnorm_key = next(iter(finetune_stats))
                            source = "dataset_statistics.json"
                    except (json.JSONDecodeError, OSError):
                        pass

            if not cfg.unnorm_key:
                checkpoint_filename = Path(vla_checkpoint).name
                if "+" in checkpoint_filename:
                    parts = checkpoint_filename.split("+")
                    if len(parts) >= 2:
                        cfg.unnorm_key = parts[1]
                        source = "checkpoint directory name"

            if not cfg.unnorm_key:
                checkpoint_parts = Path(vla_checkpoint).parts
                if "runs" in checkpoint_parts:
                    runs_idx = checkpoint_parts.index("runs")
                    if runs_idx + 1 < len(checkpoint_parts):
                        cfg.unnorm_key = checkpoint_parts[runs_idx + 1]
                        source = "path segment after runs/"

            # Validate against what the model actually carries. Fail loudly: a wrong
            # key produces plausible-looking but wrongly scaled actions.
            if getattr(self.vla, "norm_stats", None):
                available = list(self.vla.norm_stats.keys())
                if cfg.unnorm_key not in available:
                    base_key = cfg.unnorm_key.split(":")[0] if cfg.unnorm_key else ""
                    matching = [k for k in available if base_key and k.startswith(base_key)]
                    if matching:
                        print(f"   ⚠️  unnorm_key '{cfg.unnorm_key}' not found; using '{matching[0]}'")
                        cfg.unnorm_key = matching[0]
                        source += " (prefix-matched)"
                    else:
                        raise ValueError(
                            f"Could not determine unnorm_key for checkpoint:\n"
                            f"    {vla_checkpoint}\n"
                            f"Resolved '{cfg.unnorm_key}', which is not in the model's norm_stats.\n"
                            f"Pass it explicitly with --unnorm_key. Candidates in this checkpoint's "
                            f"dataset_statistics.json are the fine-tuning datasets; the {len(available)} "
                            f"keys in norm_stats also include the base model's pretraining mixtures, "
                            f"which are NOT what you want."
                        )

            print(f"   Detected unnorm_key: {cfg.unnorm_key}   (source: {source})")

            # Import VLA execution utilities
            from lilo_vla.envs.libero_env import get_libero_wrist_image
            self.get_libero_wrist_image = get_libero_wrist_image
            self.cfg = cfg

    def _setup_recording(self):
        """Setup video and image recording directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_clean = self.task_name.replace(" ", "_").lower()

        # Add background erasing suffix if enabled
        bg_suffix = '_bg_erasing' if self.apply_background_erasing else ''

        if self.visualize_poses:
            # Pose visualization mode - only save images
            pose_dir = _LILO_ROOT / "outputs" / "pose_visualization"
            self.images_dir = pose_dir / f"{task_clean}{bg_suffix}_{timestamp}"
            self.images_dir.mkdir(parents=True, exist_ok=True)
            print(f"📷 Pose images will be saved to: {self.images_dir}")
        else:
            # Normal mode - video + images
            video_base = _LILO_ROOT / "outputs" / "videos" / "long_eval"
            self.video_dir = video_base / f"{task_clean}{bg_suffix}_{timestamp}"
            self.video_dir.mkdir(parents=True, exist_ok=True)
            # Video filename will be set in execute() with trial number
            self.video_filename = None

            images_base = _LILO_ROOT / "outputs" / "images" / "long_eval"
            self.images_dir = images_base / f"{task_clean}{bg_suffix}_{timestamp}"
            self.images_dir.mkdir(parents=True, exist_ok=True)
            print(f"📹 Videos will be saved to: {self.video_dir}")

    def _start_video(self, obs):
        """Start video recording with side-by-side agentview and wrist camera."""
        if self.visualize_poses or obs is None or not isinstance(obs, dict):
            return
        if 'agentview_image' not in obs:
            return

        # Get dimensions from agentview (both cameras should have same size)
        h, w = obs['agentview_image'].shape[:2]

        # Create video with double width for side-by-side frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(str(self.video_filename), fourcc, 20, (w * 2, h))

    def _record_frame(self, obs):
        """Record side-by-side frame (agentview + wrist) to video."""
        # Guard against non-dict observations (e.g., strings)
        if self.video_writer is None or obs is None or not isinstance(obs, dict):
            return
        if 'agentview_image' not in obs or 'robot0_eye_in_hand_image' not in obs:
            return

        # Get both camera views
        agentview = obs['agentview_image']
        # Use masked wrist image if available (from distractor/background masking), otherwise use raw obs
        # Note: current_masked_wrist_img is already rotated 180° by get_libero_wrist_image()
        wrist = self.current_masked_wrist_img if self.current_masked_wrist_img is not None else obs['robot0_eye_in_hand_image']
        wrist_already_rotated = (self.current_masked_wrist_img is not None)

        # Ensure uint8 format
        if agentview.dtype != np.uint8:
            agentview = (agentview * 255).astype(np.uint8)
        if wrist.dtype != np.uint8:
            wrist = (wrist * 255).astype(np.uint8)

        # Rotate agentview 180 degrees (always needed)
        agentview = cv2.rotate(agentview, cv2.ROTATE_180)

        # Only rotate wrist if it's raw obs (not already rotated from masking pipeline)
        if not wrist_already_rotated:
            wrist = cv2.rotate(wrist, cv2.ROTATE_180)

        # Concatenate side-by-side (agentview on left, wrist on right)
        combined = np.concatenate([agentview, wrist], axis=1)

        # Write to video (convert RGB to BGR for OpenCV)
        self.video_writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    def _save_mp_images(self, obs, skill_idx: int, skill_name: str):
        """Save images after motion planner completes."""
        if self.images_dir is None or obs is None or not isinstance(obs, dict):
            return

        prefix = f"skill_{skill_idx:02d}_{skill_name.replace(' ', '_')}"

        # Save agent view
        if 'agentview_image' in obs:
            img = obs['agentview_image']
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            cv2.imwrite(str(self.images_dir / f"{prefix}_agent.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Save wrist view (use masked if available)
        if 'robot0_eye_in_hand_image' in obs:
            img = self.current_masked_wrist_img if self.current_masked_wrist_img is not None else obs['robot0_eye_in_hand_image']
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            cv2.imwrite(str(self.images_dir / f"{prefix}_wrist.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _stop_video(self):
        """Stop video recording."""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"📹 Video saved: {self.video_filename}")

    def _get_skill_info(self, skill: str) -> tuple:
        """
        Get target object and VLA language from skill name using tasks_and_skills.json.

        Args:
            skill: Skill name (e.g., "pick black bowl 1")

        Returns:
            (target_object, vla_language) tuple or (None, None) if not found
        """
        # Direct lookup
        if skill in self.skill_mappings:
            skill_info = self.skill_mappings[skill]
            return skill_info.get("target_object"), skill_info.get("language")

        # Try case-insensitive lookup
        skill_lower = skill.lower()
        for skill_name, skill_info in self.skill_mappings.items():
            if skill_name.lower() == skill_lower:
                return skill_info.get("target_object"), skill_info.get("language")

        print(f"   ⚠️  Skill '{skill}' not found in tasks_and_skills.json")
        return None, None

    def _print_recovery_summary(self, recovery_stats: dict, success_count: int, total_skills: int):
        """Print detailed recovery statistics."""
        print(f"\n{'='*80}")
        print(f"📊 RECOVERY STATISTICS SUMMARY")
        print(f"{'='*80}")

        # Overall progress
        print(f"\n🎯 Overall Progress: {success_count}/{total_skills} skills completed")

        # Final failure cause
        if recovery_stats['final_failure']:
            ff = recovery_stats['final_failure']
            print(f"\n❌ Final Failure:")
            print(f"   Skill: '{ff['skill']}' (#{ff['skill_idx']}, type: {ff['skill_type']})")
            print(f"   Failed Stage: {ff['stage'].upper()}")
            print(f"   Recovery Attempts:")
            for stage, count in ff['attempted_recoveries'].items():
                if count > 0:
                    stage_name = {'above_pose': 'Above Pose Calc', 'mp': 'Motion Planning', 'vla': 'VLA Execution', 'pick_recovery': 'Pick Recovery'}[stage]
                    print(f"      - {stage_name}: {count} attempts")

        # Per-skill recovery breakdown
        print(f"\n📋 Per-Skill Recovery Breakdown:")
        print(f"{'─'*80}")

        for skill_stat in recovery_stats['skills']:
            status = '✅' if skill_stat['success'] else '❌'
            print(f"{status} Skill #{skill_stat['skill_idx']}: {skill_stat['skill_name']} ({skill_stat['skill_type']})")

            # Show recovery attempts if any
            had_retries = False
            if skill_stat['pose_retries'] > 0:
                print(f"      🔄 Above Pose Calc retries: {skill_stat['pose_retries']}")
                had_retries = True
            if skill_stat['mp_retries'] > 0:
                print(f"      🔄 Motion Planning retries: {skill_stat['mp_retries']}")
                had_retries = True
            if skill_stat['vla_retries'] > 0:
                print(f"      🔄 VLA Execution retries: {skill_stat['vla_retries']}")
                had_retries = True
            if skill_stat['pick_recovery_attempts'] > 0:
                recovery_status = '✅' if skill_stat['pick_recovery_success'] else '❌'
                print(f"      🔄 Pick Recovery attempts: {skill_stat['pick_recovery_attempts']} {recovery_status}")
                had_retries = True

            if not had_retries and skill_stat['success']:
                print(f"      ✨ Success on first attempt!")

        # Aggregate statistics
        print(f"\n📈 Aggregate Statistics:")
        total_pose_retries = sum(s['pose_retries'] for s in recovery_stats['skills'])
        total_mp_retries = sum(s['mp_retries'] for s in recovery_stats['skills'])
        total_vla_retries = sum(s['vla_retries'] for s in recovery_stats['skills'])
        total_pick_recovery = sum(s['pick_recovery_attempts'] for s in recovery_stats['skills'])
        successful_pick_recovery = sum(1 for s in recovery_stats['skills'] if s['pick_recovery_success'])

        print(f"   Total Above Pose Calc retries: {total_pose_retries}")
        print(f"   Total Motion Planning retries: {total_mp_retries}")
        print(f"   Total VLA Execution retries: {total_vla_retries}")
        if total_pick_recovery > 0:
            print(f"   Total Pick Recovery attempts: {total_pick_recovery} ({successful_pick_recovery} successful)")

        # Skills requiring recovery
        skills_with_recovery = sum(1 for s in recovery_stats['skills']
                                   if s['pose_retries'] > 0 or s['mp_retries'] > 0 or
                                      s['vla_retries'] > 0 or s['pick_recovery_attempts'] > 0)
        print(f"   Skills requiring recovery: {skills_with_recovery}/{total_skills}")

        print(f"{'='*80}\n")

    def _should_skip_reset(self, retry_count: int, prev_failure_stage) -> bool:
        """Whether this attempt re-invokes the policy in place (naive ablations).

        Attempt 1 always runs the full pipeline. 'naive' skips pose re-estimation and
        the approach-pose reset on every retry; 'naive_strict' only does so when the
        previous attempt actually reached and failed at the policy stage, so that
        motion-planning retries are not ablated along with the recovery strategy.
        """
        if retry_count == 0:
            return False
        if self.recovery_mode == 'naive':
            return True
        if self.recovery_mode == 'naive_strict':
            return prev_failure_stage == 'vla'
        return False

    def _execute_skill_once(self, skill: str, skill_idx: int, skill_type: str, skip_reset: bool = False) -> tuple:
        """
        Execute a single skill attempt: above pose calc → MP to above → VLA from above.
        No retry logic - just one attempt.

        Args:
            skill: Skill name
            skill_idx: Skill index for logging/saving
            skill_type: 'pick', 'place', or 'other'

        Returns:
            (success, task_done, vla_language, failure_stage, target_object)
            - success: True if all 3 stages passed
            - task_done: True if overall task completed
            - vla_language: Language instruction for VLA
            - failure_stage: 'above_pose', 'mp', 'vla', or None if success
            - target_object: MuJoCo object name (e.g., "akita_black_bowl_1_main")
        """
        # Clear masked wrist image - motion planner should use original wrist cam
        self.current_masked_wrist_img = None

        # Stage 1: Get target object and calculate above pose
        # Get target_object and vla_language from task config
        target_object, vla_language = self._get_skill_info(skill)

        if target_object is None or vla_language is None:
            print(f"  ✗ Skill info not found: {skill}")
            return False, False, None, 'above_pose', None

        # Naive retry: skip Stage 1 (pose re-estimation) and Stage 2 (reset to approach
        # pose); re-invoke the skill policy in place from the current EE state.
        if skip_reset:
            print(f"  ↻ [naive] re-invoking policy in place (no pose re-est, no MP reset)")
            grasped_obj = self._get_grasped_object(skill_idx) if skill_type == 'place' else None
            obs = self.env.env._get_observations()
            vla_success, task_done = self._execute_vla(
                skill, vla_language, obs, target_object, grasped_obj)
            self.current_masked_wrist_img = None
            if not vla_success:
                return False, task_done, vla_language, 'vla', target_object
            return True, task_done, vla_language, None, target_object

        # Get grasped object for place skills (needed for above height calculation)
        grasped_obj = self._get_grasped_object(skill_idx) if skill_type == 'place' else None

        # Get BDDL filename from skill_config using vla_language
        bddl_filename = None
        if vla_language:
            skill_key = vla_language.replace(' ', '_')
            skill_info = self.skill_config.get(skill_key)
            if skill_info and 'bddl_files' in skill_info:
                bddl_files = skill_info['bddl_files']
                bddl_filename = bddl_files[0] if bddl_files else None

        # Fallback to skill name if BDDL not found
        if bddl_filename is None:
            bddl_filename = skill

        # Get object pose and calculate above pose
        try:
            print(f"  ✓ Getting object pose for {target_object} by grasping {grasped_obj}...")
            obj_pos, _, above_height, bbox_z = get_special_object_handling(
                self.env, target_object, bddl_filename, self.above_height,
                skill_type=skill_type, grasped_object_name=grasped_obj
            )

            print(f"  ✓ Object pose: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}] (h={above_height:.3f}m)")

            if obj_pos is None:
                print(f"  ✗ Object pose not found: {target_object}")
                return False, False, vla_language, 'above_pose', target_object

            # Calculate above pose (no shift for evaluation)
            above_pos, above_quat, shift_info = calculate_above_pose(
                obj_pos,
                above_height=above_height,
                shift=False,  # No shift during evaluation
                xy_range=0.0,
                z_range=0.0,
                ori_range=0.0,
                bbox_z=bbox_z
            )

            print(f"  ✓ Above pose: [{above_pos[0]:.3f}, {above_pos[1]:.3f}, {above_pos[2]:.3f}] (h={above_height:.3f}m)")

        except Exception as e:
            print(f"  ✗ Above pose calculation error: {e}")
            return False, False, vla_language, 'above_pose', target_object

        # Stage 2: Move to above pose using motion planner
        # Only MPlib supports collision-aware planning with grasped objects
        if isinstance(self.motion_planner, MPlibMotionPlanner):
            move_success, result = self.motion_planner.move_to_pose(
                above_pos, above_quat,
                position_threshold=self.mp_pos_threshold,
                orientation_threshold=self.mp_ori_threshold,
                skill_type=skill_type,
                grasped_object_name=grasped_obj,
                save_pointcloud=self.mplib_save_pointcloud,
                pointcloud_output_dir=str(self.mplib_pointcloud_dir) if self.mplib_pointcloud_dir else None)
            observations = result[0]  # Extract observations from [obs, actions, rewards, dones]
        else:
            # Standard planner fallback (no grasped object support)
            move_success, observations = self.motion_planner.move_to_pose(
                above_pos, above_quat,
                position_threshold=self.mp_pos_threshold,
                orientation_threshold=self.mp_ori_threshold,
                skill_type=skill_type)

        # Normalize observations to a list and record frames
        obs_list = observations if isinstance(observations, list) else ([observations] if observations is not None else [])
        for obs in obs_list:
            self._record_frame(obs)

        if obs_list:
            self._save_mp_images(obs_list[-1], skill_idx, skill)

        obs = self.env.env._get_observations()

        if not move_success:
            print(f"  ✗ Motion planning failed")
            return False, False, vla_language, 'mp', target_object

        print(f"  ✓ Motion planning succeeded")

        # Visualization mode: skip VLA
        if self.visualize_poses:
            self._save_mp_images(obs, skill_idx, skill)
            return True, False, vla_language, None, target_object

        # Stage 3: VLA execution from above pose
        vla_success, task_done = self._execute_vla(skill, vla_language, obs, target_object, grasped_obj)

        # Clear masked wrist image after VLA execution - next skill's motion planning should use original
        self.current_masked_wrist_img = None

        if not vla_success:
            print(f"  ✗ VLA execution failed")
            return False, task_done, vla_language, 'vla', target_object

        print(f"  ✓ VLA execution succeeded")
        return True, task_done, vla_language, None, target_object

    def _get_grasped_object(self, current_skill_idx: int) -> Optional[str]:
        """Return mujoco name for previously picked object if available."""
        if not self.skill_sequence or current_skill_idx <= 1:
            return None

        if getattr(self, "_skill_metadata_cache", None) is None:
            config_path = _LILO_ROOT / "configs" / "tasks_and_skills.json"
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return None

            skill_mappings = config.get("skill_mappings", {})
            self._skill_metadata_cache = skill_mappings
            self._skill_metadata_lower = {name.lower(): meta for name, meta in skill_mappings.items()}

        for idx in range(current_skill_idx - 2, -1, -1):
            prev_skill = self.skill_sequence[idx]
            if "pick" not in prev_skill.lower():
                continue

            skill_meta = self._skill_metadata_cache.get(prev_skill)
            if skill_meta is None:
                skill_meta = self._skill_metadata_lower.get(prev_skill.lower())

            if not skill_meta:
                continue

            target_object = skill_meta.get("target_object")
            if target_object:
                return target_object

        return None

    def _execute_vla(self, skill: str, language: str, obs, target_object: str, grasped_object_name: Optional[str] = None) -> tuple:
        """
        Execute VLA skill.

        Args:
            skill: Skill name with numbers (e.g., "pick black bowl 2") - for success checking
            language: Clean VLA language (e.g., "pick black bowl") - for VLA execution
            obs: Current observation
            target_object: Target object name
            grasped_object_name: Grasped object name (for place skills) or None

        Returns:
            (skill_success, overall_task_done)
        """
        from collections import deque
        from lilo_vla.interaction.backends.robot_utils import get_action, normalize_gripper_action, invert_gripper_action, get_image_resize_size
        from lilo_vla.envs.libero_env import quat2axisangle
        from lilo_vla.interaction.backends.openvla_oft import resize_image_for_policy

        # Get fresh observation
        obs = self.env.env._get_observations()
        self._record_frame(obs)

        action_queue = deque(maxlen=self.cfg.num_open_loop_steps)
        resize_size = get_image_resize_size(self.cfg)
        overall_task_done = False

        for t in range(self.vla_horizon):
            if len(action_queue) == 0:
                wrist_img = self.get_libero_wrist_image(obs)

                # Apply background erasing if enabled
                if self.apply_background_erasing and target_object:
                    try:
                        seg_mask = create_wrist_segmentation_mask(self.env, target_object)
                        erased_mask = apply_random_erasing_to_mask(seg_mask, erasing_ratio=0.25, num_rectangles=5)
                        mask_3ch = (erased_mask[:, :, None] / 255.0).astype(np.float32)
                        wrist_img = (wrist_img * mask_3ch).astype(np.uint8)
                    except Exception as e:
                        pass  # Use original wrist_img if masking fails

                # Apply distractor masking if enabled
                if self.apply_distractor_masking and target_object:
                    try:
                        wrist_img = apply_distractor_masking(
                            self.env,
                            wrist_img,
                            target_object,
                            grasped_object_name=grasped_object_name,
                            maskable_objects=self.task_config.get("maskable_objects"),
                            resolution=256
                        )
                    except Exception as e:
                        pass  # Use original wrist_img if masking fails

                # Store masked wrist image for video/image recording (if any masking was applied)
                if self.apply_background_erasing or self.apply_distractor_masking:
                    self.current_masked_wrist_img = wrist_img.copy()

                wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

                # Generate mask as separate image if mask_as_agentview is enabled
                mask_img_resized = None
                if self.mask_as_agentview:
                    if target_object:
                        try:
                            # Use appropriate mask generation function based on skill type
                            is_place_skill = "place" in skill.lower()
                            if is_place_skill:
                                mask_img = create_wrist_segmentation_mask_with_grasped(
                                    self.env, target_object, grasped_object_name, resolution=256
                                )
                            else:
                                mask_img = create_wrist_segmentation_mask(
                                    self.env, target_object, resolution=256
                                )

                            # Convert 2D mask to 3-channel if needed
                            if len(mask_img.shape) == 2:
                                mask_img = np.stack([mask_img] * 3, axis=-1)

                            mask_img_resized = resize_image_for_policy(mask_img, resize_size)
                        except Exception as e:
                            print(f"Warning: Failed to generate mask as agentview: {e}")
                            mask_img_resized = None
                    else:
                        # No target object - create blank white mask
                        print(f"  ⚠️  No target object for {skill}, using blank mask")
                        mask_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
                        mask_img_resized = resize_image_for_policy(mask_img, resize_size)

                # Use axis-angle representation (matches data generation script line 450)
                if self.use_relative_pose:
                    # Get object pose
                    obj_pos, obj_quat_wxyz = get_object_pose(self.env, target_object)

                    # Calculate relative position
                    ee_pos = obs["robot0_eef_pos"]
                    relative_pos = ee_pos - obj_pos

                    # Calculate relative orientation
                    # Convert quaternions to scipy format [x,y,z,w]
                    ee_quat_xyzw = obs["robot0_eef_quat"]  # Already [x,y,z,w]
                    obj_quat_xyzw = np.array([obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]])

                    # Calculate relative rotation
                    obj_rot = R.from_quat(obj_quat_xyzw)
                    ee_rot = R.from_quat(ee_quat_xyzw)
                    relative_rot = obj_rot.inv() * ee_rot
                    relative_quat_xyzw = relative_rot.as_quat()

                    # Convert to axis-angle
                    relative_ori = quat2axisangle(relative_quat_xyzw)

                    observation = {
                        "full_image": wrist_img_resized,
                        "state": np.concatenate((
                            relative_pos,
                            relative_ori,
                            obs["robot0_gripper_qpos"]
                        ))
                    }
                else:
                    # Use absolute pose (backward compatibility)
                    observation = {
                        "full_image": wrist_img_resized,
                        "state": np.concatenate((
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]
                        ))
                    }

                # Add mask as second view if mask_as_agentview is enabled (must contain "wrist" in key name)
                if self.mask_as_agentview and mask_img_resized is not None:
                    observation["wrist_mask"] = mask_img_resized

                # Use clean language for VLA execution
                actions = get_action(
                    self.cfg,
                    self.vla,
                    observation,
                    language,  # ← Use clean language here
                    processor=self.processor,
                    action_head=self.action_head,
                    proprio_projector=self.proprio_projector,
                    noisy_action_projector=self.noisy_action_projector,
                    use_film=self.cfg.use_film
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)

            obs, _, done, _ = self.env.step(action.tolist())
            self._record_frame(obs)

            if done:
                overall_task_done = True

            # Check if skill is complete using BDDL predicates
            # Use skill (with numbers) for success checking
            if self.skill_checker.check_skill_success_by_language(self.env, skill, debug=False):
                print(f"   ✅ VLA skill completed in {t+1} steps (goal achieved)")
                return True, overall_task_done

        print(f"   ❌ VLA skill failed after {self.vla_horizon} steps (goal not achieved)")
        
        # Print detailed diagnostic analysis if verbose flag is enabled
        if self.verbose_vla_failure:
            print(f"\n   📊 VLA Failure Analysis:")
            self.skill_checker.check_skill_success_by_language(self.env, skill, debug=True)
        
        return False, overall_task_done

    def _create_env(self) -> bool:
        """Create LIBERO environment using task config."""
        try:
            from libero.libero.envs import OffScreenRenderEnv

            from lilo_vla.benchmark import get_bddl_path

            # Load task from tasks_and_skills.json
            # Match by both name AND benchmark to handle duplicate task names
            task_config = None
            for task in self.task_config.get('long_horizon_tasks', []):
                if task['name'] == self.task_name and task.get('benchmark') == self.benchmark_name:
                    task_config = task
                    break

            # Fallback: if no exact match, try matching by name only (backward compatibility)
            if task_config is None:
                for task in self.task_config.get('long_horizon_tasks', []):
                    if task['name'] == self.task_name:
                        task_config = task
                        break

            if task_config is None:
                available_tasks = [f"{t['name']} ({t.get('benchmark', 'N/A')})" for t in self.task_config.get('long_horizon_tasks', [])]
                print(f"❌ Task '{self.task_name}' with benchmark '{self.benchmark_name}' not found in config")
                print(f"📋 Available tasks: {available_tasks}")
                return False

            # Get BDDL file and benchmark from config
            bddl_file = task_config.get('bddl_file')
            task_benchmark = task_config.get('benchmark', self.benchmark_name)

            if not bddl_file:
                print(f"❌ No bddl_file specified for task '{self.task_name}'")
                return False

            print(f"📚 Task: {self.task_name}")
            print(f"📚 Benchmark: {task_benchmark}")
            print(f"📄 BDDL file: {bddl_file}")

            # Construct BDDL path
            # benchmark name matches folder name directly
            problem_folder = task_benchmark

            bddl_path = str(get_bddl_path(problem_folder, bddl_file))

            env_args = {
                "bddl_file_name": bddl_path,
                "camera_heights": 256, "camera_widths": 256,
                "has_renderer": False, "has_offscreen_renderer": True,
                "ignore_done": True, "use_camera_obs": True, "control_freq": 20,
                "camera_names": ["agentview", "robot0_eye_in_hand"]
            }

            env = OffScreenRenderEnv(**env_args)
            env.reset()

            # Set pipeline reference for motion planner to access recording functions
            env.pipeline = self

            self.env = env
            # Initialize motion planner based on type
            if self.motion_planner_type == "standard":
                self.motion_planner = StandardMotionPlanner(
                    env, method=self.motion_planner_method, num_steps=self.mp_steps,
                    pos_gain=self.mp_pos_gain, ori_gain=self.mp_ori_gain,
                    pos_success_threshold=self.mp_pos_success_threshold,
                    ori_success_threshold=self.mp_ori_success_threshold
                )
                print(f"🔧 Using Standard Motion Planner")
            elif self.motion_planner_type == "mplib":
                # Match data generation initialization exactly (line 568 in generate_above_augmented_demos.py)
                self.motion_planner = MPlibMotionPlanner(
                    env=self.env,
                    collision_aware=self.collision_aware,
                    velocity_factor=self.mp_velocity_factor,
                    time_step=0.025,  # Match data generation default
                    safety_margin=self.mplib_safety_margin,
                    verbose=self.mplib_verbose
                )
                print(f"🔧 Using MPlib Motion Planner (RRT-based path planning)")
            else:
                raise ValueError(f"Unknown motion planner type: {self.motion_planner_type}. Use 'standard' or 'mplib'.")

            return True
        except Exception as e:
            print(f"Failed to create env: {e}")
            import traceback
            traceback.print_exc()
            return False

    def execute(self) -> bool:
        """Execute pipeline.

        Returns:
            bool: True if overall task succeeded

        Note: self.success_count tracks number of successfully completed skills (for eval purposes)
        """
        # Initialize tracking attributes early (for eval script)
        self.success_count = 0
        self.total_skills = 0
        self.recovery_stats = None  # Will store detailed skill-level statistics

        # Reset video recording for each trial (important for multi-trial eval)
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Create new video filename in the same subfolder for this trial
        if self.video_dir is not None and not self.visualize_poses:
            task_clean = self.task_name.replace(" ", "_").lower()
            bg_suffix = '_bg_erasing' if self.apply_background_erasing else ''
            self.video_filename = self.video_dir / f"{task_clean}{bg_suffix}_trial{self.trial_num:02d}.mp4"

        if not self._create_env():
            print("❌ Failed to create environment")
            return False

        print("✅ Environment created")

        # Stabilize and start video
        for i in range(20):
            obs, _, _, _ = self.env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
            if i == 0:
                self._start_video(obs)
            self._record_frame(obs)

        # Get skills
        skills = self.task_planner.plan_task_sequence(self.task_name)
        if not skills:
            return False

        # Update total_skills now that we have the skill list
        self.skill_sequence = skills
        self.total_skills = len(skills)

        overall_task_done = False
        previous_skill = None  # Track for place recovery
        previous_skill_idx = None

        # Recovery statistics tracking
        recovery_stats = {
            'skills': [],  # Per-skill recovery info
            'final_failure': None  # What caused the final failure
        }

        print(f"\n{'='*80}")
        print(f"🎯 TASK: {self.task_name}")
        print(f"   Total skills: {len(skills)}")
        print(f"{'='*80}")

        for i, skill in enumerate(skills, 1):
            print(f"\n{'─'*80}")
            print(f"Skill {i}/{len(skills)}: {skill}")

            # Initialize skill recovery stats
            skill_stats = {
                'skill_name': skill,
                'skill_idx': i,
                'skill_type': None,
                'success': False,
                'pose_retries': 0,
                'mp_retries': 0,
                'vla_retries': 0,
                'pick_recovery_attempts': 0,
                'pick_recovery_success': False
            }

            # Determine skill type
            skill_lower = skill.lower()
            if 'pick' in skill_lower:
                skill_type = 'pick'
            elif 'place' in skill_lower or 'stack' in skill_lower:
                skill_type = 'place'
            elif 'open' in skill_lower or 'close' in skill_lower or 'turn' in skill_lower:
                skill_type = 'other'
            else:
                skill_type = 'pick'  # Default

            skill_stats['skill_type'] = skill_type

            # Place skills: double retries with pick recovery on VLA failure
            if skill_type == 'place':
                max_retries = self.max_vla_retries * 1
                vla_success = False
                vla_language = None
                retry_count = 0
                failure_stage = None

                while retry_count < max_retries and not vla_success:
                    if retry_count > 0:
                        print(f"   🔄 Place retry {retry_count + 1}/{max_retries}")

                    # Execute place skill
                    success, task_done, vla_lang, failure_stage, target_obj = self._execute_skill_once(
                        skill, i, skill_type,
                        skip_reset=self._should_skip_reset(retry_count, failure_stage))

                    if task_done:
                        overall_task_done = True

                    if success:
                        vla_success = True
                        vla_language = vla_lang
                        print(f"   ✅ Place succeeded on attempt {retry_count + 1}/{max_retries}")
                        break

                    # Track failure stage
                    if failure_stage == 'above_pose':
                        skill_stats['pose_retries'] += 1
                    elif failure_stage == 'mp':
                        skill_stats['mp_retries'] += 1
                    elif failure_stage == 'vla':
                        skill_stats['vla_retries'] += 1

                    # Handle failure
                    if (self.recovery_mode not in ('naive', 'naive_strict')
                            and failure_stage == 'vla' and previous_skill and 'pick' in previous_skill.lower()):
                        # VLA failure: recover by re-executing pick
                        print(f"    ⚠️  VLA failed - attempting pick recovery: {previous_skill}")
                        skill_stats['pick_recovery_attempts'] += 1

                        # Open gripper first
                        for _ in range(5):
                            obs, _, _, _ = self.env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
                            self._record_frame(obs)

                        # Reset predicate baselines before pick recovery to prevent false positives
                        # (e.g., when object is already elevated from failed place operation)
                        self.skill_checker.reset_predicate_baselines(previous_skill)

                        # Re-execute pick with retry loop
                        pick_success = False
                        pick_retry_count = 0
                        while pick_retry_count < self.max_vla_retries and not pick_success:
                            if pick_retry_count > 0:
                                print(f"   🔄 Pick recovery retry {pick_retry_count + 1}/{self.max_vla_retries}")

                            pick_success, _, pick_lang, pick_failure_stage, pick_target_obj = self._execute_skill_once(
                                previous_skill, previous_skill_idx, 'pick')

                            if pick_success:
                                # # Pick post-actions (close gripper + move up)
                                post_actions(pick_lang, self.env, self.motion_planner,
                                           skill_succeeded=True, next_skill=None, grasped_object=pick_target_obj,
                                           max_move_up_retries=self.max_move_up_retries)
                                print(f"   ✅ Pick recovery complete on attempt {pick_retry_count + 1}/{self.max_vla_retries}")
                                skill_stats['pick_recovery_success'] = True
                                break
                            else:
                                pick_retry_count += 1

                        if not pick_success:
                            print(f"   ❌ Pick recovery failed after {self.max_vla_retries} attempts")
                    elif failure_stage in ['above_pose', 'mp']:
                        # Above pose/MP failure: just retry, no pick recovery needed
                        pass  # Retry message will be printed at loop start

                    retry_count += 1

            else:
                # Non-place skills: regular retry loop
                vla_success = False
                vla_language = None
                retry_count = 0
                failure_stage = None

                while retry_count < self.max_vla_retries and not vla_success:
                    if retry_count > 0:
                        print(f"  ↻ Retry {retry_count + 1}/{self.max_vla_retries}")

                    # Reset predicate baselines for pick skills to ensure fresh baseline at current position
                    if skill_type == 'pick':
                        self.skill_checker.reset_predicate_baselines(skill)

                    # Execute skill
                    success, task_done, vla_lang, failure_stage, target_obj = self._execute_skill_once(
                        skill, i, skill_type,
                        skip_reset=self._should_skip_reset(retry_count, failure_stage))

                    if task_done:
                        overall_task_done = True

                    if success:
                        vla_success = True
                        vla_language = vla_lang
                        if retry_count > 0:
                            print(f"  ✓ Succeeded on attempt {retry_count + 1}")
                    else:
                        # Track failure stage
                        if failure_stage == 'above_pose':
                            skill_stats['pose_retries'] += 1
                        elif failure_stage == 'mp':
                            skill_stats['mp_retries'] += 1
                        elif failure_stage == 'vla':
                            skill_stats['vla_retries'] += 1

                        if retry_count + 1 >= self.max_vla_retries:
                            print(f"  ✗ Failed after {self.max_vla_retries} attempts (final: {failure_stage})")
                        retry_count += 1

            # Post-actions ONLY if skill succeeded
            if vla_success and vla_language:
                next_skill = skills[i] if i < len(skills) else None
                # For place skills, get grasped_object from previous pick skill
                grasped_obj_for_post = self._get_grasped_object(i) if skill_type == 'place' else target_obj
                post_actions(vla_language, self.env, self.motion_planner,
                           skill_succeeded=True, next_skill=next_skill, grasped_object=grasped_obj_for_post,
                           max_move_up_retries=self.max_move_up_retries)
                self.success_count += 1
                skill_stats['success'] = True
                # Track for place recovery
                previous_skill = skill
                previous_skill_idx = i
            else:
                max_r = self.max_vla_retries * 2 if skill_type == 'place' else self.max_vla_retries
                print(f"   ⚠️  Skipping post_actions (failed after {max_r} attempts)")

                # Record final failure cause
                if skill_stats['vla_retries'] > 0:
                    final_stage = 'vla'
                elif skill_stats['mp_retries'] > 0:
                    final_stage = 'mp'
                elif skill_stats['pose_retries'] > 0:
                    final_stage = 'above_pose'
                else:
                    final_stage = 'unknown'

                recovery_stats['final_failure'] = {
                    'skill': skill,
                    'skill_idx': i,
                    'skill_type': skill_type,
                    'stage': final_stage,
                    'attempted_recoveries': {
                        'above_pose': skill_stats['pose_retries'],
                        'mp': skill_stats['mp_retries'],
                        'vla': skill_stats['vla_retries'],
                        'pick_recovery': skill_stats['pick_recovery_attempts']
                    }
                }

                # Save skill stats before exiting
                recovery_stats['skills'].append(skill_stats)

                if not self.continue_on_failure:
                    self._print_recovery_summary(recovery_stats, self.success_count, self.total_skills)
                    self.recovery_stats = recovery_stats
                    return False

            # Save skill stats
            recovery_stats['skills'].append(skill_stats)

        print(f"\n{'='*60}")
        print(f"3️⃣ Overall Long-Horizon Task: {'✅ SUCCESS' if overall_task_done else '❌ FAILED'}")
        print(f"{'='*60}")

        # Print recovery statistics summary
        self._print_recovery_summary(recovery_stats, self.success_count, self.total_skills)

        self._stop_video()

        if self.env:
            self.env.close()

        # Store recovery stats for eval script access
        self.recovery_stats = recovery_stats

        return overall_task_done


def main():
    parser = argparse.ArgumentParser(description='Execute long horizon pipeline')
    parser.add_argument('--task_name', type=str,
                       default='Complete Kitchen Organization (Original)',
                       choices=[
                           'Complete Kitchen Organization (Original)',
                           'Organize Table (Original)',
                           'Cooking Preparation Setup (Original)',
                           'Turn On The Stove And Put The Moka Pot On It (Original)',
                           'Put Both The Alphabet Soup And The Cream Cheese Box In The Basket (Original)',
                           'Put Both The Alphabet Soup And The Tomato Sauce In The Basket (Original)',
                           'Put Both The Cream Cheese Box And The Butter In The Basket (Original)',
                           'Put The White Mug On The Left Plate And Put The Yellow And White Mug On The Right Plate (Original)',
                           'Put The White Mug On The Plate And Put The Chocolate Pudding To The Right Of The Plate (Original)',
                           'Complete Kitchen Organization (Reorder 1)',
                           'Organize Table (Reorder 1)',
                           'Cooking Preparation Setup (Reorder 1)',
                           'Complete Kitchen Organization (Reorder 2)',
                           'Organize Table (Reorder 2)',
                           'Cooking Preparation Setup (Reorder 2)',
                           'Turn On The Stove And Put The Moka Pot On It (Reorder 1)',
                           'Put Both The Alphabet Soup And The Cream Cheese Box In The Basket (Reorder 1)',
                           'Put Both The Alphabet Soup And The Tomato Sauce In The Basket (Reorder 1)',
                           'Put Both The Cream Cheese Box And The Butter In The Basket (Reorder 1)',
                           'Put The White Mug On The Left Plate And Put The Yellow And White Mug On The Right Plate (Reorder 1)',
                           'Put The White Mug On The Plate And Put The Chocolate Pudding To The Right Of The Plate (Reorder 1)',
                       ])
    parser.add_argument('--vla_checkpoint', type=str,
                       default="runs/libero_above_atomic/1.0.0/openvla-7b+libero_above_atomic+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt")
    parser.add_argument('--wrist_only', action='store_true', default=True)
    parser.add_argument('--visualize_poses', action='store_true')
    parser.add_argument('--ee_backward_offset', type=float, default=0.0)
    parser.add_argument('--motion_planner_method', type=str, default='cartesian_linear',
                       choices=['ik_setjoint', 'cartesian_linear'])
    parser.add_argument('--motion_planner_type', type=str, default='mplib', choices=['standard', 'mplib'],
                       help='Motion planner type: standard (default) or mplib (RRT-based path planning)')
    parser.add_argument('--continue_on_failure', action='store_true', default=False)
    parser.add_argument('--motion_planner_steps', type=int, default=400)
    parser.add_argument('--motion_planner_pos_gain', type=float, default=5.0)
    parser.add_argument('--motion_planner_ori_gain', type=float, default=5.0)
    parser.add_argument('--motion_planner_pos_threshold', type=float, default=0.02,
                       help='Position threshold for early convergence during motion planning (default: 0.02m = 2cm)')
    parser.add_argument('--motion_planner_ori_threshold', type=float, default=0.524,
                       help='Orientation threshold for early convergence during motion planning (default: 0.524 rad ≈ 30°)')
    parser.add_argument('--motion_planner_pos_success_threshold', type=float, default=0.025,
                       help='Looser position threshold for final success check after all steps (default: 0.025m = 2.5cm)')
    parser.add_argument('--motion_planner_ori_success_threshold', type=float, default=0.524,
                       help='Looser orientation threshold for final success check after all steps (default: 0.524 rad ≈ 30°)')
    parser.add_argument('--motion_planner_velocity_factor', type=float, default=0.9,
                       help='Velocity factor to scale controller limits (0.0-1.0, default: 0.9)')

    # MPlib-specific parameters
    parser.add_argument('--mplib_collision_aware', type=lambda x: x.lower() == 'true', default=True,
                       help='Enable MPlib collision avoidance planning (default: True)')
    parser.add_argument('--mplib_joint_vel_limits', type=float, default=2.0,
                       help='MPlib joint velocity limits in rad/s (lower = smoother/more accurate, default: 2.0)')
    parser.add_argument('--mplib_joint_acc_limits', type=float, default=4.0,
                       help='MPlib joint acceleration limits in rad/s^2 (lower = smoother, default: 4.0)')
    parser.add_argument('--mplib_waypoint_target_ratio', type=int, default=50,
                       help='MPlib target number of waypoints (higher = more accurate but slower, default: 50)')
    parser.add_argument('--mplib_planning_time', type=float, default=1.0,
                       help='MPlib planning time budget in seconds (higher = better paths, default: 1.0)')
    parser.add_argument('--mplib_safety_margin', type=float, default=1.1,
                       help='MPlib AABB safety margin multiplier (default: 1.1 = 10% margin)')
    parser.add_argument('--mplib_erode_kernel_size', type=int, default=3,
                       help='MPlib mask erosion kernel size for boundary pixel removal (default: 3)')
    parser.add_argument('--mplib_outlier_std_ratio', type=float, default=0.0,
                       help='MPlib 3D outlier removal std ratio (default: 0.0 = disabled, try 2.0 for aggressive filtering)')
    parser.add_argument('--mplib_save_pointcloud', action='store_true', default=False,
                       help='Save scene pointcloud and camera images for each MPlib planning call (default: False)')
    parser.add_argument('--mplib_verbose', action='store_true', default=False,
                       help='Enable verbose logging for MPlib motion planner (default: False)')

    parser.add_argument('--obj_ee_pairs_dir', type=str,
                       default="datasets/hdf5_datasets/atomic_local_demos_augmented_pick_10_place_15_atomic_15/ee_target_obj_pairs/simple_extraction_obs_pose_v1",
                       help='Directory containing object-EE pose pairs (default: datasets/hdf5_datasets/atomic_local_demos_augmented_pick_10_place_15_atomic_15/ee_target_obj_pairs/simple_extraction_obs_pose)')
    parser.add_argument('--max_vla_retries', type=int, default=6,
                       help='Maximum number of retry attempts for VLA execution (default: 6)')
    parser.add_argument('--max_move_up_retries', type=int, default=5,
                       help='Maximum number of retry attempts for post-action move up 5cm operation (default: 5)')
    parser.add_argument('--vla_horizon', type=int, default=200,
                       help='Maximum steps per skill VLA execution before considered failure (default: 200)')
    parser.add_argument('--verbose_vla_failure', action='store_true', default=False,
                       help='Print detailed diagnostic analysis when VLA fails (default: False)')
    parser.add_argument('--above_height', type=float, default=0.1,
                       help='Height above object for initial pose (meters, default: 0.03m = 3cm)')
    parser.add_argument('--use_relative_pose', type=lambda x: x.lower() == 'true', default=True,
                       help='Use relative EE position and orientation (relative to object) for VLA input (default: True)')
    parser.add_argument('--apply_background_erasing', action='store_true', default=False,
                       help='Apply random erasing to wrist camera background (default: False)')
    parser.add_argument('--apply_distractor_masking', action='store_true', default=False,
                       help='Mask out distractor objects in wrist camera by blacking out their bounding boxes (default: False)')
    parser.add_argument('--mask_as_agentview', action='store_true', default=False,
                       help='Use segmentation mask as second image input (for dual-view models trained with wrist+mask, default: False)')
    parser.add_argument('--benchmark', type=str, default='long_horizon_tasks_v1',
                       choices=['long_horizon_tasks_v0', 'long_horizon_tasks_v1', 'long_horizon_tasks_libero_long'],
                       help='LIBERO benchmark to use (default: long_horizon_tasks_v1). Note: benchmark is also read from task config.')
    parser.add_argument('--recovery_mode', type=str, default='full', choices=['full', 'naive', 'naive_strict'],
                       help="Recovery strategy: 'full' = pose re-estimation + approach-pose reset + pick backtracking (default); 'naive' = in-place policy re-invocation only, same retry budget")

    args = parser.parse_args()

    pipeline = LongHorizonPipeline(
        vla_checkpoint=args.vla_checkpoint,
        task_name=args.task_name,
        wrist_only=args.wrist_only,
        visualize_poses=args.visualize_poses,
        ee_offset=args.ee_backward_offset,
        motion_planner_method=args.motion_planner_method,
        motion_planner_type=args.motion_planner_type,
        continue_on_failure=args.continue_on_failure,
        mp_steps=args.motion_planner_steps,
        mp_pos_gain=args.motion_planner_pos_gain,
        mp_ori_gain=args.motion_planner_ori_gain,
        mp_pos_threshold=args.motion_planner_pos_threshold,
        mp_ori_threshold=args.motion_planner_ori_threshold,
        mp_pos_success_threshold=args.motion_planner_pos_success_threshold,
        mp_ori_success_threshold=args.motion_planner_ori_success_threshold,
        mp_velocity_factor=args.motion_planner_velocity_factor,
        obj_ee_pairs_dir=args.obj_ee_pairs_dir,
        max_vla_retries=args.max_vla_retries,
        max_move_up_retries=args.max_move_up_retries,
        collision_aware=args.mplib_collision_aware,
        mplib_joint_vel_limits=args.mplib_joint_vel_limits,
        mplib_joint_acc_limits=args.mplib_joint_acc_limits,
        mplib_waypoint_target_ratio=args.mplib_waypoint_target_ratio,
        mplib_planning_time=args.mplib_planning_time,
        mplib_safety_margin=args.mplib_safety_margin,
        mplib_erode_kernel_size=args.mplib_erode_kernel_size,
        mplib_outlier_std_ratio=args.mplib_outlier_std_ratio,
        mplib_save_pointcloud=args.mplib_save_pointcloud,
        mplib_verbose=args.mplib_verbose,
        vla_horizon=args.vla_horizon,
        verbose_vla_failure=args.verbose_vla_failure,
        above_height=args.above_height,
        use_relative_pose=args.use_relative_pose,
        apply_background_erasing=args.apply_background_erasing,
        apply_distractor_masking=args.apply_distractor_masking,
        mask_as_agentview=args.mask_as_agentview,
        benchmark_name=args.benchmark,
        recovery_mode=args.recovery_mode
    )

    start = time.time()
    success = pipeline.execute()
    elapsed = time.time() - start

    print(f"\n⏱️  Total: {elapsed:.1f}s")
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())