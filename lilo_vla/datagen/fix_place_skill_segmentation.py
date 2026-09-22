#!/usr/bin/env python3
"""
Fix eye_in_hand_segmentation for place skills by adding grasped object segmentation.

This script reads place skill HDF5 files, sets the state for each step, and regenerates
correct segmentation masks that include gripper + target object + grasped object.
"""

import argparse
import os
import h5py
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# Package bootstrap (writes the LIBERO config and applies the compatibility
# patches). MUST be imported before `libero`.
import lilo_vla  # noqa: F401

from libero.libero.envs import OffScreenRenderEnv
from lilo_vla.interaction.masking import create_wrist_segmentation_mask_with_grasped
from lilo_vla.perception.oracle import get_object_pose

# Repository root (the `lilo-vla/` directory).
_LILO_ROOT = Path(__file__).resolve().parents[2]

# Paths
SKILL_CONFIG_PATH = str(_LILO_ROOT / "configs" / "skill_config.json")
BDDL_BASE_PATH = str(_LILO_ROOT / "configs" / "bddl" / "atomic_skills")

# INPUT_DIR / OUTPUT_DIR are set from --input_dir / --output_dir in main();
# there are no defaults.
INPUT_DIR = None
OUTPUT_DIR = None

def load_skill_config() -> Dict:
    """Load skill config from JSON."""
    with open(SKILL_CONFIG_PATH, 'r') as f:
        return json.load(f)


def find_grasped_object_in_env(env, grasped_object_list: Optional[List[str]]) -> Optional[str]:
    """Find which grasped object actually exists in the environment."""
    if grasped_object_list is None:
        return None
    
    for obj_name in grasped_object_list:
        try:
            pos, quat = get_object_pose(env, obj_name)
            if pos is not None:
                return obj_name
        except:
            continue
    
    return None


def find_target_object_in_env(env, target_object_list: List[str]) -> Optional[str]:
    """Find which target object exists in the environment."""
    for obj_name in target_object_list:
        try:
            pos, quat = get_object_pose(env, obj_name)
            if pos is not None:
                return obj_name
        except:
            continue
    
    return None


def get_skill_info_from_name(skill_name: str, skill_config: Dict) -> Optional[Dict]:
    """Get skill info from skill name."""
    if skill_name not in skill_config:
        return None
    
    skill_info = skill_config[skill_name]
    return {
        'target_object': skill_info.get('target_object', []),
        'grasped_object_name': skill_info.get('grasped_object_name', None),
        'bddl_files': skill_info.get('bddl_files', []),
        'skill_type': skill_info.get('skill_type', 'other')
    }


def process_single_demo(demo_group, env, target_object_list: List[str], grasped_object_list: Optional[List[str]]) -> int:
    """Process a single demo and update segmentation masks. Returns number of steps processed."""
    # Load states
    if 'states' not in demo_group:
        return 0
    
    states = demo_group['states'][:]
    obs_group = demo_group['obs']
    
    if 'eye_in_hand_segmentation' not in obs_group:
        return 0
    
    num_steps = len(states)
    if num_steps == 0:
        return 0
    
    seg_dataset = obs_group['eye_in_hand_segmentation']
    steps_processed = 0
    
    # Process each step
    for step_idx in range(num_steps):
        try:
            # Set state
            state = states[step_idx]
            env.sim.set_state_from_flattened(state)
            
            # Find target object at this state (may vary)
            actual_target = find_target_object_in_env(env, target_object_list)
            if actual_target is None:
                continue
            
            # Find grasped object at this state
            grasped_object_name = find_grasped_object_in_env(env, grasped_object_list)
            
            # Create corrected segmentation mask
            seg_mask = create_wrist_segmentation_mask_with_grasped(
                env, actual_target, grasped_object_name, resolution=256
            )
            
            # Update segmentation
            if seg_dataset.shape[0] > step_idx:
                seg_dataset[step_idx] = seg_mask
                steps_processed += 1
            
        except Exception:
            continue  # Skip this step if error occurs
    
    return steps_processed


def process_skill_hdf5(hdf5_path: str, skill_name: str, skill_config: Dict, output_path: str) -> bool:
    """Process a single place skill HDF5 file."""
    # Get skill info
    skill_info = get_skill_info_from_name(skill_name, skill_config)
    if skill_info is None or skill_info['skill_type'] != 'place':
        return False
    
    target_object_list = skill_info['target_object']
    grasped_object_list = skill_info['grasped_object_name']
    bddl_files = skill_info['bddl_files']
    
    if not bddl_files or not target_object_list:
        return False
    
    # Initialize environment with first BDDL file
    bddl_file = os.path.join(BDDL_BASE_PATH, bddl_files[0])
    if not os.path.exists(bddl_file):
        return False
    
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=256,
        camera_widths=256,
        horizon=10000
    )
    env.reset()
    
    try:
        # Open HDF5 files
        with h5py.File(hdf5_path, 'r') as f_in:
            if 'data' not in f_in:
                return False
            
            demo_keys = [k for k in f_in['data'].keys() if k.startswith('demo_')]
            if not demo_keys:
                return False
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Copy file and update segmentations
            with h5py.File(output_path, 'w') as f_out:
                # Copy top-level groups (except data)
                for key in f_in.keys():
                    if key != 'data':
                        f_in.copy(key, f_out, name=key)
                
                # Process data group
                data_in = f_in['data']
                data_out = f_out.create_group('data')
                
                # Process each demo with progress bar
                demo_pbar = tqdm(demo_keys, desc=f"  {skill_name}", leave=False, unit="demo")
                for demo_key in demo_pbar:
                    demo_group_in = data_in[demo_key]
                    demo_group_out = data_out.create_group(demo_key)
                    
                    # Copy non-obs datasets
                    for key in demo_group_in.keys():
                        if key != 'obs':
                            demo_group_in.copy(key, demo_group_out, name=key)
                    
                    # Copy obs group
                    obs_group_in = demo_group_in['obs']
                    obs_group_out = demo_group_out.create_group('obs')
                    
                    for obs_key in obs_group_in.keys():
                        if obs_key == 'eye_in_hand_segmentation':
                            # Create dataset, will update later
                            seg_data = obs_group_in[obs_key]
                            seg_dataset = obs_group_out.create_dataset(
                                obs_key,
                                shape=seg_data.shape,
                                dtype=seg_data.dtype
                            )
                            seg_dataset[:] = seg_data[:]
                        else:
                            obs_group_in.copy(obs_key, obs_group_out, name=obs_key)
                    
                    # Update segmentations
                    process_single_demo(
                        demo_group_out,
                        env,
                        target_object_list,
                        grasped_object_list
                    )
                demo_pbar.close()
        
        return True
    
    except Exception:
        return False
    
    finally:
        env.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Regenerate eye-in-hand segmentation for place-skill HDF5 files"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the per-skill HDF5 files to fix")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write the fixed HDF5 files to")
    args = parser.parse_args()

    global INPUT_DIR, OUTPUT_DIR
    INPUT_DIR = os.path.abspath(args.input_dir)
    OUTPUT_DIR = os.path.abspath(args.output_dir)

    print(f"Loading skill config...")
    skill_config = load_skill_config()
    
    # Find all place skill HDF5 files
    place_skill_files = []
    for filename in os.listdir(INPUT_DIR):
        if filename.startswith('place_') and filename.endswith('.hdf5'):
            skill_name = filename.replace('.hdf5', '')
            if skill_name in skill_config:
                hdf5_path = os.path.join(INPUT_DIR, filename)
                output_path = os.path.join(OUTPUT_DIR, filename)
                place_skill_files.append((skill_name, hdf5_path, output_path))
    
    print(f"Found {len(place_skill_files)} place skill files to process")
    
    # Process each skill with progress bar
    successful_skills = 0
    failed_skills = 0
    
    for skill_name, hdf5_path, output_path in tqdm(place_skill_files, desc="Processing skills"):
        try:
            success = process_skill_hdf5(hdf5_path, skill_name, skill_config, output_path)
            if success:
                successful_skills += 1
            else:
                failed_skills += 1
        except Exception as e:
            print(f"\nError processing {skill_name}: {e}")
            failed_skills += 1
    
    print(f"\nCompleted: {successful_skills} successful, {failed_skills} failed")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

