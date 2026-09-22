#!/usr/bin/env python3
"""
Contact Object Detector for Long Horizon Pipeline.

This module provides utilities to find all manipulable objects in the scene
that match the objects required for long horizon tasks. Used by the pipeline
to identify available objects for skill execution.

Key Functions:
- get_all_manipulable_objects(): Main pipeline function - returns scene objects
- get_object_pose(): Helper to get specific object pose
"""

import os
import sys
import csv
import re
import numpy as np
from typing import Optional, List, Tuple, Dict

from pathlib import Path

from libero.libero import benchmark, get_libero_path
from lilo_vla.envs.libero_env import get_libero_env

# Repository root (…/lilo-vla), resolved relative to this file so nothing depends on the CWD.
_LILO_ROOT = Path(__file__).resolve().parents[2]


def load_csv_object_patterns(csv_file: str) -> List[str]:
    """
    Load object patterns from CSV mapping file.
    
    Args:
        csv_file: Path to skills_objects_mapping_updated.csv
        
    Returns:
        List of unique object patterns from the 'object' column
    """
    object_patterns = set()
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                object_name = row['object'].strip()
                if object_name:
                    object_patterns.add(object_name)
        
        return list(object_patterns)
        
    except Exception as e:
        print(f"❌ Error loading CSV patterns: {e}")
        return []


def create_pattern_regex(pattern: str) -> str:
    """
    Convert object pattern to regex for wildcard matching.
    
    Args:
        pattern: Object pattern like "akita_black_bowl_1_main"
        
    Returns:
        Regex pattern like "akita_black_bowl_.*_main"
    """
    # Replace numbers with wildcard pattern
    regex_pattern = re.sub(r'_\d+_', '_.*_', pattern)
    return regex_pattern


def find_scene_objects(env) -> List[str]:
    """
    Find all objects in the current MuJoCo scene.
    
    Args:
        env: LIBERO environment with MuJoCo simulation
        
    Returns:
        List of object body names found in scene
    """
    model = env.sim.model
    scene_objects = []
    
    # Object keywords that indicate manipulable objects
    object_keywords = [
        'white_bowl', 'black_bowl', 'plate', 'mug', 'bottle', 'pan', 'pot', 'stove', 'cabinet', 
        'drawer', 'microwave', 'ketchup', 'wine', 'frying', 'moka', 'pudding',
        'chocolate', 'akita', 'white', 'black', 'red'
    ]
    
    for i in range(model.nbody):
        body_name = model.body_id2name(i)
        if body_name:
            # Check if body name contains object keywords
            body_lower = body_name.lower()
            if any(keyword in body_lower for keyword in object_keywords):
                # Skip robot parts
                if not any(robot_part in body_lower for robot_part in ['robot', 'gripper', 'link', 'joint']):
                    scene_objects.append(body_name)
    
    print(f"🎬 Found {len(scene_objects)} objects in scene: {scene_objects}")
    
    return scene_objects


def match_objects_to_patterns(scene_objects: List[str], csv_patterns: List[str]) -> List[str]:
    """
    Match scene objects to CSV patterns using wildcard matching.
    
    Args:
        scene_objects: List of object names found in scene
        csv_patterns: List of object patterns from CSV
        
    Returns:
        List of scene objects that match CSV patterns
    """
    matched_objects = []
    
    for scene_obj in scene_objects:
        for pattern in csv_patterns:
            # Try exact match first
            if scene_obj == pattern:
                matched_objects.append(scene_obj)
                break
            
            # Try wildcard pattern matching
            regex_pattern = create_pattern_regex(pattern)
            if re.match(regex_pattern, scene_obj):
                matched_objects.append(scene_obj)
                break
    
    return list(set(matched_objects))  # Remove duplicates


def get_all_manipulable_objects(env, csv_file: str = None) -> List[str]:
    """
    Main pipeline function: Find all manipulable objects in scene that match CSV patterns.
    
    This is the function called by execute_long_horizon_pipeline.py.
    
    Args:
        env: LIBERO environment with MuJoCo simulation  
        csv_file: Path to skills_objects_mapping_updated.csv (optional, uses default)
        
    Returns:
        List of manipulable object names found in scene that match required patterns
    """
    # Use default CSV path if not provided
    if csv_file is None:
        csv_file = str(_LILO_ROOT / 'configs' / 'annotations' / 'skills_objects_mapping_updated.csv')
    
    print(f"🔍 Finding manipulable objects in scene...")
    
    # Step 1: Load object patterns from CSV
    csv_patterns = load_csv_object_patterns(csv_file)
    if not csv_patterns:
        print("❌ No CSV patterns loaded")
        return []
    
    print(f"📋 Loaded {len(csv_patterns)} object patterns from CSV")
    
    # Step 2: Find all objects in scene
    scene_objects = find_scene_objects(env)
    print(f"🎬 Found {len(scene_objects)} objects in scene")
    
    # Step 3: Match scene objects to CSV patterns
    matched_objects = match_objects_to_patterns(scene_objects, csv_patterns)
    
    print(f"✅ Matched {len(matched_objects)} manipulable objects:")
    for obj in sorted(matched_objects):
        print(f"   - {obj}")
    
    return matched_objects


def get_object_pose(env, object_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get pose (position and quaternion) of a specific object in the scene.
    
    Args:
        env: LIBERO environment with MuJoCo simulation
        object_name: Name of the object to find
        
    Returns:
        Tuple of (position, quaternion) or (None, None) if not found
    """
    model = env.sim.model
    data = env.sim.data
    
    try:
        # Try to find the object by name
        if object_name in [model.body_id2name(i) for i in range(model.nbody)]:
            body_id = model.body_name2id(object_name)
            position = data.body_xpos[body_id].copy()
            quaternion = data.body_xquat[body_id].copy()
            return position, quaternion
        else:
            return None, None
            
    except Exception as e:
        print(f"❌ Error getting pose for {object_name}: {e}")
        return None, None


def create_test_environment(task_name: str = "Cooking Preparation Setup"):
    """
    Create test environment using long_horizon_tasks_v0 benchmark.
    
    Args:
        task_name: Name of the long horizon task
        
    Returns:
        LIBERO environment for testing
    """
    # Get benchmark
    bm_name = "long_horizon_tasks_v0"
    task_suite = benchmark.get_benchmark_dict()[bm_name]()
    
    # Map friendly names to LIBERO task names
    task_name_mapping = {
        "cooking preparation setup": "LONG_HORIZON_cooking_preparation_setup",
        "complete kitchen organization": "LONG_HORIZON_complete_kitchen_organization", 
        "switch table objects": "LONG_HORIZON_switch_table_objects"
    }
    
    # Find matching task
    target_task = None
    task_key = task_name.lower().strip()
    
    if task_key in task_name_mapping:
        libero_task_name = task_name_mapping[task_key]
        for task in task_suite.tasks:
            if task.name == libero_task_name:
                target_task = task
                break
    
    if target_task is None:
        print(f"❌ Task '{task_name}' not found in benchmark")
        available_tasks = [task.name for task in task_suite.tasks]
        print(f"📋 Available tasks: {available_tasks}")
        print(f"💡 Try one of: {list(task_name_mapping.keys())}")
        return None
    
    print(f"✅ Found task: {target_task.name}")
    
    # Create environment
    try:
        env, _ = get_libero_env(target_task, model_family='openvla', resolution=256)
        obs = env.reset()
        print(f"✅ Environment created and reset successfully")
        return env
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return None


def main():
    """Test the contact object detector."""
    print("🔍 Testing Contact Object Detector")
    print("=" * 50)
    
    # Test with Cooking Preparation Setup task
    task_name = "Cooking Preparation Setup"
    print(f"🎯 Testing with task: {task_name}")
    
    # Create test environment
    env = create_test_environment(task_name)
    if env is None:
        return
    
    try:
        # Test main function
        manipulable_objects = get_all_manipulable_objects(env)
        
        print(f"\n📊 Results:")
        print(f"   Total objects found: {len(manipulable_objects)}")
        
        # Test pose detection for found objects
        if manipulable_objects:
            print(f"\n🎯 Testing pose detection for first few objects:")
            for obj in manipulable_objects[:5]:  # Test first 5 objects
                pos, quat = get_object_pose(env, obj)
                if pos is not None:
                    print(f"   ✅ {obj}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                else:
                    print(f"   ❌ {obj}: pose not found")
        
        env.close()
        print(f"\n🎉 Testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        if env:
            env.close()


if __name__ == "__main__":
    main()