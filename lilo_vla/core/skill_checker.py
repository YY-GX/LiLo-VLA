#!/usr/bin/env python3
"""
Skill Success Checker for Individual Atomic Skills

This module provides utilities to check success conditions for individual atomic skills
based on BDDL goal predicates stored in tasks_and_skills.json. Unlike the environment's
_check_success() which only checks overall long-horizon task success, these functions check
individual skill goals.

Key components:
- Load BDDL predicates directly from tasks_and_skills.json
- Evaluate predicates using LIBERO's predicate functions
- Check individual skill success independent of overall task

BDDL Predicate Reference:
- Open(obj): Check if articulated object (drawer/cabinet) is open
- Close(obj): Check if articulated object is closed
- On(obj1, obj2): Check if obj1 is on top of obj2
- In(obj1, obj2): Check if obj1 is inside obj2 (requires containment region)
- PickedUp(obj): Check if object is lifted 3cm+ above initial height
- TurnOn(obj): Check if object is turned on (e.g., stove burner)
- TurnOff(obj): Check if object is turned off

Based on LIBERO's implementation:
- libero/libero/envs/predicates/base_predicates.py
- libero/libero/envs/bddl_base_domain.py
- libero/libero/envs/problems/libero_kitchen_tabletop_manipulation.py

Note: BDDL file parsing and instance number remapping functions have been removed.
All predicates with correct object names are now stored in tasks_and_skills.json.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# lilo-vla/  (this file lives at lilo-vla/lilo_vla/core/skill_checker.py)
_LILO_ROOT = Path(__file__).resolve().parents[2]

# Import config loader
from lilo_vla.utils.config import get_cached_config, get_skill_mapping, refresh_config

# Refresh config to get latest object mappings at module load
refresh_config()

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs.predicates import eval_predicate_fn


def parse_bddl_goal(bddl_file_path: str) -> List[Tuple]:
    """
    Parse BDDL file and extract goal conditions.

    Args:
        bddl_file_path: Path to BDDL file

    Returns:
        List of goal predicates as tuples, e.g.:
        - [('open', 'wooden_cabinet_1_bottom_region')]
        - [('on', 'akita_black_bowl_1_main', 'flat_stove_1_init_region')]
        - [('in', 'chocolate_pudding_1', 'wooden_cabinet_1_top_region')]
    """
    if not os.path.exists(bddl_file_path):
        raise FileNotFoundError(f"BDDL file not found: {bddl_file_path}")

    parsed_problem = BDDLUtils.robosuite_parse_problem(bddl_file_path)
    goal_state = parsed_problem.get('goal_state', [])

    return goal_state


def check_skill_success(env, bddl_file_path: str, debug: bool = False) -> bool:
    """
    Check if individual skill's goal conditions are satisfied.

    This function mimics LIBERO's _check_success() and _eval_predicate() methods
    but works for individual atomic skills rather than overall long-horizon tasks.

    Args:
        env: LIBERO gym environment with object_states_dict (may be wrapped)
        bddl_file_path: Path to skill's BDDL file
        debug: If True, print detailed predicate evaluation info

    Returns:
        True if all goal predicates are satisfied, False otherwise
    """
    try:
        # Unwrap environment if needed to access object_states_dict
        base_env = env
        while hasattr(base_env, 'env') and not hasattr(base_env, 'object_states_dict'):
            base_env = base_env.env

        goal_state = parse_bddl_goal(bddl_file_path)

        if debug:
            print(f"\n🔍 Checking skill success for: {os.path.basename(bddl_file_path)}")
            print(f"   Goal predicates: {goal_state}")

        result = True
        for state in goal_state:
            predicate_result = _eval_predicate(base_env, state, debug=debug)
            result = predicate_result and result

            if debug:
                status = "✅" if predicate_result else "❌"
                print(f"   {status} Predicate {state}: {predicate_result}")

        if debug:
            print(f"   {'✅' if result else '❌'} Overall skill success: {result}\n")

        return result

    except Exception as e:
        print(f"❌ Error checking skill success: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False


def _eval_predicate(env, state: Tuple, debug: bool = False) -> bool:
    """
    Evaluate a single BDDL predicate.

    This directly mirrors LIBERO's _eval_predicate() implementation in:
    libero/libero/envs/problems/libero_kitchen_tabletop_manipulation.py

    Args:
        env: LIBERO gym environment with object_states_dict
        state: Predicate tuple (predicate_name, arg1, [arg2])
        debug: If True, print debug info

    Returns:
        True if predicate is satisfied, False otherwise
    """
    if len(state) == 3:
        predicate_fn_name = state[0]
        object_1_name = state[1]
        object_2_name = state[2]

        if debug:
            print(f"      Binary predicate: {predicate_fn_name}({object_1_name}, {object_2_name})")

        if object_1_name not in env.object_states_dict:
            # Try object mapping from config (e.g., white_cabinet_1 -> wooden_cabinet_1)
            config = get_cached_config()
            object_mappings = config.get("object_mappings", {})
            mapped_name = object_mappings.get(object_1_name)

            if mapped_name and mapped_name in env.object_states_dict:
                object_1_name = mapped_name
                if debug:
                    print(f"   Mapped object name: {state[1]} -> {mapped_name}")
            else:
                # Try to find similar object name (without suffix)
                base_name = object_1_name.rsplit('_', 1)[0] if '_' in object_1_name else object_1_name
                found_alternative = False
                for obj_name in env.object_states_dict.keys():
                    if obj_name.startswith(base_name):
                        object_1_name = obj_name
                        found_alternative = True
                        break

                if not found_alternative:
                    print(f"⚠️  Warning: Object '{object_1_name}' not found in object_states_dict")
                    print(f"      Available objects: {list(env.object_states_dict.keys())}")
                    return False

        if object_2_name not in env.object_states_dict:
            # Try object mapping from config (e.g., white_cabinet_1_top_region -> wooden_cabinet_1_top_region)
            config = get_cached_config()
            object_mappings = config.get("object_mappings", {})
            mapped_name = object_mappings.get(object_2_name)

            if mapped_name and mapped_name in env.object_states_dict:
                object_2_name = mapped_name
                if debug:
                    print(f"   Mapped object name: {state[2]} -> {mapped_name}")
            else:
                # Try to find similar object name (without suffix)
                base_name = object_2_name.rsplit('_', 1)[0] if '_' in object_2_name else object_2_name
                found_alternative = False
                for obj_name in env.object_states_dict.keys():
                    if obj_name.startswith(base_name):
                        object_2_name = obj_name
                        found_alternative = True
                        break

                if not found_alternative:
                    print(f"⚠️  Warning: Object '{object_2_name}' not found in object_states_dict")
                    print(f"      Available objects: {list(env.object_states_dict.keys())}")
                    return False

        # Add detailed debugging for predicate evaluation
        obj1_state = env.object_states_dict[object_1_name]
        obj2_state = env.object_states_dict[object_2_name]

        if debug:
            print(f"🔍 DEBUG: Evaluating {predicate_fn_name}({object_1_name}, {object_2_name})")
            print(f"   Object 1 ({object_1_name}):")
            print(f"     Position: {getattr(obj1_state, 'position', 'N/A')}")
            print(f"     Quaternion: {getattr(obj1_state, 'quaternion', 'N/A')}")
            print(f"   Object 2 ({object_2_name}):")
            print(f"     Position: {getattr(obj2_state, 'position', 'N/A')}")
            print(f"     Quaternion: {getattr(obj2_state, 'quaternion', 'N/A')}")

        result = eval_predicate_fn(
            predicate_fn_name,
            obj1_state,
            obj2_state,
        )

        # Add detailed success/failure analysis based on predicate type
        if debug:
            import numpy as np
            predicate_lower = predicate_fn_name.lower()
            
            if predicate_lower == 'on':
                # For 'On' predicate, show distance and height difference
                pos1 = obj1_state.get_geom_state()['pos']
                pos2 = obj2_state.get_geom_state()['pos']
                height_diff = pos1[2] - pos2[2]  # Z difference
                xy_distance = np.linalg.norm(pos1[:2] - pos2[:2])
                contact = obj2_state.check_contact(obj1_state)
                
                print(f"   📊 'On' Predicate Analysis:")
                print(f"      Contact: {'✅ YES' if contact else '❌ NO'}")
                print(f"      Z-height: {height_diff:.4f}m", end='')
                if height_diff <= 0.01:
                    print(" (object at or below target ✓)")
                else:
                    print(" (object ABOVE target ❌)")
                print(f"      XY distance: {xy_distance:.4f}m", end='')
                if xy_distance < 0.03:
                    print(f" < 0.03m threshold ✓")
                else:
                    print(f" > 0.03m threshold (exceeded by {xy_distance - 0.03:.4f}m) ❌")
            
            elif predicate_lower == 'in':
                # For 'In' predicate, show containment info
                pos1 = obj1_state.get_geom_state()['pos']
                pos2 = obj2_state.get_geom_state()['pos']
                contains = obj2_state.check_contain(obj1_state)
                contact = obj2_state.check_contact(obj1_state)
                
                print(f"   📊 'In' Predicate Analysis:")
                print(f"      Contact: {'✅ YES' if contact else '❌ NO'}")
                print(f"      Contains: {'✅ YES' if contains else '❌ NO'}")
                if not contains:
                    # Show distance to container
                    distance = np.linalg.norm(pos1 - pos2)
                    print(f"      Distance to container: {distance:.4f}m")
            
            elif predicate_lower == 'up':
                # For 'Up' predicate (PickedUp), show height
                pos1 = obj1_state.get_geom_state()['pos']
                z_pos = pos1[2]
                threshold = 1.0  # From Up predicate implementation
                
                print(f"   📊 'Up' Predicate Analysis:")
                print(f"      Z position: {z_pos:.4f}m")
                print(f"      Threshold: {threshold}m")
                if z_pos >= threshold:
                    print(f"      Status: ✅ Object is high enough")
                else:
                    print(f"      Status: ❌ Object too low (need {threshold - z_pos:.4f}m more)")

        if debug:
            print(f"   Result: {'✅ SUCCESS' if result else '❌ FAILED'}")
        return result

    elif len(state) == 2:
        predicate_fn_name = state[0]
        object_name = state[1]

        if debug:
            print(f"      Unary predicate: {predicate_fn_name}({object_name})")

        if object_name not in env.object_states_dict:
            # Try object mapping from config (e.g., white_cabinet_1_top_region -> wooden_cabinet_1_top_region)
            config = get_cached_config()
            object_mappings = config.get("object_mappings", {})
            mapped_name = object_mappings.get(object_name)

            if mapped_name and mapped_name in env.object_states_dict:
                object_name = mapped_name
                if debug:
                    print(f"   Mapped object name: {state[1]} -> {mapped_name}")
            else:
                # Try to find similar object name (without suffix)
                base_name = object_name.rsplit('_', 1)[0] if '_' in object_name else object_name
                found_alternative = False
                for obj_name in env.object_states_dict.keys():
                    if obj_name.startswith(base_name):
                        object_name = obj_name
                        found_alternative = True
                        break

                if not found_alternative:
                    print(f"⚠️  Warning: Object '{object_name}' not found in object_states_dict")
                    print(f"      Available objects: {list(env.object_states_dict.keys())}")
                    return False

        obj_state = env.object_states_dict[object_name]
        
        # Add detailed diagnostics for unary predicates
        if debug:
            import numpy as np
            predicate_lower = predicate_fn_name.lower()
            
            if predicate_lower in ['open', 'close']:
                # For Open/Close predicates, show joint states
                joint_states = obj_state.get_joint_state()
                if joint_states:
                    print(f"   📊 '{predicate_fn_name.capitalize()}' Predicate Analysis:")
                    for i, joint_val in enumerate(joint_states):
                        print(f"      Joint {i}: {joint_val:.4f} rad ({np.degrees(joint_val):.2f}°)")
                else:
                    print(f"   📊 '{predicate_fn_name.capitalize()}' Predicate Analysis:")
                    print(f"      No joint states available")
            
            elif predicate_lower in ['turnon', 'turnoff']:
                # For TurnOn/TurnOff predicates
                print(f"   📊 '{predicate_fn_name.capitalize()}' Predicate Analysis:")
                print(f"      Checking if object is {'on' if predicate_lower == 'turnon' else 'off'}")

        return eval_predicate_fn(
            predicate_fn_name,
            obj_state
        )

    else:
        print(f"⚠️  Warning: Invalid predicate format: {state}")
        return False


def get_skill_bddl_file(skill_language: str, csv_file: str = None) -> Optional[str]:
    """
    Get BDDL file path for a given skill language description using exact config lookup.

    Args:
        skill_language: Language description like "pick black bowl 2"
        csv_file: Deprecated - kept for compatibility

    Returns:
        Path to BDDL file, or None if not found
    """
    config = get_cached_config()
    skill_mappings = config.get("skill_mappings", {})

    skill_clean = skill_language.strip().lower()

    # Use exact match only - no fuzzy matching since we have precise config
    if skill_clean in skill_mappings:
        skill_data = skill_mappings[skill_clean]
        bddl_files = skill_data.get("bddl_files", [])
        if bddl_files:
            first_bddl = bddl_files[0]
            bddl_dir = str(_LILO_ROOT / "configs" / "bddl" / "atomic_skills")
            bddl_path = os.path.join(bddl_dir, first_bddl)

            if os.path.exists(bddl_path):
                print(f"✅ Found BDDL file for '{skill_language}': {os.path.basename(bddl_path)}")
                return bddl_path
            else:
                print(f"⚠️  BDDL file not found: {bddl_path}")
                return None

    print(f"⚠️  Skill '{skill_language}' not found in unified config (exact match required)")
    print(f"💡 Available skills in config:")
    for available_skill in sorted(skill_mappings.keys())[:10]:
        print(f"   - {available_skill}")
    if len(skill_mappings) > 10:
        print(f"   ... and {len(skill_mappings) - 10} more")

    return None


def extract_instance_number(skill_language: str) -> Optional[int]:
    """
    Extract instance number from skill language.

    Args:
        skill_language: e.g., "place black bowl 2 on the plate 2"

    Returns:
        Instance number if found, None otherwise

    Examples:
        "pick black bowl 2" → 2
        "place black bowl 1 on plate 1" → 1
        "pick black bowl" → None
    """
    import re

    # Match trailing number in skill (e.g., "pick black bowl 2")
    # Look for space followed by digit(s) at word boundary
    match = re.search(r'\s(\d+)(?:\s|$)', skill_language)

    if match:
        return int(match.group(1))

    return None


def remap_object_names_with_instance(goal_state: List[Tuple], instance_num: Optional[int]) -> List[Tuple]:
    """
    Remap object names in BDDL goal predicates to match instance number.

    BDDL files use generic names (akita_black_bowl_1_main, plate_1_main),
    but skills operate on specific instances (akita_black_bowl_2_main, plate_2_main).

    Args:
        goal_state: List of predicates from BDDL
        instance_num: Instance number from skill (e.g., 2 from "pick black bowl 2")

    Returns:
        Remapped goal predicates

    Example:
        Input: [('on', 'akita_black_bowl_1_main', 'plate_1_main')]
        Instance: 2
        Output: [('on', 'akita_black_bowl_2_main', 'plate_2_main')]
    """
    import re

    if instance_num is None:
        return goal_state

    remapped = []

    for predicate in goal_state:
        if len(predicate) == 3:
            pred_name, obj1, obj2 = predicate

            # Remap obj1 (replace _1_ with _N_)
            obj1_remapped = re.sub(r'_1_', f'_{instance_num}_', obj1)
            obj1_remapped = re.sub(r'_1$', f'_{instance_num}', obj1_remapped)

            # Remap obj2 (replace _1_ with _N_)
            obj2_remapped = re.sub(r'_1_', f'_{instance_num}_', obj2)
            obj2_remapped = re.sub(r'_1$', f'_{instance_num}', obj2_remapped)

            remapped.append((pred_name, obj1_remapped, obj2_remapped))

        elif len(predicate) == 2:
            pred_name, obj = predicate

            # Remap obj (replace _1_ with _N_)
            obj_remapped = re.sub(r'_1_', f'_{instance_num}_', obj)
            obj_remapped = re.sub(r'_1$', f'_{instance_num}', obj_remapped)

            remapped.append((pred_name, obj_remapped))

        else:
            remapped.append(predicate)

    return remapped


def check_skill_success_by_language(env, skill_language: str, debug: bool = False) -> bool:
    """
    Convenience function to check skill success using language description.

    Uses bddl_predicates directly from unified config instead of parsing BDDL files.

    Args:
        env: LIBERO gym environment
        skill_language: Language description like "pick black bowl 2"
        debug: If True, print detailed info

    Returns:
        True if skill succeeded, False otherwise
    """
    try:
        # Get predicates directly from unified config
        config = get_cached_config()
        skill_language_lower = skill_language.lower().strip()

        if skill_language_lower not in config['skill_mappings']:
            print(f"❌ Skill '{skill_language}' not found in unified config")
            return False

        skill_data = config['skill_mappings'][skill_language_lower]
        bddl_predicates = skill_data.get("bddl_predicates", [])

        if not bddl_predicates:
            print(f"❌ No bddl_predicates found for skill '{skill_language}'")
            return False

        if debug:
            print(f"📝 BDDL predicates from config: {bddl_predicates}")

        # Check skill success using predicates from config
        return check_skill_success_with_predicates(env, bddl_predicates, debug=debug)

    except Exception as e:
        if debug:
            print(f"❌ Error checking skill success for '{skill_language}': {e}")
            import traceback
            traceback.print_exc()
        return False


def check_skill_success_with_predicates(env, goal_state: List[Tuple], debug: bool = False) -> bool:
    """
    Check skill success given explicit goal predicates.

    Args:
        env: LIBERO gym environment
        goal_state: List of goal predicates
        debug: If True, print detailed info

    Returns:
        True if all predicates satisfied
    """
    try:
        # Unwrap environment if needed
        base_env = env
        while hasattr(base_env, 'env') and not hasattr(base_env, 'object_states_dict'):
            base_env = base_env.env

        if debug:
            print(f"\n🔍 Checking skill success with custom predicates")
            print(f"   Goal predicates: {goal_state}")

        result = True
        for state in goal_state:
            predicate_result = _eval_predicate(base_env, state, debug=debug)
            result = predicate_result and result

            if debug:
                status = "✅" if predicate_result else "❌"
                print(f"   {status} Predicate {state}: {predicate_result}")

        if debug:
            print(f"   {'✅' if result else '❌'} Overall skill success: {result}\n")

        return result

    except Exception as e:
        print(f"❌ Error checking skill success: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False


def reset_predicate_baselines(skill_language: str, debug: bool = False):
    """
    Reset PickedUp predicate baseline heights for objects involved in this skill.

    This function clears stale baseline heights from the PickedUp predicate's internal state.
    This is necessary when an object's position has changed significantly (e.g., after a failed
    place operation that leaves the object elevated), and we need the predicate to record a
    new baseline height at the current position.

    Args:
        skill_language: Skill language description (e.g., "pick frying pan 1")
        debug: Whether to print debug information

    Example:
        Before pick recovery, reset baselines to prevent false positives when object is already elevated:
        >>> reset_predicate_baselines("pick frying pan 1")
    """
    try:
        from libero.libero.envs.predicates import get_predicate_fn

        # Get the PickedUp predicate instance
        pickedup_predicate = get_predicate_fn('pickedup')

        # Extract object name from skill using config
        config = get_cached_config()
        skill_language_lower = skill_language.lower().strip()

        if skill_language_lower not in config['skill_mappings']:
            if debug:
                print(f"   ⚠️  Skill '{skill_language}' not found in config, skipping baseline reset")
            return

        skill_data = config['skill_mappings'][skill_language_lower]
        target_object = skill_data.get('target_object')

        if not target_object:
            if debug:
                print(f"   ⚠️  No target_object found for skill '{skill_language}', skipping baseline reset")
            return

        # Clear this object's baseline height if it exists
        if hasattr(pickedup_predicate, 'initial_heights') and target_object in pickedup_predicate.initial_heights:
            del pickedup_predicate.initial_heights[target_object]
            print(f"   🔄 Reset baseline height for {target_object}")
        elif debug:
            print(f"   ℹ️  No baseline height stored for {target_object}, nothing to reset")

    except Exception as e:
        if debug:
            print(f"   ⚠️  Error resetting predicate baselines: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test skill success checker')
    parser.add_argument('--bddl-file', type=str, help='Path to BDDL file')
    parser.add_argument('--skill', type=str, help='Skill language description')
    parser.add_argument('--list-predicates', action='store_true', help='List available predicates')

    args = parser.parse_args()

    if args.list_predicates:
        from libero.libero.envs.predicates import get_predicate_fn_dict
        predicates = get_predicate_fn_dict()

        print("\n📋 Available BDDL Predicates:")
        print("=" * 60)
        for name in sorted(predicates.keys()):
            print(f"   - {name}")
        print("=" * 60)
        print(f"\nTotal: {len(predicates)} predicates\n")

    elif args.bddl_file:
        print(f"\n📄 Parsing BDDL file: {args.bddl_file}")
        goal_state = parse_bddl_goal(args.bddl_file)

        print(f"\n🎯 Goal conditions:")
        for i, state in enumerate(goal_state, 1):
            print(f"   {i}. {state}")
        print()

    elif args.skill:
        bddl_file = get_skill_bddl_file(args.skill)
        if bddl_file:
            print(f"\n✅ Found BDDL file: {bddl_file}")
            goal_state = parse_bddl_goal(bddl_file)
            print(f"\n🎯 Goal conditions for '{args.skill}':")
            for i, state in enumerate(goal_state, 1):
                print(f"   {i}. {state}")
            print()
        else:
            print(f"\n❌ Could not find BDDL file for skill: {args.skill}")

    else:
        parser.print_help()