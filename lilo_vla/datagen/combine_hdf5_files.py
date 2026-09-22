#!/usr/bin/env python3
"""
Combine all success HDF5 files and .init files from v1-v10 into a single folder.

This script:
1. Reads all success HDF5 files from v1-v10
2. Combines demos from all versions into one HDF5 file per skill
3. Combines .init files similarly
4. Generates statistics JSON file
"""

import os
import json
import h5py
import pickle
import glob
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm

# Base directory containing the v* folders produced by the parallel demo
# generation launcher. Set from --base_dir in main(); there is no default.
BASE_DIR = None
# Combined output directory. Set from --output_dir in main() (defaults to
# <base_dir>/all, which is what the released run used).
OUTPUT_DIR = None

# Version directories
V_DIRS = [f"v{i}" for i in range(1, 11)]

# Path to HDF5 check results (to skip invalid files). Set from
# --hdf5_check_results in main(); empty means "no files are skipped".
HDF5_CHECK_RESULTS = ""

# Only combine multi-BDDL skills (set to True to filter, False to combine all)
MULTI_BDDL_ONLY = False

# Skills with >3 BDDL files (heterogeneous demos issue)
MULTI_BDDL_SKILLS = [
    "pick_black_bowl",
    "pick_frying_pan",
    "place_black_bowl_on_the_plate",
    "place_black_bowl_on_top_of_the_cabinet"
]

# Skills to process in make-up mode (for skills with low demo counts)
MAKE_UP_BDDL_SKILLS = [
    # "place_tomato_sauce_in_basket",
    # "turn_on_the_stove",
    # "place_alphabet_soup_in_basket",
    # "place_ketchup_in_top_drawer_of_the_cabinet",
    # "place_white_bowl_on_the_plate"

    # "turn_on_the_stove",
    # "place_tomato_sauce_in_basket",    
    # "place_alphabet_soup_in_basket",
    "place_butter_in_basket",
    # "place_butter_in_basket",
    # "turn_off_the_stove",  # this has many demos, i list here cuz no seg for it, so need to regenerate
    # "place_frying_pan_on_the_stove",  # this has many demos, i list here cuz no seg for it, so need to regenerate
    # "place_moka_pot_on_the_stove"  # this has many demos, i list here cuz no seg for it, so need to regenerate
]


def load_invalid_files(check_results_file: str) -> Set[str]:
    """
    Load list of invalid HDF5 files from check results.
    
    Args:
        check_results_file: Path to hdf5_check_results.json
        
    Returns:
        Set of invalid file paths (absolute paths)
    """
    invalid_files = set()
    
    if not os.path.exists(check_results_file):
        print(f"  ⚠️  Warning: Check results file not found: {check_results_file}")
        return invalid_files
    
    try:
        with open(check_results_file, 'r') as f:
            results = json.load(f)
        
        # Get invalid file paths
        for file_info in results.get('invalid_files', []):
            file_path = file_info.get('file_path', '')
            if file_path:
                invalid_files.add(os.path.abspath(file_path))
        
        print(f"  Loaded {len(invalid_files)} invalid files to skip")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not load check results: {e}")
    
    return invalid_files




def analyze_source_distribution(hdf5_path: str) -> Dict[str, int]:
    """
    Analyze source_hdf5 distribution in an HDF5 file.

    Returns:
        dict: {source_base_name: count}
    """
    source_counts = {}

    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' not in f:
                return source_counts

            for demo_key in f['data'].keys():
                demo = f['data'][demo_key]
                if 'metadata' not in demo:
                    continue

                meta = demo['metadata']
                source_hdf5 = None

                # Try to get source_hdf5 from attrs
                if 'source_hdf5' in meta.attrs:
                    source_hdf5 = meta.attrs['source_hdf5']
                    if isinstance(source_hdf5, bytes):
                        source_hdf5 = source_hdf5.decode('utf-8')

                if source_hdf5:
                    # Extract base name from path
                    base_name = os.path.basename(source_hdf5).replace('_demo.hdf5', '')
                    source_counts[base_name] = source_counts.get(base_name, 0) + 1
    except Exception as e:
        print(f"    ⚠️  Warning: Could not analyze {hdf5_path}: {e}")

    return source_counts


def get_selected_source(skill_name: str, hdf5_path: str) -> str:
    """
    Get the source_hdf5 to use for filtering (the one with max count).

    Returns:
        str: Selected source base name, or None if no filtering
    """
    if not MULTI_BDDL_ONLY or skill_name not in MULTI_BDDL_SKILLS:
        return None

    source_counts = analyze_source_distribution(hdf5_path)

    if not source_counts:
        print(f"    ⚠️  Warning: No source_hdf5 metadata found in {skill_name}.hdf5")
        return None

    # Select source with max count
    selected_source = max(source_counts.items(), key=lambda x: x[1])

    print(f"    📊 Source distribution:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        marker = "✓ SELECTED" if source == selected_source[0] else ""
        print(f"       {source}: {count} demos {marker}")
    print(f"       Total: {sum(source_counts.values())} demos → Using {selected_source[1]} from {selected_source[0]}")

    return selected_source[0]


def get_demo_source_mapping(hdf5_path: str) -> Dict[int, str]:
    """
    Get mapping from demo index to source_hdf5 base name.

    Returns:
        dict: {demo_index: source_base_name}
    """
    demo_sources = {}

    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' not in f:
                return demo_sources

            for demo_key in sorted(f['data'].keys()):
                demo_idx = int(demo_key.split('_')[1])
                demo = f['data'][demo_key]

                if 'metadata' not in demo:
                    continue

                meta = demo['metadata']
                source_hdf5 = None

                # Try to get source_hdf5 from attrs
                if 'source_hdf5' in meta.attrs:
                    source_hdf5 = meta.attrs['source_hdf5']
                    if isinstance(source_hdf5, bytes):
                        source_hdf5 = source_hdf5.decode('utf-8')

                if source_hdf5:
                    # Extract base name from path
                    base_name = os.path.basename(source_hdf5).replace('_demo.hdf5', '')
                    demo_sources[demo_idx] = base_name
    except Exception as e:
        print(f"    ⚠️  Warning: Could not get demo sources from {hdf5_path}: {e}")

    return demo_sources


def load_init_states(file_path: str) -> List[np.ndarray]:
    """
    Load init states from a .init file.

    Args:
        file_path: Path to .init file

    Returns:
        List of init state arrays
    """
    init_states = []
    try:
        with open(file_path, 'rb') as f:
            init_states = pickle.load(f)
    except Exception as e:
        print(f"  ⚠️  Warning: Could not load {file_path}: {e}")
        return []

    return init_states


def copy_hdf5_group(source_group: h5py.Group, dest_group: h5py.Group, name: str = None):
    """
    Recursively copy an HDF5 group from source to destination.
    This is much faster than loading into memory and saving.
    
    Args:
        source_group: Source HDF5 group
        dest_group: Destination HDF5 group (parent)
        name: Name for the copied group (if None, use source_group.name)
    """
    if name is None:
        name = os.path.basename(source_group.name) if source_group.name else 'group'
    
    # Create new group in destination
    new_group = dest_group.create_group(name)
    
    # Copy all attributes
    for key, val in source_group.attrs.items():
        new_group.attrs[key] = val
    
    # Copy all datasets and subgroups
    for key in source_group.keys():
        item = source_group[key]
        if isinstance(item, h5py.Dataset):
            # Copy dataset with same properties
            source_ds = item
            # Handle compression: only pass compression_opts if compression is set
            create_kwargs = {
                'data': source_ds,
            }
            if source_ds.compression is not None:
                create_kwargs['compression'] = source_ds.compression
                if source_ds.compression_opts is not None:
                    create_kwargs['compression_opts'] = source_ds.compression_opts
            if source_ds.shuffle:
                create_kwargs['shuffle'] = source_ds.shuffle
            if source_ds.fletcher32:
                create_kwargs['fletcher32'] = source_ds.fletcher32
            
            dest_ds = new_group.create_dataset(key, **create_kwargs)
            # Copy dataset attributes
            for attr_key, attr_val in source_ds.attrs.items():
                dest_ds.attrs[attr_key] = attr_val
        elif isinstance(item, h5py.Group):
            # Recursively copy subgroup
            copy_hdf5_group(item, new_group, key)


def combine_hdf5_files_efficient(input_files: List[str], output_file: str) -> Tuple[int, int, int]:
    """
    Efficiently combine multiple HDF5 files by directly copying groups.
    Much faster than loading into memory.
    
    Args:
        input_files: List of input HDF5 file paths
        output_file: Path to output HDF5 file
        
    Returns:
        (total_demos, num_non_shifted, num_shifted) tuple
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    total_demos = 0
    num_non_shifted = 0
    num_shifted = 0
    demo_idx = 0
    
    # Open output file for writing
    with h5py.File(output_file, 'w') as out_file:
        out_data_group = out_file.create_group('data')
        
        # Process each input file
        for input_file in input_files:
            try:
                with h5py.File(input_file, 'r') as in_file:
                    if 'data' not in in_file:
                        continue
                    
                    in_data_group = in_file['data']
                    demo_keys = sorted(in_data_group.keys())
                    
                    # Copy each demo group and read metadata in the same pass
                    for demo_key in demo_keys:
                        demo_group = in_data_group[demo_key]
                        
                        # Read metadata for statistics (fast, read while file is open)
                        metadata = {}
                        if 'metadata' in demo_group:
                            meta_group = demo_group['metadata']
                            # Read attributes (fast)
                            for key in meta_group.attrs.keys():
                                try:
                                    val = meta_group.attrs[key]
                                    if isinstance(val, str) and val.startswith('{'):
                                        metadata[key] = json.loads(val)
                                    elif isinstance(val, (np.integer, np.floating)):
                                        metadata[key] = val.item()
                                    else:
                                        metadata[key] = val
                                except:
                                    pass
                            # Read scalar datasets
                            for key in meta_group.keys():
                                try:
                                    val = meta_group[key]
                                    if isinstance(val, h5py.Dataset):
                                        if val.shape == ():
                                            metadata[key] = val[()].item() if hasattr(val[()], 'item') else val[()]
                                        elif len(val.shape) == 1 and val.shape[0] == 1:
                                            metadata[key] = val[0].item() if hasattr(val[0], 'item') else val[0]
                                except:
                                    pass
                        
                        # Copy demo group to output with new index
                        new_demo_key = f"demo_{demo_idx}"
                        copy_hdf5_group(demo_group, out_data_group, new_demo_key)
                        
                        # Update statistics from metadata
                        shifted = False
                        if 'shifted' in metadata:
                            shifted = metadata.get('shifted', False)
                        elif 'iteration' in metadata:
                            iteration = metadata.get('iteration', 0)
                            shifted = (iteration > 0)
                        
                        if shifted:
                            num_shifted += 1
                        else:
                            num_non_shifted += 1
                        
                        demo_idx += 1
                        total_demos += 1
                        
            except Exception as e:
                print(f"    ⚠️  Warning: Could not process {input_file}: {e}")
                continue
    
    return total_demos, num_non_shifted, num_shifted


def save_init_states(init_states: List[np.ndarray], output_file: str):
    """
    Save init states to a .init file.
    
    Args:
        init_states: List of init state arrays
        output_file: Path to output .init file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(init_states, f)


def extract_skill_name(filename: str) -> str:
    """
    Extract skill name from filename.
    E.g., "place_white_mug_on_the_plate.hdf5" -> "place_white_mug_on_the_plate"
    
    Args:
        filename: HDF5 or .init filename
        
    Returns:
        Skill name
    """
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Remove _failure suffix if present
    if name.endswith('_failure'):
        name = name[:-8]
    return name


def main():
    """Main function to combine HDF5 and .init files."""
    parser = argparse.ArgumentParser(description='Combine HDF5 files from v1-v10')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Directory containing the per-seed v1..v10 demo folders')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory for the combined output (default: <base_dir>/all)')
    parser.add_argument('--hdf5_check_results', type=str, default='',
                       help='Optional hdf5_check_results.json listing corrupt files to skip')
    parser.add_argument('--make_up_mode', action='store_true', default=False,
                       help='Make-up mode: only combine skills in MAKE_UP_BDDL_SKILLS list')
    parser.add_argument('--start_from', type=int, default=None,
                       help='Resume from specific skill number (1-based index). Existing stats will be loaded and preserved.')
    parser.add_argument('--skill_name', type=str, default=None,
                       help='Combine files for only one specific skill (e.g., "pick_black_bowl"). Existing stats will be loaded and this skill will be updated.')
    args = parser.parse_args()

    global BASE_DIR, OUTPUT_DIR, HDF5_CHECK_RESULTS
    BASE_DIR = os.path.abspath(args.base_dir)
    OUTPUT_DIR = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(BASE_DIR, "all")
    HDF5_CHECK_RESULTS = args.hdf5_check_results

    print("=" * 80)
    print("Combine HDF5 Files from v1-v10")
    if MULTI_BDDL_ONLY:
        print("🎯 MULTI-BDDL MODE: Only combining skills with >3 BDDL files")
    if args.make_up_mode:
        print("💄 MAKE-UP MODE: Only combining skills in MAKE_UP_BDDL_SKILLS")
    if args.start_from:
        print(f"▶️  RESUME MODE: Starting from skill #{args.start_from}")
    if args.skill_name:
        print(f"🎯 SINGLE-SKILL MODE: Only combining '{args.skill_name}'")
    print("=" * 80)
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    if MULTI_BDDL_ONLY:
        print(f"Skills to combine: {', '.join(MULTI_BDDL_SKILLS)}")
    if args.make_up_mode:
        print(f"Skills to combine: {', '.join(MAKE_UP_BDDL_SKILLS)}")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load invalid files to skip
    print("Loading invalid files list...")
    invalid_files = load_invalid_files(HDF5_CHECK_RESULTS)
    print()
    
    # Collect all files by skill name
    skill_hdf5_files = defaultdict(list)  # skill_name -> list of (v_dir, file_path)
    skill_init_files = defaultdict(list)   # skill_name -> list of (v_dir, file_path)
    
    print("Collecting files from v1-v10...")
    skipped_invalid = 0
    for v_dir in V_DIRS:
        v_path = os.path.join(BASE_DIR, v_dir)
        if not os.path.exists(v_path):
            print(f"  ⚠️  Warning: {v_path} does not exist, skipping...")
            continue
        
        # Find all success HDF5 files (exclude _failure.hdf5)
        hdf5_files = glob.glob(os.path.join(v_path, "*.hdf5"))
        success_hdf5_files = [f for f in hdf5_files if "_failure" not in os.path.basename(f)]
        
        for file_path in success_hdf5_files:
            # Skip invalid files
            abs_path = os.path.abspath(file_path)
            if abs_path in invalid_files:
                skipped_invalid += 1
                continue
            
            filename = os.path.basename(file_path)
            skill_name = extract_skill_name(filename)
            skill_hdf5_files[skill_name].append((v_dir, file_path))
        
        # Find all .init files (exclude _failure.init)
        init_files = glob.glob(os.path.join(v_path, "*.init"))
        success_init_files = [f for f in init_files if "_failure" not in os.path.basename(f)]
        
        for file_path in success_init_files:
            filename = os.path.basename(file_path)
            skill_name = extract_skill_name(filename)
            skill_init_files[skill_name].append((v_dir, file_path))
    
    print(f"Found {len(skill_hdf5_files)} unique skills")
    if skipped_invalid > 0:
        print(f"Skipped {skipped_invalid} invalid files from check results")
    print()
    
    # Load existing stats.json if in make-up mode, resume mode, or single-skill mode
    stats_file = os.path.join(OUTPUT_DIR, "stats.json")
    existing_stats = None
    if (args.make_up_mode or args.start_from or args.skill_name) and os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f:
                existing_stats = json.load(f)
            print(f"  📊 Loaded existing stats.json with {len(existing_stats.get('skills', {}))} skills")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not load existing stats.json: {e}")
            existing_stats = None
    
    # Combine files for each skill
    # Note: total_skills will be updated after filtering
    stats = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_demos": 0,
        "total_skills": 0,  # Will be set after filtering
        "avg_demos_per_skill": 0.0,
        "skills": {}
    }
    
    # If in make-up mode and existing stats exist, preserve all non-makeup skills
    if args.make_up_mode and existing_stats:
        # Copy all existing skills except makeup skills (they will be updated)
        for skill_name, skill_stats in existing_stats.get('skills', {}).items():
            if skill_name not in MAKE_UP_BDDL_SKILLS:
                stats["skills"][skill_name] = skill_stats.copy()
                stats["total_demos"] += skill_stats.get("num_demos", 0)
        print(f"  📊 Preserved stats for {len(stats['skills'])} existing skills (excluding makeup skills)")
    # If in single-skill mode and existing stats exist, preserve all other skills
    elif args.skill_name and existing_stats:
        # Copy all existing skills except the one being updated
        for skill_name, skill_stats in existing_stats.get('skills', {}).items():
            if skill_name != args.skill_name:
                stats["skills"][skill_name] = skill_stats.copy()
                stats["total_demos"] += skill_stats.get("num_demos", 0)
        print(f"  📊 Preserved stats for {len(stats['skills'])} existing skills (will update '{args.skill_name}')")
    # If in resume mode and existing stats exist, preserve already processed skills
    elif args.start_from and existing_stats:
        # Copy all existing skills (they will be kept, new ones added)
        for skill_name, skill_stats in existing_stats.get('skills', {}).items():
            stats["skills"][skill_name] = skill_stats.copy()
            stats["total_demos"] += skill_stats.get("num_demos", 0)
        print(f"  📊 Preserved stats for {len(stats['skills'])} already processed skills")

    print("Combining files...")
    skill_names = sorted(skill_hdf5_files.keys())
    original_skill_count = len(skill_names)

    # Filter skills if MULTI_BDDL_ONLY is enabled
    if MULTI_BDDL_ONLY:
        skill_names = [s for s in skill_names if s in MULTI_BDDL_SKILLS]
        print(f"  Filtered to {len(skill_names)} multi-BDDL skills (from {original_skill_count} total skills)")
        print()
    
    # Filter skills if make_up_mode is enabled
    if args.make_up_mode:
        skill_names = [s for s in skill_names if s in MAKE_UP_BDDL_SKILLS]
        make_up_found = [s for s in MAKE_UP_BDDL_SKILLS if s in skill_hdf5_files]
        print(f"  💄 Make-up mode: Filtered to {len(skill_names)} skills (from {original_skill_count} total skills)")
        if len(make_up_found) > 0:
            print(f"     Skills found: {', '.join(make_up_found)}")
        if len(skill_names) == 0:
            print(f"     ⚠️  Warning: No matching skills found in MAKE_UP_BDDL_SKILLS")
        print()

    # Filter skills if skill_name is specified
    if args.skill_name:
        if args.skill_name in skill_names:
            skill_names = [args.skill_name]
            print(f"  🎯 Single-skill mode: Processing only '{args.skill_name}'")
        else:
            print(f"  ❌ Error: Skill '{args.skill_name}' not found in dataset")
            available_skills = sorted(skill_hdf5_files.keys())
            print(f"     Available skills ({len(available_skills)} total): {', '.join(available_skills[:5])}...")
            return
        print()

    # Calculate total_skills: processed skills + preserved skills (if in make-up mode or resume mode or skill_name mode)
    if args.make_up_mode and existing_stats:
        # Total = makeup skills being processed + preserved skills
        total_skills = len(skill_names) + len([s for s in stats["skills"].keys() if s not in skill_names])
    elif args.skill_name and existing_stats:
        # Total = all skills (the one being updated + all existing preserved skills)
        total_skills = len([s for s in stats["skills"].keys() if s != args.skill_name]) + len(skill_names)
    elif args.start_from and existing_stats:
        # Total = all skills (including already processed + new ones to process)
        total_skills = len(skill_names)
    else:
        total_skills = len(skill_names)
    stats["total_skills"] = total_skills

    # Print resume information
    if args.start_from:
        skills_to_skip = args.start_from - 1
        skills_to_process = total_skills - skills_to_skip
        print(f"  ⏭️  Will skip first {skills_to_skip} skills")
        print(f"  ▶️  Will process {skills_to_process} skills (from #{args.start_from} to #{total_skills})")
        print()

    for skill_idx, skill_name in enumerate(skill_names, 1):
        # Skip already processed skills if resuming
        if args.start_from and skill_idx < args.start_from:
            print(f"  [{skill_idx}/{total_skills}] ⏭️  Skipping {skill_name} (already processed)")
            continue

        print(f"  [{skill_idx}/{total_skills}] Processing {skill_name}...")

        # Combine HDF5 files efficiently (direct copy, no memory loading)
        input_files = [file_path for _, file_path in skill_hdf5_files[skill_name]]
        
        # Skip if no input files
        if not input_files:
            print(f"    ⚠️  Warning: No input files found for {skill_name}, skipping...")
            continue
        
        output_hdf5 = os.path.join(OUTPUT_DIR, f"{skill_name}.hdf5")
        
        print(f"    Combining {len(input_files)} files...")
        total_demos, num_non_shifted, num_shifted = combine_hdf5_files_efficient(
            input_files, output_hdf5
        )
        print(f"    ✓ Combined {total_demos} demos")
        
        # Get selected source for filtering (if multi-BDDL skill)
        selected_source = get_selected_source(skill_name, output_hdf5)

        # Combine .init files
        all_init_states = []
        if skill_name in skill_init_files:
            # Get demo source mapping from combined HDF5 file
            demo_sources = {}
            if selected_source:
                demo_sources = get_demo_source_mapping(output_hdf5)

            demo_idx_global = 0  # Track global demo index across all v* files
            for v_dir, file_path in skill_init_files[skill_name]:
                print(f"    Loading init states from {v_dir}/{os.path.basename(file_path)}...")
                init_states = load_init_states(file_path)

                # Filter init states if selected_source is set
                if selected_source and demo_sources:
                    filtered_init_states = []
                    for local_idx, init_state in enumerate(init_states):
                        # Map local index to global demo index
                        if demo_idx_global in demo_sources:
                            if demo_sources[demo_idx_global] == selected_source:
                                filtered_init_states.append(init_state)
                        demo_idx_global += 1

                    if filtered_init_states:
                        print(f"       Filtered: {len(filtered_init_states)}/{len(init_states)} init states from {selected_source}")
                    all_init_states.extend(filtered_init_states)
                else:
                    all_init_states.extend(init_states)
                    demo_idx_global += len(init_states)

        # Save combined .init file
        if all_init_states:
            output_init = os.path.join(OUTPUT_DIR, f"{skill_name}.init")
            print(f"    ✓ Saving {len(all_init_states)} init states to {output_init}...")
            save_init_states(all_init_states, output_init)
        elif selected_source:
            print(f"    ⚠️  Warning: No init states after filtering for {selected_source}")
        
        # Calculate statistics
        # total_demos, num_non_shifted, num_shifted are already returned from combine_hdf5_files_efficient()
        # Verify: num_non_shifted + num_shifted should equal total_demos
        if total_demos > 0 and (num_non_shifted + num_shifted) != total_demos:
            print(f"    ⚠️  Warning: Statistics mismatch for {skill_name}: "
                  f"total_demos={total_demos}, non_shifted={num_non_shifted}, shifted={num_shifted}")
        
        non_shifted_ratio = num_non_shifted / total_demos if total_demos > 0 else 0.0
        
        stats["skills"][skill_name] = {
            "num_demos": total_demos,
            "num_non_shifted": num_non_shifted,
            "num_shifted": num_shifted,
            "non_shifted_ratio": round(non_shifted_ratio, 4),  # Store as float, rounded to 4 decimals
            "num_init_states": len(all_init_states)
        }
        
        # Add demos for this skill
        stats["total_demos"] += total_demos
        
        print(f"    ✓ Combined: {total_demos} demos, {len(all_init_states)} init states")
        print(f"      Non-shifted ratio: {non_shifted_ratio:.2%}")
        print()
    
    # Calculate average
    if stats["total_skills"] > 0:
        stats["avg_demos_per_skill"] = stats["total_demos"] / stats["total_skills"]
    
    # Save statistics
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total skills: {stats['total_skills']}")
    print(f"Total demos: {stats['total_demos']}")
    print(f"Average demos per skill: {stats['avg_demos_per_skill']:.2f}")
    print()
    print(f"Statistics saved to: {stats_file}")
    print("Done!")


if __name__ == "__main__":
    main()

