#!/usr/bin/env python3
"""
Downsample combined atomic_above demos in `all/` into `downsample_{total}/`.

For each skill:
- Sample up to --max_non_shifted non-shifted demos (default: 20)
- Sample up to --max_standard standard_shift demos (default: 15)
- Sample up to --max_z_only z_only_shift demos (default: 15)
- Total target: sum of the three (default: 50)

If standard_shift or z_only_shift have fewer demos than requested, use
non-shifted demos (that weren't already sampled) to make up the difference.

Classification rules (per demo's metadata):
- Non-shifted: shifted=False OR iteration==0 OR shift_info is None
- Shifted:
    - z_only_shift: shift_info['z_only_positive'] == True
    - standard_shift: shift_info['z_only_positive'] == False (or missing -> standard)

`.init` files are downsampled consistently using the non-shifted subset only
(one init state per non-shifted demo).
"""

import os
import json
import h5py
import pickle
import argparse
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


# INPUT_DIR is set from --input_dir in main(); there is no default.
# OUTPUT_DIR is set from --output_dir in main(), and otherwise falls back to
# <parent of input_dir>/downsample_{total} (the as-run behaviour).
INPUT_DIR = None

# Skills to downsample when --skill_filter is enabled (edit this list as needed)
SKILL_FILTER_LIST = [
    # "place_tomato_sauce_in_basket",
    # "place_alphabet_soup_in_basket",
    # "place_butter_in_basket",
    "place_butter_in_basket"
]

# Skills with >3 BDDL files (heterogeneous demos issue)
# When --use_source_filtering is enabled, only demos from selected sources will be used
MULTI_BDDL_SKILLS = [
    "pick_black_bowl",
    "pick_frying_pan",
    "place_black_bowl_on_the_plate",
    "place_black_bowl_on_top_of_the_cabinet"
]

# Manual source selection for multi-BDDL skills (used when --use_source_filtering is enabled)
# Key: skill name, Value: list of source IDs (0 = source with most demos, 1 = second most, etc.)
# If a skill is in this dict, it will use the specified sources
# If a skill is not in this dict but is in MULTI_BDDL_SKILLS, it will use only the max count source
MANUAL_SOURCE_SELECTION = {
    "pick_black_bowl": [0, 1, 5],
    "pick_frying_pan": [0, 1],
    "place_black_bowl_on_the_plate": [0, 1, 2],
}


def load_init_states(init_path: str) -> List[np.ndarray]:
    if not os.path.exists(init_path):
        return []
    with open(init_path, "rb") as f:
        return pickle.load(f)


def save_init_states(init_states: List[np.ndarray], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(init_states, f)


def parse_metadata(demo_group: h5py.Group) -> Dict:
    """Extract metadata including source_hdf5 for filtering and shift classification."""
    meta = {}
    if "metadata" not in demo_group:
        return meta
    m = demo_group["metadata"]

    # attrs (shifted, iteration, shift_info as json, source_hdf5)
    for k, v in m.attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        if k == "shift_info":
            try:
                meta[k] = json.loads(v)
            except Exception:
                meta[k] = None
        else:
            meta[k] = v

    # scalar datasets (sometimes stored as datasets instead of attrs)
    for k in m.keys():
        ds = m[k]
        if not isinstance(ds, h5py.Dataset):
            continue
        try:
            val = ds[()]
            # unwrap 0-d arrays
            if hasattr(val, "item"):
                val = val.item()
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            if k == "shift_info":
                if isinstance(val, str):
                    try:
                        meta[k] = json.loads(val)
                    except Exception:
                        meta[k] = None
            else:
                meta[k] = val
        except Exception:
            continue

    return meta


def classify_demo(meta: Dict) -> str:
    """
    Return one of: 'non_shifted', 'standard_shift', 'z_only_shift'.
    """
    shifted = meta.get("shifted", False)
    iteration = meta.get("iteration", 0)
    shift_info = meta.get("shift_info", None)

    # Non-shifted if explicitly marked or first iteration or no shift_info
    if (not shifted) or iteration == 0 or shift_info is None:
        return "non_shifted"

    # Shifted demos
    z_only = False
    if isinstance(shift_info, dict):
        z_only = bool(shift_info.get("z_only_positive", False))

    return "z_only_shift" if z_only else "standard_shift"


# old params: max_non_shifted=50, max_standard=30, max_z_only=20
def downsample_skill(
    hdf5_path: str,
    init_path: str,
    rng: np.random.Generator,
    output_dir: str,
    skill_name: str = "",
    use_source_filtering: bool = False,
    max_non_shifted: int = 20,
    max_standard: int = 15,
    max_z_only: int = 15,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Downsample a single skill file.

    Returns:
        stats: dict with original and selected counts
        warnings: dict with bucket -> (available, requested) for underfilled buckets
    """
    stats = {
        "total_original": 0,
        "non_shifted_original": 0,
        "standard_shift_original": 0,
        "z_only_shift_original": 0,
        "total_selected": 0,
        "non_shifted_selected": 0,
        "standard_shift_selected": 0,
        "z_only_shift_selected": 0,
        "non_shifted_makeup_for_standard": 0,
        "non_shifted_makeup_for_z_only": 0,
    }
    warnings = {}

    if not os.path.exists(hdf5_path):
        return stats, warnings

    init_states = load_init_states(init_path)

    with h5py.File(hdf5_path, "r") as in_f:
        if "data" not in in_f:
            return stats, warnings

        data_grp = in_f["data"]
        demo_keys = sorted(k for k in data_grp.keys() if k.startswith("demo_"))

        # Source filtering: if enabled and skill is in MULTI_BDDL_SKILLS, filter by allowed sources
        allowed_sources = None
        if use_source_filtering and skill_name in MULTI_BDDL_SKILLS:
            # Analyze source distribution
            from collections import defaultdict
            source_counts = defaultdict(int)
            for key in demo_keys:
                demo_grp = data_grp[key]
                meta = parse_metadata(demo_grp)
                source_hdf5 = meta.get("source_hdf5", None)
                if source_hdf5:
                    # Extract base name from path
                    base_name = os.path.basename(source_hdf5).replace('_demo.hdf5', '')
                    source_counts[base_name] += 1

            # Sort sources by count (descending)
            sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)

            # Determine which sources to use
            if skill_name in MANUAL_SOURCE_SELECTION:
                # Use manually specified source IDs
                source_ids = MANUAL_SOURCE_SELECTION[skill_name]
                allowed_sources = set()
                for sid in source_ids:
                    if sid < len(sorted_sources):
                        allowed_sources.add(sorted_sources[sid][0])
            else:
                # Use only the max count source (index 0)
                if sorted_sources:
                    allowed_sources = {sorted_sources[0][0]}

        # Collect indices by class and map non-shifted demos to init indices
        non_shifted_indices: List[int] = []
        standard_indices: List[int] = []
        z_only_indices: List[int] = []
        non_shift_demo_to_init_idx: Dict[int, int] = {}

        non_shift_counter = 0
        for key in demo_keys:
            idx = int(key.split("_")[1])
            demo_grp = data_grp[key]
            meta = parse_metadata(demo_grp)

            # Filter by source if source filtering is enabled
            if allowed_sources is not None:
                source_hdf5 = meta.get("source_hdf5", None)
                if source_hdf5:
                    base_name = os.path.basename(source_hdf5).replace('_demo.hdf5', '')
                    if base_name not in allowed_sources:
                        continue  # Skip this demo
                else:
                    continue  # Skip demos without source_hdf5

            cls = classify_demo(meta)

            stats["total_original"] += 1
            if cls == "non_shifted":
                stats["non_shifted_original"] += 1
                non_shifted_indices.append(idx)
                non_shift_demo_to_init_idx[idx] = non_shift_counter
                non_shift_counter += 1
            elif cls == "standard_shift":
                stats["standard_shift_original"] += 1
                standard_indices.append(idx)
            else:
                stats["z_only_shift_original"] += 1
                z_only_indices.append(idx)

        # Sample within each bucket
        def sample_indices(indices: List[int], k: int, bucket_name: str) -> List[int]:
            if len(indices) <= k:
                if len(indices) < k:
                    warnings[bucket_name] = {
                        "available": len(indices),
                        "requested": k,
                    }
                return list(indices)
            return list(rng.choice(indices, size=k, replace=False))

        # First, sample the target amounts from each bucket
        selected_non_shifted = sample_indices(non_shifted_indices, max_non_shifted, "non_shifted")
        selected_standard = sample_indices(standard_indices, max_standard, "standard_shift")
        selected_z_only = sample_indices(z_only_indices, max_z_only, "z_only_shift")

        # If standard_shift or z_only_shift are short, use non-shifted demos to make up
        # Get remaining non-shifted demos that weren't already selected
        remaining_non_shifted = [idx for idx in non_shifted_indices if idx not in selected_non_shifted]
        
        # Make up for standard_shift if needed
        standard_shortage = max_standard - len(selected_standard)
        if standard_shortage > 0 and remaining_non_shifted:
            makeup_for_standard = sample_indices(
                remaining_non_shifted, 
                min(standard_shortage, len(remaining_non_shifted)),
                "non_shifted_makeup_for_standard"
            )
            selected_non_shifted.extend(makeup_for_standard)
            stats["non_shifted_makeup_for_standard"] = len(makeup_for_standard)
            # Update remaining list
            remaining_non_shifted = [idx for idx in remaining_non_shifted if idx not in makeup_for_standard]
        
        # Make up for z_only_shift if needed
        z_only_shortage = max_z_only - len(selected_z_only)
        if z_only_shortage > 0 and remaining_non_shifted:
            makeup_for_z_only = sample_indices(
                remaining_non_shifted,
                min(z_only_shortage, len(remaining_non_shifted)),
                "non_shifted_makeup_for_z_only"
            )
            selected_non_shifted.extend(makeup_for_z_only)
            stats["non_shifted_makeup_for_z_only"] = len(makeup_for_z_only)

        # Union of all selected demo indices
        selected_all = sorted(
            set(selected_non_shifted) | set(selected_standard) | set(selected_z_only)
        )

        stats["non_shifted_selected"] = len(selected_non_shifted)
        stats["standard_shift_selected"] = len(selected_standard)
        stats["z_only_shift_selected"] = len(selected_z_only)
        stats["total_selected"] = len(selected_all)
        
        # Initialize make-up stats if not set
        if "non_shifted_makeup_for_standard" not in stats:
            stats["non_shifted_makeup_for_standard"] = 0
        if "non_shifted_makeup_for_z_only" not in stats:
            stats["non_shifted_makeup_for_z_only"] = 0

        # Write downsampled HDF5
        os.makedirs(output_dir, exist_ok=True)
        out_hdf5_path = os.path.join(output_dir, os.path.basename(hdf5_path))
        with h5py.File(out_hdf5_path, "w") as out_f:
            out_data = out_f.create_group("data")
            for new_idx, orig_idx in enumerate(selected_all):
                src_key = f"demo_{orig_idx}"
                dst_key = f"demo_{new_idx}"
                in_f.copy(data_grp[src_key], out_data, dst_key)

        # Downsample init states: only for selected non-shifted demos
        if init_states:
            selected_init_states: List[np.ndarray] = []
            for orig_idx in selected_non_shifted:
                if orig_idx not in non_shift_demo_to_init_idx:
                    continue
                init_idx = non_shift_demo_to_init_idx[orig_idx]
                if init_idx < len(init_states):
                    selected_init_states.append(init_states[init_idx])

            out_init_path = os.path.join(output_dir, os.path.basename(init_path))
            if selected_init_states:
                save_init_states(selected_init_states, out_init_path)

    return stats, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Downsample combined atomic_above_fewer demos into all_downsampled/"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory holding the combined per-skill HDF5/.init files (the `all/` folder)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write the downsampled dataset to "
             "(default: <parent of --input_dir>/downsample_{total}). "
             "Point this at the directory the RLDS builder reads.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--skill_filter",
        action='store_true',
        help=f"Only downsample skills in SKILL_FILTER_LIST (currently: {', '.join(SKILL_FILTER_LIST)})",
    )
    parser.add_argument(
        "--use_source_filtering",
        action='store_true',
        default=False,
        help="Enable source filtering for multi-BDDL skills (uses MANUAL_SOURCE_SELECTION config)",
    )
    parser.add_argument(
        "--max_non_shifted",
        type=int,
        default=20,
        help="Max non-shifted demos to sample per skill (default: 20)",
    )
    parser.add_argument(
        "--max_standard",
        type=int,
        default=15,
        help="Max standard_shift demos to sample per skill (default: 15)",
    )
    parser.add_argument(
        "--max_z_only",
        type=int,
        default=15,
        help="Max z_only_shift demos to sample per skill (default: 15)",
    )
    args = parser.parse_args()

    global INPUT_DIR
    INPUT_DIR = os.path.abspath(args.input_dir)

    # Compute dynamic output directory based on downsampling params
    total_demos = args.max_non_shifted + args.max_standard + args.max_z_only
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(INPUT_DIR), f"downsample_{total_demos}")

    rng = np.random.default_rng(args.seed)

    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Downsampling params: max_non_shifted={args.max_non_shifted}, max_standard={args.max_standard}, max_z_only={args.max_z_only} (total={total_demos})")

    skill_files = sorted(
        f
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".hdf5") and not f.endswith("_failure.hdf5")
    )

    # Filter skills if --skill_filter is enabled
    if args.skill_filter:
        filtered_files = []
        for fname in skill_files:
            skill_name = os.path.splitext(fname)[0]
            if skill_name in SKILL_FILTER_LIST:
                filtered_files.append(fname)

        if len(filtered_files) == 0:
            print(f"\n⚠️  Warning: No matching skills found!")
            print(f"   Requested: {', '.join(SKILL_FILTER_LIST)}")
            print(f"   Available: {', '.join([os.path.splitext(f)[0] for f in skill_files[:5]])}...")
            return

        print(f"\n🎯 Skill filter enabled: Processing {len(filtered_files)} / {len(skill_files)} skills")
        print(f"   Skills: {', '.join([os.path.splitext(f)[0] for f in filtered_files])}")
        skill_files = filtered_files

    all_skill_stats: Dict[str, Dict[str, int]] = {}
    all_skill_warnings: Dict[str, Dict[str, Dict[str, int]]] = {}

    for fname in tqdm(skill_files, desc="Downsampling skills", unit="skill"):
        skill_name = os.path.splitext(fname)[0]
        hdf5_path = os.path.join(INPUT_DIR, fname)
        init_path = os.path.join(INPUT_DIR, f"{skill_name}.init")

        stats, warnings = downsample_skill(
            hdf5_path,
            init_path,
            rng,
            output_dir=output_dir,
            skill_name=skill_name,
            use_source_filtering=args.use_source_filtering,
            max_non_shifted=args.max_non_shifted,
            max_standard=args.max_standard,
            max_z_only=args.max_z_only,
        )
        all_skill_stats[skill_name] = stats
        if warnings:
            all_skill_warnings[skill_name] = warnings

    # Structured summary
    print("\n===== Downsampling Summary (per skill) =====")
    for skill_name in sorted(all_skill_stats.keys()):
        s = all_skill_stats[skill_name]
        print(f"Skill: {skill_name}")
        print(
            f"  original: total={s['total_original']}, "
            f"non_shifted={s['non_shifted_original']}, "
            f"standard_shift={s['standard_shift_original']}, "
            f"z_only_shift={s['z_only_shift_original']}"
        )
        print(
            f"  selected: total={s['total_selected']}, "
            f"non_shifted={s['non_shifted_selected']}, "
            f"standard_shift={s['standard_shift_selected']}, "
            f"z_only_shift={s['z_only_shift_selected']}"
        )
        if s.get('non_shifted_makeup_for_standard', 0) > 0 or s.get('non_shifted_makeup_for_z_only', 0) > 0:
            print(
                f"  makeup: non_shifted_for_standard={s.get('non_shifted_makeup_for_standard', 0)}, "
                f"non_shifted_for_z_only={s.get('non_shifted_makeup_for_z_only', 0)}"
            )
        if skill_name in all_skill_warnings:
            w = all_skill_warnings[skill_name]
            print("  warnings:")
            for bucket, info in w.items():
                print(
                    f"    {bucket}: available={info['available']}, "
                    f"requested={info['requested']}"
                )
        print()
    
    # Print skills that don't have the target number of demos
    print(f"\n===== Skills with < {total_demos} demos =====")
    skills_under_target = [
        (skill_name, stats['total_selected'])
        for skill_name, stats in all_skill_stats.items()
        if stats['total_selected'] < total_demos
    ]
    if skills_under_target:
        for skill_name, count in sorted(skills_under_target, key=lambda x: x[1]):
            print(f"  {skill_name}: {count} demos")
    else:
        print(f"  All skills have {total_demos} demos!")


if __name__ == "__main__":
    main()


