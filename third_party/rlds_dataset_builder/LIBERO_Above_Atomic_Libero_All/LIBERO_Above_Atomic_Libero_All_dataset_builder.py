from typing import Iterator, Tuple, Any

import os
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import json
from pathlib import Path
from conversion_utils import MultiThreadedDatasetBuilder

# Import random erasing utility. This is the TRAIN-time erasing kernel that
# bakes the paper's background augmentation into the RLDS dataset at BUILD
# time; it is NOT the eval-time lilo_vla.interaction.random_erasing.
from lilo_vla.datagen.random_erasing_mask_tools import generate_augmented_wrist_images


"""
Build (needs the py3.9 `rlds_env`, see docs/repro/):

    cd third_party/rlds_dataset_builder/LIBERO_Above_Atomic_Libero_All
    LILO_VLA_ATOMIC_DEMOS_PATH=<downsampled demos dir> \
        tfds build --overwrite --data_dir <lilo-vla>/datasets/rlds_datasets
"""


# Repository root (the `lilo-vla/` directory).
_LILO_ROOT = Path(__file__).resolve().parents[3]

# For atomic skills, use selected hdf5 files from the downsampled `all/` directory
# All 27 unique atomic skills from long-horizon tasks (IDs 8, 9, 10) + LIBERO composite tasks
# Override with the LILO_VLA_ATOMIC_DEMOS_PATH environment variable (tfds build
# takes no extra CLI arguments).
atomic_demos_path = os.environ.get(
    "LILO_VLA_ATOMIC_DEMOS_PATH",
    str(_LILO_ROOT / "datasets" / "hdf5_datasets" / "atomic_above_27_skills" / "all_downsampled"),
)

# Filter demos from multi-BDDL skills to use only demos from the same source file (the one with max count)
# Set to True to filter, False to use all demos
USE_SINGLE_SOURCE_FOR_MULTI_BDDL = False

# Skills with >3 BDDL files (heterogeneous demos issue)
MULTI_BDDL_SKILLS = [
    "pick_black_bowl",
    "pick_frying_pan",
    "place_black_bowl_on_the_plate",
    "place_black_bowl_on_top_of_the_cabinet"
]

# Manual source selection: manually specify which sources to use for each skill
# Key: skill name, Value: list of source IDs (0 = source with most demos, 1 = second most, etc.)
# If a skill is in this dict, it will use the specified sources instead of auto-selecting the max count source
# If a skill is not in this dict, it will use the default behavior (single source with max count)
# Example: {"pick_black_bowl": [0, 1]} would use the top 2 sources for pick_black_bowl
MANUAL_SOURCE_SELECTION = {
    "pick_black_bowl": [0, 1, 5],
    "pick_frying_pan": [0, 1],
    "place_black_bowl_on_the_plate": [0, 1, 2],
}

# ==============================================================================
# Random Erasing Augmentation Configuration
# ==============================================================================
# Each demo generates: 1 original + 2 augmented versions (3 total per demo)
# Augmented versions have random rectangular erasing applied to background
APPLY_WRIST_SEGMENTATION_MASK = True

# Augmentation parameters (only used if APPLY_WRIST_SEGMENTATION_MASK=True)
NUM_AUGMENTATIONS_PER_DEMO = 2  # Generate 2 augmented versions per demo
ERASING_AREA_RATIOS = [20, 30, 40, 60, 80]  # % of background to erase (random sample)
NUM_RECTANGLES_OPTIONS = [1, 2, 3, 4, 5]     # Number of rectangles (random sample)


# All 27 unique atomic skills (alphabetically ordered)
# Combines skills from long-horizon tasks (IDs 8, 9, 10) + LIBERO composite tasks
demos_file = [
    f"{atomic_demos_path}/close_the_bottom_drawer_of_the_cabinet.hdf5",
    f"{atomic_demos_path}/open_the_microwave.hdf5",
    f"{atomic_demos_path}/pick_alphabet_soup.hdf5",
    f"{atomic_demos_path}/pick_black_bowl.hdf5",
    f"{atomic_demos_path}/pick_butter.hdf5",
    f"{atomic_demos_path}/pick_chocolate_pudding.hdf5",
    f"{atomic_demos_path}/pick_cream_cheese.hdf5",
    f"{atomic_demos_path}/pick_frying_pan.hdf5",
    f"{atomic_demos_path}/pick_moka_pot.hdf5",
    f"{atomic_demos_path}/pick_tomato_sauce.hdf5",
    f"{atomic_demos_path}/pick_white_mug.hdf5",
    f"{atomic_demos_path}/pick_wine_bottle.hdf5",
    f"{atomic_demos_path}/pick_yellow_and_white_mug.hdf5",
    f"{atomic_demos_path}/place_alphabet_soup_in_basket.hdf5",
    f"{atomic_demos_path}/place_black_bowl_in_bottom_drawer_of_the_cabinet.hdf5",
    f"{atomic_demos_path}/place_black_bowl_on_the_black_bowl.hdf5",
    f"{atomic_demos_path}/place_black_bowl_on_the_plate.hdf5",
    f"{atomic_demos_path}/place_butter_in_basket.hdf5",
    f"{atomic_demos_path}/place_chocolate_pudding_to_right_of_plate.hdf5",
    f"{atomic_demos_path}/place_cream_cheese_in_basket.hdf5",
    f"{atomic_demos_path}/place_frying_pan_on_the_stove.hdf5",
    f"{atomic_demos_path}/place_moka_pot_on_the_stove.hdf5",
    f"{atomic_demos_path}/place_tomato_sauce_in_basket.hdf5",
    f"{atomic_demos_path}/place_white_mug_on_the_plate.hdf5",
    f"{atomic_demos_path}/place_wine_bottle_in_bottom_drawer_of_the_cabinet.hdf5",
    f"{atomic_demos_path}/place_yellow_and_white_mug_on_right_plate.hdf5",
    f"{atomic_demos_path}/turn_on_the_stove.hdf5",
]

def analyze_source_distribution(hdf5_path: str) -> dict:
    """
    Analyze source_hdf5 distribution in an HDF5 file.

    Returns:
        dict: {source_hdf5: count}
    """
    source_counts = {}
    skill_name = os.path.basename(hdf5_path).replace('.hdf5', '')

    with h5py.File(hdf5_path, 'r') as f:
        n_demos = len(f['data'])
        for i in range(n_demos):
            demo_key = f'demo_{i}'
            if demo_key not in f['data']:
                continue

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

    return source_counts


def get_selected_sources(hdf5_path: str):
    """
    Get the source_hdf5 names to use for filtering.
    
    Returns:
        set: Set of selected source base names, or None if no filtering
    """
    skill_name = os.path.basename(hdf5_path).replace('.hdf5', '')

    # Only filter for multi-BDDL skills
    if not USE_SINGLE_SOURCE_FOR_MULTI_BDDL or skill_name not in MULTI_BDDL_SKILLS:
        return None

    # Analyze source distribution
    source_counts = analyze_source_distribution(hdf5_path)

    if not source_counts:
        print(f"⚠️  Warning: No source_hdf5 metadata found in {skill_name}.hdf5")
        print(f"   NOTE: Demos may need to be regenerated with updated script")
        return None

    # Sort sources by count (descending) to assign IDs
    sorted_sources = sorted(source_counts.items(), key=lambda x: -x[1])
    source_names_by_id = [source for source, _ in sorted_sources]

    # Check if manual selection is specified
    if skill_name in MANUAL_SOURCE_SELECTION:
        source_ids = MANUAL_SOURCE_SELECTION[skill_name]
        selected_source_names = set()
        for source_id in source_ids:
            if 0 <= source_id < len(source_names_by_id):
                selected_source_names.add(source_names_by_id[source_id])
            else:
                print(f"⚠️  Warning: Source ID {source_id} out of range for {skill_name} (max: {len(source_names_by_id) - 1})")
        
        if not selected_source_names:
            print(f"⚠️  Warning: No valid sources selected for {skill_name}, falling back to max count source")
            selected_source_names = {source_names_by_id[0]}
        
        print(f"\n📊 {skill_name}.hdf5 - Source distribution:")
        for idx, (source, count) in enumerate(sorted_sources):
            marker = "✓ SELECTED" if source in selected_source_names else ""
            print(f"   Source {idx}: {source}: {count} demos {marker}")
        total_demos = sum(source_counts.values())
        selected_demos = sum(count for source, count in sorted_sources if source in selected_source_names)
        print(f"   Total: {total_demos} demos → Using {selected_demos} from {len(selected_source_names)} source(s)")
        
        return selected_source_names
    else:
        # Default: Select source with max count
        selected_source = max(source_counts.items(), key=lambda x: x[1])
        
        print(f"\n📊 {skill_name}.hdf5 - Source distribution:")
        for idx, (source, count) in enumerate(sorted_sources):
            marker = "✓ SELECTED" if source == selected_source[0] else ""
            print(f"   Source {idx}: {source}: {count} demos {marker}")
        print(f"   Total: {sum(source_counts.values())} demos → Using {selected_source[1]} from {selected_source[0]}")
        
        return {selected_source[0]}


# Selected sources for filtering (computed inside _generate_examples to avoid multiprocessing issues)
# Maps skill_name -> set of allowed source base names
SELECTED_SOURCES = {}


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths with filtering for minimum episode length."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock

    # Compute selected sources once per worker (avoids file handle conflicts in multiprocessing)
    global SELECTED_SOURCES
    if not SELECTED_SOURCES and USE_SINGLE_SOURCE_FOR_MULTI_BDDL:
        print(f"\n🎯 Source Filtering Enabled for Multi-BDDL Skills")
        if MANUAL_SOURCE_SELECTION:
            print(f"   Manual source selection configured for: {list(MANUAL_SOURCE_SELECTION.keys())}")
        for demo_file in demos_file:
            selected_sources = get_selected_sources(demo_file)
            if selected_sources:
                skill_name = os.path.basename(demo_file).replace('.hdf5', '')
                SELECTED_SOURCES[skill_name] = selected_sources
        print()

    min_episode_length = 8  # Minimum steps required for 8-action chunking

    def _parse_example(episode_path, demo_id, variant_idx=0):
        """
        Parse a single demo and generate a variant.

        Args:
            episode_path: Path to HDF5 file
            demo_id: Demo index
            variant_idx: Variant index (0=original, 1-2=augmented with random erasing)
        """
        # load raw data
        with h5py.File(episode_path, "r") as F:
            if f"demo_{demo_id}" not in F['data'].keys():
                return None # skip episode if the demo doesn't exist (e.g. due to failed demo)
            demo_group = F['data'][f"demo_{demo_id}"]
            actions = demo_group["actions"][()]
            states = demo_group["obs"]["ee_states"][()]
            gripper_states = demo_group["obs"]["gripper_states"][()]
            joint_states = demo_group["obs"]["joint_states"][()]
            images = demo_group["obs"]["agentview_rgb"][()]
            wrist_images = demo_group["obs"]["eye_in_hand_rgb"][()]

            # Apply random erasing augmentation to wrist images
            if APPLY_WRIST_SEGMENTATION_MASK and variant_idx > 0:
                # Only augment for variant_idx > 0 (variant 0 is original)
                if "eye_in_hand_segmentation" in demo_group["obs"]:
                    wrist_segmentation = demo_group["obs"]["eye_in_hand_segmentation"][()]

                    # Sample augmentation parameters (deterministic per variant)
                    aug_rng = np.random.RandomState(seed=demo_id * 10 + variant_idx)
                    erasing_area_ratio = aug_rng.choice(ERASING_AREA_RATIOS)
                    num_rectangles = aug_rng.choice(NUM_RECTANGLES_OPTIONS)

                    # Generate augmented wrist images
                    wrist_images = generate_augmented_wrist_images(
                        wrist_images=wrist_images,
                        wrist_segmentation=wrist_segmentation,
                        erasing_area_ratio=erasing_area_ratio,
                        num_rectangles=num_rectangles,
                        demo_id=demo_id,
                        variant_idx=variant_idx
                    )
                else:
                    print(f"Warning: APPLY_WRIST_SEGMENTATION_MASK=True but eye_in_hand_segmentation not found in {episode_path}")

            # Load metadata if available
            metadata = {}
            shifted = False
            iteration = 0
            shift_info = None
            if 'metadata' in demo_group:
                meta_group = demo_group['metadata']
                # Load attributes
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
                # Load scalar datasets
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
                
                # Extract shifted information
                if 'shifted' in metadata:
                    shifted = bool(metadata['shifted'])
                elif 'iteration' in metadata:
                    iteration = int(metadata.get('iteration', 0))
                    shifted = (iteration > 0)
                
                # Extract shift_info if available
                if 'shift_info' in metadata:
                    shift_info_val = metadata['shift_info']
                    # If it's a string (JSON), parse it
                    if isinstance(shift_info_val, str):
                        try:
                            shift_info = json.loads(shift_info_val)
                        except:
                            shift_info = None
                    else:
                        shift_info = shift_info_val

        # Filter based on source_hdf5 for multi-BDDL skills
        skill_name = os.path.basename(episode_path).replace('.hdf5', '')
        if skill_name in SELECTED_SOURCES:
            # This skill requires filtering
            source_hdf5 = metadata.get('source_hdf5', None)
            if source_hdf5:
                if isinstance(source_hdf5, bytes):
                    source_hdf5 = source_hdf5.decode('utf-8')
                # Extract base name from path
                source_base = os.path.basename(source_hdf5).replace('_demo.hdf5', '')
                # Skip if not from selected sources
                if source_base not in SELECTED_SOURCES[skill_name]:
                    return None  # Skip this demo

        # Check episode length - must be >= 8 for 8-action chunking
        episode_length = actions.shape[0]
        if episode_length < min_episode_length:
            return None  # Skip episodes that are too short

        # Extract skill name and language instruction
        # Get skill name from filename (remove .hdf5 extension)
        skill_name = os.path.basename(episode_path).replace('.hdf5', '')
        
        # Convert skill name to language instruction by replacing underscores with spaces
        language_instruction = skill_name.replace('_', ' ')

        # Determine demo type based on metadata
        # Note: All demos are augmented (start from above region), but we distinguish:
        # - "original" = non-shifted (iteration 0, exact above pose)
        # - "augmented" = shifted (iteration > 0, shifted above pose)
        demo_type = "original" if not shifted else "augmented"

        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(actions.shape[0]):
            episode.append({
                'observation': {
                    'image': images[i][::-1,::-1],
                    'wrist_image': wrist_images[i][::-1,::-1],
                    'state': np.asarray(np.concatenate((states[i], gripper_states[i]), axis=-1), np.float32),
                    'joint_state': np.asarray(joint_states[i], dtype=np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),
                'discount': 1.0,
                'reward': float(i == (actions.shape[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': language_instruction,
                'demo_type': demo_type,
            })

        # create output data sample
        # Store shift_info as JSON string for TFDS compatibility
        shift_info_str = json.dumps(shift_info) if shift_info else json.dumps({})
        
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path,
                'skill_name': skill_name,
                'shifted': shifted,
                'iteration': iteration,
                'shift_info': shift_info_str,
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return episode_path + f"_demo{demo_id}_v{variant_idx}", sample

    # Generate 3 variants per demo: 1 original + 2 augmented
    # Total demos = n_demos * 3 (if APPLY_WRIST_SEGMENTATION_MASK=True)
    num_variants = 1 + NUM_AUGMENTATIONS_PER_DEMO if APPLY_WRIST_SEGMENTATION_MASK else 1

    for sample in paths:
        with h5py.File(sample, "r") as F:
            n_demos = len(F['data'])

        for demo_idx in range(n_demos):
            for variant_idx in range(num_variants):
                ret = _parse_example(sample, demo_idx, variant_idx)
                if ret is not None:
                    yield ret


class LiberoAboveAtomicLiberoAll(MultiThreadedDatasetBuilder):
    """DatasetBuilder for all 27 unique atomic skills from long-horizon tasks and LIBERO composite tasks."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release with all 27 unique atomic skills.',
    }
    N_WORKERS = 40             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 80   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot EEF state (6D pose, 2D gripper).',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint angles.',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot EEF action.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'demo_type': tfds.features.Text(
                        doc='Demo type: original or augmented.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'skill_name': tfds.features.Text(
                        doc='Skill name extracted from filename.'
                    ),
                    'shifted': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='Whether this demo was shifted from original above pose.'
                    ),
                    'iteration': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='Augmentation iteration (0 = non-shifted, >0 = shifted).'
                    ),
                    'shift_info': tfds.features.Text(
                        doc='Shift parameters as JSON string (xy_shift, z_shift, ori_shift). Empty dict if not shifted.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            "train": demos_file,
        }

