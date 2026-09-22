#!/bin/bash
# Launch 10 parallel processes for demo generation in make-up mode
# Each process processes only skills in MAKE_UP_BDDL_SKILLS with different random seeds and output folders
#
# Usage:
#   scripts/data_generation/run_parallel_demo_generation.sh <BASE_OUTPUT_DIR>
#   BASE_OUTPUT_DIR=<dir> scripts/data_generation/run_parallel_demo_generation.sh
#
# BASE_OUTPUT_DIR receives one sub-folder per process (v1 .. v10); those are the
# folders that lilo_vla.datagen.combine_hdf5_files --base_dir then combines.

# ============================================================================
# CONFIGURATION - Edit these parameters as needed
# ============================================================================
# Repository root (the `lilo-vla/` directory), derived from this script's own
# location. Override with LILO_VLA_ROOT if the tree is laid out differently.
SCRIPT_DIR="${LILO_VLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
PYTHON_MODULE="lilo_vla.datagen.generate_object_centric_demos"
# No default: pass the output root as $1 or via the BASE_OUTPUT_DIR env var.
BASE_OUTPUT_DIR="${1:-${BASE_OUTPUT_DIR:-}}"
NUM_AUGMENTATIONS=3
NUM_PROCESSES=10

# Random seeds for each process (one per process)
SEEDS=(42 123 456 789 1011 2022 3033 4044 5055 6066)

# Output folder suffixes (one per process)
OUTPUT_SUFFIXES=(v1 v2 v3 v4 v5 v6 v7 v8 v9 v10)

# Conda environment and GPU used by the released run
CONDA_ENV="${CONDA_ENV:-openvla-oft}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
# ============================================================================

# Validate configuration
if [ -z "${BASE_OUTPUT_DIR}" ]; then
    echo "Error: BASE_OUTPUT_DIR is required."
    echo "Usage: $0 <BASE_OUTPUT_DIR>    (or set BASE_OUTPUT_DIR in the environment)"
    exit 1
fi

if [ ${#SEEDS[@]} -ne ${NUM_PROCESSES} ]; then
    echo "Error: Number of seeds (${#SEEDS[@]}) must match NUM_PROCESSES (${NUM_PROCESSES})"
    exit 1
fi

if [ ${#OUTPUT_SUFFIXES[@]} -ne ${NUM_PROCESSES} ]; then
    echo "Error: Number of output suffixes (${#OUTPUT_SUFFIXES[@]}) must match NUM_PROCESSES (${NUM_PROCESSES})"
    exit 1
fi

cd "${SCRIPT_DIR}"

# Launch processes in tmux sessions
# Note: Using 1-indexed loop to work with both bash and zsh array indexing
for i in $(seq 1 ${NUM_PROCESSES}); do
    ARRAY_IDX=$((i - 1))  # Convert to 0-indexed for array access
    SEED=${SEEDS[$ARRAY_IDX]}
    OUTPUT_SUFFIX=${OUTPUT_SUFFIXES[$ARRAY_IDX]}
    OUTPUT_PATH="${BASE_OUTPUT_DIR}/${OUTPUT_SUFFIX}"
    SESSION_NAME="gen_fewer_${OUTPUT_SUFFIX}"

    # Enable video saving only for v1 (first process)
    VIDEO_FLAG=""
    if [ "${OUTPUT_SUFFIX}" = "v1" ]; then
        VIDEO_FLAG="--save_videos_normal"
    fi

    # Kill existing session if it exists
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

    # Create tmux session (detached)
    tmux new-session -d -s "${SESSION_NAME}"

    # Send commands to tmux session (wait for conda activation to complete)
    tmux send-keys -t "${SESSION_NAME}" "cd ${SCRIPT_DIR}" C-m
    tmux send-keys -t "${SESSION_NAME}" "conda activate ${CONDA_ENV}" C-m
    sleep 1  # Wait for conda activation to complete
    # tmux send-keys -t "${SESSION_NAME}" "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python3 -m ${PYTHON_MODULE} --make_up_mode --num_augmentations ${NUM_AUGMENTATIONS} --random_seed ${SEED} --output_path \"${OUTPUT_PATH}\" --trigger_distance 0.05 ${VIDEO_FLAG} --use_init_states" C-m
    tmux send-keys -t "${SESSION_NAME}" "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python3 -m ${PYTHON_MODULE} --make_up_mode --num_augmentations ${NUM_AUGMENTATIONS} --random_seed ${SEED} --output_path \"${OUTPUT_PATH}\" ${VIDEO_FLAG} --use_init_states" C-m

    echo "Created tmux session: ${SESSION_NAME} (seed=${SEED}${VIDEO_FLAG:+ , videos enabled})"
done

echo ""
echo "All tmux sessions created!"
echo "Attach to a session with: tmux attach -t gen_fewer_v1"
echo "List all sessions with: tmux ls"
echo "Kill a session with: tmux kill-session -t gen_fewer_v1"
echo ""
echo "Sessions: gen_fewer_v1, gen_fewer_v2, gen_fewer_v3, gen_fewer_v4, gen_fewer_v5, gen_fewer_v6, gen_fewer_v7, gen_fewer_v8, gen_fewer_v9, gen_fewer_v10"
