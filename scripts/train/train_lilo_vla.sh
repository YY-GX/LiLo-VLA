#!/bin/bash

# Atomic Skills Above Libero All Training - OpenVLA-OFT LoRA Finetuning on Above Atomic Libero All Dataset
#
# Note: This dataset contains selected augmented demos starting from "above region" poses.
# All 27 unique atomic skills from long-horizon tasks (IDs 8, 9, 10) + LIBERO composite tasks
# All demos are augmented (some shifted, some non-shifted).
# Balanced sampling can be used to balance shifted vs non-shifted demos if needed.
#
# This is the launcher that produced the released checkpoint. Every flag below
# is byte-identical to the as-run version: `get_run_id` derives the checkpoint
# directory name from them, and `unnorm_key` is parsed back out of that name at
# evaluation time. Do not change them.
#
# Usage (from anywhere; the script cd's to the repository root itself):
#   scripts/train/train_lilo_vla.sh
#   LOG_DIR=/path/to/logs scripts/train/train_lilo_vla.sh

# Repository root (the `lilo-vla/` directory), derived from this script's own
# location. Override with LILO_VLA_ROOT if the tree is laid out differently.
LILO_VLA_ROOT="${LILO_VLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && cd .. && pwd)}"
cd "${LILO_VLA_ROOT}" || exit 1

CPUS_PER_TASK=32
GPUS=4
JOB_NAME="above_atomic_libero_all"
# Was "$ENDPOINT/pkgs_baselines/openvla-oft/logs"; $ENDPOINT is site-specific.
LOG_DIR="${LOG_DIR:-${LILO_VLA_ROOT}/logs}"
LOG_FILE="train_above_atomic_libero_all_%j.out"

mkdir -p "$LOG_DIR"

# Training on the above atomic libero all dataset
sbatch \
  --partition=h100 \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="torchrun --standalone --nnodes 1 --nproc-per-node $GPUS scripts/train/finetune.py \
    --is_local_policy True \
    --num_images_in_input 1 \
    --vla_path openvla/openvla-7b \
    --data_root_dir datasets/rlds_datasets \
    --dataset_name libero_above_atomic_libero_all \
    --run_root_dir runs/libero_above_atomic_libero_all/1.0.0 \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --use_proprio True \
    --grad_accumulation_steps 1 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 200005 \
    --save_freq 10000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity 'yygx' \
    --wandb_project 'openvla-oft-atomic-skills-above-libero-all' \
    --run_id_note 'atomic_skills_above_libero_all--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state'"
