# Data generation and training

How the LiLo-VLA training set and checkpoint were produced, and how to re-run each stage.

Nothing in the paper's evaluation tables depends on re-running these stages — the released dataset and the released checkpoint are *inputs* to this release, not
outputs of this code path.

Every stage takes its directories as explicit arguments; there are no absolute defaults and the
stages do **not** chain implicitly. `$DATA` below is whatever root you choose.

---

## Environments

Two conda environments are required (see `repro/` for the authoritative freeze):

| environment | python | used for |
|---|---|---|
| main | 3.10.16 | stages 0–4 and training (needs LIBERO pinned at `f78abd6`, robosuite, mujoco, mplib, torch) |
| `rlds_env` | 3.9.0 | stage 5 only (`tfds build`; TF 2.13.0, tfds 4.9.2, apache-beam 2.49) |

`lilo_vla` must be importable in **both** — the vendored RLDS builder imports the erasing kernel
from it. In `rlds_env`, `pip install --no-deps -e lilo-vla/` is enough; LIBERO is not needed there
(the bootstrap downgrades to a warning).

---

## Stage 0 — LIBERO-90 → no-noop HDF5

Replays the stock LIBERO demonstrations, drops no-op transitions and unsuccessful episodes, and
re-renders observations at 256×256.

```bash
python scripts/data_generation/regenerate_libero_dataset.py \
    --libero_task_suite libero_90 \
    --libero_raw_data_dir  $DATA/hdf5_datasets/libero_90 \
    --libero_target_dir    $DATA/hdf5_datasets/libero_90_no_noops
```

Prompts interactively if the target directory already exists (D-8). Writes
`libero_90_metainfo.json` into the target directory.

## Stage 1 — object-centric augmented demonstrations

The generator that produced the released dataset. For each of the 27 atomic skills it starts the
robot from a perturbed "above object" pose, plans to a demo waypoint with mplib, and replays the
rest of the original demonstration.

```bash
scripts/data_generation/run_parallel_demo_generation.sh $DATA/hdf5_datasets/atomic_above_27_skills
```

Ten tmux sessions, one per seed — `42 123 456 789 1011 2022 3033 4044 5055 6066` — each writing
`v1 … v10`, with `--make_up_mode --num_augmentations 3 --use_init_states`; `v1` additionally saves
one success and one failure video per skill. These are the parameters the as-run shell contained;
see D-6 for a discrepancy with the paper's description that must be resolved before publication.

The launcher pins `CUDA_VISIBLE_DEVICES=2` and the conda environment name `openvla-oft`, as the
original did; override with `CUDA_DEVICE=` and `CONDA_ENV=`. To run a single seed directly:

```bash
python -m lilo_vla.datagen.generate_object_centric_demos \
    --make_up_mode --use_init_states \
    --num_augmentations 3 --random_seed 42 \
    --dataset_path $DATA/hdf5_datasets/libero_90_no_noops \
    --output_path  $DATA/hdf5_datasets/atomic_above_27_skills/v1
```

The skills regenerated in `--make_up_mode` are `MAKE_UP_BDDL_SKILLS` in the module — the full
27-skill release set (D-2). Atomic-skill BDDLs and `skill_config.json` are resolved from the
bundled `configs/`, not from an external LIBERO checkout.

## Stage 2 — combine the per-seed folders

```bash
python -m lilo_vla.datagen.combine_hdf5_files \
    --base_dir $DATA/hdf5_datasets/atomic_above_27_skills
    # --output_dir defaults to <base_dir>/all
    # --hdf5_check_results <json>  optionally skips files listed as corrupt
```

Merges `v1 … v10` into one HDF5 and one `.init` file per skill under `all/`, and writes
`stats.json` beside them.

## Stage 3 — downsample to 50 demos per skill

```bash
python -m lilo_vla.datagen.downsample_combined_hdf5_files \
    --input_dir  $DATA/hdf5_datasets/atomic_above_27_skills/all \
    --output_dir $DATA/hdf5_datasets/atomic_above_27_skills/all_downsampled \
    --max_non_shifted 20 --max_standard 15 --max_z_only 15
```

20 non-shifted + 15 standard-shift + 15 z-only-shift = 50 per skill. `--output_dir` is explicit on
purpose: the file had drifted to writing `downsample_{total}/` while the RLDS builder read
`all_downsampled/` (D-5). Omitting it reproduces the drifted behaviour.

## Stage 4 — repair place-skill wrist segmentation (conditional)

```bash
python -m lilo_vla.datagen.fix_place_skill_segmentation \
    --input_dir  $DATA/hdf5_datasets/atomic_above_27_skills/all_downsampled \
    --output_dir $DATA/hdf5_datasets/atomic_above_27_skills/all_downsampled_fixed_seg
```

Re-renders `eye_in_hand_segmentation` for `place_*` skills so the mask covers gripper + target +
**grasped** object. Only the place skills are rewritten.

> **Provenance note.** The as-found constants read `all_downsampled` and wrote
> `all_downsampled_fixed_seg`, while the RLDS builder read `all_downsampled`. Whether the released
> build consumed the repaired directory (via a later rename or copy) or the unrepaired one cannot
> be determined from the source alone. Decide deliberately which directory you feed to stage 5.

## Stage 5 — build the RLDS dataset (`rlds_env`)

This is where the paper's **random erasing** is applied: 1 original + 2 erased variants per demo,
with `ERASING_AREA_RATIOS = [20, 30, 40, 60, 80]` and 1–5 rectangles. It is a **build-time**
augmentation — `--image_aug` during training is only crop + photometric jitter (D-3).

```bash
conda activate rlds_env
cd third_party/rlds_dataset_builder/LIBERO_Above_Atomic_Libero_All
LILO_VLA_ATOMIC_DEMOS_PATH=$DATA/hdf5_datasets/atomic_above_27_skills/all_downsampled \
    tfds build --overwrite --data_dir $DATA/rlds_datasets
```

`tfds build` takes no extra CLI arguments, so the demo directory is passed by environment variable;
without it the builder falls back to `<repo>/datasets/hdf5_datasets/atomic_above_27_skills/all_downsampled`.
The resulting dataset is named `libero_above_atomic_libero_all`, which is what training expects.

## Stage 6 — LoRA finetuning

```bash
scripts/train/train_lilo_vla.sh          # sbatch, 4×H100
LOG_DIR=/somewhere/logs scripts/train/train_lilo_vla.sh
```

The script `cd`s to the repository root and submits

```
torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/train/finetune.py …
```

with `--data_root_dir datasets/rlds_datasets`, `--dataset_name libero_above_atomic_libero_all`,
`--run_root_dir runs/libero_above_atomic_libero_all/1.0.0`, `--batch_size 16`,
`--learning_rate 5e-4`, `--lora_rank 32`, `--max_steps 200005`, `--num_steps_before_decay 100000`,
`--save_freq 10000`, `--use_l1_regression True`, `--use_proprio True`, `--image_aug True`,
`--num_images_in_input 1`, `--is_local_policy True`, and
`--run_id_note 'atomic_skills_above_libero_all--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state'`.

**Do not change any of these flags.** The checkpoint directory name is derived from them, and
evaluation parses `unnorm_key` back out of that name (D-9, C-3). Update `--wandb_entity` /
`--wandb_project`, or run with `WANDB_MODE=offline`.

Without Slurm, run the `torchrun` line directly from the repository root.

---

## Ablation variants

The ablation training runs differ from the above **only** in `--dataset_name` (and, for the
per-skill-count ablations, batch size): a dataset built with random erasing disabled, and datasets
downsampled to 10/20/30/50 demos per skill. They use the same training code; rebuild the RLDS
dataset with different stage-3/stage-5 settings and point `--dataset_name` at it. The ablation
shells themselves are not part of this release.
