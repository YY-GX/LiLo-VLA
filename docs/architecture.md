# Architecture — module ↔ paper map

Every claim below was read out of the code in this tree.

---

## 1. One skill, end to end

A long-horizon task is a sequence of atomic skills `a_1 … a_n` read from
`configs/tasks_and_skills.json`. One attempt at skill `a_i` is three stages plus a
verifier — `LongHorizonPipeline._execute_skill_once()`
(`lilo_vla/core/pipeline.py:574`):

```
  skill a_i
      │
      │  ┌─ Stage 1 ─ approach pose  ────────────── M_reach, paper III-C ─┐
      ├──┤  lilo_vla/perception/oracle.get_object_pose(env, obj)           │
      │  │      -> object pose straight out of MuJoCo (oracle perception)  │
      │  │  lilo_vla/reaching/approach_pose.calculate_above_pose(...)       │
      │  │      -> T_approach: object pose + above_height, rest orientation │
      │  │      + get_special_object_handling(): per-object height rules    │
      │  └─────────────────────────────────────────────────────────────────┘
      │
      │  ┌─ Stage 2 ─ reach ──────────────────────── M_reach, paper III-C ─┐
      ├──┤  lilo_vla/reaching/motion_planning/mplib_planner.MPlibMotionPlanner
      │  │      collision-aware plan to T_approach over a rendered point    │
      │  │      cloud; attaches/detaches the grasped object                 │
      │  └─────────────────────────────────────────────────────────────────┘
      │
      │  ┌─ Stage 3 ─ interact ───────────────────── M_int,  paper III-D ──┐
      ├──┤  lilo_vla/interaction/masking.create_wrist_segmentation_mask()   │
      │  │      object-centric view: everything but the target (and the     │
      │  │      grasped object) is masked out of the wrist image            │
      │  │  lilo_vla/interaction/backends/openvla_oft                       │
      │  │      OpenVLA-OFT rollout, 8-step action chunks, <=vla_horizon    │
      │  │      steps, 8-D object-RELATIVE proprio state                    │
      │  └─────────────────────────────────────────────────────────────────┘
      │
      └─ Verifier V(a_i) ───────────────────────────── paper III-E ────────┐
         lilo_vla/core/skill_checker.check_skill_success_by_language()     │
             evaluates the skill's `bddl_predicates` from                  │
             configs/tasks_and_skills.json with LIBERO's eval_predicate_fn │
         then post_actions() (`pipeline.py:122`) lifts the EE clear        │
```

The verifier fires **inside** the VLA rollout (`_execute_vla`, `pipeline.py:752`): the
rollout returns the moment `V(a_i)` becomes true, not when the horizon runs out.

---

## 2. Algorithm 1 — sequencing and recovery (paper III-E)

`LongHorizonPipeline.execute()` (`lilo_vla/core/pipeline.py:1019`).

| Algorithm 1 element | implementation |
|---|---|
| skill sequence `a_1 … a_n` | `lilo_vla/core/task_planner.plan_task_sequence(task_name)` — reads the `skills` list of the matching task in `configs/tasks_and_skills.json`. **Ordering is the only thing that distinguishes V1/V2/V3**; the three variants share one BDDL file |
| per-skill verifier `V(a_i)` | `skill_checker.check_skill_success_by_language()` over `bddl_predicates`. The evaluation opens **zero** atomic-skill BDDL files — the predicates live in the JSON |
| per-skill retry budget `r_i`, `K = 10` | `--max_vla_retries 10`; the counter is per skill and is **not** reset across skills |
| recovery on failure | in-place retry with pose re-estimation + a fresh motion-planned reset to `T_approach` for non-holding skills; for holding / Place skills the pipeline **backtracks to the most recent Pick** and re-executes it |
| ablation switch | `_should_skip_reset()` (`pipeline.py:558`) implements `--recovery_mode` ∈ {`full`, `naive`, `naive_strict`}. The paper's naive-retry control is **`naive_strict`** |
| "w/o Recovery" row | *not* a separate rollout: `scripts/evaluate_long_horizon.py:133-165` synthesises it from the per-skill retry counters of the same run |

---

## 3. Module map

| path | paper | role |
|---|---|---|
| `lilo_vla/core/pipeline.py` | III-E (Algorithm 1) | the whole execution loop: env creation, skill iteration, retries, recovery, verification, recording, metrics |
| `lilo_vla/core/task_planner.py` | III-E | task name → ordered skill list |
| `lilo_vla/core/skill_checker.py` | III-E | `V(a_i)`; also resets the `pickedup` predicate's height baseline between skills |
| `lilo_vla/reaching/approach_pose.py` | III-C | `calculate_above_pose` (shared by eval, `shift=False`, and data generation, `shift=True` = the initial-state perturbation) and the per-object height rules |
| `lilo_vla/reaching/motion_planning/` | III-C | MPLib wrapper, point-cloud scene construction, planning strategies, the Franka URDF/SRDF + meshes. `cartesian_planner.py` is an alternative planner that is imported but never executed on the published path |
| `lilo_vla/perception/oracle.py` | III-C | oracle object pose from the simulator. Only `get_object_pose` is on the live path |
| `lilo_vla/interaction/masking.py` | III-D | object-centric segmentation masks for the wrist view and distractor masking of the agent view |
| `lilo_vla/interaction/backends/openvla_oft.py` | III-D | model load, image preprocessing, action de-normalisation, checkpoint file sync |
| `lilo_vla/interaction/backends/robot_utils.py` | III-D | gripper action normalisation/inversion, observation plumbing |
| `lilo_vla/interaction/random_erasing.py` | — | **eval-time** experiment behind `--apply_background_erasing`. Not the paper's augmentation |
| `lilo_vla/envs/libero_env.py` | IV-A | LIBERO env helpers: wrist image 180° rotation, `quat2axisangle` |
| `lilo_vla/benchmark/__init__.py` | IV-A | registers `ultra_long` + `libero_long_plus_plus` into LIBERO; resolves bundled BDDL paths |
| `lilo_vla/libero_compat.py` | IV-A | the 5 LIBERO patches that define the success criteria |
| `lilo_vla/datagen/` | III-B | object-centric atomic demo generation, HDF5 combine/downsample, the **build-time** random-erasing kernel |
| `lilo_vla/training/` | III-B | optional balanced sampler (unused for the released checkpoint) |
| `prismatic/` | III-D | OpenVLA-OFT model code, verbatim, top-level on purpose |
| `third_party/rlds_dataset_builder/` | III-B | vendored TFDS builder; this is where random erasing is baked into the dataset |

---

## 4. Where the compositional information actually lives

`configs/tasks_and_skills.json` is the keystone, not the BDDL files:

* `long_horizon_tasks[*].skills` — the ordered skill sequence. V1/V2/V3 and the
  re-sequenced LIBERO-Long++ twins differ **only** here.
* `long_horizon_tasks[*].benchmark` — selects the BDDL *folder*; `long_horizon_tasks[*].bddl_file`
  selects the stem. The 21 configurations need only **9** BDDL files.
* `skill_mappings[*].bddl_predicates` — **this is `V(a_i)`**. The evaluation never opens an
  atomic-skill BDDL file.
* `skill_mappings[*]` also carries the target object and the language string handed to the VLA.
* `maskable_objects` / `object_mappings` — drive `apply_distractor_masking`.

`configs/skill_config.json` maps a skill's language string to its BDDL filename for data
generation. Six keys are **deliberately absent**; they fall through to
`bddl_filename = skill`, which still contains `"place"` and therefore still triggers the
place-specific approach-height rules. Do not "fix" them.

The 91 atomic-skill BDDL files under `configs/bddl/atomic_skills/` ship so that the
training data can be regenerated. Evaluation never reads them.

---

## 5. Data flow, end to end

```
LIBERO-90 demos
   └─ scripts/data_generation/regenerate_libero_dataset.py        (stage 0: strip no-ops)
        └─ lilo_vla/datagen/generate_object_centric_demos.py      (stage 1: 27 skills x seeds
             │                                                     x augmentations; uses
             │                                                     M_reach + perturbation)
             ├─ lilo_vla/datagen/combine_hdf5_files.py            (stage 2)
             ├─ lilo_vla/datagen/downsample_combined_hdf5_files.py(stage 3: ~50 demos/skill)
             └─ third_party/rlds_dataset_builder/…                (stage 4: RLDS +
                     lilo_vla/datagen/random_erasing_mask_tools    RANDOM ERASING is applied
                                                                   HERE, at build time)
                  └─ scripts/train/finetune.py                    (stage 5: OpenVLA-OFT LoRA)
                       └─ scripts/evaluate_long_horizon.py        (stage 6: this benchmark)
```

Full recipe: [data_generation_and_training.md](data_generation_and_training.md).

---

## 6. Import-time behaviour worth knowing

* `import lilo_vla` **bootstraps LIBERO**: it points `LIBERO_CONFIG_PATH` at
  `configs/libero/`, writes a `config.yaml` there if needed (upstream LIBERO blocks on an
  interactive `input()` without one) and applies `libero_compat`. It must run before any
  `import libero`; `scripts/evaluate_long_horizon.py` does it as its first import.
* `import prismatic` drags in the entire RLDS stack, so **tensorflow is required merely to
  evaluate**. This is why `prismatic/` had to be copied whole.
* `prismatic/vla/constants.py` picks the action-chunk and proprio dimensions by sniffing
  `sys.argv` for the substring `libero`.
