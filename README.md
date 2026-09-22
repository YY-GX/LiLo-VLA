# LiLo-VLA

**Compositional Long-Horizon Manipulation via Linked Object-Centric Policies**

[**Project page**](https://yy-gx.github.io/LiLo-VLA/) &nbsp;·&nbsp;
[**Paper**](https://yy-gx.github.io/LiLo-VLA/static/pdfs/main.pdf) &nbsp;·&nbsp;
[**Checkpoint**](https://huggingface.co/yygx/lilo-vla-openvla-oft) &nbsp;·&nbsp;
[**Dataset**](https://huggingface.co/datasets/yygx/lilo-vla-atomic-skills-rlds)

LiLo-VLA executes long-horizon manipulation by *linking* a library of object-centric
atomic policies. A **reaching module** drives the end-effector to a per-object approach
pose with a collision-aware motion planner; an **interaction module** hands control to a
VLA policy that sees an object-centric, distractor-masked wrist view; a symbolic
**verifier** checks each skill's BDDL predicates and triggers recovery when one fails.

Because transport is delegated to a planner, the learned policy only ever has to solve
one atomic skill from a canonical approach pose — so the system executes skill orderings
it never saw during training.

| | |
|---|---|
| **Benchmark** | 21 configurations. Ultra-Long (3 tasks × 3 skill orderings, 9–16 skills each) and LIBERO-Long++ (6 tasks × 2 orderings, 3–4 skills each) |
| **Policy** | OpenVLA-OFT 7B, LoRA r32, L1 regression, 8-step action chunks, wrist image + proprioception |
| **Simulator** | LIBERO / robosuite / MuJoCo |
| **Paper** | [PDF](https://yy-gx.github.io/LiLo-VLA/static/pdfs/main.pdf) · [project page](https://yy-gx.github.io/LiLo-VLA/) |
| **Checkpoint** | [`yygx/lilo-vla-openvla-oft`](https://huggingface.co/yygx/lilo-vla-openvla-oft) |
| **Training data** | [`yygx/lilo-vla-atomic-skills-rlds`](https://huggingface.co/datasets/yygx/lilo-vla-atomic-skills-rlds) |

---

## Contents

- [Results](#results) — what the system scores, and how much run-to-run variance to expect
- [Just want the benchmark?](#just-want-the-benchmark) — evaluate *your* policy on the 21 tasks, 9 dependencies
- [Installation](#installation) — LIBERO pin, compatibility patches, install extras
- [Evaluation](#evaluation) — reproduce the numbers, read the result files
- [Method](#method) — module ↔ paper-section map
- [Repository layout](#repository-layout)
- [Data generation and training](#data-generation-and-training)
- [Citation](#citation) · [License](#license)

## Results

Reported in the paper. 10 trials per configuration. **SR** = the entire skill sequence
completed, in order. **AP** (Average Progress) = mean fraction of the sequence completed
before the first out-of-order or failed skill.

#### Suite 1 — LIBERO-Long++ (visual clutter)

| | Original | Reordered | **Avg.** |
|---|---|---|---|
| SR | 78% | 85% | **82%** |
| AP | 88% | 89% | **89%** |

#### Suite 2 — Ultra-Long (temporal scalability)

| | Original | Reorder 1 | Reorder 2 | **Avg.** |
|---|---|---|---|---|
| SR | 53% | 37% | 43% | **44%** |
| AP | 79% | 84% | 74% | **79%** |

**Overall: 69% SR / 86% AP**, macro-averaged over the 9 scenarios (not the 21
configurations). For comparison, Pi0.5 reaches 28% / 31% and OpenVLA-OFT 2% / 4% on the
same benchmark; the per-baseline and ablation rows are in the paper.

**Reordering is close to free on Suite 1** — 85% SR on reordered sequences against 78% on
the original ones. The policy is never trained on a task-level ordering, only on
individual atomic skills, so no ordering is more in-distribution than another. (A
monolithic baseline collapses here: Pi0.5 goes from 83% to 0% when the same tasks are
reordered.)

**Report AP alongside SR.** A 16-skill task is rarely solved end to end, so SR alone
discards most of the signal — Suite 2 averages 44% SR but 79% AP, i.e. a typical trial
completes about four fifths of the sequence.

### On run-to-run variance

**These numbers are not bit-reproducible, and that is inherent to the benchmark.** Each
trial starts from a bare `env.reset()`, so every run draws a fresh sample from the
simulator's initial-state distribution; there is no seed that pins it. With 10 trials per
configuration, a single cell's SR carries roughly a ±25-point 95% confidence interval, and
we have measured the same configuration moving by 20 SR points across runs with no code
change at all.

Practical consequences:

* Expect your numbers to differ from the table, per configuration, by more than you might
  assume from the two-digit precision.
* **AP is the stable metric** and is what we would compare across systems. It averages over
  many skills per trial rather than collapsing each trial to a single bit.
* Compare group means (the **Avg.** columns), not individual cells.
* If you need tighter error bars, raise `--num_trials`; the cost is linear.

Exact commands, the result files behind each number, and an independent re-run of all 21
configurations from this release tree (overall 71.5% SR / 86.9% AP, against the paper's
69% / 86%) are in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

---

## Just want the benchmark?

If you only want to **evaluate your own policy** on the 21 tasks, you do not need the
LiLo-VLA method, the OpenVLA-OFT backbone, TensorFlow, or a GPU-sized dependency tree.

```bash
git clone https://github.com/YY-GX/LiLo-VLA.git && cd LiLo-VLA
pip install -e ".[libero]"          # 9 packages, not 39
```

```python
import lilo_vla.benchmark                      # registers the suites AND patches LIBERO
from libero.libero.benchmark import get_benchmark

bench = get_benchmark("ultra_long")()          # or "libero_long_plus_plus",
                                               # "ultra_long_all", "libero_long_plus_plus_all"
for i in range(bench.get_num_tasks()):
    task = bench.get_task(i)
    env, obs = bench.make_env(i)               # the published evaluation protocol
    # ... roll out YOUR policy ...
    success = env.env._check_success()
    env.close()
```

`make_env()` builds the environment exactly as every number above was measured, so your
results are comparable without having to match a protocol by hand.

To report Average Progress as well as final success, read the per-skill predicates from
`configs/tasks_and_skills.json` — each task's `skills` list carries the `bddl_predicates`
that define when that skill is done.

| suite | tasks | what it tests |
|---|---|---|
| `ultra_long` | 3 | temporal scalability — 9, 10 and 16 skills |
| `ultra_long_variant_1` / `_variant_2` | 3 + 3 | zero-shot execution of permuted skill orderings |
| `ultra_long_all` | 9 | all of the above |
| `libero_long_plus_plus` | 6 | visual robustness — cluttered backgrounds, 3–4 skills |
| `libero_long_plus_plus_variant` | 6 | the same tasks with reordered skills |
| `libero_long_plus_plus_all` | 12 | all of the above |

---

## Installation

### 1. LIBERO, pinned at `f78abd6`

```bash
pip install -e ".[libero]"
```

That extra installs LIBERO at the revision this project is built against. Importing
`lilo_vla` then applies five compatibility patches to it and verifies they took; if they
did not, the import raises `LiLoCompatError` with instructions rather than letting you
collect results that cannot be compared.

<details>
<summary>What the patches do</summary>

Three of the five change *success criteria*, so an unpatched LIBERO does not crash — it
silently scores the same rollouts differently.

| patch | effect |
|---|---|
| `PickedUp` predicate + `pickedup` registry entry | upstream LIBERO has no pick predicate; a pick succeeds when the object rises ≥ 3 cm |
| `closed` registry alias | the Ultra-Long kitchen goal uses the past-tense form |
| `ObjectState.check_ontop` | xy tolerance 0.05 and z + 0.01, vs upstream's 0.03 and 0 |
| `SiteObject.in_box` | z floor 0.05 vs upstream's 0.01 |
| `WoodenCabinet.default_close_ranges`, `Microwave.default_open_ranges` | articulation thresholds |

Check at any time:

```bash
python -c "import lilo_vla; from lilo_vla.libero_compat import verify; verify(); print('OK')"
```

</details>

### 2. LiLo-VLA

```bash
pip install -e ".[libero]"                # benchmark only     — 9 packages
pip install -e ".[libero,method]"         # + the LiLo-VLA method
pip install -e ".[libero,method,train]"   # + training
```

The `method` extra is what pulls TensorFlow, transformers, timm, MPLib and Open3D. Install
it editable from a checkout: `configs/` lives next to the package and is resolved relative
to the source tree.

Torch was built against CUDA 12.1 for the released runs:

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

flash-attn (training only) needs `pip install flash-attn==2.5.5 --no-build-isolation`.

---

## Evaluation

The checkpoint is on the Hub; pass its repo id directly.

```bash
CKPT=yygx/lilo-vla-openvla-oft

# Suite 2 — Ultra-Long
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_long_horizon.py \
    --vla_checkpoint "$CKPT" \
    --task_name 'Complete Kitchen Organization (Original)' \
    --num_trials 10 \
    --benchmark ultra_long \
    --apply_distractor_masking

# Suite 1 — LIBERO-Long++
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_long_horizon.py \
    --vla_checkpoint "$CKPT" \
    --task_name 'Put Both The Alphabet Soup And The Cream Cheese Box In The Basket (Original)' \
    --num_trials 10 \
    --benchmark libero_long_plus_plus \
    --apply_distractor_masking
```

`--task_name` takes any of the 21 configurations: a task name plus `(Original)`,
`(Reorder 1)` or `(Reorder 2)`. `python scripts/evaluate_long_horizon.py --help` lists
them all.

### Reading the results

Each run writes a JSON under `outputs/eval_results/`. The numbers you want are:

```
metrics.with_recovery.success_rate    fraction of trials completing the whole sequence
metrics.with_recovery.avg_progress    mean fraction of the sequence completed
```

`metrics.no_recovery.*` is the same run scored as if no retry had been allowed — it is
derived from the per-skill retry counters, not a second rollout.

To summarise several runs:

```bash
python scripts/analysis/score_results.py 'outputs/eval_results/*.json'
```

### Flags that matter

| flag | default | note |
|---|---|---|
| `--apply_distractor_masking` | off | **on for every published number** |
| `--max_vla_retries` | 10 | the retry budget K in Algorithm 1 |
| `--recovery_mode` | `full` | `naive_strict` is the paper's naive-retry control |
| `--motion_planner_type` | `mplib` | collision-aware planner |
| `--benchmark` | `ultra_long` | historical names (`long_horizon_tasks_v1`, `long_horizon_tasks_libero_long`) are also accepted |

The evaluation is unseeded: each trial is a bare `env.reset()` and the tables average over
the simulator's own initial-state distribution. The bundled `.pruned_init` files support a
deterministic sweep instead, via `get_task_init_states(i)` and `env.set_init_state(...)` —
a different protocol that gives different numbers.

---

## Method

| Paper | Module | Code |
|---|---|---|
| §III-C Reaching | approach pose `T_obj · T_offset`, MPLib collision-free planning, training-time pose perturbation | `lilo_vla/reaching/` |
| §III-D Interaction | wrist-only object-centric observation, distractor masking, OpenVLA-OFT backend | `lilo_vla/interaction/` |
| §III-E Algorithm 1 | skill loop, geometric verifier `V(a_i)`, closed-loop recovery (retry in place for pick/articulation, backtrack to the last pick for place) | `lilo_vla/core/pipeline.py` |

[docs/architecture.md](docs/architecture.md) maps every module to its paper section.

---

## Repository layout

```
lilo_vla/
  benchmark/        the 21-configuration suites, registered into LIBERO
  libero_compat.py  the five LIBERO patches the published numbers depend on
  core/             Algorithm 1 — skill chaining, BDDL verifier, recovery
  reaching/         approach-pose computation + MPLib motion planning
  interaction/      object-centric masking, OpenVLA-OFT backend
  perception/       oracle object-pose reader
  envs/  utils/     LIBERO env helpers, transforms, config resolution
  datagen/          object-centric demo generation, HDF5 combine/downsample
prismatic/          OpenVLA-OFT model code (vendored verbatim, MIT)
configs/            tasks_and_skills.json, skill_config.json, BDDL files, init states
scripts/            evaluate_long_horizon.py, train/, data_generation/, analysis/
third_party/        vendored RLDS dataset builder
docs/               architecture, reproduction targets, environment freeze
```

## Data generation and training

[docs/data_generation_and_training.md](docs/data_generation_and_training.md) has the full
recipe: LIBERO-90 → atomic-skill segmentation → object-centric regeneration with approach-
pose perturbation → HDF5 combine/downsample → RLDS build → OpenVLA-OFT fine-tune.

One thing that is easy to get wrong: **random erasing is applied at RLDS *build* time**,
not during training. `--image_aug` in the training script is random-resized-crop plus
brightness/contrast/saturation/hue only.

```bash
bash scripts/train/train_lilo_vla.sh     # 4 GPUs, batch 16, lr 5e-4, LoRA r32, 200k steps
```

---

## Citation

```bibtex
@article{yang2026lilo,
  title={LiLo-VLA: Compositional Long-Horizon Manipulation via Linked Object-Centric Policies},
  author={Yang, Yue and Cheng, Shuo and Fang, Yu and Bharadhwaj, Homanga and Ding, Mingyu and Bertasius, Gedas and Szafir, Daniel},
  journal={arXiv preprint arXiv:2602.21531},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE). This distribution also contains third-party code and data
from OpenVLA-OFT, LIBERO and rlds_dataset_builder; their notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
