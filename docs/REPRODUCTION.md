# Reproduction targets

Every number in this document was **recomputed from the original result JSONs** with
`scripts/analysis/score_results.py` at release time. Each cell names the file it came
from. Nothing here is copied from a spreadsheet.

---

## 0. Read this before comparing anything

**1. The `success_rate` field in a result JSON is not task success.** (It sits at
`metrics.<bucket>.success_rate`, not at the top level.) It is the environment's
*instantaneous* BDDL goal conjunction, sampled at one point inside the VLA rollout and
nowhere else. Every other `env.step` on the path discards `done`, and the
rollout returns the moment the *skill* predicate fires — so the instant the task
actually completes is never sampled. Recompute instead, over `trials[*].with_recovery`:

```
SR = mean(skills_completed == total_skills)
AP = mean(skills_completed / total_skills)
```

`avg_progress` as emitted is already AP and matches digit for digit. Do **not** switch to
`env.check_success()`; that is a different metric definition.

```bash
python scripts/analysis/score_results.py 'outputs/eval_results/*.json'
```

How wrong the field is, concretely: for Complete Kitchen Organization (Original)
(`…_20260125_001929`, §3.1) the file records `metrics.with_recovery.success_rate = 0.2`
while the true SR is **0.90**. For Organize Table (Original) it records `0.0` against a true SR
of **0.40**. The scorer prints the bogus value in its last-but-one column so the gap is
visible at a glance.

**2. No single evaluation batch reproduces a published table row.** The rows were
hand-assembled from runs made on different dates. Suite 2 "Original" draws on
2026-01-25 (Kitchen, Cooking) and 2026-01-29 (Table); V2/V3 draw on 2026-01-18,
2026-01-25 and 2026-01-29; Suite 1 is the 2026-01-26 batch. **Target the per-task cells
and the five group averages below — never a single-batch mean.**

**3. The headline "avg All 69 / 86" cell is not reproducible from its own sub-rows.**
Recomputed over all 21 released configurations: **65.7 SR / 84.8 AP**. The unweighted
mean of the two suite averages is 63 / 84. Do not use 69/86 as a validation target. (The
migration manifest quotes 65.7 / 84.5 for the same set; the AP figure recomputes to 84.78
here — see §7.)

**4. Attributing a result file to a checkpoint.** The `save_results` checkpoint-suffix
`if/elif` chain is unreachable past its first branch, so **result filenames cannot
attribute a run to a checkpoint**. The only reliable attribution is the
`config.json.back.<timestamp>` file that `check_model_logic_mismatch` writes *inside the
checkpoint directory* when an eval starts against it. Every cell below was verified that
way.

**5. Run one process per task.** The `pickedup` predicate keeps a module-level
`initial_heights` cache that is never reset across tasks. All published runs used one
process per task; whether batching tasks into one process changes the numbers has not
been measured.

**6. Initial states.** The published evaluation calls a bare `env.reset()`. It does not
call `set_init_state()`, does not `reset_to()`, and does not seed. The bundled
`.pruned_init` files were generated *after* the last table run and are not part of this
protocol.

---

## 1. The checkpoint

```
runs/libero_above_atomic_libero_all/1.0.0/
  openvla-7b+libero_above_atomic_libero_all+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_libero_all--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--200000_chkpt
```

Distributed separately. **Do not rename the directory** — `unnorm_key` is parsed out of
the directory name, and renaming it silently changes action de-normalisation instead of
failing.

---

## 2. Canonical commands

```bash
export CKPT=/path/to/…--200000_chkpt

# Suite 2 — Ultra-Long
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_long_horizon.py \
    --vla_checkpoint "$CKPT" \
    --task_name 'Complete Kitchen Organization (Original)' \
    --num_trials 10 \
    --benchmark long_horizon_tasks_v1 \
    --apply_distractor_masking

# Suite 1 — LIBERO-Long++
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_long_horizon.py \
    --vla_checkpoint "$CKPT" \
    --task_name 'Put Both The Alphabet Soup And The Cream Cheese Box In The Basket (Original)' \
    --num_trials 10 \
    --benchmark long_horizon_tasks_libero_long \
    --apply_distractor_masking
```

Neither suite passes `--mplib_collision_aware`; the default `true` is what the published
rows used. Load-bearing defaults: `--max_vla_retries 10`, `--above_height 0.1`,
`--vla_horizon 200`, `--recovery_mode full`, `--motion_planner_type mplib`,
`--wrist_only` (`store_true` with `default=True`, un-disableable), and the non-CLI
`use_relative_pose=True`.

---

## 3. Suite 2 — Ultra-Long (9 configurations)

All source files live in the research tree at
`scripts/phase3/pipeline/outputs/results/eval_results/`, named
`eval_<task>_n10_above_atomic_<timestamp>.json`. Only the `<task>_<timestamp>` part is
given below.

### 3.1 Original ordering (V1)

| task | skills | SR | AP | source |
|---|---:|---:|---:|---|
| Complete Kitchen Organization (Original) | 9 | **90.0** | **98.89** | `complete_kitchen_organization_v1_…_20260125_001929` |
| Organize Table (Original) | 16 | **40.0** | **96.25** | `organize_table_v1_…_20260129_223629` |
| Cooking Preparation Setup (Original) | 10 | **30.0** | **42.00** | `cooking_preparation_setup_v1_…_20260125_172511` |
| **group mean** (paper 53 / 79) | | **53.33** | **79.05** | |

### 3.2 Variant 1 (V2)

| task | SR | AP | source |
|---|---:|---:|---|
| Complete Kitchen Organization (Reorder 1) | **50.0** | **93.33** | `complete_kitchen_organization_v2_…_20260118_152732` |
| Organize Table (Reorder 1) | **40.0** | **94.38** | `organize_table_v2_…_20260129_223634` |
| Cooking Preparation Setup (Reorder 1) | **20.0** | **64.00** | `cooking_preparation_setup_v2_…_20260125_172519` |
| **group mean** (paper 37 / 84) | **36.67** | **83.90** | |

### 3.3 Variant 2 (V3)

| task | SR | AP | source |
|---|---:|---:|---|
| Complete Kitchen Organization (Reorder 2) | **50.0** | **76.67** | `complete_kitchen_organization_v3_…_20260118_152851` |
| Organize Table (Reorder 2) | **60.0** | **90.00** | `organize_table_v3_…_20260129_223644` |
| Cooking Preparation Setup (Reorder 2) | **20.0** | **54.00** | `cooking_preparation_setup_v3_…_20260125_172516` |
| **group mean** (paper 43 / 74) | **43.33** | **73.56** | |

**Suite 2 average over all 9: 44.44 SR / 78.84 AP** (paper 44 / 79).

---

## 4. Suite 1 — LIBERO-Long++ (12 configurations)

### 4.1 Original ordering — the Table-1 batch, 2026-01-26

| task | SR | AP | source |
|---|---:|---:|---|
| Put Both The Alphabet Soup And The Cream Cheese Box In The Basket (Original) | **90.0** | **95.00** | `…_20260126_124957` |
| Put Both The Alphabet Soup And The Tomato Sauce In The Basket (Original) | **100.0** | **100.00** | `…_20260126_125000` |
| Put Both The Cream Cheese Box And The Butter In The Basket (Original) | **100.0** | **100.00** | `…_20260126_125006` |
| Put The White Mug On The Left Plate And Put The Yellow And White Mug On The Right Plate (Original) | **60.0** | **72.50** | `…_20260126_232343` |
| Put The White Mug On The Plate And Put The Chocolate Pudding To The Right Of The Plate (Original) | **60.0** | **90.00** | `…_20260126_125051` |
| Turn On The Stove And Put The Moka Pot On It (Original) | **60.0** | **73.33** | `…_20260126_232343` |
| **group mean** (sheet 78 / 88) | **78.33** | **88.47** | |

Two tasks share the `232343` timestamp — they were re-run together late that evening;
the four `1249xx/1250xx` files are the midday batch.

### 4.2 Re-sequenced ordering (` V1` names) — batch 2026-01-29 11:16

| task | SR | AP | source |
|---|---:|---:|---|
| Put Both The Alphabet Soup And The Cream Cheese Box In The Basket (Original) (Reorder 1) | **90.0** | **95.00** | `…_v1_…_20260129_111604` |
| Put Both The Alphabet Soup And The Tomato Sauce In The Basket (Original) (Reorder 1) | **100.0** | **100.00** | `…_v1_…_20260129_111610` |
| Put Both The Cream Cheese Box And The Butter In The Basket (Original) (Reorder 1) | **90.0** | **95.00** | `…_v1_…_20260129_111618` |
| Put The White Mug On The Left Plate And Put The Yellow And White Mug On The Right Plate (Original) (Reorder 1) | **70.0** | **90.00** | `…_v1_…_20260129_111628` |
| Put The White Mug On The Plate And Put The Chocolate Pudding To The Right Of The Plate (Original) (Reorder 1) | **100.0** | **100.00** | `…_v1_…_20260129_111636` |
| Turn On The Stove And Put The Moka Pot On It (Original) (Reorder 1) | **60.0** | **60.00** | `…_v1_…_20260129_111557` |
| **group mean** (sheet 85 / 89) | **85.00** | **90.00** | |

The sheet's AP cell for this group reads 89; 90.00 is what the files recompute to. The
sheet's row-1 AP is a transcription of an earlier file.

**Suite 1 average over all 12: 81.67 SR / 89.24 AP** (sheet 82 / 89).

---

## 5. Aggregates

| set | n | SR | AP |
|---|---:|---:|---:|
| Suite 2 (Ultra-Long) | 9 | 44.44 | 78.84 |
| Suite 1 original | 6 | 78.33 | 88.47 |
| Suite 1 re-sequenced | 6 | 85.00 | 90.00 |
| Suite 1 all | 12 | 81.67 | 89.24 |
| **All released configurations** | **21** | **65.71** | **84.78** |

The five **group** averages (78/88, 85/89, 53/79, 37/84, 43/74) reproduce exactly from
the files and are the validation targets.

---

## 6. Run-to-run variance — read before calling a mismatch a bug

`n = 10`. The same command on the same checkpoint moves by ±10–20 SR points between
dates. Measured, from files verified to be the main checkpoint:

| task | dates and results |
|---|---|
| Complete Kitchen Organization (Original) | 90.0/98.89 (`20260125_001929`), 90.0/98.89 (`20260322_200417`), 80.0/94.44 (`20260322_212650`) |
| Organize Table (Original) | 40.0/96.25 (`20260129_223629`), 50.0/91.88 (`20260322_194917`), 40.0/96.25 (`20260322_220142`) |
| Cooking Preparation Setup (Original) | 50.0/75.71 (`20260115_132258`), 30.0/42.00 (`20260125_172511`), 20.0/32.00 (`20260322_212619`) |

**Cooking Preparation Setup is the unstable one — ±20 SR points is normal there and a
Cooking mismatch is not by itself evidence of a code bug.**

`outputs/eval_results/` in the research tree also holds many same-named files from other
checkpoints and ablations (some scoring 0.0/14.29). Those are **not** comparable; always
attribute a file through `config.json.back.<ts>` before using it.

### Acceptance bands for a migrated run

| configuration | SR | AP |
|---|---|---|
| Complete Kitchen Organization (Original) | 80–90 | 94–99 |
| Organize Table (Original) | 40–50 | 92–96 |
| Cooking Preparation Setup (Original) | 20–30 | 32–42 |
| each Suite 1 task | within ±20 SR of §4.1 | — |
| Suite 1 mean | ≥ 70 | ≥ 85 |
| Suite 2 mean | within ±10 SR of 53 (Original group) | — |

---

## 7. Discrepancies against the source spreadsheet, recorded

| claim | status |
|---|---|
| sheet "avg All 69 / 86" | **not reproducible.** Recomputed over the 21 configurations: 65.71 / 84.78 |
| migration manifest "65.7 / 84.5" for the same 21 | SR agrees; AP recomputes to **84.78** here from the file list in §3–§4. Unresolved rounding/selection difference of 0.3 AP |
| sheet Suite-1 re-sequenced AP "89" | files give **90.00** |
| sheet Suite-1 row-1 AP "90" | a transcription of an earlier file; the 2026-01-26 file gives 95.00 |

---

## 8. Ablations

All recomputed from their own result files.

### 8.1 Recovery — which `--recovery_mode` is the paper's naive-retry control

Pooled over the three Suite-2 V1 tasks:

| mode | SR | AP | source files |
|---|---:|---:|---|
| `full` (default, the method) | 53.33 | 79.05 | §3.1 |
| **`naive_strict`** — **the paper's 23 % / 52 % row** | **23.33** | **51.76** | `complete_kitchen_organization_v1_…_20260915_142313` (0.0/17.78), `organize_table_v1_…_20260915_142313` (60.0/92.50), `cooking_preparation_setup_v1_…_20260915_142314` (10.0/45.00) |
| `naive` — **not** the published row | 10.00 | 50.96 | `…_20260915_140639` (30.0/30.00), `…_20260915_140638` (0.0/81.88), `…_20260915_140642` (0.0/41.00) |

`full` = pose re-estimation + motion-planned reset to the approach pose, and backtracking
to the most recent Pick for holding/Place skills. `naive` skips the reset on every retry;
`naive_strict` skips it only when the previous attempt failed at the policy stage.

### 8.2 w/o Recovery

Not a separate rollout — `scripts/evaluate_long_horizon.py` synthesises a `no_recovery`
metrics block from the per-skill retry counters of the very same run. Score the §3
files again against that block:

```bash
python scripts/analysis/score_results.py --bucket no_recovery \
    'outputs/eval_results/eval_*_v1_n10_above_atomic_*.json'
```

(equivalently, read `metrics.no_recovery.avg_progress` out of the same files):

| group | SR | AP |
|---|---:|---:|
| Original (V1) | 0.0 | 14.56 |
| Variant 1 (V2) | 0.0 | 20.27 |
| Variant 2 (V3) | 0.0 | 20.85 |
| **average** (paper 0 / 18) | **0.0** | **18.56** |

That block is load-bearing for a published row. It is not dead code.

### 8.3 w/o object-centric masking

Same command, drop `--apply_distractor_masking` → Suite 2 collapses to ≈ 0 SR.

> Provenance gap: `apply_distractor_masking` is **not** recorded in the result JSON's
> `pipeline_config`, so the no-masking runs cannot be identified from the files alone.
> This ablation is reproducible by re-running, but its original files are not
> attributable. Do not trust an unlabelled 0-SR file to be this ablation.

### 8.4 Collision avoidance off

Suite 1 plus `--mplib_collision_aware false` (recorded as `collision_aware: false` in
`pipeline_config`), batch 2026-01-29 17:02:

| task | SR | AP | source |
|---|---:|---:|---|
| Alphabet Soup + Cream Cheese | 100.0 | 100.00 | `…_20260129_170221` |
| Alphabet Soup + Tomato Sauce | 80.0 | 95.00 | `…_20260129_170228` |
| Cream Cheese + Butter | 100.0 | 100.00 | `…_20260129_170226` |
| White Mug + Yellow/White Mug | 60.0 | 72.50 | `…_20260129_170224` |
| White Mug + Chocolate Pudding | 60.0 | 85.00 | `…_20260129_170226` |
| Stove + Moka Pot | 40.0 | 63.33 | `…_20260129_170227` |
| **mean** | **73.33** | **85.97** | |

This is the *ablation*, **not** the Table-1 Suite-1 row. The Table-1 row is §4.1, with
collision avoidance **on**.

### 8.5 Training-set ablations

Rebuilt datasets, identical training code — e.g.
`--dataset_name libero_above_atomic_libero_all_no_random_erasing`. See
[data_generation_and_training.md](data_generation_and_training.md). The ablation training
shells were not migrated; they differ from the main launcher only in `--dataset_name` and
batch size.

---

## 9. The 21 configurations

**Ultra-Long** (`--benchmark long_horizon_tasks_v1`), 3 tasks × V1/V2/V3, sharing 3 BDDL
files; the variants differ only in skill ordering:

`Complete Kitchen Organization {V1,V2,V3}` (9 skills) ·
`Organize Table {V1,V2,V3}` (16 skills) ·
`Cooking Preparation Setup {V1,V2,V3}` (10 skills)

**LIBERO-Long++** (`--benchmark long_horizon_tasks_libero_long`), 6 tasks × {original,
` V1`}, sharing 6 BDDL files (3–4 skills each):

`Turn On The Stove And Put The Moka Pot On It (Original)` ·
`Put Both The Alphabet Soup And The Cream Cheese Box In The Basket (Original)` ·
`Put Both The Alphabet Soup And The Tomato Sauce In The Basket (Original)` ·
`Put Both The Cream Cheese Box And The Butter In The Basket (Original)` ·
`Put The White Mug On The Left Plate And Put The Yellow And White Mug On The Right Plate (Original)` ·
`Put The White Mug On The Plate And Put The Chocolate Pudding To The Right Of The Plate (Original)`

---

# Independent re-run, 2026-09

All 21 configurations were re-run from this release tree with the published checkpoint
([`yygx/lilo-vla-openvla-oft`](https://huggingface.co/yygx/lilo-vla-openvla-oft)), 10
trials each, default flags plus `--apply_distractor_masking`. Group means, at the same
granularity the paper reports:

| Group | This re-run (SR / AP) | Paper (SR / AP) |
|---|---|---|
| Suite 1 — Original | 73.3 / 87.4 | 78 / 88 |
| Suite 1 — Reordered | 86.7 / 91.8 | 85 / 89 |
| **Suite 1 — Avg.** | **80.0 / 89.6** | **82 / 89** |
| Suite 2 — Original | 60.0 / 77.2 | 53 / 79 |
| Suite 2 — Reorder 1 | 50.0 / 85.7 | 37 / 84 |
| Suite 2 — Reorder 2 | 53.3 / 82.0 | 43 / 74 |
| **Suite 2 — Avg.** | **54.4 / 81.7** | **44 / 79** |
| **Overall** (9 scenarios, macro-averaged) | **71.5 / 86.9** | **69 / 86** |

Average Progress lands within 0.9 points overall and within 8 points for every group.
Success Rate scatters more, in both directions, which is what a 10-trial binomial does —
see "On run-to-run variance" in the README. Nothing here required re-tuning: the same
commands in this document, against the published checkpoint, produce this table.

Result JSONs for every configuration are written to `outputs/eval_results/`.
