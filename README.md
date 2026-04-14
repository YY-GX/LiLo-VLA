# LiLo-VLA Benchmark

Evaluation benchmark for **LiLo-VLA: Compositional Long-Horizon Manipulation via Linked Object-Centric Policies**.

This release contains the benchmark only (BDDL task definitions + initial states). The full LiLo-VLA codebase will be released in early May 2025.

## Benchmark Overview

21 long-horizon manipulation tasks across two suites, built on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO):

| Suite | Tasks | Variants | Total | Focus |
|---|---|---|---|---|
| **Ultra-Long** | 3 | 3 per task | 9 | Temporal scalability (9-16 skills per task) |
| **LIBERO-Long++** | 6 | 2 per task | 12 | Visual robustness (cluttered backgrounds) |

## Setup

### 1. Install LIBERO

Follow the [official LIBERO installation guide](https://github.com/Lifelong-Robot-Learning/LIBERO).

### 2. Install this package

```bash
git clone <this-repo-url>
cd lilo-vla
pip install -e .
```

That's it. No manual file copying needed — BDDL files and init states are bundled and resolved automatically.

## Usage

```python
import lilo_vla.benchmark  # registers suites into LIBERO
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

# Load a suite (same API as libero_10, libero_spatial, etc.)
bench = get_benchmark("ultra_long")()

for i in range(bench.get_num_tasks()):
    task = bench.get_task(i)
    bddl_path = bench.get_task_bddl_file_path(i)
    init_states = bench.get_task_init_states(i)  # (50, state_dim)

    # Create environment and set deterministic initial state
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256, camera_widths=256,
        has_renderer=False, has_offscreen_renderer=True,
        ignore_done=True, use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    env.reset()
    obs = env.set_init_state(init_states[0])  # use init_states[trial_idx] for each trial

    # ... run your policy, check success ...

    env.close()
```

## Citation

```bibtex
@article{yang2026lilo,
  title={LiLo-VLA: Compositional Long-Horizon Manipulation via Linked Object-Centric Policies},
  author={Yang, Yue and Cheng, Shuo and Fang, Yu and Bharadhwaj, Homanga and Ding, Mingyu and Bertasius, Gedas and Szafir, Daniel},
  journal={arXiv preprint arXiv:2602.21531},
  year={2026}
}
```
