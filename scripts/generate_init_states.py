#!/usr/bin/env python3
"""
Generate .pruned_init files for LiLo-VLA benchmark suites.

For each BDDL file, creates N initial states by repeatedly resetting the
environment and capturing the flattened MuJoCo sim state. The result is a
numpy array of shape (N, state_dim) saved via torch.save(), matching the
format used by LIBERO's existing benchmark suites (e.g., libero_10).

Usage:
    python lilo-vla/scripts/generate_init_states.py
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "externals", "boss"))

from libero.libero.envs import OffScreenRenderEnv

NUM_INIT_STATES = 50  # Match LIBERO convention

# Map suite name -> (bddl_dir, list of bddl stems)
SUITE_BDDL_MAP = {
    "ultra_long": {
        "bddl_dir": os.path.join(os.path.dirname(__file__), "..", "configs", "bddl", "ultra_long"),
        "stems": [
            "LONG_HORIZON_complete_kitchen_organization",
            "LONG_HORIZON_organize_table",
            "LONG_HORIZON_cooking_preparation_setup",
        ],
    },
    "libero_long_plus_plus": {
        "bddl_dir": os.path.join(os.path.dirname(__file__), "..", "configs", "bddl", "libero_long_plus_plus"),
        "stems": [
            "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
            "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
            "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
            "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
            "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
            "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        ],
    },
}


def generate_init_states(bddl_path, num_states=NUM_INIT_STATES):
    """Generate N initial states for a single BDDL file."""
    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": 256,
        "camera_widths": 256,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_camera_obs": True,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
    }

    env = OffScreenRenderEnv(**env_args)

    states = []
    for i in range(num_states):
        env.reset()
        state = env.env.sim.get_state().flatten()
        states.append(state)

    env.close()
    return np.array(states)


def main():
    for suite_name, suite_info in SUITE_BDDL_MAP.items():
        bddl_dir = os.path.abspath(suite_info["bddl_dir"])
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "configs", "init_states", suite_name
        )
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Suite: {suite_name}")
        print(f"BDDL dir: {bddl_dir}")
        print(f"Output dir: {output_dir}")
        print(f"{'='*60}")

        for stem in suite_info["stems"]:
            bddl_path = os.path.join(bddl_dir, f"{stem}.bddl")
            output_path = os.path.join(output_dir, f"{stem}.pruned_init")

            if not os.path.exists(bddl_path):
                print(f"  [SKIP] BDDL not found: {bddl_path}")
                continue

            print(f"\n  Generating {NUM_INIT_STATES} init states for: {stem}")
            print(f"    BDDL: {bddl_path}")

            states = generate_init_states(bddl_path, NUM_INIT_STATES)
            torch.save(states, output_path)

            print(f"    Saved: {output_path}")
            print(f"    Shape: {states.shape}")

    print(f"\n{'='*60}")
    print("Done! All init states generated.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
