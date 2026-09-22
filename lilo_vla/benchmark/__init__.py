"""
LiLo-VLA Benchmark Suite Registration
======================================

Registers Ultra-Long and LIBERO-Long++ evaluation suites into LIBERO's
existing benchmark system (``libero.libero.benchmark``).

Suites registered
-----------------
- ``ultra_long``                    -- 3 Ultra-Long tasks (original sequence)
- ``ultra_long_variant_1``          -- 3 Ultra-Long tasks (permuted V2)
- ``ultra_long_variant_2``          -- 3 Ultra-Long tasks (permuted V3)
- ``ultra_long_all``                -- 9 Ultra-Long tasks combined
- ``libero_long_plus_plus``         -- 6 LIBERO-Long++ tasks (original)
- ``libero_long_plus_plus_variant`` -- 6 LIBERO-Long++ tasks (reversed order)
- ``libero_long_plus_plus_all``     -- 12 LIBERO-Long++ tasks combined

Usage::

    import lilo_vla.benchmark          # triggers registration
    from libero.libero.benchmark import get_benchmark

    b = get_benchmark("ultra_long")()
    print(b.get_num_tasks())           # 3

Importing this module has no side effects outside the process: it registers the
suites in LIBERO's in-memory ``BENCHMARK_MAPPING`` and touches nothing on disk.

Historical CLI spellings
------------------------
``long_horizon_tasks_v1`` and ``long_horizon_tasks_libero_long`` are the names every
command in the lab notebook uses.  ``BENCHMARK_ALIASES`` maps them onto the published
suite names, and :func:`get_bddl_path` accepts either spelling.
"""

import os
import pathlib
import logging

from libero.libero.benchmark import (
    Benchmark,
    Task,
    register_benchmark,
    BENCHMARK_MAPPING,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ULTRA_LONG",
    "ULTRA_LONG_VARIANT_1",
    "ULTRA_LONG_VARIANT_2",
    "ULTRA_LONG_ALL",
    "LIBERO_LONG_PLUS_PLUS",
    "LIBERO_LONG_PLUS_PLUS_VARIANT",
    "LIBERO_LONG_PLUS_PLUS_ALL",
    "BENCHMARK_ALIASES",
    "get_bddl_path",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_LILO_ROOT = _THIS_DIR.parent.parent            # lilo-vla/
_CONFIGS_BDDL = _LILO_ROOT / "configs" / "bddl"
_CONFIGS_INIT_STATES = _LILO_ROOT / "configs" / "init_states"

# ---------------------------------------------------------------------------
# Benchmark-name bridge
#
# The historical CLI / config spellings on the left are what every recorded
# evaluation command uses; the published suite names on the right are what this
# module registers and what `configs/bddl/<folder>/` is named.  Both are accepted
# everywhere.  File stems are identical on both sides -- only the folder differs.
# ---------------------------------------------------------------------------
BENCHMARK_ALIASES = {
    "long_horizon_tasks_v1": "ultra_long",
    "long_horizon_tasks_libero_long": "libero_long_plus_plus",
}

# ---------------------------------------------------------------------------
# BDDL filenames (without .bddl extension)
# ---------------------------------------------------------------------------
_ULTRA_LONG_BDDL_STEMS = [
    "LONG_HORIZON_complete_kitchen_organization",
    "LONG_HORIZON_organize_table",
    "LONG_HORIZON_cooking_preparation_setup",
]

_LIBERO_LONG_PP_BDDL_STEMS = [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
]

# ---------------------------------------------------------------------------
# Language strings  (from tasks_and_skills.json)
# ---------------------------------------------------------------------------

# Ultra-Long Original (IDs 8, 9, 10) -- paper names (short task descriptions)
_ULTRA_LONG_ORIGINAL_LANGUAGES = [
    "Complete Kitchen Organization",
    "Organize Table",
    "Cooking Preparation Setup",
]

# Ultra-Long Variant 1 / V2 (IDs 19, 20, 21) -- permuted compound instructions
_ULTRA_LONG_V2_LANGUAGES = [
    # ID 19: Complete Kitchen Organization (Reorder 1)
    "pick cream cheese, then place cream cheese in the basket, then pick black bowl, "
    "then place black bowl on the plate, then pick black bowl, then stack black bowl "
    "on black bowl, then pick black bowl, then place black bowl on the plate, then "
    "close the bottom drawer of the cabinet",
    # ID 20: Organize Table (Reorder 1)
    "pick alphabet soup, then place alphabet soup in the basket, then pick tomato sauce, "
    "then place tomato sauce in the basket, then pick alphabet soup, then place alphabet "
    "soup in the basket, then pick tomato sauce, then place tomato sauce in the basket, "
    "then pick cream cheese, then place cream cheese in the basket, then pick cream cheese, "
    "then place cream cheese in the basket, then pick black bowl, then place black bowl "
    "in bottom drawer of the cabinet, then close the bottom drawer of the cabinet",
    # ID 21: Cooking Preparation Setup (Reorder 1)
    "turn on the stove, then pick moka pot, then place moka pot on the stove, then "
    "turn on the stove, then pick frying pan, then place frying pan on the stove, "
    "then open the microwave",
]

# Ultra-Long Variant 2 / V3 (IDs 22, 23, 24) -- permuted compound instructions
_ULTRA_LONG_V3_LANGUAGES = [
    # ID 22: Complete Kitchen Organization (Reorder 2)
    "pick black bowl, then place black bowl on the plate, then pick black bowl, "
    "then stack black bowl on black bowl, then pick cream cheese, then place cream "
    "cheese in the basket, then pick black bowl, then place black bowl on the plate, "
    "then close the bottom drawer of the cabinet",
    # ID 23: Organize Table (Reorder 2)
    "pick alphabet soup, then place alphabet soup in the basket, then pick alphabet soup, "
    "then place alphabet soup in the basket, then pick tomato sauce, then place tomato "
    "sauce in the basket, then pick tomato sauce, then place tomato sauce in the basket, "
    "then pick cream cheese, then place cream cheese in the basket, then pick cream cheese, "
    "then place cream cheese in the basket, then pick black bowl, then place black bowl "
    "in bottom drawer of the cabinet, then close the bottom drawer of the cabinet",
    # ID 24: Cooking Preparation Setup (Reorder 2)
    "turn on the stove, then turn on the stove, then pick frying pan, then place "
    "frying pan on the stove, then pick moka pot, then place moka pot on the stove, "
    "then open the microwave",
]

# LIBERO-Long++ Original (IDs 12-17) -- Title Case to match tasks_and_skills.json
_LIBERO_LONG_PP_ORIGINAL_LANGUAGES = [
    "Turn On The Stove And Put The Moka Pot On It",
    "Put Both The Alphabet Soup And The Cream Cheese Box In The Basket",
    "Put Both The Alphabet Soup And The Tomato Sauce In The Basket",
    "Put Both The Cream Cheese Box And The Butter In The Basket",
    "Put The White Mug On The Left Plate And Put The Yellow And White Mug On The Right Plate",
    "Put The White Mug On The Plate And Put The Chocolate Pudding To The Right Of The Plate",
]

# LIBERO-Long++ Variant (IDs 25-30) -- reversed order
_LIBERO_LONG_PP_VARIANT_LANGUAGES = [
    # ID 25
    "Put The Moka Pot On It And Then Turn On The Stove",
    # ID 26
    "Put The Cream Cheese Box In The Basket And Then Put The Alphabet Soup In The Basket",
    # ID 27
    "Put The Tomato Sauce In The Basket And Then Put The Alphabet Soup In The Basket",
    # ID 28
    "Put The Butter In The Basket And Then Put The Cream Cheese Box In The Basket",
    # ID 29
    "Put The Yellow And White Mug On The Right Plate And Then Put The White Mug On The Left Plate",
    # ID 30
    "Put The Chocolate Pudding To The Right Of The Plate And Then Put The White Mug On The Plate",
]

# ---------------------------------------------------------------------------
# Task names (used as unique task identifiers)
# ---------------------------------------------------------------------------

_ULTRA_LONG_ORIGINAL_NAMES = [
    "LONG_HORIZON_complete_kitchen_organization_v1",
    "LONG_HORIZON_organize_table_v1",
    "LONG_HORIZON_cooking_preparation_setup_v1",
]

_ULTRA_LONG_V2_NAMES = [
    "LONG_HORIZON_complete_kitchen_organization_v2",
    "LONG_HORIZON_organize_table_v2",
    "LONG_HORIZON_cooking_preparation_setup_v2",
]

_ULTRA_LONG_V3_NAMES = [
    "LONG_HORIZON_complete_kitchen_organization_v3",
    "LONG_HORIZON_organize_table_v3",
    "LONG_HORIZON_cooking_preparation_setup_v3",
]

_LIBERO_LONG_PP_ORIGINAL_NAMES = [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
]

_LIBERO_LONG_PP_VARIANT_NAMES = [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_variant",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_variant",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_variant",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_variant",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_variant",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_variant",
]


# ---------------------------------------------------------------------------
# Helper: build Task tuples
# ---------------------------------------------------------------------------

def _make_tasks(names, languages, bddl_stems, problem_folder):
    """Build a list of Task NamedTuples."""
    tasks = []
    for name, lang, stem in zip(names, languages, bddl_stems):
        tasks.append(
            Task(
                name=name,
                language=lang,
                problem="Libero",
                problem_folder=problem_folder,
                bddl_file=f"{stem}.bddl",
                init_states_file=f"{stem}.pruned_init",
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# LiLoBenchmark base -- bypasses boss_task_map / selected_task_indexes
# ---------------------------------------------------------------------------

class LiLoBenchmark(Benchmark):
    """
    Base class for LiLo-VLA benchmark suites.

    Subclasses must set ``self._lilo_tasks`` (a list of ``Task`` tuples)
    before calling ``_make_benchmark()``.  This completely bypasses the
    ``task_maps`` / ``selected_task_indexes`` machinery in the upstream
    ``Benchmark`` base class.
    """

    def _make_benchmark(self):
        """Override to use our own task list instead of task_maps."""
        self.tasks = list(self._lilo_tasks)
        if self.n_tasks is None:
            self.n_tasks = len(self.tasks)

    def get_task_init_states(self, i):
        """Resolve init states: try our configs/init_states first, fall back to LIBERO."""
        import torch
        task = self.tasks[i]
        # 1) Our own init_states directory
        local_path = _CONFIGS_INIT_STATES / task.problem_folder / task.init_states_file
        if local_path.exists():
            # `.pruned_init` files are pickled numpy arrays; torch >= 2.6 defaults
            # `weights_only` to True and would raise UnpicklingError.
            return torch.load(str(local_path), map_location="cpu", weights_only=False)
        # 2) Fall back to LIBERO's init_states directory
        try:
            from libero.libero import get_libero_path
            libero_path = os.path.join(
                get_libero_path("init_states"),
                task.problem_folder,
                task.init_states_file,
            )
            if os.path.exists(libero_path):
                return torch.load(libero_path, map_location="cpu", weights_only=False)
        except Exception:
            pass
        raise FileNotFoundError(
            f"Init states not found for task '{task.name}'. "
            f"Looked at: {local_path} and LIBERO init_states directory. "
            f"Run: python lilo-vla/scripts/generate_init_states.py"
        )

    def get_task_bddl_file_path(self, i):
        """Resolve BDDL path: try our configs/bddl first, fall back to LIBERO."""
        task = self.tasks[i]
        # 1) Our own BDDL directory
        local_path = _CONFIGS_BDDL / task.problem_folder / task.bddl_file
        if local_path.exists():
            return str(local_path)
        # 2) Fall back to LIBERO's bddl_files directory
        try:
            from libero.libero import get_libero_path
            libero_path = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file,
            )
            if os.path.exists(libero_path):
                return libero_path
        except Exception:
            pass
        # 3) Return local path even if missing (caller can handle the error)
        return str(local_path)

    # -- evaluation protocol ------------------------------------------------

    def make_env(self, i, camera_heights=256, camera_widths=256, **kwargs):
        """Build the environment for task ``i`` using the published protocol.

        This is the exact setup every number in the paper was measured with, so
        that results obtained through it are comparable. In particular the env is
        returned **already reset**, from a bare ``env.reset()`` with no
        ``set_init_state`` / ``reset_to`` / seeding: the reported tables average
        over the simulator's own initial-state distribution.

        The bundled ``.pruned_init`` files are NOT used here. They exist for
        anyone who wants a deterministic sweep instead, via
        ``get_task_init_states(i)`` + ``env.set_init_state(...)`` -- that is a
        different protocol and gives different numbers.

        Extra keyword arguments are passed through to ``OffScreenRenderEnv``.

        Returns:
            (env, obs)
        """
        import lilo_vla  # noqa: F401  -- ensures the LIBERO patches are applied
        from libero.libero.envs import OffScreenRenderEnv

        env_args = dict(
            bddl_file_name=self.get_task_bddl_file_path(i),
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_names=["agentview", "robot0_eye_in_hand"],
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
        )
        env_args.update(kwargs)
        env = OffScreenRenderEnv(**env_args)
        obs = env.reset()
        return env, obs


# ---------------------------------------------------------------------------
# Ultra-Long suites
# ---------------------------------------------------------------------------

@register_benchmark
class ULTRA_LONG(LiLoBenchmark):
    """3 Ultra-Long tasks -- original sequence (task IDs 8-10)."""

    def __init__(self, n_tasks=None):
        self.name = "ultra_long"
        self._lilo_tasks = _make_tasks(
            _ULTRA_LONG_ORIGINAL_NAMES,
            _ULTRA_LONG_ORIGINAL_LANGUAGES,
            _ULTRA_LONG_BDDL_STEMS,
            "ultra_long",
        )
        self.n_tasks = n_tasks
        self._make_benchmark()


@register_benchmark
class ULTRA_LONG_VARIANT_1(LiLoBenchmark):
    """3 Ultra-Long tasks -- permuted V2 (task IDs 19-21)."""

    def __init__(self, n_tasks=None):
        self.name = "ultra_long_variant_1"
        self._lilo_tasks = _make_tasks(
            _ULTRA_LONG_V2_NAMES,
            _ULTRA_LONG_V2_LANGUAGES,
            _ULTRA_LONG_BDDL_STEMS,
            "ultra_long",
        )
        self.n_tasks = n_tasks
        self._make_benchmark()


@register_benchmark
class ULTRA_LONG_VARIANT_2(LiLoBenchmark):
    """3 Ultra-Long tasks -- permuted V3 (task IDs 22-24)."""

    def __init__(self, n_tasks=None):
        self.name = "ultra_long_variant_2"
        self._lilo_tasks = _make_tasks(
            _ULTRA_LONG_V3_NAMES,
            _ULTRA_LONG_V3_LANGUAGES,
            _ULTRA_LONG_BDDL_STEMS,
            "ultra_long",
        )
        self.n_tasks = n_tasks
        self._make_benchmark()


@register_benchmark
class ULTRA_LONG_ALL(LiLoBenchmark):
    """All 9 Ultra-Long tasks (original + V2 + V3)."""

    def __init__(self, n_tasks=None):
        self.name = "ultra_long_all"
        self._lilo_tasks = (
            _make_tasks(
                _ULTRA_LONG_ORIGINAL_NAMES,
                _ULTRA_LONG_ORIGINAL_LANGUAGES,
                _ULTRA_LONG_BDDL_STEMS,
                "ultra_long",
            )
            + _make_tasks(
                _ULTRA_LONG_V2_NAMES,
                _ULTRA_LONG_V2_LANGUAGES,
                _ULTRA_LONG_BDDL_STEMS,
                "ultra_long",
            )
            + _make_tasks(
                _ULTRA_LONG_V3_NAMES,
                _ULTRA_LONG_V3_LANGUAGES,
                _ULTRA_LONG_BDDL_STEMS,
                "ultra_long",
            )
        )
        self.n_tasks = n_tasks
        self._make_benchmark()


# ---------------------------------------------------------------------------
# LIBERO-Long++ suites
# ---------------------------------------------------------------------------

@register_benchmark
class LIBERO_LONG_PLUS_PLUS(LiLoBenchmark):
    """6 LIBERO-Long++ tasks -- original order (task IDs 12-17)."""

    def __init__(self, n_tasks=None):
        self.name = "libero_long_plus_plus"
        self._lilo_tasks = _make_tasks(
            _LIBERO_LONG_PP_ORIGINAL_NAMES,
            _LIBERO_LONG_PP_ORIGINAL_LANGUAGES,
            _LIBERO_LONG_PP_BDDL_STEMS,
            "libero_long_plus_plus",
        )
        self.n_tasks = n_tasks
        self._make_benchmark()


@register_benchmark
class LIBERO_LONG_PLUS_PLUS_VARIANT(LiLoBenchmark):
    """6 LIBERO-Long++ tasks -- reversed order (task IDs 25-30)."""

    def __init__(self, n_tasks=None):
        self.name = "libero_long_plus_plus_variant"
        self._lilo_tasks = _make_tasks(
            _LIBERO_LONG_PP_VARIANT_NAMES,
            _LIBERO_LONG_PP_VARIANT_LANGUAGES,
            _LIBERO_LONG_PP_BDDL_STEMS,
            "libero_long_plus_plus",
        )
        self.n_tasks = n_tasks
        self._make_benchmark()


@register_benchmark
class LIBERO_LONG_PLUS_PLUS_ALL(LiLoBenchmark):
    """All 12 LIBERO-Long++ tasks (original + reversed variant)."""

    def __init__(self, n_tasks=None):
        self.name = "libero_long_plus_plus_all"
        self._lilo_tasks = (
            _make_tasks(
                _LIBERO_LONG_PP_ORIGINAL_NAMES,
                _LIBERO_LONG_PP_ORIGINAL_LANGUAGES,
                _LIBERO_LONG_PP_BDDL_STEMS,
                "libero_long_plus_plus",
            )
            + _make_tasks(
                _LIBERO_LONG_PP_VARIANT_NAMES,
                _LIBERO_LONG_PP_VARIANT_LANGUAGES,
                _LIBERO_LONG_PP_BDDL_STEMS,
                "libero_long_plus_plus",
            )
        )
        self.n_tasks = n_tasks
        self._make_benchmark()


# ---------------------------------------------------------------------------
# Benchmark bridge: resolve a BDDL file inside this repo.
#
# Replaces the previous `setup_bddl_files()`, which symlinked our BDDL directories
# into the installed LIBERO package on import.  That mutated a third-party install
# as a side effect of `import lilo_vla.benchmark`, so it is gone; resolution is now
# explicit and local.
# ---------------------------------------------------------------------------

def get_bddl_path(problem_folder, stem):
    """Absolute path of ``configs/bddl/<problem_folder>/<stem>.bddl``.

    Args:
        problem_folder: a published suite name (``ultra_long``,
            ``libero_long_plus_plus``, ``atomic_skills``) or one of the historical
            spellings in :data:`BENCHMARK_ALIASES`.
        stem: the BDDL file stem, with or without the ``.bddl`` extension.

    Raises:
        FileNotFoundError: if the file is not bundled.
    """
    folder = BENCHMARK_ALIASES.get(str(problem_folder), str(problem_folder))
    name = str(stem)
    if name.endswith(".bddl"):
        name = name[: -len(".bddl")]
    path = _CONFIGS_BDDL / folder / f"{name}.bddl"
    if not path.exists():
        raise FileNotFoundError(
            f"BDDL file not found: {path} "
            f"(benchmark={problem_folder!r}, stem={stem!r})"
        )
    return str(path)
