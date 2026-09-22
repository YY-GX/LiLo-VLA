"""
libero_compat  --  make a stock upstream LIBERO behave like the BOSS fork,
                   for the LiLo-VLA evaluation path only.

WHY THIS EXISTS
---------------
The numbers in the paper were produced against a *fork* of LIBERO that ships with
the BOSS project (RA-L'25), installed editable as "BOSS", rather than against upstream

    github.com/Lifelong-Robot-Learning/LIBERO @ f78abd6 "fixed the venv issue"

The fork touches 20 .py files and 13 scene XMLs. A file-by-file diff (see
11_libero_fork_remaining_delta.md) shows that only FIVE of those deltas are actually
exercised by the LiLo-VLA execution path
(scripts/phase3/pipeline/evaluation/evaluate_above.py + utils/skill_checker.py).
Everything else is dead code, debug comments, BOSS-only features (SequentialEnv, the
BOSS benchmark registry), scaled "larger_/smaller_" object variants that no LiLo-VLA
BDDL references, or two extra cameras that the eval never renders.

This module reproduces exactly those five deltas by monkey-patching, so the release
can depend on `libero` installed from the upstream repo instead of vendoring a fork.

USAGE
-----
    import lilo_vla               # bootstrap: LIBERO config + patches, idempotent
    # ... then create OffScreenRenderEnv / call eval_predicate_fn as usual

or explicitly:

    from lilo_vla.libero_compat import apply_patches, verify
    apply_patches(); verify()

Import this BEFORE the first `OffScreenRenderEnv(...)` is constructed. (Importing it
after is still fine for the predicate patches -- they are looked up per call -- but
WoodenCabinet's articulation ranges are baked in at object construction time.)

NOT COVERED HERE (deliberately -- see the report for the reasoning)
------------------------------------------------------------------
* libero/libero/__init__.py: BOSS renames LIBERO_CONFIG_PATH -> BOSS_CONFIG_PATH and
  reads ./.boss/config.yaml relative to the CWD, and it comments out the interactive
  `input()` prompt that upstream fires on first import. That code runs at *import*
  time, before this module can intervene, so it cannot be monkey-patched. The release
  must instead ship/create `$LIBERO_CONFIG_PATH/config.yaml` (or `~/.libero/config.yaml`)
  before importing libero.
* libero/libero/benchmark/__init__.py: BOSS replaces the whole benchmark registry.
  evaluate_above.py does not use it (it builds the env straight from a .bddl path), so
  it is out of scope here; lilo-vla registers its own suites (phase-2a "Option B").
* the 13 assets/scenes/*.xml: BOSS only adds a `back_camera` and a `top_camera` to each.
  The eval renders ["agentview", "robot0_eye_in_hand"] only.
* the "larger_/smaller_"/egg/lemon/onion/potato/akita_green_bowl object classes and
  their assets: no LiLo-VLA task BDDL and no atomic-skill BDDL references them (verified
  over all 9 released task BDDLs + 106 atomic-skill BDDLs).
"""

_PATCHED = False

__all__ = ["apply_patches", "verify", "PickedUp"]


# ---------------------------------------------------------------------------
# PATCH 1 of 5 -- the PickedUp predicate class
#
# Provenance: BOSS libero/libero/envs/predicates/base_predicates.py, lines 121-138
#             (does not exist upstream at all).
# Copied verbatim; only the module-level import of UnaryAtomic is resolved lazily.
#
# Consumers on the LiLo-VLA path:
#   * skill_checker.check_skill_success_by_language -> eval_predicate_fn("pickedup", ...)
#     15 of the 37 skills used by the 9 released tasks are `pickedup`.
#   * skill_checker.reset_predicate_baselines() reaches into the *singleton instance*
#     and does `del pickedup_predicate.initial_heights[target_object]`, so the registry
#     must hold one long-lived INSTANCE (as BOSS does), not the class.
#   * 40/92 atomic_skills/*.bddl + 7/14 atomic_skills_additional/*.bddl goals.
# ---------------------------------------------------------------------------
def _make_pickedup_class():
    from libero.libero.envs.predicates.base_predicates import UnaryAtomic

    class PickedUp(UnaryAtomic):
        def __init__(self):
            super().__init__()
            self.initial_heights = {}  # Store initial heights of objects

        def __call__(self, arg):
            # Get object's current position
            current_pos = arg.get_geom_state()["pos"]
            current_height = current_pos[2]

            # Store initial height if not already stored
            if arg.object_name not in self.initial_heights:
                self.initial_heights[arg.object_name] = current_height
                return False  # Object not picked up at initialization

            # Check if object is lifted at least 0.03m above its initial height
            initial_height = self.initial_heights[arg.object_name]
            return current_height >= initial_height + 0.03

    return PickedUp


PickedUp = None  # bound by apply_patches()


def _patch_predicate_registry():
    """
    PATCH 2 of 5 -- register `pickedup` and the `closed` alias.

    Provenance: BOSS libero/libero/envs/predicates/__init__.py
        line 16:    "closed": Close(),  # Alias for past tense used in BDDL files
        line 19:    "pickedup": PickedUp(),

    `closed` matters because bddl lowercases goal predicate heads, so
    `(Closed wooden_cabinet_1_bottom_region)` in
    configs/bddl/ultra_long/LONG_HORIZON_complete_kitchen_organization.bddl:151
    resolves to the key "closed", which upstream does not have
    (upstream only has "close"). Without it eval_predicate_fn() raises AssertionError
    the first time _check_success() runs on that task.
    """
    global PickedUp
    from libero.libero.envs import predicates as _pred
    from libero.libero.envs.predicates import base_predicates as _bp

    PickedUp = _make_pickedup_class()
    # Also expose it where BOSS defines it, so `update_predicate_fn_dict("x", "PickedUp")`
    # and any `from ...base_predicates import *` keeps working.
    if not hasattr(_bp, "PickedUp"):
        _bp.PickedUp = PickedUp
    if not hasattr(_pred, "PickedUp"):
        _pred.PickedUp = PickedUp

    d = _pred.VALIDATE_PREDICATE_FN_DICT
    # Idempotent: never clobber an existing instance (it carries `initial_heights`
    # state that reset_predicate_baselines() mutates across an episode).
    if "pickedup" not in d:
        d["pickedup"] = PickedUp()
    if "closed" not in d:
        d["closed"] = _bp.Close()


def _patch_object_state_check_ontop():
    """
    PATCH 3 of 5 -- loosen ObjectState.check_ontop (the `On` predicate, object-on-object).

    Provenance: BOSS libero/libero/envs/object_states/base_object_states.py, lines 97-115.
    BOSS renamed the upstream body to `check_ontop_old` (line 78) and installed this one.

    Upstream (line 78-93 of the same file upstream):
        (this_z <= other_z) and check_contact and (xy_dist < 0.03)
    BOSS:
        (this_z <= other_z + 0.01) and check_contact and (xy_dist < 0.05)

    This is a SUCCESS-CRITERION change, not a bug fix: it makes `On` strictly easier.
    Running the released tasks on upstream tolerances would report *different* (lower)
    numbers than the paper.

    Consumers: `On(a, b)` dispatches to `b.check_ontop(a)`. When `b` is a real object
    (plate_1, akita_black_bowl_2, ...) that is ObjectState.check_ontop -- 9 of the 37
    released skills, plus the goals
        (On akita_black_bowl_1 plate_1) (On akita_black_bowl_2 plate_2)
        (On akita_black_bowl_3 akita_black_bowl_2)          [complete_kitchen_organization]
        (On akita_black_bowl_1 plate_1) (On akita_black_bowl_2 plate_2) [cooking_preparation_setup]
        (On porcelain_mug_1 plate_1) (On white_yellow_mug_1 plate_2)    [LIVING_ROOM_SCENE5]
        (On porcelain_mug_1 plate_1)                                    [LIVING_ROOM_SCENE6]
        (On akita_black_bowl_2 plate_1)                                 [organize_table]
    When `b` is a region site (flat_stove_1_cook_region) it is
    SiteObjectState.check_ontop, which BOSS did NOT change.
    """
    import numpy as np
    from libero.libero.envs.object_states.base_object_states import ObjectState

    # keep upstream's version reachable, exactly as BOSS does
    if not hasattr(ObjectState, "check_ontop_old"):
        ObjectState.check_ontop_old = ObjectState.check_ontop

    def check_ontop(self, other):
        this_object = self.env.get_object(self.object_name)
        this_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_name]
        ]
        other_object = self.env.get_object(other.object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[other.object_name]
        ]

        return (
            (this_object_position[2] <= other_object_position[2] + 0.01)
            and self.check_contact(other)
            and (
                np.linalg.norm(this_object_position[:2] - other_object_position[:2])
                < 0.05
            )
        )

    check_ontop._lilo_vla_compat = True
    ObjectState.check_ontop = check_ontop


def _patch_site_object_in_box():
    """
    PATCH 4 of 5 -- loosen SiteObject.in_box (the `In` predicate).

    Provenance: BOSS libero/libero/envs/objects/site_object.py, lines 53-54:
        # lb[2] -= 0.01
        lb[2] -= 0.05  # YY: old: 0.01 -> bug when object actually already in the box

    Again a success-criterion change (5x lower floor on the containment box).

    Consumers: `In(a, b)` -> `b.check_contain(a)` -> SiteObjectState.check_contain
    -> SiteObject.in_box. 10 of the 37 released skills are `in`, and the goals
        (In cream_cheese_1 basket_1_contain_region)              [complete_kitchen_organization]
        (In alphabet_soup_1/2 basket_1_contain_region) x2
        (In cream_cheese_1/2 basket_2_contain_region) x2
        (In tomato_sauce_1/2 basket_3_contain_region) x2
        (In akita_black_bowl_1 wooden_cabinet_1_bottom_region)   [organize_table]
        (In alphabet_soup_1 ...) (In cream_cheese_1 ...)         [LIVING_ROOM_SCENE1]
        (In alphabet_soup_1 ...) (In tomato_sauce_1 ...)         [LIVING_ROOM_SCENE2 a]
        (In cream_cheese_1 ...)  (In butter_1 ...)               [LIVING_ROOM_SCENE2 b]

    NOTE: BOSS also rewrote SiteObject.under() in the same file, but that rewrite is a
    no-op -- it only adds a defensive `np.array(total_size)` cast and renames two
    locals; the returned expression is identical to upstream. It is NOT reproduced here.
    """
    import numpy as np
    from libero.libero.envs.objects.site_object import SiteObject

    if not hasattr(SiteObject, "in_box_upstream"):
        SiteObject.in_box_upstream = SiteObject.in_box

    def in_box(self, this_position, this_mat, other_position):
        """
        Checks whether the object is contained within this SiteObject.
        (verbatim from BOSS site_object.py, only lb[2] differs from upstream)
        """
        total_size = np.abs(this_mat @ self.size)

        ub = this_position + total_size
        lb = this_position - total_size

        lb[2] -= 0.05  # BOSS: was 0.01 upstream
        return np.all(other_position > lb) and np.all(other_position < ub)

    in_box._lilo_vla_compat = True
    SiteObject.in_box = in_box


def _patch_articulation_ranges():
    """
    PATCH 5 of 5 -- articulation open/close thresholds.

    Provenance: BOSS libero/libero/envs/objects/articulated_objects.py
        line 183-184 (WoodenCabinet):
            # default_close_ranges = [0.0, 0.005]      <- upstream
            default_close_ranges  = [-0.02, 0.005]     <- BOSS
        line 62-63 (Microwave):
            # default_open_ranges  = [-2.094, -1.3]    <- upstream
            default_open_ranges   = [-2.094, -0.5236]  <- BOSS

    WoodenCabinet is REQUIRED: `is_close(qpos)` returns `qpos > min(close_ranges)`,
    so BOSS accepts a drawer that is still 2 cm open as "closed". That is the criterion
    behind the skill `close the bottom drawer of the cabinet 1`
    (skill_mappings -> ["close", "wooden_cabinet_1_bottom_region"]) and behind the goal
    `(Closed wooden_cabinet_1_bottom_region)` of LONG_HORIZON_complete_kitchen_organization.

    Microwave is NOT used by any of the 9 released task BDDLs, but 8 of the 106
    atomic-skill BDDLs have `(Open microwave_1)` / `(Close microwave_1)` goals
    (`is_open(qpos)` = `qpos < max(open_ranges)`, so BOSS calls the door "open" at 30 deg
    instead of 74.5 deg). Included so that the atomic-skill suite also reproduces.

    These are set in __init__, so the patch wraps __init__ rather than a method; it must
    therefore be applied before any env is constructed.
    """
    from libero.libero.envs.objects.articulated_objects import Microwave, WoodenCabinet

    def _wrap(cls, key, value):
        if getattr(cls.__init__, "_lilo_vla_compat", False):
            return
        orig = cls.__init__

        def __init__(self, *args, **kwargs):
            orig(self, *args, **kwargs)
            self.object_properties["articulation"][key] = list(value)

        __init__._lilo_vla_compat = True
        __init__.__doc__ = orig.__doc__
        cls.__init__ = __init__

    _wrap(WoodenCabinet, "default_close_ranges", [-0.02, 0.005])
    _wrap(Microwave, "default_open_ranges", [-2.094, -0.5236])


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def apply_patches():
    """Apply every BOSS delta the LiLo-VLA path depends on. Safe to call repeatedly."""
    global _PATCHED
    if _PATCHED:
        return
    _patch_predicate_registry()
    _patch_object_state_check_ontop()
    _patch_site_object_in_box()
    _patch_articulation_ranges()
    _PATCHED = True


def verify():
    """Raise AssertionError if any patch failed to take. Cheap; call it from a test."""
    from libero.libero.envs.predicates import VALIDATE_PREDICATE_FN_DICT
    from libero.libero.envs.object_states.base_object_states import ObjectState
    from libero.libero.envs.objects.site_object import SiteObject
    from libero.libero.envs.objects.articulated_objects import Microwave, WoodenCabinet

    assert "pickedup" in VALIDATE_PREDICATE_FN_DICT, "PickedUp not registered"
    assert "closed" in VALIDATE_PREDICATE_FN_DICT, "'closed' alias not registered"
    assert hasattr(VALIDATE_PREDICATE_FN_DICT["pickedup"], "initial_heights"), \
        "'pickedup' must be a stateful instance (skill_checker mutates .initial_heights)"
    assert getattr(ObjectState.check_ontop, "_lilo_vla_compat", False), \
        "ObjectState.check_ontop not patched"
    assert getattr(SiteObject.in_box, "_lilo_vla_compat", False), \
        "SiteObject.in_box not patched"
    assert WoodenCabinet().object_properties["articulation"]["default_close_ranges"] \
        == [-0.02, 0.005], "WoodenCabinet close range not patched"
    assert Microwave().object_properties["articulation"]["default_open_ranges"] \
        == [-2.094, -0.5236], "Microwave open range not patched"
    return True


apply_patches()
