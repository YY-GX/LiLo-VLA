"""
LiLo-VLA -- Compositional Long-Horizon Manipulation via Linked Object-Centric Policies.

Importing this package BOOTSTRAPS LIBERO, in this order:

1. Point ``LIBERO_CONFIG_PATH`` at ``lilo-vla/configs/libero/`` and make sure a
   ``config.yaml`` exists there.  Upstream ``libero/libero/__init__.py`` executes at
   import time and, when no config file is present, blocks on an interactive
   ``input()`` prompt.  That code cannot be monkey-patched after the fact, so the
   file has to be on disk *before* the first ``import libero``.
2. Apply ``lilo_vla.libero_compat`` -- the five BOSS-fork deltas that the published
   numbers depend on (``pickedup`` predicate, ``closed`` alias, ``ObjectState.check_ontop``,
   ``SiteObject.in_box``, ``WoodenCabinet``/``Microwave`` articulation ranges).

Both steps are idempotent, and ``import lilo_vla`` must come before any ``import
libero`` (``scripts/evaluate_long_horizon.py`` does exactly that, as its first import).

Notes
-----
* ``LIBERO_CONFIG_PATH`` is only set when it is not already in the environment, so an
  explicit user setting always wins.
* The generated ``configs/libero/config.yaml`` holds absolute, machine-local paths and
  is therefore git-ignored rather than shipped.
* If ``libero`` is not installed at all, the bootstrap warns instead of raising, so that
  training-only use of this package still works.  Anything that actually touches LIBERO
  then fails at its own import, never silently against an unpatched LIBERO.
"""

import importlib.util
import os
import pathlib
import warnings


class LiLoCompatError(RuntimeError):
    """The LIBERO compatibility patches could not be applied.

    Raised at import time. Continuing would produce success/failure decisions
    that differ from the ones the published results were measured with.
    """

__all__ = ["LILO_ROOT", "CONFIGS_DIR", "bootstrap"]

# lilo_vla/__init__.py -> lilo_vla/ -> lilo-vla/
LILO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS_DIR = LILO_ROOT / "configs"

_LIBERO_CONFIG_DIR = CONFIGS_DIR / "libero"
_BOOTSTRAPPED = False


def _libero_package_dir():
    """Locate the installed ``libero.libero`` package WITHOUT importing it.

    ``libero`` is a namespace package (no top-level ``__init__.py``), so ``find_spec``
    does not execute ``libero/libero/__init__.py`` -- which is the whole point, since
    that module is what we are trying to configure before it runs.
    Returns ``None`` when LIBERO is not installed.
    """
    try:
        spec = importlib.util.find_spec("libero.libero")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return pathlib.Path(spec.origin).resolve().parent


def _write_libero_config(libero_pkg):
    """Write ``configs/libero/config.yaml`` if it is missing, incomplete or stale."""
    import yaml

    config_file = _LIBERO_CONFIG_DIR / "config.yaml"
    wanted = {
        "assets": str(libero_pkg / "assets"),
        "benchmark_root": str(libero_pkg),
        "datasets": str(libero_pkg.parent / "datasets"),
        "bddl_files": str(CONFIGS_DIR / "bddl"),
        "init_states": str(CONFIGS_DIR / "init_states"),
    }

    current = None
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                current = yaml.load(f.read(), Loader=yaml.FullLoader)
        except Exception:
            current = None

    # Exact match, so a moved checkout or a reinstalled / switched LIBERO always
    # self-heals.  The file is a derived cache, never hand-maintained state: point
    # LIBERO_CONFIG_PATH somewhere else to take full control of it.
    if current == wanted:
        return

    _LIBERO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(wanted, f)


def bootstrap():
    """Configure and patch LIBERO for LiLo-VLA.  Safe to call repeatedly."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    _LIBERO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(_LIBERO_CONFIG_DIR))

    libero_pkg = _libero_package_dir()
    if libero_pkg is None:
        warnings.warn(
            "LIBERO is not installed; skipping the LiLo-VLA LIBERO bootstrap. "
            "Evaluation and data generation will not work until you install "
            "LIBERO (pinned at f78abd6) -- see the README.",
            RuntimeWarning,
            stacklevel=2,
        )
        _BOOTSTRAPPED = True
        return

    # Only write into the directory we actually own.
    if os.environ["LIBERO_CONFIG_PATH"] == str(_LIBERO_CONFIG_DIR):
        _write_libero_config(libero_pkg)

    from . import libero_compat

    libero_compat.apply_patches()

    # Fail loudly rather than silently producing numbers that are not comparable.
    # Three of these patches change SUCCESS CRITERIA (the "on top of", "in" and
    # "closed" thresholds), so an unpatched LIBERO does not crash -- it quietly
    # scores the same rollouts differently.
    try:
        libero_compat.verify()
    except AssertionError as exc:
        raise LiLoCompatError(
            f"The LiLo-VLA LIBERO patches did not apply: {exc}\n\n"
            f"Your LIBERO is not the supported revision. Install it with:\n"
            f"    pip install \"libero @ git+https://github.com/"
            f"Lifelong-Robot-Learning/LIBERO.git@f78abd6\"\n\n"
            f"These patches define the success criteria the published numbers use. "
            f"Running without them gives results that are not comparable."
        ) from exc

    _BOOTSTRAPPED = True


bootstrap()
