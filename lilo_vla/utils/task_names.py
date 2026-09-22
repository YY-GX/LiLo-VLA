"""Canonical task names, and aliases for the pre-2026-09 names.

The two suites used to number their variants inconsistently: an Ultra-Long task's
``V1`` meant the ORIGINAL ordering, while a LIBERO-Long++ task's ``V1`` meant the
REORDERED one. Names now state the role explicitly -- ``(Original)``,
``(Reorder 1)``, ``(Reorder 2)`` -- and mean the same thing in both suites.

The old names still work everywhere a task name is accepted.
"""

# old name -> canonical name
LEGACY_TASK_NAMES = {
    'Complete Kitchen Organization V1': 'Complete Kitchen Organization (Original)',
    'Organize Table V1': 'Organize Table (Original)',
    'Cooking Preparation Setup V1': 'Cooking Preparation Setup (Original)',
    'Turn On The Stove And Put The Moka Pot On It': 'Turn On The Stove And Put The Moka Pot On It (Original)',
    'Put Both The Alphabet Soup And The Cream Cheese Box In The Basket': 'Put Both The Alphabet Soup And The Cream Cheese Box In The Basket (Original)',
    'Put Both The Alphabet Soup And The Tomato Sauce In The Basket': 'Put Both The Alphabet Soup And The Tomato Sauce In The Basket (Original)',
    'Put Both The Cream Cheese Box And The Butter In The Basket': 'Put Both The Cream Cheese Box And The Butter In The Basket (Original)',
    'Put The White Mug On The Left Plate And Put The Yellow And White Mug On The Right Plate': 'Put The White Mug On The Left Plate And Put The Yellow And White Mug On The Right Plate (Original)',
    'Put The White Mug On The Plate And Put The Chocolate Pudding To The Right Of The Plate': 'Put The White Mug On The Plate And Put The Chocolate Pudding To The Right Of The Plate (Original)',
    'Complete Kitchen Organization V2': 'Complete Kitchen Organization (Reorder 1)',
    'Organize Table V2': 'Organize Table (Reorder 1)',
    'Cooking Preparation Setup V2': 'Cooking Preparation Setup (Reorder 1)',
    'Complete Kitchen Organization V3': 'Complete Kitchen Organization (Reorder 2)',
    'Organize Table V3': 'Organize Table (Reorder 2)',
    'Cooking Preparation Setup V3': 'Cooking Preparation Setup (Reorder 2)',
    'Turn On The Stove And Put The Moka Pot On It V1': 'Turn On The Stove And Put The Moka Pot On It (Reorder 1)',
    'Put Both The Alphabet Soup And The Cream Cheese Box In The Basket V1': 'Put Both The Alphabet Soup And The Cream Cheese Box In The Basket (Reorder 1)',
    'Put Both The Alphabet Soup And The Tomato Sauce In The Basket V1': 'Put Both The Alphabet Soup And The Tomato Sauce In The Basket (Reorder 1)',
    'Put Both The Cream Cheese Box And The Butter In The Basket V1': 'Put Both The Cream Cheese Box And The Butter In The Basket (Reorder 1)',
    'Put The White Mug On The Left Plate And Put The Yellow And White Mug On The Right Plate V1': 'Put The White Mug On The Left Plate And Put The Yellow And White Mug On The Right Plate (Reorder 1)',
    'Put The White Mug On The Plate And Put The Chocolate Pudding To The Right Of The Plate V1': 'Put The White Mug On The Plate And Put The Chocolate Pudding To The Right Of The Plate (Reorder 1)',
}


def resolve_task_name(name):
    """Map a legacy task name onto its canonical form; pass anything else through."""
    return LEGACY_TASK_NAMES.get(name, name)


def canonical_task_names():
    """The 21 canonical task names, in config order."""
    import json
    from pathlib import Path
    cfg = json.loads(
        (Path(__file__).resolve().parents[2] / "configs" / "tasks_and_skills.json").read_text()
    )
    return [t["name"] for t in cfg["long_horizon_tasks"]]
