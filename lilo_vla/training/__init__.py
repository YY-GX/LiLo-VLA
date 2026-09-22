"""Training-side helpers for LiLo-VLA.

The training entry point itself is ``scripts/train/finetune.py`` (launched by
``scripts/train/train_lilo_vla.sh``); this package holds only the optional
pieces it imports.

- ``balanced_dataset_wrapper``  optional original/augmented sampling wrapper,
  reached only via ``--use_balanced_sampling True``. The released checkpoint
  did NOT use it.
"""
