"""Data generation for LiLo-VLA.

Stage 1 of the pipeline: object-centric augmented demonstrations for the 27
atomic skills, plus the utilities that combine, downsample and repair them.

Modules (each is runnable with ``python -m lilo_vla.datagen.<module>``):

- ``generate_object_centric_demos``  the generator that produced the released
  dataset (run in parallel by ``scripts/data_generation/run_parallel_demo_generation.sh``)
- ``combine_hdf5_files``             merge the per-seed ``v1..v10`` folders
- ``downsample_combined_hdf5_files`` sample 50 demos/skill for the RLDS build
- ``fix_place_skill_segmentation``   regenerate wrist masks for place skills
- ``random_erasing_mask_tools``      the TRAIN-time background-erasing kernel,
  applied at RLDS *build* time by ``third_party/rlds_dataset_builder``.
  This is NOT ``lilo_vla.interaction.random_erasing``, which is a different,
  eval-time experiment with different parameters. Do not merge the two.

Nothing is imported eagerly: the generator pulls in MuJoCo, mplib and open3d.
"""
