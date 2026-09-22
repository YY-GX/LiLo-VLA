#!/usr/bin/env python3
"""
Random erasing utilities for data augmentation with segmentation masks.

Provides functions to apply random rectangular erasing to non-target regions
of segmentation masks for background augmentation.
"""

import numpy as np
from typing import Optional


def apply_random_erasing_augmentation(
    mask: np.ndarray,
    erasing_area_ratio: int,
    num_rectangles: int,
    random_state: Optional[np.random.RandomState] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply random rectangular erasing to background region of segmentation mask.

    Generates random rectangles that erase (zero out) parts of the background,
    leaving target object untouched. Handles rectangle overlaps correctly by
    counting union area only once.

    Args:
        mask: Grayscale segmentation mask (H, W), uint8
              - 255 = target object (never erased)
              - 0-254 = background (eligible for erasing)
        erasing_area_ratio: Target percentage of background to erase (20-80)
                           Example: 30 = erase 30% of background pixels
        num_rectangles: Number of random rectangles to generate (1-5)
        random_state: numpy RandomState for reproducibility

    Returns:
        Tuple of (erased_mask, erased_pixels):
        - erased_mask: Segmentation mask with rectangles zeroed out in background
        - erased_pixels: Boolean mask indicating which pixels were erased (for RGB masking)
    """
    if random_state is None:
        random_state = np.random.RandomState()

    # Create working copy
    erased_mask = mask.copy()
    h, w = mask.shape

    # Identify background region (pixels < 255)
    background_mask = (mask < 255)
    background_area = background_mask.sum()

    if background_area == 0:
        return erased_mask  # No background to erase

    # Calculate target area to erase
    target_erase_area = int(background_area * (erasing_area_ratio / 100.0))

    # Track erased pixels (union of all rectangles)
    erased_pixels = np.zeros_like(mask, dtype=bool)
    current_erased_area = 0

    # Generate rectangles iteratively until target area reached
    max_attempts = num_rectangles * 10  # Prevent infinite loop
    rectangles_placed = 0
    attempts = 0

    while rectangles_placed < num_rectangles and attempts < max_attempts:
        attempts += 1

        # Random rectangle size (10-40% of image dimension)
        rect_h = random_state.randint(int(h * 0.1), int(h * 0.4))
        rect_w = random_state.randint(int(w * 0.1), int(w * 0.4))

        # Random position
        y1 = random_state.randint(0, h - rect_h)
        x1 = random_state.randint(0, w - rect_w)
        y2 = y1 + rect_h
        x2 = x1 + rect_w

        # Check if rectangle overlaps with target object (mask == 255)
        rect_region = mask[y1:y2, x1:x2]
        has_target = (rect_region == 255).any()

        if has_target:
            continue  # Skip rectangles that touch target object

        # Check if rectangle is in background region
        rect_background = background_mask[y1:y2, x1:x2]
        background_pixels_in_rect = rect_background.sum()

        if background_pixels_in_rect == 0:
            continue  # Rectangle not in background

        # Mark this rectangle as erased
        temp_erased = erased_pixels.copy()
        temp_erased[y1:y2, x1:x2] = True

        # Calculate new union area (overlaps counted once)
        new_erased_area = (temp_erased & background_mask).sum()

        # Accept rectangle if it contributes to erasing
        if new_erased_area > current_erased_area:
            erased_pixels = temp_erased
            current_erased_area = new_erased_area
            rectangles_placed += 1

            # Stop if target area reached
            if current_erased_area >= target_erase_area:
                break

    # Apply erasing: zero out erased pixels in background
    erased_mask[erased_pixels] = 0

    return erased_mask, erased_pixels


def generate_augmented_wrist_images(
    wrist_images: np.ndarray,
    wrist_segmentation: np.ndarray,
    erasing_area_ratio: int,
    num_rectangles: int,
    demo_id: int,
    variant_idx: int
) -> np.ndarray:
    """
    Generate augmented wrist images with random background erasing.

    Args:
        wrist_images: Wrist camera frames (T, H, W, 3), uint8
        wrist_segmentation: Segmentation masks (T, H, W), uint8
        erasing_area_ratio: Percentage of background to erase (20-80)
        num_rectangles: Number of erasing rectangles (1-5)
        demo_id: Demo index for seeding
        variant_idx: Variant index (0=original, 1-2=augmented) for seeding

    Returns:
        Augmented wrist images (T, H, W, 3), uint8
    """
    masked_frames = []

    for frame_idx in range(wrist_images.shape[0]):
        # Get frame and segmentation mask
        frame = wrist_images[frame_idx]
        seg_mask = wrist_segmentation[frame_idx]

        # Apply random erasing to segmentation mask
        # Seed: demo_id * 1000 + variant_idx * 100 + frame_idx
        frame_seed = demo_id * 1000 + variant_idx * 100 + frame_idx
        frame_rng = np.random.RandomState(seed=frame_seed)

        erased_mask, erased_pixels = apply_random_erasing_augmentation(
            seg_mask,
            erasing_area_ratio=erasing_area_ratio,
            num_rectangles=num_rectangles,
            random_state=frame_rng
        )

        # Apply erasing to frame: black out only the erased rectangles
        # Keep background and important objects visible (consistent with apply_distractor_masking)
        masked_frame = frame.copy()
        masked_frame[erased_pixels] = 0
        masked_frames.append(masked_frame)

    return np.array(masked_frames)
