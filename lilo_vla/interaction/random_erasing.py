"""
Random erasing for segmentation masks.

Applies random rectangular erasing to non-target regions in segmentation masks
as a form of data augmentation during training.
"""

import numpy as np
from typing import Tuple


def apply_random_erasing_to_mask(
    seg_mask: np.ndarray,
    erasing_ratio: float = 0.25,
    num_rectangles: int = 5,
    min_aspect_ratio: float = 0.3,
    max_aspect_ratio: float = 3.0,
    random_state: np.random.RandomState = None
) -> np.ndarray:
    """
    Apply random rectangular erasing to background regions of a segmentation mask.

    This function:
    1. Keeps target region (mask==255) unchanged (always visible)
    2. Makes most of background visible by setting it to 255
    3. Then blacks out random rectangles (sets to 0) in background region

    Final result: erasing_ratio of background is blacked out with rectangles.

    Args:
        seg_mask: Segmentation mask (H, W), where 255 = target region (keep), 0 = background
        erasing_ratio: Ratio of background area to black out (e.g., 0.25 = black out 25%, keep 75% visible)
        num_rectangles: Number of random rectangles to generate
        min_aspect_ratio: Minimum aspect ratio (height/width) for rectangles
        max_aspect_ratio: Maximum aspect ratio (height/width) for rectangles
        random_state: NumPy random state for reproducibility (if None, uses global state)

    Returns:
        Modified mask where:
        - Target region (original 255) stays 255 (visible)
        - Most background becomes 255 (visible)
        - Random rectangles in background are 0 (blacked out)
    """
    if random_state is None:
        random_state = np.random

    # Copy mask to avoid modifying original
    modified_mask = seg_mask.copy()

    # Find background region (where mask == 0)
    background_mask = (seg_mask == 0)
    background_area = np.sum(background_mask)

    # If no background region or erasing_ratio is 0, make all background visible
    if background_area == 0 or erasing_ratio <= 0:
        # Set all background to 255 (visible)
        modified_mask[background_mask] = 255
        return modified_mask

    # First, make ALL background visible (set to 255)
    modified_mask[background_mask] = 255

    # Now black out random rectangles (erasing_ratio of background)
    target_erase_area = background_area * erasing_ratio

    # Get image dimensions
    height, width = seg_mask.shape

    # Track total erased area
    total_erased = 0
    attempts = 0
    max_attempts = num_rectangles * 10  # Avoid infinite loop

    while total_erased < target_erase_area and attempts < max_attempts:
        attempts += 1

        # Randomly sample a rectangle size
        # Area of this rectangle (as ratio of remaining area to erase)
        remaining_ratio = (target_erase_area - total_erased) / background_area
        rect_area_ratio = random_state.uniform(0.05, min(0.3, remaining_ratio * 2))
        rect_area = int(height * width * rect_area_ratio)

        if rect_area < 4:  # Too small, skip
            continue

        # Random aspect ratio
        aspect_ratio = random_state.uniform(min_aspect_ratio, max_aspect_ratio)

        # Calculate rectangle dimensions
        rect_h = int(np.sqrt(rect_area * aspect_ratio))
        rect_w = int(rect_area / rect_h)

        # Ensure dimensions are valid
        rect_h = min(rect_h, height)
        rect_w = min(rect_w, width)

        if rect_h < 2 or rect_w < 2:
            continue

        # Randomly sample position
        y = random_state.randint(0, height - rect_h + 1)
        x = random_state.randint(0, width - rect_w + 1)

        # Extract the patch
        patch = background_mask[y:y+rect_h, x:x+rect_w]

        # Only erase if this patch overlaps with background region
        patch_background_area = np.sum(patch)

        if patch_background_area > 0:
            # Black out this rectangle (set to 0) in the modified mask
            modified_mask[y:y+rect_h, x:x+rect_w] = np.where(
                patch,  # Only black out where background
                0,      # Set to 0 (black out)
                modified_mask[y:y+rect_h, x:x+rect_w]  # Keep original otherwise
            )

            total_erased += patch_background_area

    return modified_mask


def apply_random_erasing_to_image(
    image: np.ndarray,
    seg_mask: np.ndarray,
    erasing_ratio: float = 0.25,
    num_rectangles: int = 5,
    random_state: np.random.RandomState = None
) -> np.ndarray:
    """
    Apply random erasing to image based on segmentation mask.

    This is a convenience function that:
    1. Generates modified mask (target always visible, most background visible, some background blacked out)
    2. Applies it to the image

    Args:
        image: RGB image (H, W, 3), dtype uint8
        seg_mask: Segmentation mask (H, W), where 255 = target region, 0 = background
        erasing_ratio: Ratio of background to black out (e.g., 0.25 = black out 25% of background)
        num_rectangles: Number of random rectangles to generate
        random_state: NumPy random state for reproducibility

    Returns:
        Image with random erasing applied (H, W, 3), dtype uint8
        Final image shows:
        - Target region: fully visible
        - Background: (1-erasing_ratio) visible, erasing_ratio blacked out with rectangles
    """
    # Generate modified mask with random erasing
    # Target stays 255, background becomes mostly 255 except random rectangles set to 0
    modified_mask = apply_random_erasing_to_mask(
        seg_mask,
        erasing_ratio=erasing_ratio,
        num_rectangles=num_rectangles,
        random_state=random_state
    )

    # Apply mask to image: keep pixels where mask==255, black out where mask==0
    # Convert mask to 0-1 range and expand to 3 channels
    mask_3ch = (modified_mask[:, :, None] / 255.0).astype(np.float32)
    masked_image = (image * mask_3ch).astype(np.uint8)

    return masked_image
