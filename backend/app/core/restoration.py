"""
Pixel restoration algorithms for removing satellite trails from FITS images.

This module provides inpainting algorithms that fill in the pixels where
trails have been detected, using information from surrounding pixels.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
import cv2

logger = logging.getLogger(__name__)


def restore_pixels_local_background(
        image_data: np.ndarray,
        trails: List[Dict[str, Any]],
        sample_radius: int = 30,
        expand_mask: int = 5
) -> np.ndarray:
    """
    Restore trail pixels using local background estimation with noise matching.

    1. Expand trail mask slightly to catch edges
    2. For each trail pixel, sample nearby non-trail pixels
    3. Estimate local background (median) and noise (std)
    4. Replace trail pixel with background + matched Gaussian noise
    """
    logger.info(f"Restoring {len(trails)} trail(s) using local background method")

    if len(trails) == 0:
        return image_data.copy()

    # Create combined mask
    combined_mask = np.zeros(image_data.shape, dtype=np.uint8)
    for trail in trails:
        mask = trail.get('mask')
        if mask is not None:
            binary_mask = (mask > 0.1).astype(np.uint8) * 255
            combined_mask = np.maximum(combined_mask, binary_mask)

    # Expand mask to catch trail edges
    if expand_mask > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (expand_mask * 2 + 1, expand_mask * 2 + 1)
        )
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    mask_bool = combined_mask > 0
    trail_pixel_count = np.count_nonzero(mask_bool)

    if trail_pixel_count == 0:
        return image_data.copy()

    logger.info(f"Restoring {trail_pixel_count:,} pixels")

    # Create output image
    restored = image_data.copy()

    # Get coordinates of trail pixels
    trail_coords = np.where(mask_bool)

    # Create a dilated mask for sampling (area around trail but not in trail)
    sample_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (sample_radius * 2 + 1, sample_radius * 2 + 1)
    )
    sample_region = cv2.dilate(combined_mask, sample_kernel, iterations=1)
    sample_mask = (sample_region > 0) & ~mask_bool  # Around trail but not in it

    # Calculate global background statistics as fallback
    background_pixels = image_data[~mask_bool]
    global_median = np.median(background_pixels)
    global_std = np.std(background_pixels)

    # For efficiency, process in blocks
    height, width = image_data.shape
    block_size = 64

    for by in range(0, height, block_size):
        for bx in range(0, width, block_size):
            # Get block bounds
            by_end = min(by + block_size, height)
            bx_end = min(bx + block_size, width)

            # Check if any trail pixels in this block
            block_mask = mask_bool[by:by_end, bx:bx_end]
            if not np.any(block_mask):
                continue

            # Sample region around this block
            sample_by = max(0, by - sample_radius)
            sample_by_end = min(height, by_end + sample_radius)
            sample_bx = max(0, bx - sample_radius)
            sample_bx_end = min(width, bx_end + sample_radius)

            # Get nearby non-trail pixels for statistics
            region_mask = mask_bool[sample_by:sample_by_end, sample_bx:sample_bx_end]
            region_data = image_data[sample_by:sample_by_end, sample_bx:sample_bx_end]

            nearby_pixels = region_data[~region_mask]

            if len(nearby_pixels) > 10:
                local_median = np.median(nearby_pixels)
                local_std = np.std(nearby_pixels)
            else:
                local_median = global_median
                local_std = global_std

            # Fill trail pixels in this block
            block_trail_coords = np.where(block_mask)
            num_pixels = len(block_trail_coords[0])

            # Generate random noise matching local statistics
            noise = np.random.normal(0, local_std, num_pixels)
            fill_values = local_median + noise

            # Apply to restored image
            for i in range(num_pixels):
                y = by + block_trail_coords[0][i]
                x = bx + block_trail_coords[1][i]
                restored[y, x] = fill_values[i]

    logger.info("Local background restoration complete")
    return restored


def restore_pixels_local_background_iterative(
        image_data: np.ndarray,
        trails: List[Dict[str, Any]],
        sample_radius: int = 40,
        base_expand: int = 8,
        passes: int = 3
) -> np.ndarray:
    """
    Multiple passes with increasing mask expansion.
    Catches the faint trail halo that single pass misses.
    """
    current = image_data.copy()

    for i in range(passes):
        # Each pass expands mask a bit more
        expand = base_expand + (i * 4)  # 8, 12, 16
        logger.info(f"Restoration pass {i + 1}/{passes}, expand={expand}px")

        current = restore_pixels_local_background(
            current,
            trails,
            sample_radius=sample_radius,
            expand_mask=expand
        )

    return current

