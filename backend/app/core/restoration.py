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


def restore_pixels(
    image_data: np.ndarray,
    trails: List[Dict[str, Any]],
    method: str = 'telea',
    inpaint_radius: int = 30
) -> np.ndarray:
    """
    Restore pixels where satellite trails have been detected.

    Uses inpainting algorithms to fill in trail regions based on
    surrounding pixel values.

    Args:
        image_data: Original FITS image data (H, W)
        trails: List of detected trails with 'mask' key containing binary masks
        method: Inpainting method - 'telea', 'ns', or 'biharmonic'
        inpaint_radius: Radius around each pixel to consider for inpainting

    Returns:
        Restored image with trails removed

    Methods:
        - 'telea': Fast Marching Method (Telea 2004) - Fast, good for thin features
        - 'ns': Navier-Stokes based (Bertalmio 2001) - Slower, better for larger regions
        - 'biharmonic': Biharmonic equation - Smooth, works well for astronomy
    """
    logger.info(f"Restoring pixels for {len(trails)} trail(s) using {method} method")

    if len(trails) == 0:
        logger.info("No trails to restore, returning original image")
        return image_data.copy()

    # Create combined mask from all trails
    combined_mask = np.zeros(image_data.shape, dtype=np.uint8)

    for trail in trails:
        mask = trail.get('mask')
        if mask is not None:
            # Convert float mask to binary uint8
            binary_mask = (mask > 0.1).astype(np.uint8) * 255
            combined_mask = np.maximum(combined_mask, binary_mask)

    trail_pixel_count = np.count_nonzero(combined_mask)
    total_pixels = combined_mask.size
    trail_percentage = (trail_pixel_count / total_pixels) * 100

    logger.info(f"Restoring {trail_pixel_count:,} pixels ({trail_percentage:.3f}% of image)")

    if trail_pixel_count == 0:
        logger.warning("Combined mask is empty, returning original image")
        return image_data.copy()

    # Perform restoration
    try:
        if method == 'biharmonic':
            restored = _restore_biharmonic(image_data, combined_mask)
        elif method == 'ns':
            restored = _restore_opencv_ns(image_data, combined_mask, inpaint_radius)
        elif method == 'telea':
            restored = _restore_opencv_telea(image_data, combined_mask, inpaint_radius)
        else:
            logger.warning(f"Unknown method '{method}', falling back to 'telea'")
            restored = _restore_opencv_telea(image_data, combined_mask, inpaint_radius)

        logger.info("Pixel restoration complete")
        return restored

    except Exception as e:
        logger.error(f"Error during restoration: {e}", exc_info=True)
        logger.warning("Returning original image due to restoration failure")
        return image_data.copy()


def restore_pixels_iterative(
    image_data: np.ndarray,
    trails: List[Dict[str, Any]],
    method: str = 'telea',
    inpaint_radius: int = 20,
    expand_mask: int = 5,
    iterations: int = 3
) -> np.ndarray:
    """
    Restore pixels using multiple passes for better results on faint trails.

    Each iteration:
    1. Expands the mask slightly more than previous iteration
    2. Runs inpainting on current image
    3. Uses result as input for next iteration

    This progressive approach works better for wide, faint trails.

    Args:
        image_data: Original FITS image data
        trails: List of detected trails
        method: Inpainting method ('telea', 'ns', 'biharmonic')
        inpaint_radius: Radius for inpainting algorithm
        expand_mask: Initial mask expansion in pixels
        iterations: Number of restoration passes (2-5 recommended)

    Returns:
        Restored image
    """
    logger.info(f"Iterative restoration: {iterations} passes, method={method}, radius={inpaint_radius}")

    if len(trails) == 0:
        return image_data.copy()

    current_image = image_data.copy()

    for i in range(iterations):
        logger.debug(f"Restoration pass {i+1}/{iterations}")

        # Expand masks progressively for this iteration
        trails_expanded = []
        for trail in trails:
            trail_copy = trail.copy()
            mask = trail['mask']

            # Progressive expansion: more on each iteration
            expansion = expand_mask + (i * 2)

            if expansion > 0:
                binary = (mask > 0.1).astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (expansion*2+1, expansion*2+1)
                )
                expanded = cv2.dilate(binary, kernel, iterations=1)
                trail_copy['mask'] = expanded.astype(np.float32) / 255.0

            trails_expanded.append(trail_copy)

        # Restore this iteration
        current_image = restore_pixels(
            current_image,
            trails_expanded,
            method=method,
            inpaint_radius=inpaint_radius
        )

        logger.debug(f"Pass {i+1} complete (expansion={expansion}px)")

    logger.info("Iterative restoration complete")
    return current_image


def _restore_opencv_telea(
    image_data: np.ndarray,
    mask: np.ndarray,
    radius: int = 5
) -> np.ndarray:
    """
    Restore using OpenCV's Telea inpainting algorithm.

    Fast Marching Method - fast and works well for thin trails.
    """
    logger.debug(f"Using Telea inpainting with radius={radius}")

    # Normalize to 8-bit for OpenCV
    img_min, img_max = image_data.min(), image_data.max()
    if img_max > img_min:
        normalized = ((image_data - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image_data, dtype=np.uint8)

    # Perform inpainting
    inpainted = cv2.inpaint(normalized, mask, radius, cv2.INPAINT_TELEA)

    # Convert back to original scale
    if img_max > img_min:
        restored = (inpainted.astype(np.float32) / 255.0) * (img_max - img_min) + img_min
    else:
        restored = image_data.copy()

    return restored.astype(image_data.dtype)


def _restore_opencv_ns(
    image_data: np.ndarray,
    mask: np.ndarray,
    radius: int = 5
) -> np.ndarray:
    """
    Restore using OpenCV's Navier-Stokes inpainting algorithm.

    Fluid dynamics based method - better for larger regions.
    """
    logger.debug(f"Using Navier-Stokes inpainting with radius={radius}")

    # Normalize to 8-bit for OpenCV
    img_min, img_max = image_data.min(), image_data.max()
    if img_max > img_min:
        normalized = ((image_data - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image_data, dtype=np.uint8)

    # Perform inpainting
    inpainted = cv2.inpaint(normalized, mask, radius, cv2.INPAINT_NS)

    # Convert back to original scale
    if img_max > img_min:
        restored = (inpainted.astype(np.float32) / 255.0) * (img_max - img_min) + img_min
    else:
        restored = image_data.copy()

    return restored.astype(image_data.dtype)


def _restore_biharmonic(
    image_data: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Restore using biharmonic inpainting (scikit-image).

    Solves the biharmonic equation to smoothly interpolate missing regions.
    Works very well for astronomical images.
    """
    try:
        from skimage.restoration import inpaint

        logger.debug("Using biharmonic inpainting")

        # Convert mask to boolean
        mask_bool = (mask > 0).astype(bool)

        # Perform inpainting
        restored = inpaint.inpaint_biharmonic(
            image_data,
            mask_bool,
            channel_axis=None
        )

        return restored.astype(image_data.dtype)

    except ImportError:
        logger.warning("scikit-image not available, falling back to Telea method")
        return _restore_opencv_telea(image_data, mask, radius=5)


def restore_pixels_advanced(
    image_data: np.ndarray,
    trails: List[Dict[str, Any]],
    expand_mask: int = 5,
    blend_edges: bool = True
) -> np.ndarray:
    """
    Advanced restoration with mask expansion and edge blending.

    This method:
    1. Expands trail masks to ensure complete removal
    2. Performs inpainting with larger radius
    3. Blends edges for smoother transitions

    Args:
        image_data: Original FITS image data
        trails: List of detected trails
        expand_mask: Pixels to expand mask by (helps remove trail edges)
        blend_edges: Whether to blend the edges of inpainted regions

    Returns:
        Restored image
    """
    logger.info(f"Advanced restoration for {len(trails)} trail(s)")

    if len(trails) == 0:
        return image_data.copy()

    # Create combined mask
    combined_mask = np.zeros(image_data.shape, dtype=np.uint8)

    for trail in trails:
        mask = trail.get('mask')
        if mask is not None:
            binary_mask = (mask > 0.1).astype(np.uint8) * 255
            combined_mask = np.maximum(combined_mask, binary_mask)

    # Expand mask to ensure trail edges are captured
    if expand_mask > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_mask*2+1, expand_mask*2+1))
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        logger.debug(f"Expanded mask by {expand_mask} pixels")

    # Perform inpainting with larger radius for better results
    restored = _restore_opencv_telea(image_data, combined_mask, radius=25)

    # Optional: Blend edges for smoother transition
    if blend_edges:
        restored = _blend_inpainted_edges(image_data, restored, combined_mask)

    logger.info("Advanced restoration complete")
    return restored


def _blend_inpainted_edges(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
    blend_width: int = 3
) -> np.ndarray:
    """
    Blend the edges between inpainted and original regions.

    Creates a smooth transition to avoid sharp boundaries.
    """
    # Create a distance transform from the mask edges
    mask_binary = (mask > 0).astype(np.uint8)

    # Erode to get inner boundary
    kernel = np.ones((blend_width*2+1, blend_width*2+1), np.uint8)
    inner_mask = cv2.erode(mask_binary, kernel, iterations=1)

    # Blend region is the difference
    blend_region = mask_binary - inner_mask

    if blend_region.sum() == 0:
        return inpainted

    # Create blend weights (0 to 1)
    blend_weights = cv2.distanceTransform(blend_region, cv2.DIST_L2, 3)
    if blend_weights.max() > 0:
        blend_weights = blend_weights / blend_weights.max()

    # Blend
    result = original.copy()
    blend_mask = blend_region > 0
    result[blend_mask] = (
        inpainted[blend_mask] * blend_weights[blend_mask] +
        original[blend_mask] * (1 - blend_weights[blend_mask])
    )

    # Replace fully inpainted region
    full_mask = mask_binary & ~blend_region
    result[full_mask > 0] = inpainted[full_mask > 0]

    return result


def evaluate_restoration_quality(
    original: np.ndarray,
    restored: np.ndarray,
    mask: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate the quality of restoration.

    Computes metrics on the restored region to assess quality.

    Returns:
        Dictionary with quality metrics:
        - mean_absolute_difference: MAD between original and restored (outside mask)
        - restored_mean: Mean value in restored region
        - restored_std: Standard deviation in restored region
        - noise_ratio: Ratio of restored std to surrounding std
    """
    mask_bool = mask > 0

    # Get surrounding region (dilate mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    surrounding_mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    surrounding_mask = surrounding_mask & ~mask_bool

    # Metrics in surrounding region (unchanged)
    surrounding_mean = original[surrounding_mask].mean()
    surrounding_std = original[surrounding_mask].std()

    # Metrics in restored region
    restored_mean = restored[mask_bool].mean()
    restored_std = restored[mask_bool].std()

    # How different is restored from original in unchanged regions?
    unchanged_mask = ~mask_bool
    mad = np.abs(original[unchanged_mask] - restored[unchanged_mask]).mean()

    # Noise ratio - restored should have similar noise to surroundings
    noise_ratio = restored_std / surrounding_std if surrounding_std > 0 else 0

    return {
        'mean_absolute_difference': float(mad),
        'restored_mean': float(restored_mean),
        'restored_std': float(restored_std),
        'surrounding_mean': float(surrounding_mean),
        'surrounding_std': float(surrounding_std),
        'noise_ratio': float(noise_ratio)
    }