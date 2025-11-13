"""
Trail detection algorithms for identifying satellite trails in FITS images.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import uuid

from ml.inference import TrailDetector, create_detector

logger = logging.getLogger(__name__)

# Global detector instance (loaded once at startup)
_detector: Optional[TrailDetector] = None


def initialize_detector(
    model_path: Optional[Path] = None,
    device: Optional[str] = None,
    confidence_threshold: float = 0.5,
    min_trail_pixels: int = 50
) -> None:
    """
    Initialize the global detector instance.
    Call this once at application startup.

    Args:
        model_path: Path to trained model (.pth file)
        device: 'cuda' or 'cpu' (auto-detected if None)
        confidence_threshold: Minimum confidence for detection
        min_trail_pixels: Minimum pixels for a valid trail
    """
    global _detector

    try:
        _detector = create_detector(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            min_trail_pixels=min_trail_pixels
        )
        logger.info("Trail detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        raise


def get_detector() -> TrailDetector:
    """
    Get the global detector instance.
    Raises an error if not initialized.
    """
    if _detector is None:
        raise RuntimeError(
            "Detector not initialized. Call initialize_detector() first."
        )
    return _detector


def preprocess_fits_data(image_data: np.ndarray) -> np.ndarray:
    """
    Convert FITS image data to RGB format expected by the model.

    Applies 2-98 percentile histogram stretch to match training data.
    """
    # Ensure 2D
    if len(image_data.shape) != 2:
        raise ValueError(f"Expected 2D image, got shape {image_data.shape}")

    # Apply 2-98 percentile histogram stretch (matching training data)
    vmin = np.percentile(image_data, 2)
    vmax = np.percentile(image_data, 98)

    # Clip and normalize
    stretched = np.clip(image_data, vmin, vmax)

    if vmax > vmin:
        normalized = (stretched - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(stretched)

    logger.debug(f"Applied 2-98% histogram stretch: [{vmin:.1f}, {vmax:.1f}] -> [0, 1]")

    # Convert to RGB by duplicating channels
    rgb_image = np.stack([normalized, normalized, normalized], axis=-1)

    return rgb_image.astype(np.float32)


def create_trail_mask(
    image_shape: tuple,
    contour: List[List[int]],
    confidence: float
) -> np.ndarray:
    """
    Create a binary mask from trail contour points.

    Args:
        image_shape: Shape of original image (H, W)
        contour: List of [x, y] coordinates
        confidence: Confidence value to fill mask with

    Returns:
        Binary mask (H, W) with trail pixels marked
    """
    import cv2

    mask = np.zeros(image_shape, dtype=np.float32)

    if len(contour) > 0:
        # Convert to format cv2 expects
        contour_array = np.array(contour, dtype=np.int32)
        cv2.fillPoly(mask, [contour_array], confidence)

    return mask


def detect_trails(image_data: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect satellite trails using the trained U-Net model.
    Uses tile-based processing for large images.
    """
    logger.info(f"Running ML trail detection on image of shape {image_data.shape}")

    try:
        detector = get_detector()

        # Check if image is large - if so, use tile-based processing
        TILE_SIZE = 512
        OVERLAP = 64  # Overlap to catch trails at tile boundaries

        if image_data.shape[0] > TILE_SIZE * 2 or image_data.shape[1] > TILE_SIZE * 2:
            logger.info("Large image detected, using tile-based processing")
            return _detect_trails_tiled(image_data, detector, TILE_SIZE, OVERLAP)
        else:
            # Small image - process normally
            return _detect_trails_single(image_data, detector)

    except Exception as e:
        logger.error(f"Error during trail detection: {e}", exc_info=True)
        return []


def _detect_trails_single(image_data: np.ndarray, detector) -> List[Dict[str, Any]]:
    """Process a single image without tiling."""
    rgb_image = preprocess_fits_data(image_data)
    logger.info(f"Preprocessed image to RGB: {rgb_image.shape}")

    detection_result = detector.detect(rgb_image, return_mask=True)
    num_trails = detection_result['num_trails']
    logger.info(f"Model detected {num_trails} trail(s)")

    detected_trails = []
    for trail_info in detection_result['trails']:
        mask = create_trail_mask(
            image_shape=image_data.shape,
            contour=trail_info['contour'],
            confidence=trail_info['confidence']
        )

        trail_dict = {
            'trail_id': f"trail_{uuid.uuid4()}",
            'start_point': list(trail_info['start_point']),
            'end_point': list(trail_info['end_point']),
            'width': trail_info['width'],
            'confidence': trail_info['confidence'],
            'mask': mask
        }
        detected_trails.append(trail_dict)

    return detected_trails


def _merge_duplicate_trails(trails: List[Dict[str, Any]], tolerance: float = 50.0) -> List[Dict[str, Any]]:
    """
    Merge duplicate trail detections that are very close to each other.

    This is necessary because large images are processed in overlapping tiles,
    and the same trail may be detected in multiple tiles.

    Args:
        trails: List of detected trails
        tolerance: Maximum distance (pixels) between endpoints to consider duplicates

    Returns:
        Deduplicated list of trails
    """
    if len(trails) <= 1:
        return trails

    # Sort by confidence (highest first) - keep the best detection of duplicates
    trails_sorted = sorted(trails, key=lambda t: t['confidence'], reverse=True)

    merged = []
    used = set()

    for i, trail1 in enumerate(trails_sorted):
        if i in used:
            continue

        # Check if this trail is a duplicate of any already merged trail
        is_duplicate = False

        for trail2 in merged:
            # Calculate distance between start and end points
            start_dist = np.linalg.norm(
                np.array(trail1['start_point']) - np.array(trail2['start_point'])
            )
            end_dist = np.linalg.norm(
                np.array(trail1['end_point']) - np.array(trail2['end_point'])
            )

            # If both endpoints are close, it's a duplicate
            if start_dist < tolerance and end_dist < tolerance:
                is_duplicate = True
                logger.debug(
                    f"Filtered duplicate trail: "
                    f"start_dist={start_dist:.1f}px, end_dist={end_dist:.1f}px"
                )
                break

        if not is_duplicate:
            merged.append(trail1)
            used.add(i)

    logger.info(f"Merged {len(trails)} detections into {len(merged)} unique trail(s)")
    return merged


def _detect_trails_tiled(
    image_data: np.ndarray,
    detector,
    tile_size: int = 512,
    overlap: int = 64
) -> List[Dict[str, Any]]:
    """
    Process large image in tiles with overlap.
    Combines detections from all tiles and filters false positives.
    """
    import cv2

    height, width = image_data.shape
    stride = tile_size - overlap

    # Create full-size prediction mask
    full_mask = np.zeros(image_data.shape, dtype=np.float32)
    weight_mask = np.zeros(image_data.shape, dtype=np.float32)

    tiles_processed = 0

    # Process tiles with overlap
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile = image_data[y:y + tile_size, x:x + tile_size]
            rgb_tile = preprocess_fits_data(tile)
            result = detector.detect(rgb_tile, return_mask=True)
            full_mask[y:y + tile_size, x:x + tile_size] += result['mask']
            weight_mask[y:y + tile_size, x:x + tile_size] += 1.0
            tiles_processed += 1

    # Handle right edge
    if width % stride != 0:
        x = width - tile_size
        for y in range(0, height - tile_size + 1, stride):
            tile = image_data[y:y + tile_size, x:x + tile_size]
            rgb_tile = preprocess_fits_data(tile)
            result = detector.detect(rgb_tile, return_mask=True)
            full_mask[y:y + tile_size, x:x + tile_size] += result['mask']
            weight_mask[y:y + tile_size, x:x + tile_size] += 1.0
            tiles_processed += 1

    # Handle bottom edge
    if height % stride != 0:
        y = height - tile_size
        for x in range(0, width - tile_size + 1, stride):
            tile = image_data[y:y + tile_size, x:x + tile_size]
            rgb_tile = preprocess_fits_data(tile)
            result = detector.detect(rgb_tile, return_mask=True)
            full_mask[y:y + tile_size, x:x + tile_size] += result['mask']
            weight_mask[y:y + tile_size, x:x + tile_size] += 1.0
            tiles_processed += 1

    # Average overlapping predictions
    weight_mask[weight_mask == 0] = 1  # Avoid division by zero
    full_mask = full_mask / weight_mask

    logger.info(f"Processed {tiles_processed} tiles")
    logger.info(f"Combined mask - max value: {full_mask.max():.3f}")

    # Extract trails from combined mask
    from ml.inference import extract_trail_contours, fit_line_to_contour, estimate_trail_width

    contours = extract_trail_contours(
        full_mask,
        min_pixels=detector.min_trail_pixels,
        confidence_threshold=detector.confidence_threshold
    )

    logger.info(f"Extracted {len(contours)} contour(s) from combined mask")

    # Convert to trail format with length filtering
    MIN_TRAIL_LENGTH = 50  # pixels - filters out stars and hot pixels

    detected_trails = []
    filtered_count = 0

    for idx, contour in enumerate(contours):
        start_point, end_point = fit_line_to_contour(contour)

        # Calculate length FIRST
        length = np.linalg.norm(np.array(end_point) - np.array(start_point))

        # Skip if too short (filter false positives like stars)
        if length < MIN_TRAIL_LENGTH:
            logger.debug(f"Filtered short detection: {length:.1f}px < {MIN_TRAIL_LENGTH}px")
            filtered_count += 1
            continue

        width = estimate_trail_width(full_mask, contour)

        # Calculate average confidence
        confidence_values = []
        for point in contour[::max(1, len(contour) // 10)]:
            x, y = point
            if 0 <= y < full_mask.shape[0] and 0 <= x < full_mask.shape[1]:
                confidence_values.append(full_mask[y, x])

        avg_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0

        # Create mask for this trail
        mask = create_trail_mask(
            image_shape=image_data.shape,
            contour=contour.tolist() if isinstance(contour, np.ndarray) else contour,
            confidence=avg_confidence
        )

        trail_dict = {
            'trail_id': f"trail_{uuid.uuid4()}",
            'start_point': list(start_point),
            'end_point': list(end_point),
            'width': width,
            'confidence': avg_confidence,
            'mask': mask
        }

        detected_trails.append(trail_dict)

        logger.debug(
            f"Trail {trail_dict['trail_id']}: "
            f"confidence={avg_confidence:.3f}, "
            f"length={length:.1f}px"
        )

    logger.info(f"Filtered {filtered_count} short detections (< {MIN_TRAIL_LENGTH}px)")
    logger.info(f"Found {len(detected_trails)} trail(s) before deduplication")

    # Merge duplicate detections
    detected_trails = _merge_duplicate_trails(detected_trails, tolerance=50.0)

    logger.info(f"Returning {len(detected_trails)} unique trail(s)")

    return detected_trails




















