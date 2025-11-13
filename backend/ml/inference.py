"""
This is the inference system for the satellite trail detection.

It will load the trained U-Net model and provide an interface for detecting
trails in astronomical images
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import logging
from scipy import ndimage

from .model import UNet

logger = logging.getLogger(__name__)

def extract_trail_contours(
        mask: np.ndarray,
        min_pixels: int = 50,
        confidence_threshold: float = 0.5
) -> List[np.ndarray]:

    """
    This extracts trail contours from a prediction mask. It converts the continuous probability mask
    into discrete trail objects by finding connected components above a threshold.

    It takes in a binary mask (H, W) with values 0-1.
    It takes in min_pixels, which is the minimum number of pixels for a valid trail.
    It takes in a confidence_threshold, which is the threshold for binarizing predictions.

    It will return a list of contours, each as an array of (x,y) coordinates
    """

    # Here we binarize the mask
    binary_mask = (mask > confidence_threshold).astype(np.uint8)

    # Now we find connected components
    labeled_mask, num_features = ndimage.label(binary_mask)

    contours = []
    for i in range(1, num_features+1):
        # Extract this component
        component = (labeled_mask == i).astype(np.uint8)

        # check size
        if component.sum() < min_pixels:
            continue

        # Find contour using OpenCV
        component_contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if component_contours:
            # Take the largest contour for this component
            largest = max(component_contours, key=cv2.contourArea)
            contours.append(largest.squeeze())

    return contours

def fit_line_to_contour(contour: np.ndarray) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """
    Function to fit a line to a trail contour

    Trails are linear, so we fit a line and find the endpoints.
    """

    if len(contour) < 2:
        if len(contour) == 1:
            pt = tuple(contour[0])
            return (pt, pt)
        return ((0,0), (0,0))

    # Fit lines using least squares
    # Line: y = mx + b
    x = contour[:, 0]
    y = contour[:, 1]

    # Here we handle vertical lines
    if x.std() < 1:
        x_avg = int(x.mean())
        y_min, y_max = int(y.min()), int(y.max())
        return ((x_avg, y_min), (x_avg, y_max))

    # Fit line
    coeffs = np.polyfit(x, y, 1)
    m, b = coeffs

    # Find endpoints along the contour
    x_min, x_max = int(x.min()), int(x.max())
    y_start = int(m * x_min + b)
    y_end = int(m * x_max + b)

    return ((x_min, y_start), (x_max, y_end))


def estimate_trail_width(mask: np.ndarray, contour: np.ndarray) -> float:
    if len(contour) < 2:
        return 1.0

    # Unindent everything below - it should be at function level
    num_samples = min(len(contour), 10)
    indices = np.linspace(0, len(contour) - 1, num_samples).astype(int)
    sample_points = contour[indices]

    widths = []
    for point in sample_points:
        x, y = point

        # Sample a small region around this point
        y_min = max(0, y - 10)
        y_max = min(mask.shape[0], y + 10)
        x_min = max(0, x - 10)
        x_max = min(mask.shape[1], x + 10)

        region = mask[y_min:y_max, x_min:x_max]

        # Count pixels above threshold
        width = (region > 0.5).sum()
        widths.append(width)

    # Return median width
    return float(np.median(widths)) if widths else 1.0


class TrailDetector:
    """
    Trail detection system.
    """

    def __init__(
            self,
            model_path: Path,
            device: str = 'cuda',
            confidence_threshold: float = 0.5,
            min_trail_pixels: int = 50
    ):
        """
        Initialize the trail detector.

        Args:
            model_path: Path to trained model checkpoint (.pth file)
            device: 'cuda' or 'cpu'
            confidence_threshold: Minimum confidence for trail detection
            min_trail_pixels: Minimum pixels for a valid trail
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.min_trail_pixels = min_trail_pixels

        # Load model
        self.model = UNet(in_channels=3, out_channels=1).to(device)
        self._load_checkpoint(model_path)
        self.model.eval()  # Set to evaluation mode

        logger.info(f"TrailDetector initialized on {device}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        logger.info(f"Min trail pixels: {min_trail_pixels}")

    def _load_checkpoint(self, checkpoint_path: Path):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('best_val_loss', 'unknown')
        logger.info(f"Loaded model from epoch {epoch}, val_loss: {val_loss}")

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: RGB image (H, W, 3) with values 0-1

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Ensure image is float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Ensure RGB and normalized
        if image.max() > 1.0:
            image = image / 255.0

        # Convert to tensor and add batch dimension
        # (H, W, 3) -> (3, H, W) -> (1, 3, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        return tensor.to(self.device)

    def detect(
            self,
            image: np.ndarray,
            return_mask: bool = False
    ) -> Dict[str, Any]:
        """
        Detect trails in an image.

        Returns:
            Dictionary containing:
                - num_trails: Number of detected trails
                - trails: List of trail dictionaries, each containing:
                    - id: Trail ID (0, 1, 2, ...)
                    - confidence: Average confidence score
                    - start_point: (x, y) start coordinates
                    - end_point: (x, y) end coordinates
                    - width: Estimated width in pixels
                    - length: Length in pixels
                    - contour: Array of contour points
                - mask: Probability mask (only if return_mask=True)
        """
        # Preprocess
        input_tensor = self.preprocess_image(image)

        # Run inference
        with torch.no_grad():
            prediction = self.model(input_tensor)

        # Convert to numpy
        mask = prediction.cpu().squeeze().numpy()

        # Extract trails
        contours = extract_trail_contours(
            mask,
            min_pixels=self.min_trail_pixels,
            confidence_threshold=self.confidence_threshold
        )

        # Process each trail
        trails = []
        for idx, contour in enumerate(contours):
            # Fit line to get endpoints
            start_point, end_point = fit_line_to_contour(contour)

            # Calculate length
            length = np.linalg.norm(
                np.array(end_point) - np.array(start_point)
            )

            # Estimate width
            width = estimate_trail_width(mask, contour)

            # Calculate average confidence
            # Sample confidence values along the contour
            confidence_values = []
            for point in contour[::max(1, len(contour) // 10)]:
                x, y = point
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    confidence_values.append(mask[y, x])

            avg_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0

            trail_info = {
                'id': idx,
                'confidence': avg_confidence,
                'start_point': start_point,
                'end_point': end_point,
                'width': width,
                'length': float(length),
                'contour': contour.tolist()
            }
            trails.append(trail_info)

        result = {
            'num_trails': len(trails),
            'trails': trails
        }

        if return_mask:
            result['mask'] = mask

        return result

    def detect_batch(
            self,
            images: List[np.ndarray],
            return_masks: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect trails in a batch of images.
        """
        results = []
        for image in images:
            result = self.detect(image, return_mask=return_masks)
            results.append(result)
        return results


def create_detector(
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        min_trail_pixels: int = 50
) -> TrailDetector:
    """
    Factory function to create a TrailDetector with sensible defaults.
    """
    # Default model path
    if model_path is None:
        model_path = Path(__file__).parent.parent / 'data' / 'models' / 'best_model.pth'

    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Validate model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    return TrailDetector(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold,
        min_trail_pixels=min_trail_pixels
    )