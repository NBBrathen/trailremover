"""
Image processing service for FITS files.
Handles loading, saving, and restoration operations.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging
from astropy.io import fits

from app.config import settings
from app.core.restoration import restore_pixels_local_background_iterative


logger = logging.getLogger(__name__)


class ImageProcessor:
    """Service for processing FITS astronomical images."""

    def load_fits_image(self, file_path: Path) -> np.ndarray:
        """
        Load FITS image data.

        Args:
            file_path: Path to FITS file

        Returns:
            Image data as numpy array
        """
        try:
            with fits.open(file_path) as hdul:
                image_data = hdul[0].data.astype(np.float32)
                logger.info(f"Loaded FITS image: {image_data.shape}")
                return image_data
        except Exception as e:
            logger.error(f"Failed to load FITS file: {e}")
            raise

    def save_fits_image(
            self,
            image_data: np.ndarray,
            output_path: Path,
            original_path: Path = None
    ) -> None:
        """
        Save image data as FITS file.

        Args:
            image_data: Image data to save
            output_path: Output file path
            original_path: Optional original file to copy header from
        """
        try:
            # Get header from original if provided
            header = None
            if original_path and original_path.exists():
                with fits.open(original_path) as hdul:
                    header = hdul[0].header.copy()

            # Create HDU
            hdu = fits.PrimaryHDU(image_data, header=header)

            # Add processing history
            hdu.header['HISTORY'] = 'Processed by TrailRemover'

            # Write file
            hdu.writeto(output_path, overwrite=True)
            logger.info(f"Saved FITS image to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save FITS file: {e}")
            raise

    def apply_restoration(self, image_data, trails):
        """Apply pixel restoration to remove trails from image."""
        logger.info(f"Applying restoration to {len(trails)} trail(s)")

        if len(trails) == 0:
            return image_data.copy()

        try:
            restored = restore_pixels_local_background_iterative(
                image_data,
                trails,
                sample_radius=30,
                base_expand=8,
                passes=3
            )

            logger.info("Restoration complete")
            return restored

        except Exception as e:
            logger.error(f"Restoration failed: {e}", exc_info=True)
            return image_data.copy()