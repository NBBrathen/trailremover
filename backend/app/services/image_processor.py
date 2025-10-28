# This is where we handle all FITS image processing operations

from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from astropy.io import fits
import logging

from app.core.detection import detect_trails
from app.core.restoration import restore_pixels

# The logger is used for debugging and error tracking throughout the process
# for example, logger.error(some error) logs when things go wrong
# and logger.info(some progress) logs when things go right
logger = logging.getLogger(__name__)

class ImageProcessingService:
    """
    This is the service class for processing the FITS images.
    It handles loading, trail detection, and pixel restoration
    """

    def __init__(self):
        self.supported_extensions = ['.fits', '.fit', '.fts']

    def load_fits_image(selfself, file_path: Path) -> Optional[np.ndarray]:
        """
        Here we load a FITS image file and return the image data as a numpy array
        """

        try:
            with fits.open(file_path) as hdul:
                # FITS files have multiple HDUs (header data units)
                # The main image data is in the first HDU

                image_data = hdul[0].data

                if image_data is None:
                    logger.error(f"No image data found in {file_path}")
                    return None

                # We have to have the data as float for processing
                return image_data.astype(np.float32)

        except Exception as e:
            logger.error(f"Error loading FITS file {file_path}: {str(e)}")
            return None

    def save_fits_image(
            self,
            image_data: np.ndarray,
            output_path: Path,
            original_header: Optional[fits.Header] = None
    ) -> bool:
        """
        This saves the image data as a FITS file.
        It takes in a numpy array of image data, the output path,
        and the optional FITS header to preserve metadata.

        It will return True if it saved successfully and False otherwise.
        """
        try:
            # Here we create a new HDU with the image data
            hdu = fits.PrimaryHDU(image_data)

            if original_header is not None:
                hdu.header = original_header
                hdu.header['HISTORY'] = "Processed by TrailRemover"

            hdul = fits.HDUList([hdu])
            hdul.writeto(output_path, overwrite=True)

            logger.info(f"Successfully saved processed image to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving FITS file {output_path}: {str(e)}")
            return False

    def run_detection(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Here we run the trail detection algorithm on the image data
        We will take in the NumPy array of image data and return a dictionary
        with detection results with trail information
        """
        try:
            trails = detect_trails(image_data)

            return {
                'success': True,
                'trails': trails,
                'trail_count': len(trails)
            }

        except Exception as e:
            logger.error(f"Error during trail detection: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'trails': [],
                'trail_count': 0
            }

    def apply_restoration(
            self,
            image_data: np.ndarray,
            trails_to_correct: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """
        Here we apply the pixel restoration to remove trails.
        It takes the original image data and list of trail dictionaries to remove
        and then returns the corrected image data, or None if it fails
        """
        try:
            # We need to create a copy and not modify the original
            corrected_image = image_data.copy()

            corrected_image = restore_pixels(corrected_image, trails_to_correct)

            logger.info(f"Successfully restored {len(trails_to_correct)} trails")
            return corrected_image

        except Exception as e:
            logger.error(f"Error during pixel restoration: {str(e)}")
            return None

