"""
API client for communicating with Trail Remover backend.
Handles all HTTP requests to the FastAPI server.
"""

import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TrailRemoverAPIClient:
    """Client for interacting with Trail Remover backend API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"

    def health_check(self) -> bool:
        """Check if API is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def upload_image(self, file_path: Path) -> Optional[str]:
        """
        Upload a FITS image for processing.
        """
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'application/octet-stream')}
                response = requests.post(
                    f"{self.api_base}/images/upload",
                    files=files,
                    timeout=120
                )

            if response.status_code == 200:
                data = response.json()
                job_id = data['job_id']
                logger.info(f"Upload successful: job_id={job_id}")
                return job_id
            else:
                logger.error(f"Upload failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Upload error: {e}")
            return None

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a processing job.
        """
        try:
            response = requests.get(
                f"{self.api_base}/jobs/{job_id}",
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Get job status failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Get job status error: {e}")
            return None

    def get_detections(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detected trails for a job.
        """
        try:
            response = requests.get(
                f"{self.api_base}/jobs/{job_id}/detections",
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                logger.info(f"Got {data['trail_count']} trails for job {job_id}")
                return data
            else:
                logger.error(f"Get detections failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Get detections error: {e}")
            return None

    def submit_corrections(self, job_id: str, trail_ids: List[str]) -> bool:
        """
        Submit corrections after user review.
        """
        try:
            payload = {'trails_to_correct': trail_ids}
            response = requests.post(
                f"{self.api_base}/jobs/{job_id}/correct",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"Submitted corrections for {len(trail_ids)} trails")
                return True
            else:
                logger.error(f"Submit corrections failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Submit corrections error: {e}")
            return False

    def download_result(self, job_id: str, output_path: Path) -> bool:
        """
        Download corrected FITS image.
        """
        try:
            response = requests.get(
                f"{self.api_base}/jobs/{job_id}/download",
                timeout=60
            )

            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded corrected image to {output_path}")
                return True
            else:
                logger.error(f"Download failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Download error: {e}")
            return False


# Convenience function to create client
def create_api_client(base_url: str = "http://localhost:8000") -> TrailRemoverAPIClient:
    """Create an API client instance."""
    return TrailRemoverAPIClient(base_url)