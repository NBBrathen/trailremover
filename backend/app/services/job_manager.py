"""
This is the job manager for tracking image processing jobs.
"""

from enum import Enum
from typing import Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import uuid
import logging

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    DETECTING = "DETECTING"
    AWAITING_REVIEW = "AWAITING_REVIEW"
    CORRECTING = "CORRECTING"
    DONE = "DONE"
    FAILED = "FAILED"

class JobManager:
    """ This is the job tracker """

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, filename: str, file_path: Path) -> str:
        """ This creates a new processing job """
        job_id = str(uuid.uuid4())

        self.jobs[job_id] = {
            'job_id': job_id,
            'status': JobStatus.QUEUED,
            'filename': filename,
            'file_path': str(file_path),
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'detected_trails': None,
            'corrected_image_path': None,
            'error_message': None
        }

        logger.info(f"Created job {job_id} for file {filename}")
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """ This retrieves job information """
        return self.jobs.get(job_id)

    def update_job_status(
            self,
            job_id: str,
            status: JobStatus,
            **kwargs
    ) -> bool:
        """ This updates job status and additional fields """
        if job_id not in self.jobs:
            return False

        self.jobs[job_id]['status'] = status
        self.jobs[job_id]['updated_at'] = datetime.utcnow().isoformat()

        # Update any additional fields
        for key, value in kwargs.items():
            self.jobs[job_id][key] = value

        logger.info(f"Job {job_id} updated to status {status}")
        return True

    def set_detected_trails(self, job_id: str, trails: list) -> bool:
        """ This stores detected trails for a job """
        if job_id not in self.jobs:
            return False

        self.jobs[job_id]['detected_trails'] = trails
        return True

job_manager = JobManager()
