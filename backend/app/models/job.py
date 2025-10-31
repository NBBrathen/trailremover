""" These are the Pydantic models for job-related data structures """

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    DETECTING = "DETECTING"
    AWAITING_REVIEW = "AWAITING_REVIEW"
    CORRECTING = "CORRECTING"
    DONE = "DONE"
    FAILED = "FAILED"

class JobResponse(BaseModel):
    """ Response model for job information """
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    filename: str = Field(..., description="Original filename")
    created_at: str = Field(..., description="Job creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    error_message: Optional[str] = Field(None, description="Error message if job failed")

    # This config class is for adding examples for the API docs
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "AWAITING_REVIEW",
                "filename": "galaxy_m31.fits",
                "created_at": "2025-10-26T10:30:00Z",
                "updated_at": "2025-10-26T10:31:30Z",
                "error_message": None
            }
        }

class CreateJobResponse(BaseModel):
    """ Response when a new job is created """
    job_id: str
    message: str = "Job created successfully"