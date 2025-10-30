""" Pydantic models for trail data structures """

from pydantic import BaseModel, Field
from typing import List, Tuple

class Trail(BaseModel):
    """ This is the model representing a detected satellite trail """
    trail_id: str = Field(..., description="Unique trail identifier")
    start_point: Tuple[int, int] = Field(..., description="Starting coordinates (x, y)")
    end_point: Tuple[int, int] = Field(..., description="Ending coordinates (x, y)")
    width: float = Field(..., description="Trail width in pixels")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence (0-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "trail_id": "trail_001",
                "start_point": [100, 200],
                "end_point": [500, 600],
                "width": 3.5,
                "confidence": 0.92
            }
        }

class DetectionResponse(BaseModel):
    """ Response containing detected trails """
    job_id: str
    trail_count: int
    trails: List[Trail]

class CorrectionRequest(BaseModel):
    """ Request model for submitting correction with reviewed trails"""
    trails_to_correct: List[str] = Field(..., description="List of trail IDs to correct (excluding false positives)")

    class Config:
        json_schema_extra = {
            "example": {
                "trails_to_correct": ["trail_001", "trail_003"]
            }
        }