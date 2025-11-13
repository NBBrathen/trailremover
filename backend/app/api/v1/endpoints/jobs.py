"""
Job management endpoints.
Handles job status queries, trail review, and correction submission.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pathlib import Path
from typing import List
import logging

from app.config import settings
from app.models.job import JobResponse
from app.models.trail import DetectionResponse, Trail, CorrectionRequest
from app.services.job_manager import job_manager, JobStatus
from app.services.image_processor import ImageProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# Create image processor instance
image_processor = ImageProcessor()


async def apply_corrections_job(job_id: str, trails_to_correct: List[str]):
    """Background task to apply corrections and restore pixels."""
    try:
        # Update status
        job_manager.update_job_status(job_id, JobStatus.CORRECTING)

        # Get job data
        job = job_manager.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        # Load original image
        file_path = Path(job['file_path'])
        image_data = image_processor.load_fits_image(file_path)

        # Get all detected trails
        all_trails = job.get('detected_trails', [])

        # Filter to only trails the user wants to correct
        trails_to_apply = [
            trail for trail in all_trails
            if trail['trail_id'] in trails_to_correct
        ]

        logger.info(
            f"Applying corrections for {len(trails_to_apply)} trails "
            f"(user excluded {len(all_trails) - len(trails_to_apply)} false positives)"
        )

        # Apply restoration
        corrected_image = image_processor.apply_restoration(
            image_data,
            trails_to_apply
        )

        # Save corrected image
        output_filename = f"corrected_{job['filename']}"
        output_path = settings.PROCESSED_DIR / output_filename

        # Just call save - it will raise an exception if it fails
        image_processor.save_fits_image(
            corrected_image,
            output_path,
            original_path=file_path
        )

        # Update job with corrected image path
        job_manager.update_job_status(
            job_id,
            JobStatus.DONE,
            corrected_image_path=str(output_path)
        )

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Error applying corrections for job {job_id}: {e}", exc_info=True)
        job_manager.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=str(e)
        )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a processing job.

    Returns job metadata including status, timestamps, and any error messages.
    """
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    return JobResponse(
        job_id=job['job_id'],
        status=job['status'],
        filename=job['filename'],
        created_at=job['created_at'],
        updated_at=job['updated_at'],
        error_message=job.get('error_message')
    )


@router.get("/{job_id}/detections", response_model=DetectionResponse)
async def get_detections(job_id: str):
    """
    Get detected trails for a job.

    This endpoint is called after detection completes (status = AWAITING_REVIEW)
    to retrieve the list of detected trails for user review.
    """
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    if job['status'] != JobStatus.AWAITING_REVIEW:
        raise HTTPException(
            status_code=400,
            detail=f"Job not ready for review. Current status: {job['status']}"
        )

    detected_trails = job.get('detected_trails', [])

    # Convert to Trail models (exclude mask data for API response)
    trails = [
        Trail(
            trail_id=trail['trail_id'],
            start_point=tuple(trail['start_point']),
            end_point=tuple(trail['end_point']),
            width=trail['width'],
            confidence=trail['confidence']
        )
        for trail in detected_trails
    ]

    return DetectionResponse(
        job_id=job_id,
        trail_count=len(trails),
        trails=trails
    )


@router.post("/{job_id}/correct")
async def submit_corrections(
        job_id: str,
        correction: CorrectionRequest,
        background_tasks: BackgroundTasks
):
    """
    Submit corrections after reviewing detected trails.

    The user reviews the detected trails and submits a list of trail IDs
    to correct (excluding false positives). This endpoint starts the
    pixel restoration process.
    """
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    if job['status'] != JobStatus.AWAITING_REVIEW:
        raise HTTPException(
            status_code=400,
            detail=f"Job not in review state. Current status: {job['status']}"
        )

    # Validate that all trail IDs exist
    detected_trails = job.get('detected_trails', [])
    detected_ids = {trail['trail_id'] for trail in detected_trails}

    for trail_id in correction.trails_to_correct:
        if trail_id not in detected_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid trail ID: {trail_id}"
            )

    # Start background correction
    background_tasks.add_task(
        apply_corrections_job,
        job_id,
        correction.trails_to_correct
    )

    return {
        "message": "Corrections submitted successfully",
        "job_id": job_id,
        "trails_to_correct": len(correction.trails_to_correct)
    }


@router.get("/{job_id}/download")
async def download_result(job_id: str):
    """
    Download the corrected FITS image.

    This endpoint is called after processing completes (status = DONE)
    to retrieve the final corrected image.
    """
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    if job['status'] != JobStatus.DONE:
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Current status: {job['status']}"
        )

    corrected_path = job.get('corrected_image_path')
    if not corrected_path or not Path(corrected_path).exists():
        raise HTTPException(
            status_code=500,
            detail="Corrected image not found"
        )

    from fastapi.responses import FileResponse

    return FileResponse(
        corrected_path,
        media_type="application/octet-stream",
        filename=f"corrected_{job['filename']}"
    )