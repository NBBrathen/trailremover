"""
Image upload endpoints.
Handles FITS file uploads and initiates processing jobs.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from pathlib import Path
import logging
import shutil

from app.config import settings
from app.models.job import CreateJobResponse
from app.services.job_manager import job_manager, JobStatus
from app.services.image_processor import ImageProcessingService

logger = logging.getLogger(__name__)
router = APIRouter()

# Create image processor instance
image_processor = ImageProcessingService()


async def process_image_job(job_id: str, file_path: Path):
    """
    Background task to process an image and detect trails.

    This runs asynchronously after the upload completes.
    """
    try:
        # Update status to DETECTING
        job_manager.update_job_status(job_id, JobStatus.DETECTING)

        # Load the FITS image
        image_data = image_processor.load_fits_image(file_path)

        if image_data is None:
            job_manager.update_job_status(
                job_id,
                JobStatus.FAILED,
                error_message="Failed to load FITS image"
            )
            return

        # Run detection
        detection_result = image_processor.run_detection(image_data)

        if not detection_result['success']:
            job_manager.update_job_status(
                job_id,
                JobStatus.FAILED,
                error_message=detection_result.get('error', 'Detection failed')
            )
            return

        # Store detected trails
        job_manager.set_detected_trails(job_id, detection_result['trails'])

        # Update status to AWAITING_REVIEW
        job_manager.update_job_status(
            job_id,
            JobStatus.AWAITING_REVIEW
        )

        logger.info(
            f"Job {job_id} completed detection: "
            f"found {detection_result['trail_count']} trail(s)"
        )

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
        job_manager.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=str(e)
        )


@router.post("/upload", response_model=CreateJobResponse)
async def upload_image(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...)
):
    """
    Upload a FITS image for trail detection.

    This endpoint:
    1. Validates the file
    2. Saves it to disk
    3. Creates a processing job
    4. Starts background detection
    5. Returns the job ID for tracking

    The client can poll the job status endpoint to monitor progress.
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )

    # Validate file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to start

    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE / 1024 / 1024:.0f} MB"
        )

    try:
        # Create a job
        job_id = job_manager.create_job(
            filename=file.filename,
            file_path=settings.UPLOAD_DIR / file.filename
        )

        # Save uploaded file
        file_path = settings.UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File uploaded: {file.filename} -> Job {job_id}")

        # Start background processing
        background_tasks.add_task(process_image_job, job_id, file_path)

        return CreateJobResponse(
            job_id=job_id,
            message=f"File uploaded successfully. Job {job_id} created."
        )

    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}"
        )
    finally:
        file.file.close()