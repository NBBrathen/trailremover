"""
FastAPI application entry point for Trail Remover backend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from app.config import settings
from app.api.v1.routes import api_router
from app.core.detection import initialize_detector

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    This runs once when the application starts and cleans up on shutdown.
    """
    # Startup
    logger.info("Starting Trail Remover API...")

    # Create necessary directories
    settings.create_directories()
    logger.info("Data directories verified")

    # Initialize the ML detector
    try:
        initialize_detector(
            model_path=settings.MODEL_PATH,
            device=settings.DEVICE,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD,
            min_trail_pixels=settings.MIN_TRAIL_PIXELS
        )
        logger.info("ML detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        logger.warning("API will start but trail detection will not work")

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down Trail Remover API...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API for detecting and removing satellite trails from astronomical images",
    lifespan=lifespan
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "message": "Trail Remover API",
        "version": settings.APP_VERSION,
        "status": "online"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )