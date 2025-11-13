"""
API v1 routes configuration.
This file wires together all the endpoint modules.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import images, jobs

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    images.router,
    prefix="/images",
    tags=["images"]
)

api_router.include_router(
    jobs.router,
    prefix="/jobs",
    tags=["jobs"]
)