"""
Application configuration settings.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings using pydantic for validation."""

    # Application
    APP_NAME: str = "Trail Remover"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # API
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    TEMP_DIR: Path = DATA_DIR / "temp"
    MODELS_DIR: Path = DATA_DIR / "models"

    # Model settings
    MODEL_PATH: Optional[Path] = None  # Will default to MODELS_DIR / 'best_model.pth'
    DEVICE: Optional[str] = None  # 'cuda' or 'cpu', auto-detected if None
    CONFIDENCE_THRESHOLD: float = 0.5
    MIN_TRAIL_PIXELS: int = 50

    # File upload settings
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100 MB
    ALLOWED_EXTENSIONS: list = ['.fits', '.fit', '.fts']

    # Logging
    LOG_LEVEL: str = "INFO"

    # Restoration settings
    RESTORATION_METHOD: str = "ns"  # 'telea', 'ns', or 'biharmonic'
    RESTORATION_RADIUS: int = 25
    RESTORATION_EXPAND_MASK: int = 10
    RESTORATION_ITERATIONS: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set default model path if not specified
        if self.MODEL_PATH is None:
            self.MODEL_PATH = self.MODELS_DIR / 'best_model.pth'

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [
            self.DATA_DIR,
            self.UPLOAD_DIR,
            self.PROCESSED_DIR,
            self.TEMP_DIR,
            self.MODELS_DIR
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()