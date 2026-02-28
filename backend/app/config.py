import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    HF_TOKEN: str = ""
    WHISPER_MODEL: str = "medium"
    WHISPER_MODEL_LOW_MEMORY: str = "small"
    LOW_MEMORY_MODE: bool = False
    DEVICE: str = "cpu"
    
    # Chunked diarization for long audio (avoids OOM on 16GB)
    # If audio duration > this (seconds), use chunked processing
    DIARIZATION_CHUNK_MAX_DURATION: float = 600.0  # 10 min
    DIARIZATION_CHUNK_DURATION: float = 360.0  # 6 min per chunk
    DIARIZATION_CHUNK_OVERLAP: float = 90.0  # 90 sec overlap for speaker matching
    
    # Storage dir
    STORAGE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'storage')
    # Override log file path if needed (e.g. LOG_FILE=/tmp/whisper.log)
    LOG_FILE: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra='ignore')

settings = Settings()
