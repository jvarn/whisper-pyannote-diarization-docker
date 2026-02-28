from enum import Enum
from typing import Dict, Optional, Any
from pydantic import BaseModel
import uuid
import datetime
import json
import logging
import os
import shutil
import threading
from .config import settings

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

class Job(BaseModel):
    id: str
    status: JobStatus
    created_at: datetime.datetime
    filename: str
    progress: str = ""
    error: Optional[str] = None
    result: Optional[Any] = None

# File-based storage for jobs to survive restarts
JOBS_FILE = os.path.join(settings.STORAGE_DIR, "jobs.json")
JOBS_BACKUP = JOBS_FILE + ".bak"

_memory_jobs: Dict[str, Job] = {}
_jobs_loaded = False
_lock = threading.RLock()

def _load_jobs_once():
    global _jobs_loaded, _memory_jobs
    if _jobs_loaded:
        return
    with _lock:
        if _jobs_loaded:
            return
        if not os.path.exists(JOBS_FILE):
            _jobs_loaded = True
            return
        try:
            with open(JOBS_FILE, "r") as f:
                data = json.load(f)
                _memory_jobs = {k: Job(**v) for k, v in data.items()}
        except Exception as e:
            logger.error("Failed to load jobs.json: %s. Trying backup.", e)
            if os.path.exists(JOBS_BACKUP):
                try:
                    with open(JOBS_BACKUP, "r") as f:
                        data = json.load(f)
                        _memory_jobs = {k: Job(**v) for k, v in data.items()}
                    logger.info("Loaded jobs from backup.")
                except Exception as backup_e:
                    logger.error("Failed to load jobs backup: %s", backup_e)
            else:
                logger.warning("No backup available; starting with empty jobs.")
        _jobs_loaded = True

def _save_jobs():
    with _lock:
        os.makedirs(os.path.dirname(JOBS_FILE), exist_ok=True)
        if os.path.exists(JOBS_FILE):
            try:
                shutil.copy2(JOBS_FILE, JOBS_BACKUP)
            except OSError as e:
                logger.warning("Could not create backup: %s", e)
        temp_file = JOBS_FILE + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(
                {k: v.model_dump(mode="json") for k, v in _memory_jobs.items()},
                f,
                indent=2,
            )
        os.replace(temp_file, JOBS_FILE)

def create_job(filename: str) -> Job:
    _load_jobs_once()
    with _lock:
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.datetime.utcnow(),
            filename=filename,
            progress="Added to queue"
        )
        _memory_jobs[job_id] = job
        _save_jobs()
    return job

def _reload_from_disk() -> None:
    """Reload jobs from disk. Used when worker runs in subprocess and updates jobs.json."""
    if not os.path.exists(JOBS_FILE):
        return
    try:
        with open(JOBS_FILE, "r") as f:
            data = json.load(f)
        with _lock:
            _memory_jobs.clear()
            _memory_jobs.update({k: Job(**v) for k, v in data.items()})
    except Exception as e:
        logger.warning("Could not reload jobs from disk: %s", e)

def get_job(job_id: str) -> Optional[Job]:
    _load_jobs_once()
    _reload_from_disk()
    with _lock:
        return _memory_jobs.get(job_id)

def update_job(job_id: str, status: Optional[JobStatus] = None, progress: Optional[str] = None,
               error: Optional[str] = None, result: Optional[Any] = None) -> None:
    _load_jobs_once()
    with _lock:
        job = _memory_jobs.get(job_id)
        if not job:
            return
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = progress
        if error is not None:
            job.error = error
        if result is not None:
            job.result = result
        _save_jobs()

def get_all_jobs() -> Dict[str, Job]:
    _load_jobs_once()
    with _lock:
        return dict(_memory_jobs)

def cleanup_old_jobs(days: int = 7) -> None:
    _load_jobs_once()
    with _lock:
        now = datetime.datetime.utcnow()
        to_delete = []
        for job_id, job in _memory_jobs.items():
            if job.status in (JobStatus.RUNNING, JobStatus.QUEUED):
                continue
            if (now.replace(tzinfo=None) - job.created_at.replace(tzinfo=None)).days > days:
                to_delete.append(job_id)
        if to_delete:
            for job_id in to_delete:
                del _memory_jobs[job_id]
            _save_jobs()
