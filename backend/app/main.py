import asyncio
import logging
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from .config import settings
from .jobs import create_job, get_job, cleanup_old_jobs
from .worker import spawn_process_job, check_finished_processes
from .logging_config import configure_logging

logger = logging.getLogger(__name__)

async def _process_monitor():
    """Periodically check if any worker processes have died and mark jobs FAILED."""
    while True:
        await asyncio.sleep(5)
        check_finished_processes()

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Backend starting up")
    os.makedirs(os.path.join(settings.STORAGE_DIR, 'uploads'), exist_ok=True)
    os.makedirs(os.path.join(settings.STORAGE_DIR, 'outputs'), exist_ok=True)
    cleanup_old_jobs()
    monitor_task = asyncio.create_task(_process_monitor())
    yield
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Whisper Diarization API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    # Basic validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    job = create_job(file.filename)
    logger.info("Created job %s for file %s", job.id, file.filename)
    
    upload_dir = os.path.join(settings.STORAGE_DIR, 'uploads')
    file_path = os.path.join(upload_dir, f"{job.id}_orig_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Spawn worker in separate process so OOM does not kill API server
    spawn_process_job(job.id, file_path, upload_dir)
    
    return {"job_id": job.id, "status": job.status, "message": "File uploaded and processing started."}

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
        
    # Build download URLs
    txt_url = f"/api/jobs/{job_id}/download?format=txt"
    json_url = f"/api/jobs/{job_id}/download?format=json"
    
    return {
        "job_id": job.id,
        "transcript": job.result.get("transcript", ""),
        "segments": job.result.get("segments", []),
        "download_txt": txt_url,
        "download_json": json_url
    }

@app.get("/api/jobs/{job_id}/download")
async def download_result(job_id: str, format: str = "txt"):
    if format not in ["txt", "json"]:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'txt' or 'json'")
        
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=400, detail="Job is not completed yet")

    output_dir = os.path.join(settings.STORAGE_DIR, 'outputs', job_id)
    file_path = os.path.join(output_dir, f"transcript.{format}")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")
        
    return FileResponse(path=file_path, filename=f"transcript_{job_id}.{format}")
