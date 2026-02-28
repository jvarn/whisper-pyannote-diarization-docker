import logging
import os
import multiprocessing
import traceback
from .jobs import update_job, JobStatus
from .pipeline.audio import normalize_audio
from .pipeline.transcribe import process_audio
from .config import settings
from .logging_config import configure_logging

logger = logging.getLogger(__name__)
_active_processes: dict[str, multiprocessing.Process] = {}

def process_job(job_id: str, input_path: str, upload_dir: str):
    """
    Background worker that runs the audio pipeline.
    Runs in a separate process; configure logging on entry.
    """
    configure_logging()
    logger.info("Worker started for job %s, input=%s", job_id, input_path)
    try:
        update_job(job_id, status=JobStatus.RUNNING, progress="Normalizing audio...")
        logger.info("Job %s: normalizing audio", job_id)
        
        # 1. Normalize audio
        wav_path = os.path.join(upload_dir, f"{job_id}.wav")
        normalize_audio(input_path, wav_path)
        logger.info("Job %s: audio normalized, wav=%s", job_id, wav_path)
        
        # 2. Transcribe and diarize
        update_job(job_id, progress="Transcribing and diarizing...")
        output_dir = os.path.join(settings.STORAGE_DIR, 'outputs', job_id)
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Job %s: starting transcription and diarization", job_id)
        
        result_payload = process_audio(
            audio_path=wav_path,
            output_dir=output_dir,
            job_id=job_id,
            progress_callback=lambda msg: update_job(job_id, progress=msg)
        )
        
        logger.info("Job %s: completed successfully", job_id)
        update_job(job_id, status=JobStatus.DONE, progress="Completed", result=result_payload)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error("Job %s failed: %s\n%s", job_id, e, error_trace)
        update_job(job_id, status=JobStatus.FAILED, error=str(e), progress="Failed")
        
    finally:
        # Cleanup original upload and wav file
        if os.path.exists(input_path):
            os.remove(input_path)
        try:
            wav_path = os.path.join(upload_dir, f"{job_id}.wav")
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except OSError:
            pass

def spawn_process_job(job_id: str, input_path: str, upload_dir: str) -> None:
    """Spawn process_job in a separate process so OOM does not kill the API server."""
    p = multiprocessing.Process(target=process_job, args=(job_id, input_path, upload_dir))
    p.start()
    _active_processes[job_id] = p

def check_finished_processes() -> None:
    """Mark jobs as FAILED if their worker process has died."""
    for job_id, p in list(_active_processes.items()):
        if not p.is_alive():
            exitcode = getattr(p, 'exitcode', '?')
            logger.warning(
                "Worker process died for job %s (exitcode=%s). "
                "Exitcode -9 or -11 often indicates OOM or segfault. Check storage/logs/backend.log for last activity.",
                job_id, exitcode
            )
            p.join()
            del _active_processes[job_id]
            update_job(job_id, status=JobStatus.FAILED, error="Worker process died (possibly OOM)", progress="Failed")
