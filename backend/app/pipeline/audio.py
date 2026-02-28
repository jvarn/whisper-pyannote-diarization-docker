import subprocess
import os
import json
import logging

logger = logging.getLogger(__name__)


def get_audio_duration_seconds(path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-i", path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def extract_audio_segment(input_path: str, output_path: str, start_sec: float, end_sec: float) -> None:
    """Extract a segment of audio using ffmpeg. start_sec and end_sec are in seconds."""
    duration = end_sec - start_sec
    logger.debug("Extracting segment %.1f-%.1fs from %s", start_sec, end_sec, input_path)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start_sec),
        "-t", str(duration),
        "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def normalize_audio(input_path: str, output_path: str) -> None:
    """
    Normalizes the input audio file to 16kHz mono WAV format using ffmpeg.
    This guarantees compatibility with faster-whisper and pyannote.
    """
    command = [
        "ffmpeg",
        "-i", input_path,
        "-vn",          # Strip video
        "-acodec", "pcm_s16le",
        "-ac", "1",     # Mono
        "-ar", "16000", # 16kHz
        "-y",           # Overwrite output
        output_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg error: {e.stderr.decode('utf-8')}"
        raise RuntimeError(error_msg)
