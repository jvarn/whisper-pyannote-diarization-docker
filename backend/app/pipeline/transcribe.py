import gc
import logging
import os
import json
import tempfile
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from typing import Callable, Any, Dict, List, Tuple

from ..config import settings
from .audio import get_audio_duration_seconds, extract_audio_segment

logger = logging.getLogger(__name__)


def _compute_speaker_overlap(
    segments_a: List[Dict], segments_b: List[Dict],
    overlap_start: float, overlap_end: float
) -> Dict[Tuple[str, str], float]:
    """
    Compute overlap duration between each (speaker_a, speaker_b) pair in the overlap region.
    Returns dict mapping (speaker_a, speaker_b) -> total overlap duration.
    """
    overlap_matrix: Dict[Tuple[str, str], float] = {}
    for sa in segments_a:
        if sa["end"] <= overlap_start or sa["start"] >= overlap_end:
            continue
        for sb in segments_b:
            if sb["end"] <= overlap_start or sb["start"] >= overlap_end:
                continue
            inter_start = max(sa["start"], sb["start"], overlap_start)
            inter_end = min(sa["end"], sb["end"], overlap_end)
            dur = max(0, inter_end - inter_start)
            if dur > 0:
                key = (sa["speaker"], sb["speaker"])
                overlap_matrix[key] = overlap_matrix.get(key, 0) + dur
    return overlap_matrix


def _diarize_chunked(
    audio_path: str,
    duration_sec: float,
    pipeline: Pipeline,
    progress_callback: Callable[[str], None],
) -> List[Dict[str, Any]]:
    """
    Run diarization in chunks for long audio. Uses overlap regions to match
    speaker identities across chunks.
    """
    chunk_dur = settings.DIARIZATION_CHUNK_DURATION
    overlap_dur = settings.DIARIZATION_CHUNK_OVERLAP
    step = chunk_dur - overlap_dur  # advance by this much each chunk

    all_diarization: List[Dict[str, Any]] = []
    global_speaker_counter = 0
    prev_speaker_map: Dict[str, str] = {}  # local_name -> global_name
    prev_segments_in_overlap: List[Dict] = []

    chunk_idx = 0
    start = 0.0
    with tempfile.TemporaryDirectory() as tmpdir:
        while start < duration_sec:
            end = min(start + chunk_dur, duration_sec)
            chunk_path = os.path.join(tmpdir, f"chunk_{chunk_idx}.wav")
            logger.info("Diarization chunk %d: %.1f-%.1fs", chunk_idx + 1, start, end)
            progress_callback(f"Performing speaker diarization... (chunk {chunk_idx + 1})")
            extract_audio_segment(audio_path, chunk_path, start, end)
            diar_result = pipeline(chunk_path)
            chunk_segments = [
                {"start": t.start + start, "end": t.end + start, "speaker": sp}
                for t, _, sp in diar_result.itertracks(yield_label=True)
            ]
            try:
                os.remove(chunk_path)
            except OSError:
                pass

            curr_speakers = sorted(set(s["speaker"] for s in chunk_segments))
            if chunk_idx == 0:
                local_to_global = {sp: sp for sp in curr_speakers}
                global_speaker_counter = len(curr_speakers)
                prev_speaker_map = dict(local_to_global)
            else:
                overlap_start = start
                overlap_end = min(start + overlap_dur, end)
                curr_in_overlap = [
                    s for s in chunk_segments
                    if s["end"] > overlap_start and s["start"] < overlap_end
                ]
                overlap_matrix = _compute_speaker_overlap(
                    prev_segments_in_overlap,
                    curr_in_overlap,
                    overlap_start, overlap_end
                )
                local_to_global = {}
                for curr_sp in curr_speakers:
                    best_prev = None
                    best_dur = 0.0
                    for (prev_sp, c_sp), d in overlap_matrix.items():
                        if c_sp == curr_sp and d > best_dur:
                            best_dur = d
                            best_prev = prev_sp
                    if best_prev:
                        local_to_global[curr_sp] = best_prev
                    else:
                        global_speaker_counter += 1
                        local_to_global[curr_sp] = f"SPEAKER_{global_speaker_counter - 1:02d}"
                prev_speaker_map = {sp: local_to_global[sp] for sp in curr_speakers}

            for seg in chunk_segments:
                seg["speaker"] = local_to_global[seg["speaker"]]
            all_diarization.extend(chunk_segments)

            overlap_region_start = start + step
            overlap_region_end = min(start + chunk_dur, end)
            prev_segments_in_overlap = [
                {"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
                for s in chunk_segments
                if s["end"] > overlap_region_start and s["start"] < overlap_region_end
            ]
            if not prev_segments_in_overlap:
                prev_segments_in_overlap = [
                    {"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
                    for s in chunk_segments
                ]
            start += step
            chunk_idx += 1
            if end >= duration_sec:
                break

    return all_diarization


def process_audio(audio_path: str, output_dir: str, job_id: str, progress_callback: Callable[[str], None]) -> Dict[str, Any]:
    """
    Main pipeline:
    1. Transcribe audio with Faster Whisper.
    2. Diarize with pyannote.audio.
    3. Merge transcription segments with diarization labels.
    """
    
    # --- 1. Whisper Transcription ---
    progress_callback("Loading Whisper model...")
    device = "cuda" if "cuda" in settings.DEVICE and torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    whisper_model = settings.WHISPER_MODEL_LOW_MEMORY if settings.LOW_MEMORY_MODE else settings.WHISPER_MODEL
    logger.info("Loading Whisper model %s (device=%s)", whisper_model, device)
    
    model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
    
    progress_callback("Transcribing audio...")
    whisper_segments, info = model.transcribe(audio_path, beam_size=5)
    
    # Consume generator into a list
    transcription = []
    for segment in whisper_segments:
        transcription.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "speaker": "Unknown" # Will be updated by diarization
        })
    
    # Unload Whisper to free memory before loading Pyannote (~1-2GB)
    del model
    gc.collect()
    logger.info("Whisper done, freed memory. Loading Pyannote...")
        
    # --- 2. Pyannote Diarization ---
    progress_callback("Loading Pyannote diarization model...")
    if not settings.HF_TOKEN:
        raise ValueError("HF_TOKEN is not configured. Pyannote requires a HuggingFace token.")
        
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.HF_TOKEN
        )
    except Exception as e:
        raise RuntimeError(f"Could not load Pyannote pipeline. Did you accept the conditions on HF? Error: {e}")
    
    # Minimize batch sizes to reduce peak memory (critical for 16GB RAM)
    if hasattr(diarization_pipeline, "segmentation_batch_size"):
        diarization_pipeline.segmentation_batch_size = 1
    if hasattr(diarization_pipeline, "embedding_batch_size"):
        diarization_pipeline.embedding_batch_size = 1
    
    # Use MPS on Apple Silicon for better efficiency (pyannote supports it; faster-whisper does not)
    if device == "cuda":
        diarization_pipeline.to(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        diarization_pipeline.to(torch.device("mps"))
        
    progress_callback("Performing speaker diarization...")
    duration_sec = get_audio_duration_seconds(audio_path)
    logger.info("Audio duration=%.1fs, chunk_threshold=%.1fs", duration_sec, settings.DIARIZATION_CHUNK_MAX_DURATION)
    if duration_sec > settings.DIARIZATION_CHUNK_MAX_DURATION:
        logger.info("Using chunked diarization (%d chunks estimated)", int(duration_sec / (settings.DIARIZATION_CHUNK_DURATION - settings.DIARIZATION_CHUNK_OVERLAP)) + 1)
        diarization = _diarize_chunked(
            audio_path, duration_sec, diarization_pipeline, progress_callback
        )
        logger.info("Chunked diarization complete, %d segments", len(diarization))
    else:
        logger.info("Running full-file diarization")
        diarization_result = diarization_pipeline(audio_path)
        diarization = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization_result.itertracks(yield_label=True)
        ]
        logger.info("Diarization complete, %d segments", len(diarization))
        
    # --- 3. Merging ---
    progress_callback("Merging transcript with speaker labels...")
    merged_segments = []
    
    for seg in transcription:
        # Find all diarization intervals that overlap with this segment
        overlapping = []
        for d in diarization:
            overlap_start = max(seg["start"], d["start"])
            overlap_end = min(seg["end"], d["end"])
            overlap_duration = overlap_end - overlap_start
            
            if overlap_duration > 0:
                overlapping.append({"speaker": d["speaker"], "duration": overlap_duration})
        
        # Determine dominant speaker for this segment
        if overlapping:
            # Group by speaker and sum durations
            speaker_durations = {}
            for o in overlapping:
                speaker_durations[o["speaker"]] = speaker_durations.get(o["speaker"], 0) + o["duration"]
            dominant_speaker = max(speaker_durations.items(), key=lambda x: x[1])[0]
            seg["speaker"] = dominant_speaker
        else:
            seg["speaker"] = "Unknown"
            
        merged_segments.append(seg)
        
    progress_callback("Saving outputs...")
    
    # Group consecutive segments by the same speaker
    final_output = []
    current_speaker = None
    current_text = []
    current_start = 0
    current_end = 0
    
    for seg in merged_segments:
        if seg["speaker"] != current_speaker:
            if current_speaker is not None:
                final_output.append({
                    "start": current_start,
                    "end": current_end,
                    "speaker": current_speaker,
                    "text": " ".join(current_text)
                })
            current_speaker = seg["speaker"]
            current_start = seg["start"]
            current_text = [seg["text"]]
            current_end = seg["end"]
        else:
            current_text.append(seg["text"])
            current_end = seg["end"]
            
    if current_speaker is not None:
        final_output.append({
            "start": current_start,
            "end": current_end,
            "speaker": current_speaker,
            "text": " ".join(current_text)
        })
        
    # Generate TXT
    def format_time(seconds: float) -> str:
        s = int(seconds)
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"

    txt_path = os.path.join(output_dir, "transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in final_output:
            start_str = format_time(seg["start"])
            end_str = format_time(seg["end"])
            f.write(f"[{start_str} - {end_str}] {seg['speaker']}: {seg['text']}\n\n")

    # Generate JSON
    json_path = os.path.join(output_dir, "transcript.json")
    json_content = {
        "job_id": job_id,
        "language": info.language,
        "segments": final_output
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_content, f, indent=2)

    return json_content
