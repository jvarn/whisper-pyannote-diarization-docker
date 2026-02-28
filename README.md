# Local Audio Transcription App
A powerful, fully-local audio transcription and speaker diarization tool using Whisper and pyannote.audio.

## Features
- Fully local processing (no paid APIs).
- Faster-Whisper for fast and accurate transcripiton.
- Speaker diarization (Speaker 1, Speaker 2, etc.) using `pyannote.audio`.
- Supports multiple languages including English and Arabic.
- Minimal React UI and FastAPI backend with background job processing.
- Outputs human-readable `.txt` and machine-readable `.json`.

## Prerequisites
- **Docker** and **Docker Compose** (recommended).
- If running locally without Docker:
  - Python 3.10+
  - Node.js 18+
  - `ffmpeg` installed system-wide.

## Setting up HuggingFace Token
The `pyannote.audio` diarization model requires downloading pre-trained weights from HuggingFace.
1. Create a Hugging Face account and accept the user conditions for:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
2. Generate an access token from your HuggingFace settings.
3. Copy `.env.example` to `.env` and configure your token:
   ```bash
   cp .env.example .env
   # Edit .env and paste your HF_TOKEN
   ```

## Running with Docker (Recommended)
You can run the full stack effortlessly with Docker:

```bash
docker compose up -d
```

The application will be available at [http://localhost:3000](http://localhost:3000).

*Note: The first time you process an audio file, it will take some time to download the Whisper and Pyannote models.*

## Running Locally (without Docker)

### 1. Backend Setup
Make sure `ffmpeg` is installed (`brew install ffmpeg` on macOS, or `apt install ffmpeg` on Linux).

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # Or run uv pip install if using uv
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
> Ensure `.env` is created in the root directory prior to starting up.

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:3000`.

## Example Usage with cURL (Backend)
You can upload to the backend directly via the API.

```bash
# Upload a file
curl -X POST -F "file=@your_audio.mp3" http://localhost:8000/api/upload

# Check job status using the returned job_id
curl http://localhost:8000/api/jobs/YOUR_JOB_ID

# Get transcript when complete
curl http://localhost:8000/api/jobs/YOUR_JOB_ID/download?fmt=txt
```

## Troubleshooting
- **Debugging crashes / 404s**: On startup, the backend prints `Logging to: /path/to/backend.log` — use that path for `tail -f`. If nothing appears, set `LOG_FILE=/tmp/whisper.log` in `.env` and tail that file instead. After a crash, check the last lines to see which step failed (e.g. "Diarization chunk 3" before an OOM kill).
- **ffmpeg not found**: Ensure `ffmpeg` is accessible in your system's PATH. On macOS, `brew install ffmpeg`.
- **Pyannote/HuggingFace unauthorized error**: Ensure you accepted the terms for the Pyannote models listed above, and your token has read access.
- **Killed / OOM**: Diarization and transcription can consume a lot of memory. If Docker kills your container, increase Docker's memory limit.
- **16GB RAM (e.g. M4 MacBook Air)**: Set `LOW_MEMORY_MODE=true` in your `.env` to use a smaller Whisper model and reduce peak memory. For files longer than 10 minutes, diarization automatically runs in chunks to avoid OOM; speaker identity is preserved across chunks using overlap-based matching.
- **Docker exit code 137 (OOM kill)**: The backend container was killed for using too much memory. Increase Docker Desktop's memory: **Docker Desktop → Settings → Resources → Memory** — set to at least **12 GB** (16 GB for 20+ minute files). Then restart: `docker compose down && docker compose up -d`.

## Privacy Note
All data processing and inference run completely locally on your hardware. The HuggingFace token is only used once to download the model weights. Do not share your `.env` file or commit it to version control.
