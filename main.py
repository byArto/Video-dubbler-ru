import os
import ssl
import uuid
import json
import asyncio
import logging
import urllib.request
from datetime import datetime
from typing import Optional
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Disable SSL verification globally for Python 3.14+ on macOS
ssl._create_default_https_context = ssl._create_unverified_context

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directories
JOBS_DIR = os.path.join(os.path.dirname(__file__), "jobs")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="Video RU Dubber")


class JobStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    DUBBING = "dubbing"
    MERGING = "merging"
    DONE = "done"
    ERROR = "error"


class SubmitRequest(BaseModel):
    url: str


class SubmitResponse(BaseModel):
    job_id: str


class StatusResponse(BaseModel):
    status: str
    error: Optional[str] = None


# In-memory job storage
jobs: dict = {}


def save_job(job_id: str, data: dict):
    """Save job state to file and memory."""
    jobs[job_id] = data
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    with open(job_file, 'w') as f:
        json.dump(data, f, default=str)


def load_job(job_id: str) -> Optional[dict]:
    """Load job from memory or file."""
    if job_id in jobs:
        return jobs[job_id]

    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    if os.path.exists(job_file):
        with open(job_file, 'r') as f:
            jobs[job_id] = json.load(f)
            return jobs[job_id]

    return None


def update_job_status(job_id: str, status: JobStatus, error: str = None):
    """Update job status."""
    job = load_job(job_id)
    if job:
        job['status'] = status.value
        if error:
            job['error'] = error
        job['updated_at'] = datetime.now().isoformat()
        save_job(job_id, job)
        logger.info(f"Job {job_id}: {status.value}")


async def process_video(job_id: str, url: str):
    """Main video processing pipeline."""
    from services.downloader import download_video
    from services.transcriber import extract_audio, transcribe_audio
    from services.translator import translate_segments
    from services.tts import generate_voiceover
    from services.merger import merge_video_with_voiceover

    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    try:
        # Step 1: Download video
        update_job_status(job_id, JobStatus.DOWNLOADING)
        video_path = await asyncio.to_thread(download_video, url, job_dir, job_id)

        # Step 2: Extract and transcribe audio
        update_job_status(job_id, JobStatus.TRANSCRIBING)
        audio_path = await asyncio.to_thread(extract_audio, video_path, job_dir, job_id)
        segments = await asyncio.to_thread(transcribe_audio, audio_path)

        if not segments:
            raise ValueError("No speech detected in video")

        # Step 3: Translate segments
        update_job_status(job_id, JobStatus.TRANSLATING)
        translated_segments = await asyncio.to_thread(translate_segments, segments)

        # Step 4: Generate Russian voiceover
        update_job_status(job_id, JobStatus.DUBBING)
        voiceover_path = await asyncio.to_thread(generate_voiceover, translated_segments, job_dir, job_id)

        # Step 5: Merge video with voiceover
        update_job_status(job_id, JobStatus.MERGING)
        output_path = os.path.join(OUTPUTS_DIR, f"{job_id}.mp4")
        await asyncio.to_thread(merge_video_with_voiceover, video_path, voiceover_path, output_path)

        # Done
        job = load_job(job_id)
        job['output_path'] = output_path
        save_job(job_id, job)
        update_job_status(job_id, JobStatus.DONE)

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        update_job_status(job_id, JobStatus.ERROR, str(e))


@app.post("/submit", response_model=SubmitResponse)
async def submit_job(request: SubmitRequest, background_tasks: BackgroundTasks):
    """Submit a new video for processing."""
    url = request.url.strip()

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    # Validate URL format (basic check)
    valid_domains = ['twitter.com', 'x.com', 'youtube.com', 'youtu.be', 'www.youtube.com', 'www.twitter.com', 'www.x.com']
    if not any(domain in url.lower() for domain in valid_domains):
        raise HTTPException(status_code=400, detail="URL must be from Twitter/X or YouTube")

    job_id = str(uuid.uuid4())

    # Create initial job record
    job_data = {
        'job_id': job_id,
        'url': url,
        'status': JobStatus.QUEUED.value,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'error': None,
        'output_path': None
    }
    save_job(job_id, job_data)

    # Start background processing
    background_tasks.add_task(process_video, job_id, url)

    logger.info(f"Job {job_id} created for URL: {url}")
    return SubmitResponse(job_id=job_id)


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """Get job status."""
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return StatusResponse(status=job['status'], error=job.get('error'))


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Download the result video."""
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job['status'] != JobStatus.DONE.value:
        raise HTTPException(status_code=400, detail=f"Job not ready. Status: {job['status']}")

    output_path = job.get('output_path')
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"dubbed_{job_id}.mp4"
    )


@app.get("/")
async def index():
    """Serve the frontend."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
