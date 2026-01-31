import os
import logging
import subprocess
from typing import List, Dict
from openai import OpenAI

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_dir: str, job_id: str) -> str:
    """Extract audio from video using ffmpeg."""
    audio_path = os.path.join(output_dir, f"{job_id}_audio.wav")

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        audio_path
    ]

    logger.info(f"Extracting audio: {video_path} -> {audio_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        raise RuntimeError(f"Failed to extract audio: {result.stderr}")

    logger.info(f"Audio extracted: {audio_path}")
    return audio_path


def transcribe_audio(audio_path: str) -> List[Dict]:
    """
    Transcribe audio using OpenAI Whisper API.
    Returns list of segments with start, end, text.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    client = OpenAI(api_key=api_key)

    logger.info(f"Transcribing audio with OpenAI Whisper API: {audio_path}")

    # Check file size (OpenAI limit: 25 MB)
    file_size = os.path.getsize(audio_path)
    if file_size > 25 * 1024 * 1024:
        logger.warning(f"Audio file is {file_size / (1024*1024):.1f} MB (limit: 25 MB)")
        # TODO: Implement audio compression or splitting

    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        segments = []
        for segment in response.segments:
            segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
            logger.debug(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}")

        logger.info(f"Transcription complete: {len(segments)} segments")
        return segments

    except Exception as e:
        logger.error(f"OpenAI Whisper API error: {e}")
        raise RuntimeError(f"Failed to transcribe audio: {e}")
