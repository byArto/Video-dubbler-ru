import os
import logging
import subprocess
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

logger = logging.getLogger(__name__)

# Количество параллельных запросов к TTS API
MAX_PARALLEL_TTS = int(os.getenv("MAX_PARALLEL_TTS", "8"))

# OpenAI TTS голоса
# alloy - нейтральный
# echo - мужской
# fable - британский акцент
# onyx - глубокий мужской (рекомендуется для русского)
# nova - женский
# shimmer - женский мягкий
VOICE = os.getenv("OPENAI_TTS_VOICE", "onyx")


def _get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _generate_single_segment(
    client: OpenAI,
    index: int,
    seg: Dict,
    output_dir: str,
    job_id: str,
    model: str,
    voice: str
) -> Optional[Tuple[int, Dict]]:
    """Generate TTS for a single segment. Returns (index, segment_info) or None on failure."""
    ru_text = seg.get('text_ru', seg.get('text', ''))
    if not ru_text.strip():
        return None

    segment_path = os.path.join(output_dir, f"{job_id}_tts_seg_{index:04d}.mp3")
    slot_duration = seg['end'] - seg['start']

    # Определение скорости по длине текста и времени слота
    speed = 1.0
    chars_per_second = len(ru_text) / slot_duration if slot_duration > 0 else 10
    if chars_per_second > 15:
        speed = min(chars_per_second / 12, 4.0)
    elif chars_per_second < 8:
        speed = max(chars_per_second / 10, 0.25)

    try:
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=ru_text,
            speed=speed,
            response_format="mp3"
        )
        response.stream_to_file(segment_path)

        logger.debug(f"Generated segment {index}: {ru_text[:30]}... (speed={speed:.2f})")

        return (index, {
            'path': segment_path,
            'start': seg['start'],
            'end': seg['end']
        })

    except Exception as e:
        logger.error(f"Failed to generate TTS for segment {index}: {e}")
        return None


def generate_voiceover(segments: List[Dict], output_dir: str, job_id: str) -> str:
    """
    Generate Russian voiceover using OpenAI TTS API with parallel processing.
    """
    if not segments:
        raise ValueError("No segments to generate voiceover for")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_TTS_MODEL", "tts-1")

    logger.info(f"Generating voiceover for {len(segments)} segments using OpenAI TTS API (parallel: {MAX_PARALLEL_TTS})")

    segment_files = []

    # Параллельная генерация TTS сегментов
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TTS) as executor:
        futures = {
            executor.submit(
                _generate_single_segment,
                client, i, seg, output_dir, job_id, model, VOICE
            ): i for i, seg in enumerate(segments)
        }

        completed = 0
        total = len(segments)

        for future in as_completed(futures):
            completed += 1
            if completed % 10 == 0 or completed == total:
                logger.info(f"TTS progress: {completed}/{total} segments")

            result = future.result()
            if result is not None:
                segment_files.append(result)

    # Сортируем по индексу для правильного порядка
    segment_files.sort(key=lambda x: x[0])
    segment_files = [seg_info for _, seg_info in segment_files]

    if not segment_files:
        raise RuntimeError("No TTS segments generated")

    # Создаем финальный voiceover с правильной синхронизацией
    voiceover_path = os.path.join(output_dir, f"{job_id}_voiceover.mp3")
    _create_timed_voiceover(segment_files, voiceover_path)

    logger.info(f"Voiceover generated: {voiceover_path}")
    return voiceover_path


def _create_timed_voiceover(segment_files: List[Dict], output_path: str):
    """Create a single audio file with segments placed at correct timestamps."""
    if not segment_files:
        raise ValueError("No segment files to process")

    filter_parts = []
    inputs = []

    for i, seg in enumerate(segment_files):
        inputs.extend(['-i', seg['path']])
        delay_ms = int(seg['start'] * 1000)
        filter_parts.append(f"[{i}]adelay={delay_ms}|{delay_ms}[a{i}]")

    mix_inputs = ''.join(f'[a{i}]' for i in range(len(segment_files)))
    filter_parts.append(f"{mix_inputs}amix=inputs={len(segment_files)}:duration=longest[out]")

    filter_complex = ';'.join(filter_parts)

    cmd = ['ffmpeg', '-y'] + inputs + [
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-ar', '44100',
        '-ac', '1',
        '-b:a', '192k',
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        raise RuntimeError(f"Failed to create timed voiceover: {result.stderr}")
