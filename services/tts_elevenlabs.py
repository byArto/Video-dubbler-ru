import os
import logging
import subprocess
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Количество параллельных запросов к ElevenLabs API
MAX_PARALLEL_TTS = int(os.getenv("MAX_PARALLEL_TTS", "5"))

# ElevenLabs голоса (Multilingual v2)
# Популярные мужские голоса для русского:
# - Adam: pNInz6obpgDQGcFmaJgB - глубокий, уверенный
# - Antoni: ErXwobaYiN019PkySvjV - молодой, энергичный
# - Arnold: VR6AewLTigWG4xSOukaG - сильный, командный
# - Clyde: 2EiwWnXFnvU5JabPnv8n - разговорный, дружелюбный
# - Daniel: onwK4e9ZLuTAKqWW03F9 - британский акцент
DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Adam


def _generate_single_segment_elevenlabs(
    client,
    index: int,
    seg: Dict,
    output_dir: str,
    job_id: str,
    voice_id: str,
    model_id: str
) -> Optional[Tuple[int, Dict]]:
    """Generate TTS for a single segment using ElevenLabs. Returns (index, segment_info) or None on failure."""
    ru_text = seg.get('text_ru', seg.get('text', ''))
    if not ru_text.strip():
        return None

    segment_path = os.path.join(output_dir, f"{job_id}_tts_seg_{index:04d}.mp3")

    try:
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            model_id=model_id,
            text=ru_text,
            output_format="mp3_44100_128"
        )

        # Записываем аудио в файл
        with open(segment_path, 'wb') as f:
            for chunk in audio:
                f.write(chunk)

        logger.debug(f"Generated segment {index}: {ru_text[:30]}...")

        return (index, {
            'path': segment_path,
            'start': seg['start'],
            'end': seg['end']
        })

    except Exception as e:
        logger.error(f"Failed to generate ElevenLabs TTS for segment {index}: {e}")
        return None


def generate_voiceover_elevenlabs(segments: List[Dict], output_dir: str, job_id: str) -> str:
    """
    Generate Russian voiceover using ElevenLabs TTS API with parallel processing.
    """
    from elevenlabs import ElevenLabs

    if not segments:
        raise ValueError("No segments to generate voiceover for")

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set in environment")

    client = ElevenLabs(api_key=api_key)
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)
    model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

    logger.info(f"Generating voiceover for {len(segments)} segments using ElevenLabs TTS (parallel: {MAX_PARALLEL_TTS})")
    logger.info(f"Voice ID: {voice_id}, Model: {model_id}")

    segment_files = []

    # Параллельная генерация TTS сегментов
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TTS) as executor:
        futures = {
            executor.submit(
                _generate_single_segment_elevenlabs,
                client, i, seg, output_dir, job_id, voice_id, model_id
            ): i for i, seg in enumerate(segments)
        }

        completed = 0
        total = len(segments)

        for future in as_completed(futures):
            completed += 1
            if completed % 10 == 0 or completed == total:
                logger.info(f"ElevenLabs TTS progress: {completed}/{total} segments")

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

    logger.info(f"ElevenLabs voiceover generated: {voiceover_path}")
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


def list_available_voices() -> List[Dict]:
    """List all available ElevenLabs voices."""
    from elevenlabs import ElevenLabs

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return []

    try:
        client = ElevenLabs(api_key=api_key)
        voices = client.voices.get_all()
        return [
            {
                'voice_id': v.voice_id,
                'name': v.name,
                'category': v.category,
                'labels': v.labels
            }
            for v in voices.voices
        ]
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        return []
