import os
import logging
import subprocess

logger = logging.getLogger(__name__)


def merge_video_with_voiceover(
    video_path: str,
    voiceover_path: str,
    output_path: str,
    volume_boost: float = 3.6
) -> str:
    """
    Merge original video with Russian voiceover.
    Replaces original audio completely with Russian voiceover.
    Boosts volume by 260% (3.6x) by default.
    """
    logger.info(f"Merging video with voiceover")
    logger.info(f"  Video: {video_path}")
    logger.info(f"  Voiceover: {voiceover_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Volume boost: {volume_boost}x")

    # Replace original audio with voiceover and boost volume
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', voiceover_path,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-af', f'volume={volume_boost}',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg merge error: {result.stderr}")
        raise RuntimeError(f"Merge failed: {result.stderr}")

    logger.info(f"Video merged successfully: {output_path}")
    return output_path
