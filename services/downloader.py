import os
import logging
import yt_dlp

logger = logging.getLogger(__name__)


def download_video(url: str, output_dir: str, job_id: str) -> str:
    """
    Download video from Twitter/X or YouTube using yt-dlp.
    Returns path to downloaded video file.
    """
    output_template = os.path.join(output_dir, f"{job_id}.%(ext)s")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best',
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': False,
        'extract_flat': False,
        'nocheckcertificate': True,
        'cookiesfrombrowser': ('chrome',),  # Use cookies from Chrome to bypass YouTube blocks
        # Fix YouTube 403 errors
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Sec-Fetch-Mode': 'navigate',
        },
        'extractor_args': {
            'youtube': {
                'player_client': ['ios'],
            }
        },
    }

    logger.info(f"Downloading video from: {url}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        # Get the actual output filename
        if info.get('requested_downloads'):
            video_path = info['requested_downloads'][0]['filepath']
        else:
            # Fallback: construct path from info
            ext = info.get('ext', 'mp4')
            video_path = os.path.join(output_dir, f"{job_id}.{ext}")

    if not os.path.exists(video_path):
        # Try to find the downloaded file
        for f in os.listdir(output_dir):
            if f.startswith(job_id) and f.endswith(('.mp4', '.webm', '.mkv')):
                video_path = os.path.join(output_dir, f)
                break

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Downloaded video not found for job {job_id}")

    logger.info(f"Video downloaded: {video_path}")
    return video_path
