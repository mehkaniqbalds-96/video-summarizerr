"""Stage 1: extract audio from a video file."""

import logging
import subprocess

from . import config

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_audio_path: str) -> str:
    """Extract a mono audio track from a video file using ffmpeg."""
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", str(config.AUDIO_CHANNELS),
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-vn",
        output_audio_path,
    ]
    logger.info("Extracting audio from %s", video_path)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return output_audio_path
