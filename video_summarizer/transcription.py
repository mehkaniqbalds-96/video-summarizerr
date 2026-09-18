"""Stage 2: speech-to-text transcription using Whisper."""

import functools
import logging

from . import config

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def load_whisper_model(model_size: str = config.DEFAULT_WHISPER_MODEL):
    """Load and cache a Whisper model.

    The import happens inside this function, not at the top of the file, so
    that lightweight functions like clean_transcript() below can be tested
    without needing the (large) whisper package installed at all.

    @lru_cache means this only actually loads the model from disk the FIRST
    time it's called with a given model_size. Every call after that reuses
    the already-loaded model instead of reloading it.
    """
    import whisper
    logger.info("Loading Whisper model: %s", model_size)
    return whisper.load_model(model_size)


def transcribe_audio(audio_path: str, model_size: str = config.DEFAULT_WHISPER_MODEL) -> str:
    """Transcribe an audio file to raw text."""
    model = load_whisper_model(model_size)
    logger.info("Transcribing %s", audio_path)
    result = model.transcribe(audio_path)
    return result["text"].strip()


def clean_transcript(text: str) -> str:
    """Light cleanup: collapse extra whitespace from raw ASR output."""
    return " ".join(text.split())
