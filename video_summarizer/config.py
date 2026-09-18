"""Central configuration for the video summarization pipeline.

Keeping these values in one place (instead of scattered across files) means
you only need to change a setting once, and it's obvious where to look.
"""

WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium", "large"]
DEFAULT_WHISPER_MODEL = "base"

SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"
SUMMARY_MAX_LENGTH = 150
SUMMARY_MIN_LENGTH = 40
SUMMARY_MAX_CHUNK_CHARS = 3000

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
