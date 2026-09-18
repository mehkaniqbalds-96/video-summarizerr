"""
End-to-End Video Summarizer — pipeline orchestrator and CLI entry point.

Pipeline: Video -> Audio extraction -> Speech-to-Text (Whisper) -> Text cleaning -> Summarization (BART)

Usage:
    python -m video_summarizer.pipeline --video path/to/video.mp4
    python -m video_summarizer.pipeline --video path/to/video.mp4 --whisper_model small --output results.txt
"""

import argparse
import logging
import os
import sys
import tempfile

from . import config
from .audio import extract_audio
from .transcription import transcribe_audio, clean_transcript
from .summarization import summarize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(video_path: str, whisper_model: str = config.DEFAULT_WHISPER_MODEL) -> dict:
    """Run the full video -> transcript -> summary pipeline end to end."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = os.path.join(tmp_dir, "audio.wav")

        logger.info("Step 1/4: Extracting audio")
        extract_audio(video_path, audio_path)

        logger.info("Step 2/4: Transcribing speech to text")
        raw_transcript = transcribe_audio(audio_path, whisper_model)

        logger.info("Step 3/4: Cleaning transcript")
        transcript = clean_transcript(raw_transcript)

        logger.info("Step 4/4: Generating summary")
        summary = summarize_text(transcript)

    return {"transcript": transcript, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="Summarize the spoken content of a video.")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument(
        "--whisper_model", default=config.DEFAULT_WHISPER_MODEL,
        choices=config.WHISPER_MODEL_SIZES,
        help="Whisper model size (bigger = more accurate, slower)",
    )
    parser.add_argument("--output", default=None, help="Optional path to save results as a .txt file")
    args = parser.parse_args()

    try:
        results = run_pipeline(args.video, args.whisper_model)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    print("\n" + "=" * 60)
    print("TRANSCRIPT")
    print("=" * 60)
    print(results["transcript"])
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(results["summary"])

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("TRANSCRIPT\n\n")
            f.write(results["transcript"])
            f.write("\n\nSUMMARY\n\n")
            f.write(results["summary"])
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
