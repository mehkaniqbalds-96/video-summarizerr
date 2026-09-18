"""Stage 3: abstractive summarization using BART."""

import functools
import logging

from . import config

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def load_summarizer(model_name: str = config.SUMMARIZATION_MODEL_NAME):
    """Load and cache the tokenizer + model (only loaded once per process)."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    logger.info("Loading summarization model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def _summarize_chunk(chunk, tokenizer, model, max_length, min_length):
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_text(
    text: str,
    max_length: int = config.SUMMARY_MAX_LENGTH,
    min_length: int = config.SUMMARY_MIN_LENGTH,
) -> str:
    """Summarize text using a pretrained BART model, chunking long input."""
    tokenizer, model = load_summarizer()

    chunk_size = config.SUMMARY_MAX_CHUNK_CHARS
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    summaries = [
        _summarize_chunk(chunk, tokenizer, model, max_length, min_length)
        for chunk in chunks
    ]
    combined = " ".join(summaries)

    if len(chunks) > 1:
        logger.info("Combining %d chunk summaries into one final summary", len(chunks))
        return _summarize_chunk(combined, tokenizer, model, max_length, min_length)

    return combined
