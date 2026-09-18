from video_summarizer.transcription import clean_transcript


def test_clean_transcript_collapses_whitespace():
    raw = "  hello   world  \n\n this  is  a test  "
    assert clean_transcript(raw) == "hello world this is a test"


def test_clean_transcript_handles_empty_string():
    assert clean_transcript("") == ""
