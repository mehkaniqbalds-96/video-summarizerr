"""
Streamlit demo app for the video summarizer.

Run with:
    streamlit run streamlit_app.py
"""

import os
import tempfile

import streamlit as st

from video_summarizer import config
from video_summarizer.audio import extract_audio
from video_summarizer.transcription import transcribe_audio, clean_transcript
from video_summarizer.summarization import summarize_text

st.set_page_config(page_title="Video Summarizer", page_icon="🎬")

st.title("🎬 Video Speech-to-Text Summarizer")
st.write("Upload a video and get a transcript plus a concise summary of what's spoken.")

whisper_model_size = st.selectbox("Whisper model size", config.WHISPER_MODEL_SIZES, index=1)

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "mkv", "avi"])

if uploaded_file is not None and st.button("Summarize"):
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        audio_path = os.path.join(tmp_dir, "audio.wav")

        with st.spinner("Extracting audio..."):
            extract_audio(video_path, audio_path)

        with st.spinner("Transcribing speech (first run also loads the model)..."):
            raw_transcript = transcribe_audio(audio_path, whisper_model_size)
            transcript = clean_transcript(raw_transcript)

        with st.spinner("Generating summary (first run also loads the model)..."):
            summary = summarize_text(transcript)

    st.subheader("📝 Transcript")
    st.write(transcript)

    st.subheader("✨ Summary")
    st.write(summary)

    st.download_button(
        "Download results as .txt",
        data=f"TRANSCRIPT\n\n{transcript}\n\nSUMMARY\n\n{summary}",
        file_name="summary_results.txt",
    )
