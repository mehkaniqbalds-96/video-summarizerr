# Video Speech-to-Text Summarizer

A proper, modular pipeline: video → audio → transcript (Whisper) → summary (BART).

## Project structure

```
video-summarizer/
├── video_summarizer/          # the actual pipeline package
│   ├── __init__.py            # exposes run_pipeline()
│   ├── config.py              # all settings in one place
│   ├── audio.py               # Stage 1: video -> audio (ffmpeg)
│   ├── transcription.py       # Stage 2: audio -> text (Whisper)
│   ├── summarization.py       # Stage 3: text -> summary (BART)
│   └── pipeline.py            # orchestrates all stages + CLI entry point
├── streamlit_app.py           # web demo, reuses the package above
├── tests/
│   └── test_transcription.py  # unit test for a pure-function stage
├── video_summarizer.ipynb     # exploratory notebook version
├── requirements.txt
└── README.md
```

**Why this shape:** the CLI script and the Streamlit app both import from the
same `video_summarizer` package instead of duplicating logic — fix a bug or
improve a stage once, and both entry points benefit. Each stage lives in its
own file so it can be understood, tested, and changed independently.

## 1. Install ffmpeg (system dependency)

- **Windows:** `winget install -e --id Gyan.FFmpeg`, or download from gyan.dev/ffmpeg/builds
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

Verify: `ffmpeg -version`

## 2. Install Python dependencies

```
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run it — three ways

**Command line:**
```
python -m video_summarizer.pipeline --video path/to/video.mp4 --output results.txt
```

**Web demo:**
```
streamlit run streamlit_app.py
```

**Notebook (for exploring/learning each stage):**
Open `video_summarizer.ipynb` in VS Code or Jupyter and run cells top to bottom.

## 4. Run the tests

```
pytest tests/ -v
```

`clean_transcript()` is tested directly with no ML models involved — that's only possible because the heavy `whisper` and `transformers` imports are deliberately deferred to inside the functions that need them, not loaded at the top of the file.

## Whisper model size guide

| Model  | Relative speed | Accuracy   | Good for |
|--------|-----------------|------------|----------|
| tiny   | Fastest         | Lowest     | Quick tests |
| base   | Fast            | Decent     | Default — start here |
| small  | Moderate        | Good       | Better accuracy, still CPU-feasible |
| medium | Slow on CPU     | Very good  | If you have a GPU |
| large  | Slowest         | Best       | GPU strongly recommended |

## Next steps

- **Evaluation:** WER for the transcript, ROUGE for the summary, so results have numbers behind them
- **Speaker diarization:** "who said what" via a model like `pyannote-audio` layered on top of Whisper
