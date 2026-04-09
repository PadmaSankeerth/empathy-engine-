<<<<<<< HEAD
# The Empathy Engine
### Dynamically Expressive Text-to-Speech via Emotion Detection

---

## Architecture Overview

```
Text Input
    │
    ▼
EmotionEngine  ──── HuggingFace (distilroberta, 7 emotions)
                 └── VADER (compound intensity score)
    │
    ▼
VoiceMapper    ──── emotion × intensity → rate / pitch / volume / break / emphasis
    │
    ▼
SSMLBuilder    ──── <speak><prosody rate="..." pitch="..." volume="...">text</prosody></speak>
    │
    ▼
TTSEngine      ──── Google Cloud TTS (primary) → gTTS + pydub (fallback)
    │
    ▼
Audio Output (.mp3)
```

---

## Step-by-Step Setup Guide

### Step 1 — Prerequisites

Make sure the following are installed on your system:

**Python 3.10 or higher**
```bash
python --version   # should print 3.10+
```

**ffmpeg** (required by pydub for audio processing)
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
# Add ffmpeg/bin to your PATH
```

---

### Step 2 — Create a Virtual Environment

```bash
# Navigate to the project folder
cd empathy_engine

# Create the virtual environment
python -m venv venv

# Activate it
# macOS / Linux:
source venv/bin/activate

# Windows (CMD):
venv\Scripts\activate.bat

# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt.

---

### Step 3 — Install PyTorch (CPU-only, saves 2GB disk space)

```bash
# CPU-only build (recommended for hackathons — no GPU needed)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Verify installation
python -c "import torch; print('PyTorch OK:', torch.__version__)"
```

> **Why separate?** The default `pip install torch` downloads a 2GB+ CUDA build.
> The CPU-only build is ~250MB and works identically for text classification.

---

### Step 4 — Install All Dependencies

```bash
pip install -r requirements.txt
```

Expected output: All packages install without errors.

> **Note on HuggingFace model:** The first time you run the app, it will
> automatically download the emotion classifier model (~300MB) from
> HuggingFace Hub. This only happens once and is cached at `~/.cache/huggingface/`.

---

### Step 5 — Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your settings
nano .env  # or use any text editor
```

**Minimum .env for gTTS (no API key needed):**
```env
FLASK_DEBUG=false
PORT=5000
```

**Optional: Google Cloud TTS (best quality)**
1. Create a GCP project at https://console.cloud.google.com
2. Enable the "Cloud Text-to-Speech API"
3. Create a Service Account → download JSON key
4. Add to .env:
```env
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/your-service-account.json
```

---

### Step 6 — Run the Web Application

```bash
python app.py
```

Expected output:
```
INFO - Loading HuggingFace emotion classifier...
INFO - HuggingFace model loaded successfully.
INFO - VADER analyzer loaded successfully.
INFO - TTS backend selected: gtts
INFO - All subsystems ready.
INFO - Starting Empathy Engine on http://0.0.0.0:5000
```

**Open your browser:** http://localhost:5000

---

### Step 7 — Verify Everything Works

Open http://localhost:5000/health in your browser. You should see:

```json
{
  "status": "ok",
  "emotion_engine": {
    "huggingface": true,
    "vader": true
  },
  "tts_engine": {
    "backend": "gtts",
    "available": true,
    "supports_ssml": false
  }
}
```

> If `huggingface` is `false`, the transformer model failed to download.
> Check your internet connection and re-run `python app.py`.

---

### Step 8 — (Optional) Run the CLI Interface

```bash
# Single text
python cli.py "I just won the championship! This is incredible!"

# Specify output file
python cli.py --text "I miss home so much." --output sadness.mp3

# Interactive mode (type multiple sentences)
python cli.py --interactive

# Health check
python cli.py --health
```

---

### Step 9 — (Optional) Run Tests

```bash
pytest tests/ -v
```

Expected: All tests pass. Tests marked as `xfail` indicate optional subsystems
(e.g., Google Cloud TTS) that are not configured.

---

## Emotion → Voice Mapping Reference

| Emotion  | Rate   | Pitch   | Volume | Break  | Emphasis |
|----------|--------|---------|--------|--------|----------|
| Joy      | +25%   | +3.5st  | +5dB   | 0ms    | moderate |
| Anger    | +22%   | +2.0st  | +8dB   | 0ms    | strong   |
| Sadness  | −28%   | −3.0st  | −4dB   | 400ms  | reduced  |
| Fear     | +18%   | +2.5st  | −2dB   | 200ms  | moderate |
| Surprise | +8%    | +4.5st  | +4dB   | 0ms    | strong   |
| Disgust  | −12%   | −2.0st  | +2dB   | 100ms  | moderate |
| Neutral  | 0%     | 0st     | 0dB    | 0ms    | none     |

All values are multiplied by intensity (0.0–1.0) before applying.

---

## Project Structure

```
empathy_engine/
├── app.py                    ← Flask application (main entry point)
├── cli.py                    ← Command-line interface
├── requirements.txt          ← Python dependencies
├── .env.example              ← Environment variables template
│
├── modules/
│   ├── emotion_engine.py     ← HuggingFace + VADER emotion detection
│   ├── voice_mapper.py       ← Emotion → vocal parameter mapping
│   ├── ssml_builder.py       ← SSML markup generation
│   └── tts_engine.py         ← TTS synthesis (Google/gTTS/pyttsx3)
│
├── templates/
│   └── index.html            ← Web UI
│
├── static/
│   └── audio/                ← Generated .mp3 files (auto-created)
│
└── tests/
    └── test_all.py           ← Full test suite
```

---

## TTS Backend Priority

The app auto-detects the best available backend:

| Priority | Backend           | SSML Prosody | Cost      | Quality |
|----------|-------------------|--------------|-----------|---------|
| 1st      | Google Cloud TTS  | ✓ Full       | ~$4/1M chars | ★★★★★ |
| 2nd      | gTTS + pydub      | ✗ Post-proc  | Free      | ★★★☆☆  |
| 3rd      | pyttsx3           | ✗ Basic      | Free offline | ★★☆☆☆ |

---

## Common Issues

**ImportError: No module named 'transformers'**
```bash
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
```

**ffmpeg not found (pydub error)**
```bash
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: download from ffmpeg.org and add to PATH
```

**Model download stuck / slow**
The HuggingFace model downloads once (~300MB). If it stalls, check internet
connectivity. The model caches to `~/.cache/huggingface/hub/`.

**Port 5000 already in use**
```bash
PORT=5001 python app.py
# Then open http://localhost:5001
```

**Google Cloud TTS: credentials error**
Make sure `GOOGLE_APPLICATION_CREDENTIALS` in your `.env` points to the absolute
path of your service account JSON file, and the Cloud TTS API is enabled in GCP.

---

## API Reference

### POST /analyze
```json
// Request
{ "text": "I just got promoted!" }

// Response
{
  "success": true,
  "emotion": "joy",
  "intensity": 0.82,
  "confidence": 0.94,
  "compound": 0.78,
  "all_scores": { "joy": 0.94, "neutral": 0.03, "surprise": 0.02, ... },
  "voice_params": {
    "rate": "+20%", "pitch": "+2.9st",
    "volume": "+4.1dB", "break_ms": 0, "emphasis": "moderate"
  },
  "ssml": "<speak><prosody rate=\"+20%\" ...>I just got promoted!</prosody></speak>",
  "audio_url": "/audio/abc123def.mp3",
  "processing_ms": 1240
}
```

### GET /health
Returns subsystem status for all three components.

### GET /mapping
Returns the full emotion-to-voice mapping table.

### GET /audio/{filename}
Serves a generated MP3 audio file.
=======
# empathy-engine-
>>>>>>> c428673bd6bf3f6a35b66402a3f6d0c482eb6934
