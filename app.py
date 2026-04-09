"""
Empathy Engine — Flask Application
The Empathy Engine: Giving AI a Human Voice

Routes:
  GET  /           → Web UI
  POST /analyze    → Analyze text, synthesize speech, return JSON
  GET  /health     → Health check for all subsystems
  GET  /mapping    → Emotion-to-voice mapping table
  GET  /audio/<fn> → Serve generated audio files
"""

import logging
import os
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from emotion_engine import EmotionEngine
from ssml_builder import SSMLBuilder
from tts_engine import TTSEngine
from voice_mapper import VoiceMapper

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("empathy_engine")

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

AUDIO_DIR = Path("static/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ── Load Models (once at startup) ─────────────────────────────────────────────
logger.info("Initializing Empathy Engine subsystems...")
emotion_engine = EmotionEngine()
voice_mapper   = VoiceMapper()
ssml_builder   = SSMLBuilder()
tts_engine     = TTSEngine(voice_gender="female")
logger.info("All subsystems ready.")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main pipeline endpoint.

    Request JSON:  { "text": "I am so happy today!" }
    Response JSON: {
        "success": true,
        "text": "...",
        "emotion": "joy",
        "intensity": 0.82,
        "confidence": 0.94,
        "compound": 0.78,
        "all_scores": { "joy": 0.94, "neutral": 0.03, ... },
        "voice_params": { "rate": "+20%", "pitch": "+2.9st", ... },
        "ssml": "<speak>...</speak>",
        "audio_url": "/audio/abc123.mp3",
        "processing_ms": 843
    }
    """
    start_time = time.time()

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"success": False, "error": "No text provided."}), 400

    if len(text) > 2000:
        return jsonify({"success": False, "error": "Text too long (max 2000 characters)."}), 400

    try:
        # Step 1: Detect emotion
        emotion_result = emotion_engine.analyze(text)
        logger.info(
            f"Emotion: {emotion_result.label} | "
            f"Intensity: {emotion_result.intensity:.2f} | "
            f"Confidence: {emotion_result.confidence:.2f}"
        )

        # Step 2: Map to voice parameters
        voice_params = voice_mapper.map(emotion_result.label, emotion_result.intensity)

        # Step 3: Build SSML
        ssml = ssml_builder.build(text, voice_params)
        logger.info(f"SSML: {ssml[:120]}...")

        # Step 4: Synthesize speech
        filename = f"{uuid.uuid4().hex}.mp3"
        output_path = str(AUDIO_DIR / filename)
        success = tts_engine.synthesize(ssml, output_path, voice_params)

        if not success:
            return jsonify({"success": False, "error": "TTS synthesis failed."}), 500

        elapsed_ms = int((time.time() - start_time) * 1000)

        return jsonify({
            "success": True,
            "text": text,
            "emotion": emotion_result.label,
            "intensity": round(emotion_result.intensity, 3),
            "confidence": round(emotion_result.confidence, 3),
            "compound": round(emotion_result.compound, 3),
            "all_scores": {k: round(v, 3) for k, v in emotion_result.all_scores.items()},
            "voice_params": {
                "rate": voice_params.rate,
                "pitch": voice_params.pitch,
                "volume": voice_params.volume,
                "break_ms": voice_params.break_ms,
                "emphasis": voice_params.emphasis,
            },
            "ssml": ssml,
            "audio_url": f"/audio/{filename}",
            "processing_ms": elapsed_ms,
        })

    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serve generated audio files."""
    return send_from_directory(AUDIO_DIR, filename, mimetype="audio/mpeg")


@app.route("/health")
def health():
    """Health check for all subsystems."""
    em_health = emotion_engine.health_check()
    tts_health = tts_engine.health_check()
    all_ok = em_health["vader"] and tts_health["available"]
    return jsonify({
        "status": "ok" if all_ok else "degraded",
        "emotion_engine": em_health,
        "tts_engine": tts_health,
    }), 200 if all_ok else 207


@app.route("/mapping")
def mapping():
    """Return the emotion-to-voice mapping table."""
    return jsonify({"mapping": voice_mapper.get_mapping_table()})


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Empathy Engine on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
