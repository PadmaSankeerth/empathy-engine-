"""
TTS Engine — Text-to-Speech synthesis with SSML support.

Backend priority (auto-detected at startup):
  1. Google Cloud TTS  — full SSML prosody, best quality (needs GCP credentials)
  2. gTTS + pydub      — good quality, needs internet, post-process modulation
  3. pyttsx3 + espeak  — fully offline, always works, native rate/volume control

The engine tries each in order and picks the first one that actually works.
"""

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from voice_mapper import VoiceParams

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Synthesizes speech from SSML or plain text.
    Auto-selects best available backend at startup.
    """

    # Google Cloud TTS Neural2 voices
    VOICE_MAP = {
        "default": "en-US-Neural2-F",
        "male":    "en-US-Neural2-D",
        "female":  "en-US-Neural2-F",
    }

    def __init__(self, voice_gender: str = "female"):
        self.voice_gender = voice_gender
        self._gclient     = None
        self._tts_module  = None
        self._backend     = self._detect_backend()
        logger.info(f"TTS backend selected: {self._backend}")

    # ── Backend Detection ────────────────────────────────────────────────────

    def _detect_backend(self) -> str:
        """
        Try each backend in priority order.
        Return the name of the first one that is fully functional.
        """
        # 1. Google Cloud TTS
        if self._try_google_cloud():
            return "google_cloud"

        # 2. gTTS (needs internet — verify connectivity before selecting)
        if self._try_gtts():
            return "gtts"

        # 3. pyttsx3 offline (always available after: pip install pyttsx3 + apt install espeak-ng)
        if self._try_pyttsx3():
            return "pyttsx3"

        logger.error(
            "NO TTS backend available. "
            "Fix: pip install pyttsx3 gTTS pydub  AND  apt install espeak-ng"
        )
        return "none"

    def _try_google_cloud(self) -> bool:
        try:
            from google.cloud import texttospeech
            client = texttospeech.TextToSpeechClient()
            self._gclient    = client
            self._tts_module = texttospeech
            logger.info("Google Cloud TTS: available")
            return True
        except Exception as e:
            logger.info(f"Google Cloud TTS: not available ({e})")
            return False

    def _try_gtts(self) -> bool:
        try:
            import gtts  # noqa: F401
            # Quick connectivity check — gTTS is useless without internet
            import urllib.request
            urllib.request.urlopen("https://translate.google.com", timeout=4)
            logger.info("gTTS: available (internet OK)")
            return True
        except ImportError:
            logger.info("gTTS: not installed")
            return False
        except Exception:
            logger.info("gTTS: installed but internet unreachable — skipping")
            return False

    def _try_pyttsx3(self) -> bool:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.stop()
            logger.info("pyttsx3: available (offline)")
            return True
        except ImportError:
            logger.info("pyttsx3: not installed — run: pip install pyttsx3")
            return False
        except RuntimeError as e:
            if "espeak" in str(e).lower():
                logger.info(
                    "pyttsx3: espeak-ng not installed — "
                    "run: sudo apt install espeak-ng"
                )
            else:
                logger.info(f"pyttsx3: init failed ({e})")
            return False
        except Exception as e:
            logger.info(f"pyttsx3: not available ({e})")
            return False

    # ── Public Interface ─────────────────────────────────────────────────────

    def synthesize(self, ssml: str, output_path: str, params: VoiceParams = None) -> bool:
        """
        Synthesize speech from an SSML string.
        Saves audio to output_path as MP3.
        Returns True on success, False on failure.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self._backend == "google_cloud":
            return self._synthesize_google(ssml, output_path)

        elif self._backend == "gtts":
            return self._synthesize_gtts(ssml, output_path, params)

        elif self._backend == "pyttsx3":
            return self._synthesize_pyttsx3(ssml, output_path, params)

        else:
            logger.error(
                "TTS backend is 'none'. "
                "Install a backend first:\n"
                "  pip install pyttsx3 gTTS pydub\n"
                "  sudo apt install espeak-ng"
            )
            return False

    # ── Google Cloud TTS ─────────────────────────────────────────────────────

    @staticmethod
    def _strip_emphasis(ssml: str) -> str:
        """
        Remove <emphasis> tags (keeping their content).
        Neural2 voices do NOT support <emphasis> and return a 400 API error
        when it is present — stripping the tags preserves the text while
        keeping all supported prosody/break markup intact.
        """
        ssml = re.sub(r'<emphasis[^>]*>', '', ssml)
        ssml = re.sub(r'</emphasis>', '', ssml)
        return ssml

    def _synthesize_google(self, ssml: str, output_path: str) -> bool:
        """Full SSML prosody via Google Cloud TTS Neural2 voices."""
        try:
            tts = self._tts_module
            ssml = self._strip_emphasis(ssml)   # Neural2 rejects <emphasis>
            synthesis_input = tts.SynthesisInput(ssml=ssml)
            voice = tts.VoiceSelectionParams(
                language_code="en-US",
                name=self.VOICE_MAP.get(self.voice_gender, self.VOICE_MAP["female"]),
            )
            audio_config = tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.MP3,
                effects_profile_id=["headphone-class-device"],
            )
            response = self._gclient.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )
            with open(output_path, "wb") as f:
                f.write(response.audio_content)
            logger.info(f"Google Cloud TTS: saved {output_path}")
            return True
        except Exception as e:
            logger.error(f"Google Cloud TTS failed: {e}")
            return False

    # ── gTTS + pydub ─────────────────────────────────────────────────────────

    def _synthesize_gtts(self, ssml: str, output_path: str, params: VoiceParams) -> bool:
        """
        gTTS: strips SSML tags, synthesizes plain text, then applies
        speed + volume modulation via pydub post-processing.
        """
        try:
            from gtts import gTTS
            plain_text = self._strip_ssml(ssml)
            tts = gTTS(text=plain_text, lang="en", slow=False)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            tts.save(tmp_path)

            if params:
                self._apply_pydub_modulation(tmp_path, output_path, params)
            else:
                shutil.copy(tmp_path, output_path)

            os.unlink(tmp_path)
            logger.info(f"gTTS: saved {output_path}")
            return True

        except Exception as e:
            logger.error(f"gTTS synthesis failed: {e}")
            return False

    def _apply_pydub_modulation(self, input_path: str, output_path: str, params: VoiceParams):
        """Adjust volume and playback speed on an existing MP3 file."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(input_path)

            # Volume (dB)
            if params.volume and params.volume != "0dB":
                db_val = float(params.volume.replace("dB", "").replace("+", ""))
                audio = audio + db_val

            # Speed (rate)
            if params.rate and params.rate != "0%":
                rate_pct    = float(params.rate.replace("%", "").replace("+", ""))
                speed       = max(0.5, min(1.0 + rate_pct / 100.0, 2.0))
                audio       = self._change_speed(audio, speed)

            audio.export(output_path, format="mp3")

        except Exception as e:
            logger.warning(f"pydub modulation failed ({e}) — saving unmodulated audio")
            shutil.copy(input_path, output_path)

    @staticmethod
    def _change_speed(audio, speed: float):
        """Shift playback speed by resampling the frame rate."""
        altered = audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": int(audio.frame_rate * speed)},
        )
        return altered.set_frame_rate(audio.frame_rate)

    # ── pyttsx3 (offline) ────────────────────────────────────────────────────

    def _synthesize_pyttsx3(self, ssml: str, output_path: str, params: VoiceParams) -> bool:
        """
        Fully offline synthesis via pyttsx3 + espeak-ng.
        Applies rate and volume modulation natively via engine properties.
        Converts WAV → MP3 via pydub (ffmpeg).
        """
        try:
            import pyttsx3
            plain_text = self._strip_ssml(ssml)

            # Always create a brand-new engine instance per call.
            # pyttsx3's espeak driver has a known weakref bug when the same
            # engine object is reused across calls — a fresh init avoids it.
            try:
                engine = pyttsx3.init()
            except Exception:
                import gc; gc.collect()
                engine = pyttsx3.init()

            base_rate = engine.getProperty("rate") or 200  # words per minute

            # --- Apply rate ---
            if params and params.rate and params.rate != "0%":
                rate_pct = float(params.rate.replace("%", "").replace("+", ""))
                new_rate = int(base_rate * (1.0 + rate_pct / 100.0))
                engine.setProperty("rate", max(80, min(new_rate, 420)))

            # --- Apply volume ---
            if params and params.volume and params.volume != "0dB":
                db_val = float(params.volume.replace("dB", "").replace("+", ""))
                # pyttsx3 volume is 0.0–1.0; convert dB offset from a 0.85 baseline
                vol = min(1.0, max(0.1, 0.85 + db_val / 20.0))
                engine.setProperty("volume", vol)

            # Save as WAV (pyttsx3 native format)
            wav_path = output_path.replace(".mp3", ".wav")
            engine.save_to_file(plain_text, wav_path)
            engine.runAndWait()
            engine.stop()
            del engine

            # Small delay — lets espeak's background thread finish writing
            # the WAV file before we try to read it. Without this, rapid
            # sequential calls occasionally produce a 0-byte file.
            import time
            time.sleep(0.08)

            if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                logger.error(f"pyttsx3 produced no WAV output at {wav_path}")
                return False

            # Convert WAV → MP3
            try:
                from pydub import AudioSegment
                AudioSegment.from_wav(wav_path).export(output_path, format="mp3")
                os.unlink(wav_path)
            except Exception as conv_err:
                logger.warning(f"WAV→MP3 conversion failed ({conv_err}) — serving WAV as-is")
                shutil.move(wav_path, output_path)

            logger.info(f"pyttsx3: saved {output_path}")
            return True

        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            return False

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_ssml(ssml: str) -> str:
        """Remove all XML/SSML tags and return clean plain text."""
        text = re.sub(r"<[^>]+>", " ", ssml)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def health_check(self) -> dict:
        return {
            "backend":      self._backend,
            "available":    self._backend != "none",
            "supports_ssml": self._backend == "google_cloud",
        }