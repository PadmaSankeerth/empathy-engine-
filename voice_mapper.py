"""
Voice Mapper — maps emotion labels + intensity to SSML vocal parameters.

SSML prosody parameters for Google Cloud TTS:
  rate:   percentage string e.g. "+20%", "-15%", "0%"
  pitch:  semitone string   e.g. "+3st", "-2st", "0st"
  volume: dB or keyword     e.g. "+6dB", "-3dB", "medium"
"""

from dataclasses import dataclass


@dataclass
class VoiceParams:
    rate: str       # Speaking rate modifier
    pitch: str      # Pitch modifier in semitones
    volume: str     # Volume modifier in dB
    break_ms: int   # Pre-sentence pause in milliseconds
    emphasis: str   # SSML emphasis level: none | reduced | moderate | strong
    emotion: str    # Original emotion label
    intensity: float  # 0.0 to 1.0


class VoiceMapper:
    """
    Maps (emotion, intensity) → VoiceParams for SSML generation.

    Each emotion has a BASE configuration. The intensity scalar (0.0–1.0)
    linearly interpolates between neutral (0%) and the full base value (100%).

    Example: joy base rate = +25%, intensity = 0.5 → rate = +12.5% → "+13%"
    """

    # Base parameter values at full intensity (1.0)
    # (rate_pct, pitch_st, volume_db, break_ms, emphasis)
    EMOTION_BASE = {
        "joy": {
            "rate_pct": +25,
            "pitch_st": +3.5,
            "volume_db": +5,
            "break_ms": 0,
            "emphasis": "moderate",
        },
        "anger": {
            "rate_pct": +22,
            "pitch_st": +2.0,
            "volume_db": +8,
            "break_ms": 0,
            "emphasis": "strong",
        },
        "sadness": {
            "rate_pct": -28,
            "pitch_st": -3.0,
            "volume_db": -4,
            "break_ms": 400,
            "emphasis": "reduced",
        },
        "fear": {
            "rate_pct": +18,
            "pitch_st": +2.5,
            "volume_db": -2,
            "break_ms": 200,
            "emphasis": "moderate",
        },
        "surprise": {
            "rate_pct": +8,
            "pitch_st": +4.5,
            "volume_db": +4,
            "break_ms": 0,
            "emphasis": "strong",
        },
        "disgust": {
            "rate_pct": -12,
            "pitch_st": -2.0,
            "volume_db": +2,
            "break_ms": 100,
            "emphasis": "moderate",
        },
        "neutral": {
            "rate_pct": 0,
            "pitch_st": 0.0,
            "volume_db": 0,
            "break_ms": 0,
            "emphasis": "none",
        },
    }

    # Minimum intensity — ensure some modulation is always visible
    MIN_INTENSITY = 0.25

    def map(self, emotion: str, intensity: float) -> VoiceParams:
        """
        Map emotion + intensity to VoiceParams.

        intensity is clamped to [MIN_INTENSITY, 1.0] for non-neutral emotions
        so there is always some detectable modulation.
        """
        emotion = emotion.lower()
        base = self.EMOTION_BASE.get(emotion, self.EMOTION_BASE["neutral"])

        # Clamp intensity
        if emotion == "neutral":
            effective_intensity = 0.0
        else:
            effective_intensity = max(intensity, self.MIN_INTENSITY)
            effective_intensity = min(effective_intensity, 1.0)

        # Scale parameters by intensity
        rate_pct  = base["rate_pct"]  * effective_intensity
        pitch_st  = base["pitch_st"]  * effective_intensity
        volume_db = base["volume_db"] * effective_intensity
        break_ms  = int(base["break_ms"] * effective_intensity)
        emphasis  = base["emphasis"] if effective_intensity > 0.3 else "none"

        # Format SSML strings
        rate_str   = self._format_pct(rate_pct)
        pitch_str  = self._format_st(pitch_st)
        volume_str = self._format_db(volume_db)

        return VoiceParams(
            rate=rate_str,
            pitch=pitch_str,
            volume=volume_str,
            break_ms=break_ms,
            emphasis=emphasis,
            emotion=emotion,
            intensity=round(effective_intensity, 3),
        )

    @staticmethod
    def _format_pct(value: float) -> str:
        rounded = round(value)
        if rounded > 0:
            return f"+{rounded}%"
        elif rounded < 0:
            return f"{rounded}%"
        return "0%"

    @staticmethod
    def _format_st(value: float) -> str:
        rounded = round(value, 1)
        if rounded > 0:
            return f"+{rounded}st"
        elif rounded < 0:
            return f"{rounded}st"
        return "0st"

    @staticmethod
    def _format_db(value: float) -> str:
        rounded = round(value, 1)
        if rounded > 0:
            return f"+{rounded}dB"
        elif rounded < 0:
            return f"{rounded}dB"
        return "0dB"

    def get_mapping_table(self) -> list:
        """Return the full mapping table for display/debugging."""
        rows = []
        for emotion, base in self.EMOTION_BASE.items():
            params = self.map(emotion, 1.0)
            rows.append({
                "emotion": emotion,
                "rate": params.rate,
                "pitch": params.pitch,
                "volume": params.volume,
                "break_ms": params.break_ms,
                "emphasis": params.emphasis,
            })
        return rows
