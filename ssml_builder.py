"""
SSML Builder — converts text + VoiceParams into SSML markup.

Supports:
  - <prosody> for rate / pitch / volume
  - <break> for emotional pauses
  - <emphasis> for vocal stress
  - Optional sentence-level annotation
"""

import re
from voice_mapper import VoiceParams


class SSMLBuilder:
    """
    Builds SSML strings for Google Cloud TTS.

    Google Cloud TTS SSML reference:
      https://cloud.google.com/text-to-speech/docs/ssml
    """

    def build(self, text: str, params: VoiceParams) -> str:
        """
        Build SSML from text + voice parameters.

        Returns a full SSML document string.
        """
        text = text.strip()
        inner = self._apply_emphasis(text, params)
        inner = self._apply_prosody(inner, params)
        inner = self._apply_break(inner, params)
        return f"<speak>{inner}</speak>"

    def _apply_break(self, text: str, params: VoiceParams) -> str:
        """Inject a leading pause for reflective emotions (sadness, fear)."""
        if params.break_ms > 0:
            return f'<break time="{params.break_ms}ms"/>{text}'
        return text

    def _apply_emphasis(self, text: str, params: VoiceParams) -> str:
        """
        Apply <emphasis> to the last meaningful word for surprise/joy,
        or wrap the whole text for anger/sadness.
        """
        level = params.emphasis
        if level == "none":
            return text

        emotion = params.emotion

        if emotion in ("surprise", "joy") and params.intensity > 0.5:
            # Emphasize the last word
            words = text.rstrip(".!?").split()
            if len(words) > 1:
                last_word = words[-1]
                rest = " ".join(words[:-1])
                punctuation = self._trailing_punctuation(text)
                return f'{rest} <emphasis level="{level}">{last_word}</emphasis>{punctuation}'

        elif emotion in ("anger",) and params.intensity > 0.6:
            # Emphasize the whole sentence
            return f'<emphasis level="{level}">{text}</emphasis>'

        return text

    def _apply_prosody(self, text: str, params: VoiceParams) -> str:
        """Wrap text in <prosody> with rate, pitch, volume."""
        attrs = []

        if params.rate != "0%":
            attrs.append(f'rate="{params.rate}"')
        if params.pitch != "0st":
            attrs.append(f'pitch="{params.pitch}"')
        if params.volume != "0dB":
            attrs.append(f'volume="{params.volume}"')

        if not attrs:
            return text

        attr_str = " ".join(attrs)
        return f"<prosody {attr_str}>{text}</prosody>"

    @staticmethod
    def _trailing_punctuation(text: str) -> str:
        match = re.search(r'[.!?,;:]+$', text)
        return match.group(0) if match else ""

    def preview(self, text: str, params: VoiceParams) -> str:
        """Return a human-readable SSML preview (indented)."""
        ssml = self.build(text, params)
        # Basic pretty-print
        ssml = ssml.replace("><", ">\n  <")
        ssml = ssml.replace("<speak>\n  ", "<speak>\n  ")
        return ssml
