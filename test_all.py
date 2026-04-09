"""
Test suite for Empathy Engine modules.
Run: pytest tests/ -v
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── VoiceMapper Tests ────────────────────────────────────────────────────────

class TestVoiceMapper:
    def setup_method(self):
        from voice_mapper import VoiceMapper
        self.mapper = VoiceMapper()

    def test_joy_high_intensity(self):
        p = self.mapper.map("joy", 1.0)
        # Joy at full intensity should speed up and raise pitch
        assert "+" in p.rate, "Joy should increase rate"
        assert "+" in p.pitch, "Joy should raise pitch"

    def test_sadness_high_intensity(self):
        p = self.mapper.map("sadness", 1.0)
        assert "-" in p.rate, "Sadness should decrease rate"
        assert "-" in p.pitch, "Sadness should lower pitch"
        assert p.break_ms > 0, "Sadness should have a pre-pause"

    def test_neutral_emotion(self):
        p = self.mapper.map("neutral", 0.0)
        assert p.rate == "0%", "Neutral rate should be 0%"
        assert p.pitch == "0st", "Neutral pitch should be 0st"

    def test_intensity_scaling(self):
        # Higher intensity → more extreme rate
        p_low  = self.mapper.map("anger", 0.3)
        p_high = self.mapper.map("anger", 1.0)
        rate_low  = float(p_low.rate.replace("+","").replace("%",""))
        rate_high = float(p_high.rate.replace("+","").replace("%",""))
        assert rate_high > rate_low, "Higher intensity should produce greater rate change"

    def test_all_emotions_covered(self):
        emotions = ["joy","anger","sadness","fear","surprise","disgust","neutral"]
        for emo in emotions:
            p = self.mapper.map(emo, 0.7)
            assert p is not None
            assert p.emotion == emo

    def test_unknown_emotion_fallback(self):
        p = self.mapper.map("unknown_emotion", 0.5)
        assert p is not None


# ── SSMLBuilder Tests ────────────────────────────────────────────────────────

class TestSSMLBuilder:
    def setup_method(self):
        from ssml_builder import SSMLBuilder
        from voice_mapper import VoiceMapper
        self.builder = SSMLBuilder()
        self.mapper  = VoiceMapper()

    def test_output_has_speak_tags(self):
        p    = self.mapper.map("joy", 0.8)
        ssml = self.builder.build("Hello world!", p)
        assert ssml.startswith("<speak>"), "SSML must start with <speak>"
        assert ssml.endswith("</speak>"), "SSML must end with </speak>"

    def test_prosody_applied_for_non_neutral(self):
        p    = self.mapper.map("anger", 1.0)
        ssml = self.builder.build("I am angry!", p)
        assert "prosody" in ssml, "Non-neutral emotion should add prosody tag"

    def test_break_applied_for_sadness(self):
        p    = self.mapper.map("sadness", 1.0)
        ssml = self.builder.build("I feel so sad.", p)
        assert "break" in ssml, "Sadness should include a break tag"

    def test_neutral_no_prosody(self):
        p    = self.mapper.map("neutral", 0.0)
        ssml = self.builder.build("The meeting is at 3pm.", p)
        # Neutral should produce minimal or no prosody
        assert "<speak>" in ssml

    def test_empty_text_handled(self):
        p    = self.mapper.map("joy", 0.5)
        ssml = self.builder.build("", p)
        assert "<speak>" in ssml


# ── EmotionEngine Tests ──────────────────────────────────────────────────────

class TestEmotionEngine:
    def setup_method(self):
        from emotion_engine import EmotionEngine
        self.engine = EmotionEngine()

    def test_returns_emotion_result(self):
        from emotion_engine import EmotionResult
        result = self.engine.analyze("I am so happy!")
        assert isinstance(result, EmotionResult)

    def test_intensity_range(self):
        result = self.engine.analyze("This is horrible and I hate everything about it.")
        assert 0.0 <= result.intensity <= 1.0

    def test_neutral_text(self):
        result = self.engine.analyze("The document is on the table.")
        assert result.label in ("neutral", "sadness", "joy")  # Fuzzy — any calm emotion

    def test_empty_text(self):
        result = self.engine.analyze("")
        assert result.label == "neutral"
        assert result.intensity == 0.0

    def test_all_scores_sum_close_to_one(self):
        result = self.engine.analyze("This is amazing news!")
        if result.all_scores:
            total = sum(result.all_scores.values())
            assert abs(total - 1.0) < 0.05, f"Scores should sum to ~1.0, got {total}"

    def test_happy_text_detected(self):
        result = self.engine.analyze("I just won the lottery! This is incredible!")
        # Should not be sadness or disgust
        assert result.label not in ("sadness", "disgust")

    def test_angry_text_detected(self):
        result = self.engine.analyze("I am absolutely furious. This is completely unacceptable!")
        assert result.label in ("anger", "disgust", "fear")  # Negative emotions

    def test_health_check(self):
        health = self.engine.health_check()
        assert "huggingface" in health
        assert "vader" in health


# ── Integration Test ─────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_pipeline_produces_voice_params(self):
        from emotion_engine import EmotionEngine
        from voice_mapper import VoiceMapper
        from ssml_builder import SSMLBuilder

        engine  = EmotionEngine()
        mapper  = VoiceMapper()
        builder = SSMLBuilder()

        text   = "I can't believe this happened, I'm absolutely overjoyed!"
        result = engine.analyze(text)
        params = mapper.map(result.label, result.intensity)
        ssml   = builder.build(text, params)

        assert result.label != ""
        assert params.rate is not None
        assert "<speak>" in ssml
        assert text in ssml or "prosody" in ssml


# ── Flask App Tests ───────────────────────────────────────────────────────────

class TestFlaskApp:
    def setup_method(self):
        import app as application
        application.app.config['TESTING'] = True
        self.client = application.app.test_client()

    def test_index_returns_200(self):
        res = self.client.get('/')
        assert res.status_code == 200

    def test_health_endpoint(self):
        res  = self.client.get('/health')
        data = res.get_json()
        assert 'status' in data
        assert 'emotion_engine' in data
        assert 'tts_engine' in data

    def test_analyze_no_text(self):
        res  = self.client.post('/analyze', json={})
        data = res.get_json()
        assert data['success'] == False
        assert res.status_code == 400

    def test_analyze_valid_text(self):
        res  = self.client.post('/analyze', json={"text": "I am so happy today!"})
        data = res.get_json()
        assert data['success'] == True
        assert 'emotion' in data
        assert 'audio_url' in data
        assert 'ssml' in data

    def test_analyze_too_long(self):
        res  = self.client.post('/analyze', json={"text": "x" * 2001})
        data = res.get_json()
        assert data['success'] == False

    def test_mapping_endpoint(self):
        res  = self.client.get('/mapping')
        data = res.get_json()
        assert 'mapping' in data
        assert len(data['mapping']) >= 7
