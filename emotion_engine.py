"""
Emotion Engine — dual-layer emotion detection.
Layer 1: HuggingFace transformer for granular emotion labels (7 classes).
Layer 2: VADER for compound intensity score (-1 to +1).
Layer 3: Keyword fallback — activates when HF is unavailable and VADER is weak.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    label: str          # Primary emotion label
    intensity: float    # 0.0 to 1.0 (from VADER compound score magnitude)
    confidence: float   # 0.0 to 1.0 (from HuggingFace softmax score)
    compound: float     # Raw VADER compound (-1 to +1)
    all_scores: dict    # All HuggingFace emotion scores


class EmotionEngine:
    """
    Detects emotion from text using:
    - HuggingFace distilroberta model for label classification
    - VADER for intensity scaling
    - Keyword/lexicon fallback when HF is unavailable
    """

    # Map HuggingFace model labels to our standard labels
    LABEL_MAP = {
        "joy": "joy",
        "anger": "anger",
        "fear": "fear",
        "sadness": "sadness",
        "surprise": "surprise",
        "disgust": "disgust",
        "neutral": "neutral",
    }

    # --- Layer 3: Keyword lexicon for offline fallback ---
    # Each entry: (pattern, emotion, base_confidence)
    # Patterns are checked case-insensitively. Ordered from specific → general.
    KEYWORD_LEXICON = {
        "anger": [
            r"\b(furious|infuriated|enraged|livid|outraged|seething)\b",
            r"\b(angry|anger|rage|mad|hate|hatred|despise|loathe|damn|blast)\b",
            r"\b(how dare|screw this|this is ridiculous|i can't stand)\b",
            r"[!]{2,}",  # Multiple exclamation marks (mild signal)
        ],
        "fear": [
            r"\b(terrified|horrified|petrified|paralyzed with fear)\b",
            r"\b(afraid|scared|frightened|fearful|panic|horror|dread|dreading)\b",
            r"\b(anxious|anxiety|nervous|dreaded|nightmare|phobia|trembling)\b",
            r"\b(oh no|help me|i'm doomed|what if something)\b",
        ],
        "surprise": [
            r"\b(unbelievable|astonishing|astounding|flabbergasted|gobsmacked)\b",
            r"\b(wow|whoa|omg|oh my god|oh my goodness|shocking|shocked|stunned)\b",
            r"\b(can't believe|couldn't believe|never expected|out of nowhere)\b",
            r"\b(no way|seriously\?|wait what|what the|holy)\b",
        ],
        "disgust": [
            r"\b(disgusting|revolting|repulsive|nauseating|vile|appalling|repugnant)\b",
            r"\b(gross|nasty|yuck|eww|ew|foul|putrid|sickening|hideous)\b",
            r"\b(makes me sick|can't stomach|repelled|horrifying)\b",
        ],
        "joy": [
            r"\b(ecstatic|overjoyed|elated|thrilled|euphoric|exhilarated)\b",
            r"\b(happy|joyful|delighted|excited|wonderful|fantastic|amazing|love|loving)\b",
            r"\b(great|awesome|brilliant|terrific|superb|excellent|celebrate)\b",
            r"\b(so happy|so excited|best day|best ever|made my day)\b",
        ],
        "sadness": [
            # Tier 0 — explicit, intense sadness words
            r"\b(devastated|heartbroken|despondent|inconsolable|grief-stricken|bereaved|distraught)\b",
            # Tier 1 — common sadness vocabulary
            r"\b(sad|crying|cry|cried|tears|depressed|depression|miserable|lonely|loneliness|hopeless|grief|mourning|sorrow|sorrowful|anguish|melancholy|gloomy|gloom|wretched|suffering|suffer|struggling|struggle|meaningless|worthless)\b",
            # Tier 2 — common implicit phrases and feeling descriptors
            r"\b(feel(ing)?\s+(so\s+)?(down|low|lost|alone|empty|numb|worthless|terrible|awful|broken|heavy|drained|defeated|helpless|useless|forgotten|invisible))\b",
            r"\b(i('m| am)\s+(not okay|not fine|falling apart|struggling|hurting|broken|lost|so tired|exhausted|done))\b",
            r"\b(everything (is |feels?\s*)?(falling apart|so hard|(so )?heavy|so dark|pointless|meaningless|hopeless))\b",
            r"\b(miss (you|him|her|them|it)|i miss|lost everything|(can'?t|cannot) go on|feel empty|feel blue|feeling blue)\b",
            # Tier 3 — broader catch-all for low-arousal sad expressions
            r"\b(nothing (is |feels? )?(right|okay|good|the same|worth it)|nobody (understands?|cares?|loves? me)|no one (cares?|understands?))\b",
            r"\b(want to (cry|give up|disappear|hide)|wish (i was|i were|things were)|like giving up|(can'?t|cannot) take (it|this) anymore|don't see the point|what's the point)\b",
            r"\b(heart (is |feels? )?(heavy|broken|shattered)|feel(ing)? (hollow|void|numb|dead inside|so alone)|tired of (everything|life|trying))\b",
        ],
    }

    # Confidence scores assigned at each keyword tier (specific → general).
    # Sadness uses up to 7 tiers; all other emotions use 4.
    # Values beyond index 3 are intentionally lower (broad/implicit signals).
    KEYWORD_CONFIDENCE = [0.82, 0.72, 0.62, 0.55, 0.50, 0.48, 0.46]

    def __init__(self):
        self._hf_classifier = None
        self._vader_analyzer = None
        self._load_models()

    def _load_models(self):
        """Load models once at startup — expensive operation."""
        try:
            from transformers import pipeline
            logger.info("Loading HuggingFace emotion classifier...")
            self._hf_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,  # Return all scores
                truncation=True,
                max_length=512,
            )
            logger.info("HuggingFace model loaded successfully.")
        except Exception as e:
            logger.warning(f"HuggingFace model failed to load: {e}. Falling back to VADER + keywords.")
            self._hf_classifier = None

        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER analyzer loaded successfully.")
        except Exception as e:
            logger.error(f"VADER failed to load: {e}")
            self._vader_analyzer = None

    def analyze(self, text: str) -> EmotionResult:
        """
        Analyze text and return EmotionResult with label + intensity.
        """
        text = text.strip()
        if not text:
            return EmotionResult("neutral", 0.0, 1.0, 0.0, {"neutral": 1.0})

        # --- Layer 1: VADER for intensity ---
        compound = 0.0
        if self._vader_analyzer:
            scores = self._vader_analyzer.polarity_scores(text)
            compound = scores["compound"]

        intensity = min(abs(compound) * 1.3, 1.0)  # Scale up slightly, cap at 1.0

        # --- Layer 2: HuggingFace for label ---
        if self._hf_classifier:
            try:
                raw = self._hf_classifier(text)[0]  # List of {label, score}
                all_scores = {item["label"].lower(): item["score"] for item in raw}
                top = max(raw, key=lambda x: x["score"])
                label = self.LABEL_MAP.get(top["label"].lower(), "neutral")
                confidence = top["score"]

                # Intensity boost: if HF is confident and VADER is weak, use HF score
                if intensity < 0.2 and confidence > 0.7 and label != "neutral":
                    intensity = confidence * 0.6

                return EmotionResult(label, intensity, confidence, compound, all_scores)
            except Exception as e:
                logger.warning(f"HuggingFace inference failed: {e}. Using VADER + keyword fallback.")

        # --- Layer 3: Keyword-based fallback (runs when HF unavailable) ---
        keyword_result = self._keyword_detect(text)
        if keyword_result:
            kw_label, kw_confidence = keyword_result
            # Blend VADER intensity with keyword confidence
            # If VADER is strong, respect it; otherwise use keyword confidence
            effective_intensity = max(intensity, kw_confidence * 0.7)
            effective_intensity = min(effective_intensity, 1.0)
            # If VADER strongly disagrees (e.g. positive compound but anger keyword),
            # still trust the keyword label but moderate the intensity
            if abs(compound) > 0.5 and kw_label in ("anger", "disgust", "fear"):
                effective_intensity = max(effective_intensity, abs(compound) * 0.8)
            all_scores = {kw_label: kw_confidence, "neutral": 1.0 - kw_confidence}
            return EmotionResult(kw_label, effective_intensity, kw_confidence, compound, all_scores)

        # --- VADER-only fallback (when keywords don't match) ---
        if self._vader_analyzer and compound != 0.0:
            if compound >= 0.5:
                label, confidence = "joy", min(compound, 1.0)
            elif compound <= -0.5:
                label, confidence = "sadness", min(abs(compound), 1.0)
            elif compound > 0:
                label, confidence = "joy", compound
            elif compound < 0:
                label, confidence = "sadness", abs(compound)
            else:
                label, confidence = "neutral", 1.0
            return EmotionResult(label, intensity, confidence, compound, {label: confidence})

        # Final fallback: neutral
        return EmotionResult("neutral", 0.0, 1.0, 0.0, {"neutral": 1.0})

    def _keyword_detect(self, text: str) -> tuple | None:
        """
        Scan text against the keyword lexicon.
        Returns (emotion_label, confidence) for the best match, or None.

        Each emotion's patterns are ordered from most-specific to least-specific.
        The confidence decreases with tier index so precise matches score higher.
        """
        text_lower = text.lower()
        best_label = None
        best_confidence = 0.0

        for emotion, patterns in self.KEYWORD_LEXICON.items():
            for tier_index, pattern in enumerate(patterns):
                if re.search(pattern, text_lower):
                    confidence = self.KEYWORD_CONFIDENCE[
                        min(tier_index, len(self.KEYWORD_CONFIDENCE) - 1)
                    ]
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_label = emotion
                    break  # Stop at first matching tier for this emotion

        if best_label and best_confidence >= 0.45:
            return best_label, best_confidence

        return None

    def health_check(self) -> dict:
        return {
            "huggingface": self._hf_classifier is not None,
            "vader": self._vader_analyzer is not None,
            "keyword_fallback": True,  # Always available — no dependencies
        }