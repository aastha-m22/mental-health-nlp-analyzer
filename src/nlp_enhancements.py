"""
src/nlp_enhancements.py
-----------------------
Advanced NLP enrichment features:
  1. Named Entity Recognition (spaCy) — detect references to people,
     places, organisations in mental health posts
  2. Emotion lexicon scoring — maps tokens to emotion categories
  3. Linguistic pattern detection — hedging, absolutist language,
     negation density (all clinically relevant signals)
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# ── Try importing spaCy (optional dependency) ─────────────────────────
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. NER features will be skipped.")


# ══════════════════════════════════════════════════════════════════
# NER FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════
class NERFeatureExtractor:
    """
    Uses spaCy to extract named entity counts as features.

    Why NER for mental health?
    - Posts mentioning PERSON entities may reflect interpersonal conflict
    - ORG/GPE entities can signal work/social stressors
    - DATE/TIME references may indicate rumination patterns
    """

    ENTITY_TYPES = ["PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY", "EVENT"]

    def __init__(self, model: str = "en_core_web_sm"):
        self.model_name = model
        self.nlp = None
        self._load_model()

    def _load_model(self):
        if not SPACY_AVAILABLE:
            return
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"spaCy model '{self.model_name}' loaded.")
        except OSError:
            logger.warning(
                f"spaCy model '{self.model_name}' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            self.nlp = None

    def extract(self, text: str) -> Dict[str, int]:
        """Return entity type counts for a single text."""
        counts = {ent_type: 0 for ent_type in self.ENTITY_TYPES}
        counts["total_entities"] = 0
        if self.nlp is None:
            return counts
        doc = self.nlp(text[:1000])   # cap length for performance
        for ent in doc.ents:
            if ent.label_ in counts:
                counts[ent.label_] += 1
            counts["total_entities"] += 1
        return counts

    def batch_extract(self, texts: List[str]) -> np.ndarray:
        """Extract entity features for a list of texts → (n, n_features) array."""
        results = [self.extract(t) for t in texts]
        keys = self.ENTITY_TYPES + ["total_entities"]
        return np.array([[r[k] for k in keys] for r in results])

    @property
    def feature_names(self) -> List[str]:
        return [f"ner_{e.lower()}" for e in self.ENTITY_TYPES] + ["ner_total"]


# ══════════════════════════════════════════════════════════════════
# EMOTION LEXICON SCORER  (rule-based, no external model needed)
# ══════════════════════════════════════════════════════════════════

# Minimal curated lexicons — extend with NRC Emotion Lexicon in production
EMOTION_LEXICONS: Dict[str, List[str]] = {
    "sadness": [
        "sad", "cry", "tear", "grief", "mourn", "sorrow", "depress",
        "hopeless", "lonely", "alone", "empty", "numb", "hurt", "pain",
        "miss", "lost", "broken", "hollow", "dark", "gloomy", "miserable",
        "worthless", "useless", "fail", "disappoint", "regret",
    ],
    "fear_anxiety": [
        "afraid", "fear", "worry", "anxious", "panic", "terror", "dread",
        "nervous", "frighten", "scare", "phobia", "uneasy", "apprehensive",
        "overwhelm", "catastrophe", "danger", "threat", "unsafe", "alarm",
        "racing", "tremble", "shake", "sweat",
    ],
    "anger": [
        "angry", "rage", "furious", "hate", "resent", "frustrat",
        "irritat", "annoy", "bitter", "hostile", "aggress", "mad",
    ],
    "joy": [
        "happy", "joy", "excit", "delight", "love", "wonderful", "great",
        "amazing", "fantastic", "thrilled", "content", "grateful", "glad",
        "cheer", "laugh", "smile", "fun", "enjoy",
    ],
    "disgust": [
        "disgust", "gross", "sick", "repuls", "revolting", "nausea",
        "loath", "abhor",
    ],
}


def score_emotions(tokens: List[str]) -> Dict[str, float]:
    """
    Compute normalised emotion scores for a tokenized text.
    Returns proportion of tokens matching each emotion lexicon.
    """
    n = max(len(tokens), 1)
    scores = {}
    for emotion, lexicon in EMOTION_LEXICONS.items():
        count = sum(
            1 for tok in tokens
            if any(tok.startswith(lex[:4]) for lex in lexicon)   # stem-aware matching
        )
        scores[emotion] = count / n
    return scores


def batch_emotion_scores(tokenized_texts: List[List[str]]) -> np.ndarray:
    """Compute emotion features for all texts → (n, 5) array."""
    emotions = list(EMOTION_LEXICONS.keys())
    results = []
    for tokens in tokenized_texts:
        s = score_emotions(tokens)
        results.append([s[e] for e in emotions])
    return np.array(results)

EMOTION_FEATURE_NAMES = [f"emotion_{e}" for e in EMOTION_LEXICONS.keys()]


# ══════════════════════════════════════════════════════════════════
# LINGUISTIC PATTERN DETECTOR
# ══════════════════════════════════════════════════════════════════

# Absolutist thinking is a clinically validated marker of depression
ABSOLUTIST_WORDS = [
    "always", "never", "nothing", "everything", "everyone", "nobody",
    "completely", "totally", "absolutely", "forever", "impossible",
    "worthless", "perfect", "terrible", "awful", "worst",
]

HEDGING_WORDS = [
    "maybe", "perhaps", "probably", "might", "could", "possibly",
    "sometimes", "often", "usually", "guess", "think", "feel like",
    "seem", "appear", "sort of", "kind of",
]

FIRST_PERSON_PATTERN = re.compile(r"\b(i|me|my|myself|mine)\b", re.IGNORECASE)
NEGATION_PATTERN = re.compile(
    r"\b(not|no|never|neither|nobody|nothing|nowhere|nor|"
    r"don't|doesn't|didn't|won't|wouldn't|can't|cannot|couldn't|"
    r"shouldn't|isn't|aren't|wasn't|weren't|hardly|barely|scarcely)\b",
    re.IGNORECASE
)


def extract_linguistic_patterns(text: str) -> Dict[str, float]:
    """
    Extract clinically-motivated linguistic pattern features.
    """
    tokens = text.lower().split()
    n = max(len(tokens), 1)

    absolutist_count = sum(1 for t in tokens if t in ABSOLUTIST_WORDS)
    hedging_count = sum(1 for t in tokens if t in HEDGING_WORDS)
    first_person = len(FIRST_PERSON_PATTERN.findall(text))
    negation = len(NEGATION_PATTERN.findall(text))

    return {
        "absolutist_ratio":   absolutist_count / n,
        "hedging_ratio":      hedging_count / n,
        "first_person_ratio": first_person / n,
        "negation_ratio":     negation / n,
        "text_length_norm":   min(n / 100, 1.0),   # normalised length
    }


def batch_linguistic_patterns(texts: List[str]) -> np.ndarray:
    keys = ["absolutist_ratio", "hedging_ratio", "first_person_ratio",
            "negation_ratio", "text_length_norm"]
    results = [extract_linguistic_patterns(t) for t in texts]
    return np.array([[r[k] for k in keys] for r in results])

LINGUISTIC_PATTERN_NAMES = [
    "absolutist_ratio", "hedging_ratio", "first_person_ratio",
    "negation_ratio", "text_length_norm",
]


# ══════════════════════════════════════════════════════════════════
# COMBINED NLP ENHANCEMENT FEATURES
# ══════════════════════════════════════════════════════════════════
def get_all_nlp_features(
    texts: List[str],
    tokenized_texts: List[List[str]],
    use_ner: bool = False,   # set True if spaCy model is available
) -> Tuple[np.ndarray, List[str]]:
    """
    Concatenate all NLP enhancement features into one matrix.

    Returns:
        features: np.ndarray of shape (n_samples, n_features)
        names:    list of feature names
    """
    feature_parts = []
    feature_names = []

    # Emotion scores
    em = batch_emotion_scores(tokenized_texts)
    feature_parts.append(em)
    feature_names.extend(EMOTION_FEATURE_NAMES)

    # Linguistic patterns
    lp = batch_linguistic_patterns(texts)
    feature_parts.append(lp)
    feature_names.extend(LINGUISTIC_PATTERN_NAMES)

    # NER (optional)
    if use_ner:
        ner = NERFeatureExtractor()
        if ner.nlp is not None:
            ner_feats = ner.batch_extract(texts)
            feature_parts.append(ner_feats)
            feature_names.extend(ner.feature_names)

    features = np.hstack(feature_parts)
    return features, feature_names


if __name__ == "__main__":
    samples = [
        "i never feel anything it is always dark inside nothing matters",
        "had an amazing day hiking with friends loved every moment",
        "heart racing cannot breathe everyone must think i am crazy",
    ]
    tokenized = [s.split() for s in samples]

    feats, names = get_all_nlp_features(samples, tokenized, use_ner=False)
    print("NLP enhancement feature names:", names)
    print("Feature matrix shape:", feats.shape)
    for i, s in enumerate(samples):
        print(f"\nText: {s[:60]}…")
        for name, val in zip(names, feats[i]):
            if val > 0:
                print(f"  {name}: {val:.4f}")
