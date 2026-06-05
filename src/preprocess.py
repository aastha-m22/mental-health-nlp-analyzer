"""
src/preprocess.py
-----------------
Full text cleaning and normalization pipeline for social media text.
Handles: noise, URLs, emojis, slang contractions, stopwords,
         lemmatization/stemming for mental health NLP.
"""

import re
import string
import logging
from typing import List, Optional

import nltk
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK assets (safe to call repeatedly)
def download_nltk_assets():
    assets = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger",
              "punkt_tab", "omw-1.4"]
    for asset in assets:
        try:
            nltk.download(asset, quiet=True)
        except Exception:
            pass

download_nltk_assets()

logger = logging.getLogger(__name__)

# Mental-health–relevant negation words we keep even though they're stopwords
NEGATION_PRESERVE = {
    "no", "not", "never", "nor", "neither", "nobody", "nothing",
    "nowhere", "hardly", "barely", "scarcely", "don't", "doesn't",
    "didn't", "won't", "wouldn't", "can't", "cannot", "couldn't",
    "shouldn't", "isn't", "aren't", "wasn't", "weren't"
}

# Mental-health–specific slang normalization
SLANG_MAP = {
    r"\bngl\b": "not gonna lie",
    r"\bidk\b": "i do not know",
    r"\bfr\b": "for real",
    r"\bsmh\b": "shaking my head",
    r"\bbc\b": "because",
    r"\brly\b": "really",
    r"\bsmth\b": "something",
    r"\bbtw\b": "by the way",
    r"\bimo\b": "in my opinion",
    r"\bafaik\b": "as far as i know",
    r"\bbrb\b": "be right back",
    r"\blmao\b": "laughing",
    r"\blol\b": "laughing",
    r"\bomg\b": "oh my god",
    r"\bwtf\b": "what the",
    r"\baf\b": "very",
    r"\blit\b": "exciting",
    r"\bslay\b": "doing great",
    r"\bvibes\b": "feelings",
    r"\bfeel\b": "feel",
}


class TextPreprocessor:
    """
    Full NLP preprocessing pipeline for social media mental health text.

    Steps (configurable):
      1. Lowercase
      2. Expand contractions  (don't → do not)
      3. Normalise slang
      4. Remove URLs
      5. Remove email addresses
      6. Remove mentions / hashtags (optionally keep hashtag text)
      7. Remove emojis / special unicode
      8. Remove punctuation (keep negation-relevant punctuation optionally)
      9. Remove digits
     10. Tokenise
     11. Remove stopwords  (preserving negation words)
     12. Lemmatise / stem
     13. Remove very short tokens
    """

    def __init__(
        self,
        lowercase: bool = True,
        expand_contractions: bool = True,
        normalize_slang: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        keep_hashtag_text: bool = True,
        remove_emojis: bool = True,
        remove_digits: bool = True,
        remove_stopwords: bool = True,
        preserve_negation: bool = True,
        method: str = "lemmatize",   # "lemmatize" | "stem" | "none"
        min_token_len: int = 2,
    ):
        self.lowercase = lowercase
        self.expand_contractions = expand_contractions
        self.normalize_slang = normalize_slang
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.keep_hashtag_text = keep_hashtag_text
        self.remove_emojis = remove_emojis
        self.remove_digits = remove_digits
        self.remove_stopwords = remove_stopwords
        self.preserve_negation = preserve_negation
        self.method = method
        self.min_token_len = min_token_len

        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        base_stops = set(stopwords.words("english"))
        if preserve_negation:
            self.stop_words = base_stops - NEGATION_PRESERVE
        else:
            self.stop_words = base_stops

    # ── Individual cleaning steps ──────────────────────────────────────

    @staticmethod
    def _remove_urls(text: str) -> str:
        return re.sub(r"http\S+|www\.\S+", " ", text)

    @staticmethod
    def _remove_emails(text: str) -> str:
        return re.sub(r"\S+@\S+", " ", text)

    @staticmethod
    def _handle_hashtags(text: str, keep_text: bool = True) -> str:
        if keep_text:
            return re.sub(r"#(\w+)", r"\1", text)   # keep word, drop #
        return re.sub(r"#\w+", " ", text)

    @staticmethod
    def _remove_mentions(text: str) -> str:
        return re.sub(r"@\w+", " ", text)

    @staticmethod
    def _remove_emojis(text: str) -> str:
        # Remove all non-ASCII and emoji unicode blocks
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F9FF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub(" ", text)
        # Remove remaining non-ASCII
        text = text.encode("ascii", "ignore").decode("ascii")
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _normalize_slang(self, text: str) -> str:
        for pattern, replacement in SLANG_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _tokenize_and_normalize(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        result = []
        for tok in tokens:
            # Remove pure punctuation tokens
            if all(c in string.punctuation for c in tok):
                continue
            # Remove digits
            if self.remove_digits and tok.isdigit():
                continue
            # Remove stopwords (keeping negation if configured)
            if self.remove_stopwords and tok.lower() in self.stop_words:
                continue
            # Length filter
            if len(tok) < self.min_token_len:
                continue
            # Lemmatize / stem
            if self.method == "lemmatize":
                tok = self.lemmatizer.lemmatize(tok.lower())
            elif self.method == "stem":
                tok = self.stemmer.stem(tok.lower())
            else:
                tok = tok.lower()
            result.append(tok)
        return result

    # ── Main public API ────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """Return cleaned text as a single string (for TF-IDF / BoW)."""
        if not isinstance(text, str) or not text.strip():
            return ""

        if self.lowercase:
            text = text.lower()
        if self.expand_contractions:
            try:
                text = contractions.fix(text)
            except Exception:
                pass
        if self.normalize_slang:
            text = self._normalize_slang(text)
        if self.remove_urls:
            text = self._remove_urls(text)
        text = self._remove_emails(text)
        if self.remove_mentions:
            text = self._remove_mentions(text)
        text = self._handle_hashtags(text, keep_text=self.keep_hashtag_text)
        if self.remove_emojis:
            text = self._remove_emojis(text)

        tokens = self._tokenize_and_normalize(text)
        return self._normalize_whitespace(" ".join(tokens))

    def tokenize(self, text: str) -> List[str]:
        """Return list of cleaned tokens (for Word2Vec)."""
        cleaned = self.clean(text)
        return cleaned.split() if cleaned else []

    def batch_clean(self, texts: List[str]) -> List[str]:
        """Clean a list of texts."""
        return [self.clean(t) for t in texts]

    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenize a list of texts (for Word2Vec training)."""
        return [self.tokenize(t) for t in texts]


# ── Convenience function ───────────────────────────────────────────────

def get_preprocessor(method: str = "lemmatize") -> TextPreprocessor:
    """Factory: returns a fully configured preprocessor."""
    return TextPreprocessor(method=method)


if __name__ == "__main__":
    sample_texts = [
        "I can't stop crying... everything feels hopeless 😢 http://t.co/xyz #depression",
        "Had an amazing day ngl!! went hiking w/ friends lol 🎉 @bestfriend",
        "My heart is racing rly bad rn, panic attack again bc of work smh",
    ]

    preprocessor = get_preprocessor(method="lemmatize")
    for text in sample_texts:
        print("ORIGINAL:", text)
        print("CLEANED: ", preprocessor.clean(text))
        print("TOKENS:  ", preprocessor.tokenize(text))
        print()
