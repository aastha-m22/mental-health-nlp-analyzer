"""
src/features.py
---------------
Feature extraction pipeline implementing and comparing:
  1. Bag of Words (BoW)       — frequency-based, ignores word order
  2. TF-IDF                   — penalises common words, rewards discriminative ones
  3. Word2Vec (mean pooling)  — semantic, dense, captures context

Also adds:
  4. Sentiment features       — TextBlob polarity + subjectivity
  5. Linguistic features      — text length, avg word length, punctuation density
"""

import numpy as np
import pandas as pd
import logging
import joblib
import os
from typing import List, Tuple, Optional, Dict

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix

from gensim.models import Word2Vec
from textblob import TextBlob

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 1. BAG OF WORDS
# ══════════════════════════════════════════════════════════════════
class BagOfWordsExtractor:
    """
    Bag of Words: counts raw term frequencies.

    WHEN TO USE:
    - Baseline comparison
    - Short texts where word counts are informative
    - Fast training required

    LIMITATION: ignores word importance across documents;
                "the" and "worthless" treated equally.
    """

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,          # ignore very rare terms
            max_df=0.95,       # ignore near-universal terms
            
        )
        self.name = "BoW"

    def fit(self, texts: List[str]) -> "BagOfWordsExtractor":
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        return self.vectorizer.fit_transform(texts)

    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()


# ══════════════════════════════════════════════════════════════════
# 2. TF-IDF
# ══════════════════════════════════════════════════════════════════
class TFIDFExtractor:
    """
    TF-IDF: down-weights common words, up-weights rare discriminative ones.

    WHEN TO USE:
    - Primary feature method for most text classification tasks
    - Words like "feel", "sad", "hopeless" should score higher than "the"
    - Better than BoW for imbalanced or noisy text

    ENHANCEMENT: sublinear_tf=True applies log(1+tf) to dampen
    extremely frequent terms.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=2,
            max_df=0.95,
            analyzer="word",
        )
        self.name = "TF-IDF"

    def fit(self, texts: List[str]) -> "TFIDFExtractor":
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        return self.vectorizer.fit_transform(texts)

    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()

    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """Return top n features by mean TF-IDF score (requires fitted vectorizer)."""
        feature_names = self.get_feature_names()
        # Proxy: use idf scores (higher idf = more discriminative)
        idf_scores = self.vectorizer.idf_
        top_idx = np.argsort(idf_scores)[-n:][::-1]
        return pd.DataFrame({
            "feature": [feature_names[i] for i in top_idx],
            "idf_score": [idf_scores[i] for i in top_idx],
        })


# ══════════════════════════════════════════════════════════════════
# 3. WORD2VEC (mean-pooled sentence embeddings)
# ══════════════════════════════════════════════════════════════════
class Word2VecExtractor:
    """
    Word2Vec: trains on the corpus, then represents each text as
    the mean of its word vectors.

    WHEN TO USE:
    - Capture semantic meaning ("hopeless" ≈ "worthless" ≈ "empty")
    - When vocabulary is rich and corpus is large enough
    - When context and word relationships matter

    LIMITATION: mean pooling loses word order; consider LSTM/BERT
    for sequence-aware representations.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model: Optional[Word2Vec] = None
        self.name = "Word2Vec"

    def fit(self, tokenized_texts: List[List[str]]) -> "Word2VecExtractor":
        """Train Word2Vec on tokenized corpus."""
        logger.info(f"Training Word2Vec on {len(tokenized_texts)} documents…")
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            sg=1,   # Skip-gram (better for rare words)
        )
        logger.info("Word2Vec training complete.")
        return self

    def _text_to_vector(self, tokens: List[str]) -> np.ndarray:
        """Mean-pool word vectors for a single document."""
        if self.model is None:
            raise RuntimeError("Word2Vec model not trained yet.")
        vecs = [
            self.model.wv[tok]
            for tok in tokens
            if tok in self.model.wv
        ]
        if not vecs:
            return np.zeros(self.vector_size)
        return np.mean(vecs, axis=0)

    def transform(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """Transform a list of tokenized documents to a 2D numpy array."""
        return np.vstack([self._text_to_vector(tokens) for tokens in tokenized_texts])

    def fit_transform(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        self.fit(tokenized_texts)
        return self.transform(tokenized_texts)

    def save(self, path: str):
        if self.model:
            self.model.save(path)

    def load(self, path: str):
        self.model = Word2Vec.load(path)
        return self

    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Explore semantic neighbours of a word."""
        if self.model and word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        return []


# ══════════════════════════════════════════════════════════════════
# 4. SENTIMENT FEATURES
# ══════════════════════════════════════════════════════════════════
def extract_sentiment_features(texts: List[str]) -> np.ndarray:
    """
    TextBlob sentiment: returns (polarity, subjectivity) per text.
    - polarity:     -1 (very negative) to +1 (very positive)
    - subjectivity:  0 (objective)     to  1 (subjective)

    Rationale: depression posts tend to be highly negative + subjective.
    """
    features = []
    for text in texts:
        blob = TextBlob(text)
        features.append([blob.sentiment.polarity, blob.sentiment.subjectivity])
    return np.array(features)


# ══════════════════════════════════════════════════════════════════
# 5. LINGUISTIC / STRUCTURAL FEATURES
# ══════════════════════════════════════════════════════════════════
def extract_linguistic_features(texts: List[str]) -> np.ndarray:
    """
    Hand-crafted features capturing writing style signals:
      - char_count         : raw text length
      - word_count         : number of words
      - avg_word_len       : average word length
      - exclamation_count  : number of '!'
      - question_count     : number of '?'
      - ellipsis_count     : number of '...'
      - capital_ratio      : proportion of uppercase letters
      - unique_word_ratio  : lexical diversity

    Rationale: depressed/anxious users write differently —
    shorter sentences, more ellipses, lower lexical diversity.
    """
    results = []
    for text in texts:
        words = text.split()
        n_words = max(len(words), 1)
        n_chars = len(text)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        excl = text.count("!")
        quest = text.count("?")
        ellip = text.count("...")
        n_upper = sum(1 for c in text if c.isupper())
        cap_ratio = n_upper / max(n_chars, 1)
        unique_ratio = len(set(words)) / n_words

        results.append([
            n_chars, n_words, avg_word_len,
            excl, quest, ellip,
            cap_ratio, unique_ratio,
        ])
    return np.array(results)


LINGUISTIC_FEATURE_NAMES = [
    "char_count", "word_count", "avg_word_len",
    "exclamation_count", "question_count", "ellipsis_count",
    "capital_ratio", "unique_word_ratio",
]


# ══════════════════════════════════════════════════════════════════
# 6. COMBINED FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════
class FeatureBuilder:
    """
    Combines TF-IDF + sentiment + linguistic features into one
    feature matrix, with optional Word2Vec concatenation.
    """

    def __init__(self, use_tfidf: bool = True, use_w2v: bool = False):
        self.tfidf = TFIDFExtractor(max_features=5000, ngram_range=(1, 2))
        self.w2v = Word2VecExtractor() if use_w2v else None
        self.use_tfidf = use_tfidf
        self.use_w2v = use_w2v

    def fit(self, cleaned_texts: List[str], tokenized_texts: Optional[List[List[str]]] = None):
        if self.use_tfidf:
            self.tfidf.fit(cleaned_texts)
        if self.use_w2v and tokenized_texts:
            self.w2v.fit(tokenized_texts)
        return self

    def transform(
        self,
        cleaned_texts: List[str],
        tokenized_texts: Optional[List[List[str]]] = None,
    ) -> np.ndarray:
        parts = []

        if self.use_tfidf:
            tfidf_feats = self.tfidf.transform(cleaned_texts)  # sparse
            parts.append(tfidf_feats)

        sent_feats = csr_matrix(extract_sentiment_features(cleaned_texts))
        ling_feats = csr_matrix(extract_linguistic_features(cleaned_texts))
        parts.extend([sent_feats, ling_feats])

        if self.use_w2v and self.w2v and tokenized_texts:
            w2v_feats = csr_matrix(self.w2v.transform(tokenized_texts))
            parts.append(w2v_feats)

        return hstack(parts).toarray()

    def fit_transform(
        self,
        cleaned_texts: List[str],
        tokenized_texts: Optional[List[List[str]]] = None,
    ) -> np.ndarray:
        self.fit(cleaned_texts, tokenized_texts)
        return self.transform(cleaned_texts, tokenized_texts)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.tfidf.vectorizer, f"{path}/tfidf.pkl")
        if self.w2v and self.w2v.model:
            self.w2v.save(f"{path}/word2vec.model")
        logger.info(f"Feature extractors saved to {path}")

    def load(self, path: str):
        self.tfidf.vectorizer = joblib.load(f"{path}/tfidf.pkl")
        if self.use_w2v:
            self.w2v.load(f"{path}/word2vec.model")
        return self


if __name__ == "__main__":
    # Quick sanity check
    texts = [
        "i feel so empty and hopeless cannot get out bed",
        "had amazing day hiking friends sunshine wonderful",
        "heart racing panic attack cannot breathe terrified",
    ]
    tokenized = [t.split() for t in texts]

    bow = BagOfWordsExtractor(max_features=50)
    X_bow = bow.fit_transform(texts)
    print("BoW shape:", X_bow.shape)

    tfidf = TFIDFExtractor(max_features=50)
    X_tfidf = tfidf.fit_transform(texts)
    print("TF-IDF shape:", X_tfidf.shape)

    w2v = Word2VecExtractor(vector_size=50, min_count=1)
    X_w2v = w2v.fit_transform(tokenized)
    print("Word2Vec shape:", X_w2v.shape)

    sent = extract_sentiment_features(texts)
    print("Sentiment features:\n", sent)

    ling = extract_linguistic_features(texts)
    print("Linguistic features shape:", ling.shape)
