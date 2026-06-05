"""
End-to-end training pipeline for mental health related text classification.

Run:
    python notebooks/train_pipeline.py

Recommended real-data workflow:
    python scripts/download_dataset.py --dataset dreaddit
    python notebooks/train_pipeline.py

The pipeline keeps the original synthetic dataset generator as a fallback, but
prefers a real processed dataset when data/processed/mental_health_dataset.csv
exists.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.generate_dataset import generate_dataset
from src.explainability import (
    SHAPExplainer,
    plot_feature_importance,
    plot_lr_coefficients,
    plot_wordclouds,
)
from src.features import (
    LINGUISTIC_FEATURE_NAMES,
    BagOfWordsExtractor,
    TFIDFExtractor,
    Word2VecExtractor,
    extract_linguistic_features,
    extract_sentiment_features,
)
from src.models import MentalHealthClassifier, plot_confusion_matrix
from src.nlp_enhancements import (
    EMOTION_FEATURE_NAMES,
    LINGUISTIC_PATTERN_NAMES,
    batch_emotion_scores,
    batch_linguistic_patterns,
)
from src.preprocess import get_preprocessor


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CFG = {
    "n_samples": 3000,
    "test_size": 0.20,
    "random_state": 42,
    "max_tfidf_feat": 5000,
    "w2v_size": 100,
    "tune_models": False,
    "use_smote": True,
    "cv_folds": 5,
    "save_dir": os.path.join(ROOT, "models"),
    "feat_save_dir": os.path.join(ROOT, "models", "feature_extractors"),
    "fig_save_dir": os.path.join(ROOT, "models", "figures"),
    "results_dir": os.path.join(ROOT, "results"),
    "assets_dir": os.path.join(ROOT, "assets", "images"),
}

for path in [
    CFG["save_dir"],
    CFG["feat_save_dir"],
    CFG["fig_save_dir"],
    CFG["results_dir"],
    CFG["assets_dir"],
    os.path.join(ROOT, "data", "raw"),
    os.path.join(ROOT, "data", "processed"),
]:
    os.makedirs(path, exist_ok=True)


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def normalize_dataset(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize common public dataset schemas to text, label, label_name, source."""
    df = df.copy()
    text_candidates = ["text", "post", "sentence", "content", "message"]
    text_col = next((col for col in text_candidates if col in df.columns), None)
    if text_col is None:
        raise ValueError(f"No text column found in {source}. Expected one of {text_candidates}.")

    if "label" not in df.columns:
        label_candidates = ["target", "class", "category"]
        label_col = next((col for col in label_candidates if col in df.columns), None)
        if label_col is None:
            raise ValueError(f"No label column found in {source}.")
        df = df.rename(columns={label_col: "label"})

    output = df[[text_col, "label"]].rename(columns={text_col: "text"})
    output["text"] = output["text"].astype(str).str.strip()
    output = output[output["text"].ne("")]

    if not np.issubdtype(output["label"].dtype, np.number):
        class_names = sorted(output["label"].dropna().astype(str).unique())
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        output["label_name"] = output["label"].astype(str)
        output["label"] = output["label_name"].map(class_to_id).astype(int)
    else:
        output["label"] = output["label"].astype(int)
        if "label_name" in df.columns:
            label_lookup = (
                df[["label", "label_name"]]
                .dropna()
                .drop_duplicates()
                .assign(label=lambda frame: frame["label"].astype(int))
                .set_index("label")["label_name"]
                .astype(str)
                .to_dict()
            )
            output["label_name"] = output["label"].map(label_lookup)
        else:
            default_lookup = {0: "normal", 1: "stress_signal", 2: "anxiety"}
            output["label_name"] = output["label"].map(default_lookup).fillna(
                output["label"].map(lambda value: f"class_{value}")
            )

    output["source"] = source
    return output.reset_index(drop=True)


def load_project_dataset() -> pd.DataFrame:
    """Prefer real processed data and fall back to the educational generator."""
    candidates = [
        (os.path.join(ROOT, "data", "processed", "mental_health_dataset.csv"), "processed_real_dataset"),
        (os.path.join(ROOT, "data", "raw", "dreaddit_merged.csv"), "dreaddit"),
        (os.path.join(ROOT, "data", "raw", "mental_health_dataset.csv"), "synthetic_generator"),
    ]
    for path, source in candidates:
        if os.path.exists(path):
            logger.info("Loading dataset from %s", path)
            return normalize_dataset(pd.read_csv(path), source)

    logger.info("No dataset found. Generating educational synthetic dataset.")
    df = generate_dataset(n_samples=CFG["n_samples"])
    raw_path = os.path.join(ROOT, "data", "raw", "mental_health_dataset.csv")
    df.to_csv(raw_path, index=False)
    return normalize_dataset(df, "synthetic_generator")


def build_full_features(cleaned_texts, tokenized_texts, tfidf_vec, fit=False):
    """Build the final TF-IDF + sentiment + linguistic + emotion feature matrix."""
    if fit:
        tfidf_features = csr_matrix(tfidf_vec.fit_transform(cleaned_texts))
    else:
        tfidf_features = csr_matrix(tfidf_vec.transform(cleaned_texts))

    sentiment = csr_matrix(extract_sentiment_features(cleaned_texts))
    linguistic = csr_matrix(extract_linguistic_features(cleaned_texts))
    patterns = csr_matrix(batch_linguistic_patterns(cleaned_texts))
    emotions = csr_matrix(batch_emotion_scores(tokenized_texts))
    return hstack([tfidf_features, sentiment, linguistic, patterns, emotions]).toarray()


def save_model_comparison_plot(leaderboard_df: pd.DataFrame) -> None:
    plot_df = leaderboard_df.copy()
    plot_df["F1 (weighted)"] = plot_df["F1 (weighted)"].astype(float)
    plot_df = plot_df.sort_values("F1 (weighted)", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_df["Model"], plot_df["F1 (weighted)"], color="#4c78a8")
    ax.set_xlabel("Weighted F1")
    ax.set_title("Model Comparison by Weighted F1", fontweight="bold")
    ax.set_xlim(0, 1)
    for idx, val in enumerate(plot_df["F1 (weighted)"]):
        ax.text(min(val + 0.01, 0.98), idx, f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["assets_dir"], "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    banner("STEP 1 - DATA LOADING")
    df = load_project_dataset()
    label_names = (
        df[["label", "label_name"]]
        .drop_duplicates()
        .sort_values("label")["label_name"]
        .astype(str)
        .tolist()
    )
    print(f"Dataset shape: {df.shape}")
    print("Labels:", label_names)
    print(df["label_name"].value_counts())

    banner("STEP 2 - TEXT PREPROCESSING")
    preprocessor = get_preprocessor(method="lemmatize")
    df["cleaned"] = preprocessor.batch_clean(df["text"].tolist())
    df["tokens"] = preprocessor.batch_tokenize(df["text"].tolist())

    processed_path = os.path.join(ROOT, "data", "processed", "cleaned_dataset.csv")
    df[["text", "cleaned", "label", "label_name", "source"]].to_csv(processed_path, index=False)
    logger.info("Processed data saved to %s", processed_path)

    banner("STEP 3 - TRAIN / TEST SPLIT")
    X_text = df["cleaned"].tolist()
    X_tokens = df["tokens"].tolist()
    y = df["label"].values

    X_train_text, X_test_text, X_train_tok, X_test_tok, y_train, y_test = train_test_split(
        X_text,
        X_tokens,
        y,
        test_size=CFG["test_size"],
        random_state=CFG["random_state"],
        stratify=y,
    )
    print(f"Train: {len(X_train_text)} | Test: {len(X_test_text)}")

    banner("STEP 4 - FEATURE REPRESENTATION COMPARISON")
    bow = BagOfWordsExtractor(max_features=CFG["max_tfidf_feat"])
    X_bow_train = bow.fit_transform(X_train_text)
    X_bow_test = bow.transform(X_test_text)

    tfidf = TFIDFExtractor(max_features=CFG["max_tfidf_feat"], sublinear_tf=True)
    X_tfidf_train = tfidf.fit_transform(X_train_text)
    X_tfidf_test = tfidf.transform(X_test_text)

    w2v = Word2VecExtractor(vector_size=CFG["w2v_size"], min_count=2)
    X_w2v_train = w2v.fit_transform(X_train_tok)
    X_w2v_test = w2v.transform(X_test_tok)

    feature_rows = []
    for name, X_train_part, X_test_part in [
        ("BoW", X_bow_train, X_bow_test),
        ("TF-IDF", X_tfidf_train, X_tfidf_test),
        ("Word2Vec", X_w2v_train, X_w2v_test),
    ]:
        baseline = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        baseline.fit(X_train_part, y_train)
        predictions = baseline.predict(X_test_part)
        feature_rows.append({
            "Representation": name,
            "Weighted F1": f1_score(y_test, predictions, average="weighted"),
        })

    feature_comparison = pd.DataFrame(feature_rows)
    feature_comparison.to_csv(os.path.join(CFG["results_dir"], "feature_comparison.csv"), index=False)
    print(feature_comparison)

    banner("STEP 5 - FINAL FEATURE MATRIX")
    tfidf = TFIDFExtractor(max_features=CFG["max_tfidf_feat"], sublinear_tf=True)
    X_train = build_full_features(X_train_text, X_train_tok, tfidf, fit=True)
    X_test = build_full_features(X_test_text, X_test_tok, tfidf, fit=False)

    tfidf_feature_names = tfidf.get_feature_names()
    all_feature_names = (
        tfidf_feature_names
        + ["sentiment_polarity", "sentiment_subjectivity"]
        + LINGUISTIC_FEATURE_NAMES
        + LINGUISTIC_PATTERN_NAMES
        + EMOTION_FEATURE_NAMES
    )
    joblib.dump(tfidf.vectorizer, os.path.join(CFG["feat_save_dir"], "tfidf.pkl"))

    banner("STEP 6 - MODEL TRAINING")
    trainer = MentalHealthClassifier(
        use_smote=CFG["use_smote"],
        cv_folds=CFG["cv_folds"],
        label_names=label_names,
    )
    results = trainer.train_all(
        X_train,
        y_train,
        X_test,
        y_test,
        tune=CFG["tune_models"],
    )

    banner("STEP 7 - RESULTS")
    leaderboard_df = trainer.print_leaderboard()
    leaderboard_df.to_csv(os.path.join(CFG["save_dir"], "leaderboard.csv"), index=False)
    leaderboard_df.to_csv(os.path.join(CFG["results_dir"], "leaderboard.csv"), index=False)
    leaderboard_df.to_csv(os.path.join(CFG["results_dir"], "model_comparison.csv"), index=False)

    best_name = trainer.best_model_name
    best_model = trainer.get_best_model()
    best_predictions = results[best_name]["y_pred"]

    cm_path = os.path.join(CFG["fig_save_dir"], "confusion_matrix_best.png")
    plot_confusion_matrix(
        y_test,
        best_predictions,
        model_name=best_name,
        label_names=label_names,
        save_path=cm_path,
    )
    shutil.copyfile(cm_path, os.path.join(CFG["assets_dir"], "confusion_matrix.png"))
    save_model_comparison_plot(leaderboard_df)
    # print("REACHED AFTER STEP 7")
    # trainer.save_best_model(CFG["save_dir"])
    # trainer.save_all_models(CFG["save_dir"])
    # print("MODEL SAVED")
    # return

    banner("STEP 8 - EXPLAINABILITY")
    plot_wordclouds(X_train_text, y_train, save_path=os.path.join(CFG["fig_save_dir"], "wordclouds.png"))

    if "Logistic Regression" in trainer.trained_models:
        plot_lr_coefficients(
            trainer.trained_models["Logistic Regression"],
            all_feature_names,
            save_path=os.path.join(CFG["fig_save_dir"], "lr_coefficients.png"),
        )

    if "Random Forest" in trainer.trained_models:
        plot_feature_importance(
            trainer.trained_models["Random Forest"],
            all_feature_names,
            model_name="Random Forest",
            save_path=os.path.join(CFG["assets_dir"], "feature_importance.png"),
        )

    try:
        shap_explainer = SHAPExplainer(best_model, X_train[:100], model_name=best_name)
        shap_explainer.global_summary(
            X_test[:30],
            all_feature_names,
            n_samples=min(30, len(X_test)),
            save_path=os.path.join(CFG["assets_dir"], "shap_summary.png"),
        )
    except Exception as exc:
        logger.warning("SHAP summary generation skipped: %s", exc)

    banner("STEP 9 - SAVE MODELS")
    trainer.save_best_model(CFG["save_dir"])
    trainer.save_all_models(CFG["save_dir"])

    label_display = [name.replace("_", " ").title() for name in label_names]
    print(f"Best model: {best_name}")
    print(f"Best weighted F1: {results[best_name]['f1_score']:.4f}")
    print(f"Labels: {label_display}")
    print("Next step: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
