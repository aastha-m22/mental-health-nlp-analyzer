"""
src/explainability.py
---------------------
Model explainability using:
  1. SHAP (SHapley Additive exPlanations)  — model-agnostic feature attribution
  2. Feature importance (tree-based models) — built-in Gini importance
  3. Logistic Regression coefficients       — per-class top features
  4. Prediction explanation for single samples

Ethical note:
  Explainability is crucial in mental health AI — clinicians and users
  deserve to understand WHY a model flags a post. Opaque predictions
  in sensitive domains erode trust and may cause harm.
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import List, Dict, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

LABEL_NAMES = ["normal", "depression", "anxiety"]
LABEL_COLORS = {"normal": "#2ecc71", "depression": "#3498db", "anxiety": "#e74c3c"}


# ══════════════════════════════════════════════════════════════════
# 1. LOGISTIC REGRESSION COEFFICIENTS
# ══════════════════════════════════════════════════════════════════

def plot_lr_coefficients(
    model,
    feature_names: List[str],
    top_n: int = 20,
    save_path: Optional[str] = None,
):
    """
    Plot top positive and negative coefficients per class for
    a fitted Logistic Regression model.
    """
    from sklearn.linear_model import LogisticRegression
    if not hasattr(model, "coef_"):
        print("Model does not have coefficients.")
        return

    n_classes = model.coef_.shape[0]
    fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, 8))
    if n_classes == 1:
        axes = [axes]

    for cls_idx, ax in enumerate(axes):
        coefs = model.coef_[cls_idx]
        top_pos = np.argsort(coefs)[-top_n:]
        top_neg = np.argsort(coefs)[:top_n]
        indices = np.concatenate([top_neg, top_pos])

        values = coefs[indices]
        names = [feature_names[i] for i in indices]
        colors = ["#e74c3c" if v < 0 else "#2980b9" for v in values]

        ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"Top Features: {LABEL_NAMES[cls_idx].upper()}",
            fontsize=13, fontweight="bold",
            color=LABEL_COLORS.get(LABEL_NAMES[cls_idx], "black")
        )
        ax.set_xlabel("Coefficient")

    plt.suptitle("Logistic Regression Feature Coefficients", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════
# 2. FEATURE IMPORTANCE (TREE-BASED MODELS)
# ══════════════════════════════════════════════════════════════════

def plot_feature_importance(
    model,
    feature_names: List[str],
    model_name: str = "Model",
    top_n: int = 25,
    save_path: Optional[str] = None,
):
    """Plot top-N feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} does not expose feature_importances_.")
        return

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-top_n:]
    top_vals = importances[top_idx]
    top_names = [feature_names[i] for i in top_idx]

    palette = sns.color_palette("viridis", top_n)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(top_n), top_vals, color=palette, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Gini Importance", fontsize=11)
    ax.set_title(f"Top {top_n} Features — {model_name}", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════
# 3. SHAP EXPLANATIONS
# ══════════════════════════════════════════════════════════════════

class SHAPExplainer:
    """
    Wraps SHAP for model-agnostic feature attribution.

    SHAP values tell us: for a given prediction, how much did each
    feature PUSH the probability up or down compared to the baseline?

    This is critical for mental health AI — we want to surface WHICH
    words or features triggered a 'depression' classification.
    """

    def __init__(self, model, X_background: np.ndarray, model_name: str = "Model"):
        self.model = model
        self.model_name = model_name
        self.explainer = None
        self._init_explainer(X_background)

    def _init_explainer(self, X_background: np.ndarray):
        try:
            import shap
            if hasattr(self.model, "predict_proba"):
                # Use KernelExplainer for general models
                # Sample background for efficiency
                bg_size = min(100, len(X_background))
                background = shap.sample(X_background, bg_size)
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, background
                )
            else:
                # LinearSVC → use LinearExplainer with decision_function
                background = shap.sample(X_background, min(50, len(X_background)))
                self.explainer = shap.KernelExplainer(
                    self.model.decision_function, background
                )
            logger.info("SHAP KernelExplainer initialised.")
        except ImportError:
            logger.warning("SHAP not available. Install with: pip install shap")
        except Exception as e:
            logger.warning(f"SHAP initialisation failed: {e}")

    def explain_sample(
        self,
        X_sample: np.ndarray,
        feature_names: List[str],
        class_idx: int = 1,
        save_path: Optional[str] = None,
    ):
        """Generate a SHAP waterfall / force plot for a single sample."""
        if self.explainer is None:
            print("SHAP explainer not available.")
            return

        import shap
        print("Computing SHAP values (may take ~30 seconds for KernelExplainer)…")
        shap_values = self.explainer.shap_values(X_sample, nsamples=100)

        if isinstance(shap_values, list):
            sv = shap_values[class_idx][0]
        else:
            sv = shap_values[0]

        # Get top contributing features
        top_idx = np.argsort(np.abs(sv))[-15:]
        top_names = [feature_names[i] for i in top_idx]
        top_vals = sv[top_idx]

        colors = ["#e74c3c" if v > 0 else "#2980b9" for v in top_vals]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(range(len(top_names)), top_vals, color=colors)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP Value (feature contribution)")
        ax.set_title(
            f"SHAP Explanation — Predicting: {LABEL_NAMES[class_idx].upper()}",
            fontsize=13, fontweight="bold"
        )
        red_patch = mpatches.Patch(color="#e74c3c", label="Pushes toward this class")
        blue_patch = mpatches.Patch(color="#2980b9", label="Pushes away from this class")
        ax.legend(handles=[red_patch, blue_patch], loc="lower right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def global_summary(
        self,
        X: np.ndarray,
        feature_names: List[str],
        n_samples: int = 50,
        save_path: Optional[str] = None,
    ):
        """Generate a global SHAP summary bar plot."""
        if self.explainer is None:
            return

        import shap
        X_sub = X[:n_samples]
        print(f"Computing SHAP values for {n_samples} samples…")
        shap_values = self.explainer.shap_values(X_sub, nsamples=100)

        if isinstance(shap_values, list):
            mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        top_idx = np.argsort(mean_abs_shap)[-20:]
        top_names = [feature_names[i] for i in top_idx]
        top_vals = mean_abs_shap[top_idx]

        fig, ax = plt.subplots(figsize=(10, 7))
        palette = sns.color_palette("magma", 20)
        ax.barh(range(20), top_vals, color=palette)
        ax.set_yticks(range(20))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Global SHAP Feature Importance", fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


# ══════════════════════════════════════════════════════════════════
# 4. WORD CLOUD VISUALISATION
# ══════════════════════════════════════════════════════════════════

def plot_wordclouds(
    texts: List[str],
    labels: np.ndarray,
    save_path: Optional[str] = None,
):
    """Generate word clouds for each mental health category."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("wordcloud not installed: pip install wordcloud")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for cls_idx, (ax, label_name) in enumerate(zip(axes, LABEL_NAMES)):
        class_texts = " ".join([t for t, l in zip(texts, labels) if l == cls_idx])
        if not class_texts.strip():
            ax.set_visible(False)
            continue
        wc = WordCloud(
            width=600, height=400,
            background_color="white",
            colormap=["Greens", "Blues", "Reds"][cls_idx],
            max_words=80,
            collocations=False,
        ).generate(class_texts)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            label_name.upper(), fontsize=15, fontweight="bold",
            color=LABEL_COLORS[label_name]
        )
    plt.suptitle("Word Clouds by Mental Health Category", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════
# 5. SINGLE PREDICTION EXPLANATION (human-readable)
# ══════════════════════════════════════════════════════════════════

def explain_prediction(
    text: str,
    prediction: int,
    probabilities: Optional[np.ndarray],
    top_features: Optional[List[Tuple[str, float]]] = None,
) -> str:
    """
    Generate a human-readable explanation for a single prediction.
    Used in the Streamlit app.
    """
    label = LABEL_NAMES[prediction]
    lines = [
        f"🔍 Prediction: **{label.upper()}**",
        "",
    ]
    if probabilities is not None:
        lines.append("📊 Confidence Scores:")
        for i, lname in enumerate(LABEL_NAMES):
            bar = "█" * int(probabilities[i] * 20)
            lines.append(f"   {lname:<12} {bar:<20} {probabilities[i]*100:.1f}%")
        lines.append("")

    if top_features:
        lines.append("🔑 Key contributing words:")
        for word, score in top_features[:5]:
            lines.append(f"   • '{word}' (score: {score:.4f})")

    return "\n".join(lines)
