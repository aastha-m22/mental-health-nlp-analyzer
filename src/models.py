"""
src/models.py
-------------
Train, tune, and compare 7 classifiers for mental health text classification.

Models:
  1. Logistic Regression
  2. Naive Bayes (Multinomial + Complement)
  3. Support Vector Machine (Linear kernel)
  4. Random Forest
  5. XGBoost
  6. Gradient Boosting
  7. Multi-Layer Perceptron (Neural baseline)

Includes:
  - Class imbalance handling (SMOTE + class_weight)
  - Cross-validation (Stratified K-Fold)
  - Hyperparameter tuning (RandomizedSearchCV)
  - Full evaluation: accuracy, precision, recall, F1, confusion matrix
"""

import numpy as np
import pandas as pd
import logging
import joblib
import os
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, RandomizedSearchCV, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

DEFAULT_LABEL_NAMES = ["normal", "depression", "anxiety"]
LABEL_NAMES = DEFAULT_LABEL_NAMES


def _labels_and_names(y_true, y_pred=None, label_names: Optional[List[str]] = None):
    arrays = [np.asarray(y_true)]
    if y_pred is not None:
        arrays.append(np.asarray(y_pred))
    labels = np.unique(np.concatenate(arrays)).astype(int)
    names = label_names or DEFAULT_LABEL_NAMES
    if len(names) > int(labels.max(initial=0)):
        target_names = [names[i] for i in labels]
    else:
        target_names = [f"class_{i}" for i in labels]
    return labels, target_names


# ══════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════

def get_model_registry() -> Dict[str, Any]:
    """
    Returns a dict of {name: estimator} for all models.
    All estimators handle multi-class natively.
    """
    registry = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
    
            random_state=42,
        ),
        

        "SVM (Linear)": LinearSVC(
            class_weight="balanced",
            C=1.0,
            max_iter=2000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        ),
        "MLP Neural Net": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            alpha=0.001,
            max_iter=300,
            early_stopping=True,
            random_state=42,
        ),
    }
    if XGBOOST_AVAILABLE:
        registry["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
    return registry


# ══════════════════════════════════════════════════════════════════
# HYPERPARAMETER SEARCH SPACES
# ══════════════════════════════════════════════════════════════════

PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 5, 10],
        "solver": ["lbfgs", "saga"],
    },
    "SVM (Linear)": {
        "C": [0.1, 0.5, 1.0, 2.0, 5.0],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [4, 6, 8],
    },
}


# ══════════════════════════════════════════════════════════════════
# EVALUATION UTILITIES
# ══════════════════════════════════════════════════════════════════

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    label_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute and optionally print full evaluation metrics."""
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if verbose:
        labels, target_names = _labels_and_names(y_true, y_pred, label_names)
        print(f"\n{'='*55}")
        print(f"  {model_name}")
        print(f"{'='*55}")
        for k, v in metrics.items():
            print(f"  {k:<15}: {v:.4f}")
        print("\n  Classification Report:")
        print(classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        ))
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    label_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """Plot and optionally save a confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels, target_names = _labels_and_names(y_true, y_pred, label_names)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names, ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# MAIN TRAINER CLASS
# ══════════════════════════════════════════════════════════════════

class MentalHealthClassifier:
    """
    End-to-end trainer and evaluator for all models.

    Usage:
        clf = MentalHealthClassifier()
        results = clf.train_all(X_train, y_train, X_test, y_test)
        clf.print_leaderboard(results)
        clf.save_best_model("models/")
    """

    def __init__(self, use_smote: bool = True, cv_folds: int = 5, label_names: Optional[List[str]] = None):
        self.use_smote = use_smote
        self.cv_folds = cv_folds
        self.label_names = label_names or DEFAULT_LABEL_NAMES
        self.models = get_model_registry()
        self.results: Dict[str, Dict] = {}
        self.trained_models: Dict[str, Any] = {}
        self.best_model_name: Optional[str] = None

    def _apply_smote(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling to handle class imbalance."""
        if not self.use_smote:
            return X, y
        # Ensure minimum samples per class for SMOTE
        min_class_count = np.min(np.bincount(y))
        k_neighbors = min(5, min_class_count - 1)
        if k_neighbors < 1:
            logger.warning("Not enough samples for SMOTE, skipping.")
            return X, y
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        try:
            X_res, y_res = smote.fit_resample(X, y)
            logger.info(f"SMOTE: {len(y)} → {len(y_res)} samples")
            return X_res, y_res
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
            return X, y

    def _ensure_non_negative(self, X: np.ndarray) -> np.ndarray:
        """MinMaxScaler for models requiring non-negative inputs (NB)."""
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)

    def cross_validate_model(
        self, model, X: np.ndarray, y: np.ndarray, model_name: str
    ) -> Dict[str, float]:
        """Run stratified K-fold CV and return mean metrics."""
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
        cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        return {
            "cv_accuracy":  cv_results["test_accuracy"].mean(),
            "cv_f1":        cv_results["test_f1_weighted"].mean(),
            "cv_precision": cv_results["test_precision_weighted"].mean(),
            "cv_recall":    cv_results["test_recall_weighted"].mean(),
            "cv_f1_std":    cv_results["test_f1_weighted"].std(),
        }

    def tune_model(
        self, model_name: str, model, X: np.ndarray, y: np.ndarray
    ) -> Any:
        """Run RandomizedSearchCV if param grid is defined."""
        if model_name not in PARAM_GRIDS:
            return model
        param_grid = PARAM_GRIDS[model_name]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            model, param_grid, n_iter=10, cv=skf,
            scoring="f1_weighted", n_jobs=-1, random_state=42
        )
        search.fit(X, y)
        logger.info(f"Best params for {model_name}: {search.best_params_}")
        return search.best_estimator_

    def train_single(
        self,
        model_name: str,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        tune: bool = False,
    ) -> Dict:
        """Train, evaluate, and log a single model."""
        print(f"\n▶ Training: {model_name}")
        t0 = time.time()

        # Prepare training data
        X_tr = X_train
        # Naive Bayes requires non-negative features
        if "Naive Bayes" in model_name:
            X_tr = self._ensure_non_negative(X_train)
            X_te = self._ensure_non_negative(X_test)
        else:
            X_te = X_test

        # SMOTE
        X_smote, y_smote = self._apply_smote(X_tr, y_train)

        # Optional tuning
        if tune and model_name in PARAM_GRIDS:
            model = self.tune_model(model_name, model, X_smote, y_smote)

        # Fit
        model.fit(X_smote, y_smote)
        elapsed = time.time() - t0

        # Predict
        y_pred = model.predict(X_te)

        # Evaluate
        test_metrics = evaluate_model(
            y_test,
            y_pred,
            model_name=model_name,
            label_names=self.label_names,
            verbose=True,
        )

        # Cross-validate (on original unsmoted data for honest CV)
        print(f"  Running {self.cv_folds}-fold cross-validation…")
        cv_metrics = self.cross_validate_model(model, X_train, y_train, model_name)
        print(f"  CV F1 (weighted): {cv_metrics['cv_f1']:.4f} ± {cv_metrics['cv_f1_std']:.4f}")

        result = {
            "model": model,
            "model_name": model_name,
            "train_time_s": elapsed,
            **test_metrics,
            **cv_metrics,
            "y_pred": y_pred,
        }
        self.trained_models[model_name] = model
        return result

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        tune: bool = False,
    ) -> Dict[str, Dict]:
        """Train all models and collect results."""
        print(f"\n{'█'*60}")
        print("  TRAINING ALL MODELS")
        print(f"  Train size: {X_train.shape}, Test size: {X_test.shape}")
        print(f"{'█'*60}")

        for name, model in self.models.items():
            result = self.train_single(name, model, X_train, y_train, X_test, y_test, tune=tune)
            self.results[name] = result

        # Identify best model by weighted F1 on test set
        self.best_model_name = max(self.results, key=lambda k: self.results[k]["f1_score"])
        print(f"\n🏆 Best model: {self.best_model_name} "
              f"(F1={self.results[self.best_model_name]['f1_score']:.4f})")
        return self.results

    def print_leaderboard(self, results: Optional[Dict] = None):
        """Print a sorted comparison table of all models."""
        results = results or self.results
        if not results:
            print("No results yet. Run train_all() first.")
            return

        rows = []
        for name, r in results.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{r['accuracy']:.4f}",
                "Precision": f"{r['precision']:.4f}",
                "Recall": f"{r['recall']:.4f}",
                "F1 (weighted)": f"{r['f1_score']:.4f}",
                "F1 (macro)": f"{r['f1_macro']:.4f}",
                "CV F1": f"{r['cv_f1']:.4f} ± {r['cv_f1_std']:.4f}",
                "Train (s)": f"{r['train_time_s']:.2f}",
            })

        df = pd.DataFrame(rows).sort_values("F1 (weighted)", ascending=False)
        df = df.reset_index(drop=True)
        df.index += 1
        print("\n" + "="*100)
        print("  MODEL LEADERBOARD")
        print("="*100)
        print(df.to_string(index=True))
        print("="*100)
        return df

    def get_best_model(self):
        """Return the best-performing trained model."""
        if self.best_model_name:
            return self.trained_models[self.best_model_name]
        return None

    # def save_best_model(self, save_dir: str = "models/"):
    #     """Persist the best model to disk."""
    #     os.makedirs(save_dir, exist_ok=True)
    #     if self.best_model_name:
    #         path = os.path.join(save_dir, "best_model.pkl")
    #         joblib.dump(self.trained_models[self.best_model_name], path)
    #         # Save model name
    #         with open(os.path.join(save_dir, "best_model_name.txt"), "w") as f:
    #             f.write(self.best_model_name)
    #         with open(os.path.join(save_dir, "label_names.txt"), "w") as f:
    #             f.write("\n".join(self.label_names))
    #         logger.info(f"Best model '{self.best_model_name}' saved to {path}")
    #         print(f"\n✅ Best model saved: {path}")

    def save_best_model(self, save_dir="models/"):
       print("DEBUG best_model_name =", self.best_model_name)
       print("DEBUG trained_models keys =", list(self.trained_models.keys()))

       os.makedirs(save_dir, exist_ok=True)

       if self.best_model_name:
        path = os.path.join(save_dir, "best_model.pkl")

        joblib.dump(
            self.trained_models[self.best_model_name],
            path
        )

        with open(os.path.join(save_dir, "best_model_name.txt"), "w") as f:
            f.write(self.best_model_name)

        with open(os.path.join(save_dir, "label_names.txt"), "w") as f:
            f.write("\n".join(self.label_names))

        print(f"Saved model to {path}")
       else:
        print("ERROR: best_model_name is empty")

    def save_all_models(self, save_dir: str = "models/"):
        """Save all trained models."""
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.trained_models.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            path = os.path.join(save_dir, f"{safe_name}.pkl")
            joblib.dump(model, path)
        logger.info(f"All models saved to {save_dir}")

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Predict with a specific trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained yet.")
        return self.trained_models[model_name].predict(X)

    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Get probability estimates (not available for LinearSVC)."""
        model = self.trained_models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not trained yet.")
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        raise AttributeError(f"{model_name} does not support predict_proba.")
