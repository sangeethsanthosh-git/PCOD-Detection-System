"""Model evaluation, ranking, and plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .utils import extract_positive_class_scores


def evaluate_models(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, object]]:
    """Evaluate all models on the test set."""
    results: Dict[str, Dict[str, object]] = {}
    is_binary = len(np.unique(y_test)) == 2

    for model_name, model in models.items():
        predictions = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, predictions)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="binary" if is_binary else "weighted", zero_division=0),
            "recall": recall_score(y_test, predictions, average="binary" if is_binary else "weighted", zero_division=0),
            "f1": f1_score(y_test, predictions, average="binary" if is_binary else "weighted", zero_division=0),
            "roc_auc": None,
            "confusion_matrix": conf_matrix,
            "roc": None,
        }

        score_vector = extract_positive_class_scores(model, X_test)
        if is_binary and score_vector is not None:
            fpr, tpr, _ = roc_curve(y_test, score_vector)
            auc_score = roc_auc_score(y_test, score_vector)
            metrics["roc_auc"] = float(auc_score)
            metrics["roc"] = {"fpr": fpr, "tpr": tpr, "auc": float(auc_score)}

        results[model_name] = metrics

    return results


def select_best_model(
    results: Dict[str, Dict[str, object]],
    candidate_models: Sequence[str] | None = None,
) -> Tuple[str, Dict[str, object]]:
    """Select best model using F1 score."""
    candidates = list(candidate_models) if candidate_models is not None else list(results.keys())
    available = [name for name in candidates if name in results]
    if not available:
        raise ValueError("No candidate models available for selection.")

    best_name = max(available, key=lambda name: float(results[name]["f1"]))
    return best_name, results[best_name]


def save_model_metrics_csv(
    results: Dict[str, Dict[str, object]],
    output_path: Path,
) -> pd.DataFrame:
    """Persist model metrics table to CSV and return it."""
    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )
    metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df


def save_confusion_matrix_plot(conf_matrix: np.ndarray, output_path: Path, model_name: str) -> None:
    """Save confusion matrix plot."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_roc_curve_plot(fpr: np.ndarray, tpr: np.ndarray, auc_score: float, output_path: Path, model_name: str) -> None:
    """Save ROC curve plot."""
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_feature_importance_plot(
    model: object,
    feature_names: List[str],
    output_path: Path,
    top_n: int = 20,
) -> bool:
    """Save feature-importance plot for models exposing `feature_importances_`."""
    if not hasattr(model, "feature_importances_"):
        return False

    importances = np.asarray(model.feature_importances_)
    if importances.size == 0:
        return False

    top_n = min(top_n, len(feature_names))
    ranked_indices = np.argsort(importances)[::-1][:top_n]

    plot_df = pd.DataFrame(
        {
            "feature": np.array(feature_names)[ranked_indices],
            "importance": importances[ranked_indices],
        }
    )

    plt.figure(figsize=(10, max(4, int(0.45 * len(plot_df)))))
    sns.barplot(data=plot_df, x="importance", y="feature", orient="h")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return True


def save_feature_importance_from_table(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
    title: str = "Feature Importance",
) -> bool:
    """Save feature importance plot from a table with `feature` and `importance` columns."""
    required = {"feature", "importance"}
    if not required.issubset(set(importance_df.columns)):
        return False

    plot_df = (
        importance_df[["feature", "importance"]]
        .dropna()
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    if plot_df.empty:
        return False

    plt.figure(figsize=(10, max(4, int(0.45 * len(plot_df)))))
    sns.barplot(data=plot_df, x="importance", y="feature", orient="h")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return True
