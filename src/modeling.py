"""Predict default from the derived risk metrics.

    python -m src.modeling

Logistic Regression first as the baseline, then a Random Forest that has to
beat it. Figures are written to figures/.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import TARGET, MissingDataError, build_features

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RANDOM_STATE = 42
TEST_SIZE = 0.3

FEATURES = [
    "AVG_UTIL", "AVG_PAY_DELAY", "DELAY_TREND",
    "BILL_VOLATILITY", "MAX_DELAY_STREAK", "AGE", "LIMIT_BAL",
]


def build_models() -> dict:
    """Candidates, each correcting the ~78/22 class imbalance.

    Without class_weight the models optimise overall accuracy, which on this
    split is maximised by predicting "no default" almost every time — high
    accuracy, near-useless recall on the class that matters.
    """
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        # Trees are scale-invariant, so no scaler is needed; the pipeline keeps
        # both models behind one interface.
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=20,
                class_weight="balanced", n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def tune_threshold(y_true, probabilities) -> tuple[float, float]:
    """Pick the probability cut-off that maximises F1 on the default class.

    The default 0.5 is arbitrary here: with the positive class at 22% it
    trades away most of the recall that makes the model useful.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    best = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best]), float(f1[best])


def evaluate(name: str, pipeline, X_test, y_test) -> dict:
    """Score one fitted model at both the default and the tuned threshold."""
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)
    threshold, tuned_f1 = tune_threshold(y_test, probabilities)

    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    print(f"\n  {name}")
    print(f"    ROC-AUC {roc_auc:.4f}   PR-AUC {pr_auc:.4f}")
    print(f"    at default t=0.50: F1 "
          f"{f1_score(y_test, (probabilities >= 0.5).astype(int)):.4f}")
    print(f"    at tuned   t={threshold:.3f}: F1 {tuned_f1:.4f}")
    print(f"    caught {tp:,} of {tp + fn:,} defaulters "
          f"({tp / (tp + fn):.1%}); flagged {tp + fp:,}, "
          f"{tp / (tp + fp):.1%} of them real")

    return {
        "name": name, "roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
        "threshold": threshold, "f1": tuned_f1,
        "probabilities": probabilities,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run(verbose: bool = True) -> dict:
    df = build_features(verbose=verbose)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  train {len(X_train):,} rows   test {len(X_test):,} rows   "
          f"default rate {y.mean():.2%}")

    print("\n" + "=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)

    results = {}
    for name, pipeline in build_models().items():
        pipeline.fit(X_train, y_train)
        results[name] = evaluate(name, pipeline, X_test, y_test)
        results[name]["pipeline"] = pipeline

    best = max(results.values(), key=lambda r: r["f1"])
    print(f"\n  best by tuned F1: {best['name']}")

    print("\n  classification report at the tuned threshold:")
    predictions = (best["probabilities"] >= best["threshold"]).astype(int)
    print(classification_report(y_test, predictions,
                                target_names=["repaid", "defaulted"],
                                digits=3))

    _plot_curves(y_test, results)
    _plot_importance(results["random_forest"]["pipeline"])
    return results


def _plot_curves(y_test, results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result["probabilities"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {result['roc_auc']:.3f})")

        precision, recall, _ = precision_recall_curve(
            y_test, result["probabilities"]
        )
        axes[1].plot(recall, precision,
                     label=f"{name} (AP {result['pr_auc']:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC")
    axes[0].legend()

    axes[1].axhline(y_test.mean(), color="grey", linestyle="--", linewidth=1,
                    label=f"base rate ({y_test.mean():.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall")
    axes[1].legend()

    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "model_curves.png", dpi=150)
    plt.close(fig)
    print("  saved figures/model_curves.png")


def _plot_importance(pipeline) -> None:
    importances = pipeline.named_steps["clf"].feature_importances_
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([FEATURES[i] for i in order], importances[order],
            color="steelblue", alpha=0.85)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importance")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("  saved figures/feature_importance.png")


def main() -> None:
    try:
        run()
    except MissingDataError as exc:
        print(f"\n{exc}\n")
        raise SystemExit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
