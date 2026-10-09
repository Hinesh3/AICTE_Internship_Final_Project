# -*- coding: utf-8 -*-
import sys, io
# Force UTF-8 output on Windows so emoji in print() don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
"""
train.py
========
Training script for the Water Quality Prediction Agent.

Pipeline:
  1. Load data/water_potability.csv
  2. Impute missing values (median strategy)
  3. Scale features (StandardScaler)
  4. Compare Logistic Regression, Random Forest, XGBoost via stratified 5-fold CV
  5. Pick best model by ROC-AUC
  6. Evaluate on held-out test set → print full report
  7. Save best model → models/model.pkl
  8. Save feature importance → outputs/feature_importance.png
  9. Save confusion matrix → outputs/confusion_matrix.png
 10. Save metrics CSV → outputs/metrics.csv

Run:
    python train.py
"""

import os
import sys
import warnings
import pathlib

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend for saving PNGs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.pipeline         import Pipeline
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).parent
DATA_DIR = ROOT / "data"
MDL_DIR  = ROOT / "models"
OUT_DIR  = ROOT / "outputs"

for d in (DATA_DIR, MDL_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATA_PATH  = DATA_DIR / "water_potability.csv"
MODEL_PATH = MDL_DIR  / "model.pkl"

# ── Colour palette (ocean theme) ─────────────────────────────────────────
OCEAN_BLUE   = "#0077B6"
OCEAN_TEAL   = "#00B4D8"
OCEAN_LIGHT  = "#ADE8F4"
OCEAN_DARK   = "#03045E"
RED_WARN     = "#E63946"
GREEN_SAFE   = "#2DC653"

plt.rcParams.update({
    "figure.facecolor": OCEAN_DARK,
    "axes.facecolor":   "#023E8A",
    "axes.edgecolor":   OCEAN_TEAL,
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "text.color":       "white",
    "grid.color":       "#0096C7",
    "grid.alpha":       0.3,
    "font.family":      "DejaVu Sans",
})

# ── 1. Load data ──────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        sys.exit(
            f"\n[ERROR] Dataset not found at '{DATA_PATH}'.\n"
            "Please download 'water_potability.csv' from Kaggle and place it in the 'data/' folder.\n"
        )

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded {len(df)} rows, {df.shape[1]} columns.")
    print(f"[INFO] Class distribution:\n{df['Potability'].value_counts()}\n")

    X = df.drop(columns=["Potability"])
    y = df["Potability"]
    return X, y


# ── 2. Build pipelines ────────────────────────────────────────────────────
def build_pipelines() -> dict:
    """Return dict of {name: sklearn Pipeline}."""

    # Shared preprocessing steps
    preprocessor = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]

    # Imbalance ratio  (approx 60 : 40 → weight unsafe class higher)
    lr_weight = {0: 1.5, 1: 1.0}

    pipelines = {
        "Logistic Regression": Pipeline(
            preprocessor + [
                ("clf", LogisticRegression(
                    class_weight=lr_weight,
                    max_iter=1000,
                    random_state=42,
                ))
            ]
        ),
        "Random Forest": Pipeline(
            preprocessor + [
                ("clf", RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ))
            ]
        ),
        "XGBoost": Pipeline(
            preprocessor + [
                ("clf", _get_xgboost())
            ]
        ),
    }
    return pipelines


def _get_xgboost():
    """Return XGBClassifier if available, else None."""
    try:
        from xgboost import XGBClassifier  # type: ignore
        return XGBClassifier(
            n_estimators=300,
            scale_pos_weight=1.5,   # handle imbalance
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        print("[WARN] XGBoost not installed — skipping.")
        return None


# ── 3. Cross-validation ───────────────────────────────────────────────────
SCORING = {
    "accuracy":  "accuracy",
    "precision": "precision",
    "recall":    "recall",
    "f1":        "f1",
    "roc_auc":   "roc_auc",
}


def cross_validate_all(pipelines: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    records = []

    print("=" * 60)
    print("  STRATIFIED 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)

    for name, pipe in pipelines.items():
        if pipe.named_steps.get("clf") is None:
            continue
        print(f"\n► {name}")
        scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"Model": name}
        for metric, key in [
            ("Accuracy",  "test_accuracy"),
            ("Precision", "test_precision"),
            ("Recall",    "test_recall"),
            ("F1",        "test_f1"),
            ("ROC-AUC",   "test_roc_auc"),
        ]:
            mean = scores[key].mean()
            std  = scores[key].std()
            row[metric] = mean
            print(f"  {metric:12s}: {mean:.4f} ± {std:.4f}")
        records.append(row)

    return pd.DataFrame(records).set_index("Model")


# ── 4. Final evaluation on test set ───────────────────────────────────────
def evaluate_on_test(
    pipe: Pipeline,
    X_train, X_test, y_train, y_test,
    model_name: str,
) -> dict:
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
    }

    print(f"\n{'=' * 60}")
    print(f"  FINAL TEST-SET EVALUATION  —  {model_name}")
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Unsafe (0)", "Safe (1)"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n{cm}\n")

    # Recall for unsafe class (index 0)
    unsafe_recall = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    print(f"  Recall for UNSAFE class (Potability=0): {unsafe_recall:.4f}")
    metrics["Unsafe_Recall"] = unsafe_recall

    return metrics, cm


# ── 5. Save artefacts ─────────────────────────────────────────────────────
def save_confusion_matrix(cm: np.ndarray, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(OCEAN_DARK)
    ax.set_facecolor("#023E8A")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=["Unsafe", "Safe"],
        yticklabels=["Unsafe", "Safe"],
        linewidths=0.5,
        linecolor=OCEAN_DARK,
        ax=ax,
        annot_kws={"size": 16, "weight": "bold", "color": "white"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=8)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=8)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=14, color=OCEAN_LIGHT)
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = OUT_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved → {path}")


def save_feature_importance(pipe: Pipeline, feature_names: list[str], model_name: str):
    clf = pipe.named_steps["clf"]

    # Extract importance
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        label = "Feature Importance (Gini)"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        label = "|Coefficient| (Logistic Regression)"
    else:
        print("[WARN] Model has no feature importances — skipping plot.")
        return

    # Sort descending
    idx   = np.argsort(importances)
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    # Colour gradient: ocean blues
    colours = plt.cm.cool(np.linspace(0.3, 1.0, len(names)))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(OCEAN_DARK)
    ax.set_facecolor("#023E8A")

    bars = ax.barh(names, vals, color=colours, edgecolor="none", height=0.6)

    # Annotate bars
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9, color="white"
        )

    ax.set_xlabel(label, fontsize=11)
    ax.set_title(f"Feature Importances — {model_name}", fontsize=13, color=OCEAN_LIGHT, pad=12)
    ax.tick_params(axis="both", colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(OCEAN_TEAL)
    ax.spines["left"].set_color(OCEAN_TEAL)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color=OCEAN_TEAL)

    plt.tight_layout()
    path = OUT_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Feature importance saved → {path}")


def save_metrics(metrics: dict, model_name: str):
    df = pd.DataFrame([{"Model": model_name, **metrics}])
    path = OUT_DIR / "metrics.csv"
    df.to_csv(path, index=False)
    print(f"[INFO] Metrics saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("\n[*] Water Quality Prediction -- Training Pipeline\n")

    # 1. Load
    X, y = load_data()
    feature_names = list(X.columns)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"[INFO] Train size: {len(X_train)}  |  Test size: {len(X_test)}\n")

    # 3. Build pipelines (drop None classifiers)
    pipelines = {
        name: pipe
        for name, pipe in build_pipelines().items()
        if pipe.named_steps.get("clf") is not None
    }

    # 4. Cross-validate
    cv_results = cross_validate_all(pipelines, X_train, y_train)
    print(f"\n{'=' * 60}")
    print("  CV SUMMARY TABLE")
    print(f"{'=' * 60}")
    print(cv_results.round(4).to_string())

    # 5. Pick best by ROC-AUC
    best_name = cv_results["ROC-AUC"].idxmax()
    best_pipe = pipelines[best_name]
    print(f"\n[INFO] Best model by CV ROC-AUC: {best_name}")

    # 6. Final eval
    metrics, cm = evaluate_on_test(
        best_pipe, X_train, X_test, y_train, y_test, best_name
    )

    # 7. Save model
    best_pipe.fit(X_train, y_train)   # re-fit on full train set
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"[INFO] Model saved → {MODEL_PATH}")

    # 8. Save plots
    save_feature_importance(best_pipe, feature_names, best_name)
    save_confusion_matrix(cm, best_name)
    save_metrics(metrics, best_name)

    print(f"\n[DONE] Training complete. Best model: {best_name}\n")


if __name__ == "__main__":
    main()
