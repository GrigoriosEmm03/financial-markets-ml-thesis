# Importing the Libraries
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ==============================================================================
# PROJECT AEGIS - STOCKS EDITION (v4)
#
# Training orchestrator built on top of ml_preprocessing_pipeline_v4.py.
# This file is intentionally runnable as a script. It trains:
#   1) AegisTrader_Stocks_Day   -> Target_Day_Win
#   2) AegisTrader_Stocks_Swing -> Target_Swing_Win
#   3) AegisTrader_Stocks_Long  -> Target_Long_Win
# ==============================================================================

ALGORITHM_FAMILY_NAME = "AegisTrader_Stocks"
DATASET_NAME = "STOCKS"

DATASET_PATH = (
    r"C:\Users\grego\OneDrive\Υπολογιστής\Π.Μ.Σ. ΣΤΗΝ ΑΝΑΛΥΤΙΚΗ ΤΩΝ ΕΠΙΧΕΙΡΗΣΕΩΝ & "
    r"ΣΤΗΝ ΕΠΙΣΤΗΜΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ\ΔΙΠΛΩΜΑΤΙΚΗ\CSV Files\5. Long Horizon And Features Addition"
    r"\DATASET_STOCKS_ML_READY_v3.csv"
)

OUTPUT_BASE_DIR = (
    r"C:\Users\grego\OneDrive\Υπολογιστής\Π.Μ.Σ. ΣΤΗΝ ΑΝΑΛΥΤΙΚΗ ΤΩΝ ΕΠΙΧΕΙΡΗΣΕΩΝ & "
    r"ΣΤΗΝ ΕΠΙΣΤΗΜΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ\ΔΙΠΛΩΜΑΤΙΚΗ\Terminal Outputs\Model Training Evaluation Results - STOCKS"
)
GRAPH_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "Graphical Output")
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "Saved Models")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42
PERMUTATION_SAMPLE_SIZE = 5000
MIN_TICKER_HISTORY = 500
BOOTSTRAP_ITERATIONS = 300

COLOR_LOSING = "#D62728"   # red
COLOR_WINNING = "#2CA02C"  # green

HORIZON_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Day": {
        "target": "Target_Day_Win",
        "model_name": "AegisTrader_Stocks_Day",
        "description": "Short-term stocks trading model for daily signals.",
        "look_forward": 3,
        "atr_mult": 1.5,
        "reward_to_risk": 2.0,
        "purge_gap_days": 3,
        "subsampling_ratio": 0.25,
        "top_k_pct": 0.10,
    },
    "Swing": {
        "target": "Target_Swing_Win",
        "model_name": "AegisTrader_Stocks_Swing",
        "description": "Medium-term stocks trading model for within-week trading signals.",
        "look_forward": 14,
        "atr_mult": 2.5,
        "reward_to_risk": 3.0,
        "purge_gap_days": 14,
        "subsampling_ratio": 0.30,
        "top_k_pct": 0.08,
    },
    "Long": {
        "target": "Target_Long_Win",
        "model_name": "AegisTrader_Stocks_Long",
        "description": "Long-term stocks trading model for long-horizon positions.",
        "look_forward": 60,
        "atr_mult": 4.0,
        "reward_to_risk": 4.0,
        "purge_gap_days": 60,
        "subsampling_ratio": 0.35,
        "top_k_pct": 0.05,
    },
}

RF_PARAM_GRID: List[Dict[str, Any]] = [
    {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 20, "max_features": "sqrt"},
    {"n_estimators": 600, "max_depth": 8,  "min_samples_leaf": 30, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 20, "max_features": "sqrt"},
]

HGB_PARAM_GRID: List[Dict[str, Any]] = [
    {"learning_rate": 0.05, "max_iter": 300, "max_depth": 6, "min_samples_leaf": 50,  "l2_regularization": 0.10},
    {"learning_rate": 0.03, "max_iter": 400, "max_depth": 5, "min_samples_leaf": 80,  "l2_regularization": 0.20},
    {"learning_rate": 0.07, "max_iter": 220, "max_depth": 4, "min_samples_leaf": 100, "l2_regularization": 0.00},
]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    import ml_preprocessing_pipeline_v4 as prep
except ImportError as exc:
    raise ImportError(
        "Could not import 'ml_preprocessing_pipeline_v4.py'. Place it in the same folder as model_training_stocks_v4.py."
    ) from exc


@dataclass
class PreparedData4:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val_select: pd.DataFrame
    y_val_select: pd.Series
    X_val_calib: pd.DataFrame
    y_val_calib: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    metadata: Dict[str, Any]


def ensure_directories() -> None:
    for folder in (OUTPUT_BASE_DIR, GRAPH_OUTPUT_DIR, MODEL_OUTPUT_DIR):
        os.makedirs(folder, exist_ok=True)


def print_section_separator(char: str = "=", width: int = 100) -> None:
    print(char * width)


def sanitize_filename(text: str) -> str:
    bad_chars = '<>:"/\\|?*'
    return "".join("_" if ch in bad_chars else ch for ch in text).replace(" ", "_")


def primary_threshold_for(reward_to_risk: float) -> float:
    return 1.0 / (1.0 + float(reward_to_risk))


def break_even_hit_rate_for(reward_to_risk: float) -> float:
    return primary_threshold_for(reward_to_risk)


def safe_predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("Model has neither predict_proba nor decision_function.")


def compute_class_weight_dict_from_meta(metadata: Dict[str, Any]) -> Dict[int, float]:
    pos_weight = metadata.get("recommended_positive_class_weight")
    if pos_weight is None or not np.isfinite(pos_weight) or pos_weight <= 0:
        return {0: 1.0, 1: 1.0}
    return {0: 1.0, 1: float(pos_weight)}


def compute_sample_weights(y: pd.Series, class_weights: Dict[int, float]) -> np.ndarray:
    return np.array([class_weights[int(label)] for label in y], dtype=float)


def precision_at_top_k(y_true: pd.Series | np.ndarray, y_prob: np.ndarray, top_k_ratio: float) -> float:
    n = len(y_prob)
    if n == 0 or top_k_ratio <= 0:
        return float("nan")
    k = max(1, int(math.ceil(n * top_k_ratio)))
    idx = np.argsort(y_prob)[::-1][:k]
    y_top = np.asarray(y_true).astype(int)[idx]
    return float(np.mean(y_top)) if len(y_top) else float("nan")


def lift_at_top_k(y_true: pd.Series | np.ndarray, y_prob: np.ndarray, top_k_ratio: float) -> float:
    p_topk = precision_at_top_k(y_true, y_prob, top_k_ratio)
    base_rate = float(np.mean(np.asarray(y_true).astype(int)))
    if base_rate <= 0 or math.isnan(p_topk):
        return float("nan")
    return p_topk / base_rate


def threshold_from_top_k(y_prob: np.ndarray, top_k_pct: float) -> float:
    if len(y_prob) == 0:
        return 0.5
    top_k_pct = float(np.clip(top_k_pct, 1e-6, 1.0 - 1e-6))
    return float(np.quantile(y_prob, 1.0 - top_k_pct))


def expected_utility_per_signal(y_true: pd.Series | np.ndarray, y_prob: np.ndarray, threshold: float, reward_to_risk: float) -> float:
    y_true_arr = np.asarray(y_true).astype(int)
    mask = y_prob >= threshold
    if mask.sum() == 0:
        return float("nan")
    pnl = np.where(y_true_arr[mask] == 1, float(reward_to_risk), -1.0)
    return float(np.mean(pnl))


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iter: int = BOOTSTRAP_ITERATIONS,
    ci: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> Tuple[float, float]:
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return float("nan"), float("nan")
    scores: List[float] = []
    for _ in range(n_iter):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        try:
            scores.append(float(metric_fn(y_true[idx], y_prob[idx])))
        except Exception:
            continue
    if not scores:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1.0 - alpha))


def evaluate_full_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    primary_threshold: float,
    top_k_ratio: float,
    reward_to_risk: float,
    compute_bootstrap_ci: bool = False,
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred = (y_prob >= primary_threshold).astype(int)
    base_rate = float(np.mean(y_true_arr)) if len(y_true_arr) else float("nan")

    def safe_metric(fn, default=float("nan")):
        try:
            return float(fn())
        except Exception:
            return default

    roc_auc = safe_metric(lambda: roc_auc_score(y_true_arr, y_prob))
    ap = safe_metric(lambda: average_precision_score(y_true_arr, y_prob))
    p_topk = precision_at_top_k(y_true_arr, y_prob, top_k_ratio)
    lift_topk = lift_at_top_k(y_true_arr, y_prob, top_k_ratio)
    brier = safe_metric(lambda: brier_score_loss(y_true_arr, y_prob))
    ll = safe_metric(lambda: log_loss(y_true_arr, np.clip(y_prob, 1e-7, 1.0 - 1e-7), labels=[0, 1]))
    precision = safe_metric(lambda: precision_score(y_true_arr, y_pred, zero_division=0))
    recall = safe_metric(lambda: recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = safe_metric(lambda: f1_score(y_true_arr, y_pred, zero_division=0))
    bal_acc = safe_metric(lambda: balanced_accuracy_score(y_true_arr, y_pred))
    mcc = safe_metric(lambda: matthews_corrcoef(y_true_arr, y_pred))

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    break_even = break_even_hit_rate_for(reward_to_risk)
    eu = expected_utility_per_signal(y_true_arr, y_prob, primary_threshold, reward_to_risk)
    predicted_pos_rate = float(np.mean(y_pred)) if len(y_pred) else float("nan")
    pred_to_actual_ratio = predicted_pos_rate / base_rate if base_rate and base_rate > 0 else float("nan")

    metrics: Dict[str, Any] = {
        "roc_auc": roc_auc,
        "average_precision": ap,
        "precision_at_top_k": p_topk,
        "lift_at_top_k": lift_topk,
        "top_k_pct": float(top_k_ratio),
        "brier_score": brier,
        "log_loss": ll,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "balanced_accuracy": bal_acc,
        "confusion_TN": tn,
        "confusion_FP": fp,
        "confusion_FN": fn,
        "confusion_TP": tp,
        "reward_to_risk": float(reward_to_risk),
        "break_even_hit_rate": float(break_even),
        "hit_rate_vs_break_even": precision - break_even if not math.isnan(precision) else float("nan"),
        "expected_utility_per_signal": eu,
        "primary_threshold": float(primary_threshold),
        "positive_rate": base_rate,
        "predicted_positive_rate": predicted_pos_rate,
        "predicted_to_actual_ratio": pred_to_actual_ratio,
    }

    if compute_bootstrap_ci:
        metrics["roc_auc_ci95"] = stratified_bootstrap_ci(y_true_arr, y_prob, roc_auc_score)
        metrics["average_precision_ci95"] = stratified_bootstrap_ci(y_true_arr, y_prob, average_precision_score)
    return metrics


def compute_selection_score(metrics: Dict[str, Any]) -> float:
    ap = metrics.get("average_precision", float("nan"))
    lift = metrics.get("lift_at_top_k", float("nan"))
    roc = metrics.get("roc_auc", float("nan"))
    return float(
        (0 if math.isnan(ap) else ap)
        + 0.25 * (0 if math.isnan(lift) else lift)
        + 0.05 * (0 if math.isnan(roc) else roc)
    )


def make_logistic_model(class_weights: Dict[int, float]) -> LogisticRegression:
    return LogisticRegression(
        class_weight=class_weights,
        max_iter=4000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )


def make_random_forest_model(class_weights: Dict[int, float], params: Dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight=class_weights,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_hgb_model(params: Dict[str, Any]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def wrap_with_isotonic_calibration(fitted_base: Any, X_calib: pd.DataFrame, y_calib: pd.Series) -> Any:
    """Version-compatible isotonic calibration for already fitted estimators."""
    try:
        from sklearn.frozen import FrozenEstimator
        calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(fitted_base), method="isotonic", cv=None)
        calibrator.fit(X_calib, y_calib)
        return calibrator
    except Exception:
        pass

    try:
        calibrator = CalibratedClassifierCV(estimator=fitted_base, method="isotonic", cv="prefit")
    except TypeError:
        calibrator = CalibratedClassifierCV(base_estimator=fitted_base, method="isotonic", cv="prefit")
    calibrator.fit(X_calib, y_calib)
    return calibrator


def prepare_data_for_model(
    df: pd.DataFrame,
    target_col: str,
    horizon: str,
    purge_gap_days: int,
    subsampling_ratio: float,
    scale_for_model: bool,
) -> PreparedData4:
    result = prep.prepare_tabular_data(
        df=df,
        target_col=target_col,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        scale_for_model=scale_for_model,
        scaler_type="minmax",
        purge_gap_days=purge_gap_days,
        apply_subsampling=True,
        subsampling_ratio=subsampling_ratio,
        apply_winsor=True,
        horizon=horizon,
        min_ticker_history=MIN_TICKER_HISTORY,
        random_state=RANDOM_STATE,
    )
    return PreparedData4(*result)


def compute_permutation_importance_safe(model: Any, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame | None:
    try:
        if len(X) > PERMUTATION_SAMPLE_SIZE:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(len(X), size=PERMUTATION_SAMPLE_SIZE, replace=False)
            X_eval, y_eval = X.iloc[idx].copy(), y.iloc[idx].copy()
        else:
            X_eval, y_eval = X.copy(), y.copy()
        result = permutation_importance(
            model,
            X_eval,
            y_eval,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="average_precision",
            n_jobs=1,
        )
        return pd.DataFrame({
            "feature": list(X_eval.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    except Exception:
        return None


def save_class_distribution_plot(y_train, y_val_select, y_val_calib, y_test, title, save_path):
    split_names = ["Train", "Val_Select", "Val_Calib", "Test"]
    positive_rates = [float(np.mean(y_train)), float(np.mean(y_val_select)), float(np.mean(y_val_calib)), float(np.mean(y_test))]
    negative_rates = [1.0 - p for p in positive_rates]
    x = np.arange(len(split_names))
    width = 0.35
    plt.figure(figsize=(11, 6))
    plt.bar(x - width / 2, negative_rates, width=width, label="Losing (0)", color=COLOR_LOSING)
    plt.bar(x + width / 2, positive_rates, width=width, label="Winning (1)", color=COLOR_WINNING)
    for idx, val in enumerate(positive_rates):
        plt.text(idx + width / 2, val + 0.01, f"{val:.2%}", ha="center")
    plt.xticks(x, split_names)
    plt.ylim(0, 1.05)
    plt.ylabel("Class Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_model_comparison_plot(comparison_df: pd.DataFrame, title: str, save_path: str):
    plot_df = comparison_df.sort_values("selection_score", ascending=False).copy()
    metrics_to_plot = ["average_precision", "roc_auc", "precision_at_top_k", "lift_at_top_k"]
    x = np.arange(len(plot_df))
    width = 0.18
    plt.figure(figsize=(14, 7))
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + (i - 1.5) * width, plot_df[metric], width=width, label=metric)
    plt.xticks(x, plot_df["candidate_name"], rotation=18, ha="right")
    plt.ylabel("Validation Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_plot(y_true, y_prob, threshold, title, save_path):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["Pred Losing (0)", "Pred Winning (1)"])
    plt.yticks([0, 1], ["True Losing (0)", "True Winning (1)"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_roc_plot(y_true, y_prob, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}", color=COLOR_WINNING)
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_pr_plot(y_true, y_prob, title, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    base_rate = float(np.mean(y_true))
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"PR-AUC/AP = {ap:.4f}", color=COLOR_WINNING)
    plt.axhline(base_rate, linestyle="--", color="grey", label=f"Base rate = {base_rate:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_probability_histogram(y_true, y_prob, threshold, title, save_path):
    y_arr = np.asarray(y_true).astype(int)
    plt.figure(figsize=(9, 6))
    plt.hist(y_prob[y_arr == 0], bins=40, alpha=0.65, label="Losing (0)", color=COLOR_LOSING)
    plt.hist(y_prob[y_arr == 1], bins=40, alpha=0.65, label="Winning (1)", color=COLOR_WINNING)
    plt.axvline(threshold, linestyle="--", color="black", label=f"tau* = {threshold:.4f}")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_calibration_plot(y_true, y_prob, title, save_path):
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        plt.figure(figsize=(7, 6))
        plt.plot(mean_pred, frac_pos, marker="o", label="Model", color=COLOR_WINNING)
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def save_feature_importance_plot(importance_df: pd.DataFrame | None, title: str, save_path: str, top_n: int = 15):
    if importance_df is None or importance_df.empty:
        return
    plot_df = importance_df.head(top_n).sort_values("importance_mean")
    plt.figure(figsize=(9, 7))
    plt.barh(plot_df["feature"], plot_df["importance_mean"], color=COLOR_WINNING)
    plt.xlabel("Permutation Importance Mean")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def print_terminal_model_summary(config: Dict[str, Any], split_info: Dict[str, Any], feature_count: int, pos_weight: Any):
    print_section_separator()
    print(f"SUB-MODEL: {config['model_name']}")
    print(f"Target: {config['target']}")
    print(f"Description: {config['description']}")
    print(f"Purge gap: {config['purge_gap_days']} days | Subsampling ratio: {config['subsampling_ratio']:.2f} | Top-K: {config['top_k_pct']:.2%}")
    print(f"Reward-to-risk: {config['reward_to_risk']:.2f} | Primary threshold tau*: {primary_threshold_for(config['reward_to_risk']):.4f}")
    print(f"Selected features: {feature_count}")
    print(f"Recommended positive class weight: {pos_weight}")
    print("Split Info:")
    for k, v in split_info.items():
        print(f"    - {k}: {v}")
    print_section_separator()


def print_candidate_summary(candidate_name: str, metrics: Dict[str, Any], selection_score: float):
    print(f"\nTraining candidate: {candidate_name}")
    print(f"    Val_Select ROC-AUC:                {metrics['roc_auc']:.4f}")
    print(f"    Val_Select PR-AUC (AP):            {metrics['average_precision']:.4f}")
    print(f"    Val_Select Precision@Top-K:        {metrics['precision_at_top_k']:.4f}")
    print(f"    Val_Select Lift@Top-K:             {metrics['lift_at_top_k']:.4f}")
    print(f"    Val_Select Precision (tau*):       {metrics['precision']:.4f}")
    print(f"    Val_Select Recall (tau*):          {metrics['recall']:.4f}")
    print(f"    Val_Select F1 (tau*):              {metrics['f1']:.4f}")
    print(f"    Val_Select Predicted Pos Rate:     {metrics['predicted_positive_rate']:.4f}")
    print(f"    Val_Select Actual Pos Rate:        {metrics['positive_rate']:.4f}")
    print(f"    Selection Score:                   {selection_score:.6f}")


def print_tiered_test_summary(model_name: str, test_metrics: Dict[str, Any], val_select_metrics: Dict[str, Any], val_calib_metrics: Dict[str, Any], distribution_shift: Dict[str, float], artifact_path: str):
    print("\n" + "-" * 100)
    print(f"FINAL HOLDOUT TEST PERFORMANCE - {model_name}")
    print("-" * 100)
    print("\n  [Tier 1 - Discrimination]")
    print(f"    Test ROC-AUC:                  {test_metrics['roc_auc']:.4f}")
    print(f"    Test PR-AUC (AP):              {test_metrics['average_precision']:.4f}")
    print(f"    Test Precision@Top-K:          {test_metrics['precision_at_top_k']:.4f}")
    print(f"    Test Lift@Top-K:               {test_metrics['lift_at_top_k']:.4f}")
    print("\n  [Tier 2 - Probability Quality]")
    print(f"    Test Brier Score:              {test_metrics['brier_score']:.6f}")
    print(f"    Test Log-Loss:                 {test_metrics['log_loss']:.6f}")
    print("\n  [Tier 3 - Decision Quality @ tau*]")
    print(f"    Confusion Matrix:              TN={test_metrics['confusion_TN']:>6} | FP={test_metrics['confusion_FP']:>6} | FN={test_metrics['confusion_FN']:>6} | TP={test_metrics['confusion_TP']:>6}")
    print(f"    Test Precision:                {test_metrics['precision']:.4f}")
    print(f"    Test Recall:                   {test_metrics['recall']:.4f}")
    print(f"    Test F1:                       {test_metrics['f1']:.4f}")
    print(f"    Test MCC:                      {test_metrics['mcc']:.4f}")
    print(f"    Test Balanced Accuracy:        {test_metrics['balanced_accuracy']:.4f}")
    print("\n  [Tier 4 - Financial Utility]")
    print(f"    Reward-to-Risk (R/R):          {test_metrics['reward_to_risk']:.2f}")
    print(f"    Break-even Hit Rate:           {test_metrics['break_even_hit_rate']:.4f}")
    print(f"    Hit Rate (Test Precision):     {test_metrics['precision']:.4f}")
    diff = test_metrics["hit_rate_vs_break_even"]
    diff_label = "ABOVE" if isinstance(diff, float) and diff > 0 else "BELOW"
    print(f"    Hit Rate vs Break-even:        {diff:+.4f}  ({diff_label} break-even)" if isinstance(diff, float) and not math.isnan(diff) else "    Hit Rate vs Break-even:        NaN")
    print(f"    Expected Utility per Signal:   {test_metrics['expected_utility_per_signal']:.6f}")
    print("\n  [Tier 5 - Reliability]")
    if "roc_auc_ci95" in test_metrics:
        print(f"    Test ROC-AUC 95% CI:           [{test_metrics['roc_auc_ci95'][0]:.4f}, {test_metrics['roc_auc_ci95'][1]:.4f}]")
    if "average_precision_ci95" in test_metrics:
        print(f"    Test PR-AUC 95% CI:            [{test_metrics['average_precision_ci95'][0]:.4f}, {test_metrics['average_precision_ci95'][1]:.4f}]")
    print(f"    Test Predicted/Actual Ratio:   {test_metrics['predicted_to_actual_ratio']:.4f}")
    print(f"    Val_Calib Predicted/Actual:    {val_calib_metrics['predicted_to_actual_ratio']:.4f}")
    print(f"    Val_Select Predicted/Actual:   {val_select_metrics['predicted_to_actual_ratio']:.4f}")
    print("\n    Distribution Shift Indicators:")
    for k, v in distribution_shift.items():
        print(f"        - {k}: {v:.4f}" if isinstance(v, float) else f"        - {k}: {v}")
    print(f"\n    Saved Artifact: {artifact_path}")


def train_candidate(candidate_name: str, algorithm_name: str, estimator: Any, pack: PreparedData4, sample_weight: np.ndarray | None, primary_threshold: float, top_k_pct: float, reward_to_risk: float) -> Dict[str, Any]:
    print("\n" + "-" * 100)
    print(f"Now evaluating candidate model: {candidate_name}")
    print(f"Algorithm family: {algorithm_name}")
    print("Fitting estimator on the training split...")

    if sample_weight is not None:
        estimator.fit(pack.X_train, pack.y_train, sample_weight=sample_weight)
    else:
        estimator.fit(pack.X_train, pack.y_train)

    print("Generating validation probabilities on val_select...")
    val_prob = safe_predict_proba(estimator, pack.X_val_select)
    metrics = evaluate_full_metrics(pack.y_val_select, val_prob, primary_threshold, top_k_pct, reward_to_risk)
    score = compute_selection_score(metrics)
    print_candidate_summary(candidate_name, metrics, score)
    print(f"Completed evaluation for candidate model: {candidate_name}")
    print("Moving to the next candidate model, if available.")
    return {
        "candidate_name": candidate_name,
        "algorithm_name": algorithm_name,
        "estimator": estimator,
        "metrics": metrics,
        "selection_score": score,
    }


def train_single_horizon_model(df: pd.DataFrame, horizon: str, config: Dict[str, Any]) -> Dict[str, Any]:
    target_col = config["target"]
    model_name = config["model_name"]
    reward_to_risk = float(config["reward_to_risk"])
    primary_threshold = primary_threshold_for(reward_to_risk)
    top_k_pct = float(config["top_k_pct"])

    print("\n" + "=" * 100)
    print(f"TRAINING HORIZON: {horizon} | TARGET: {target_col} | MODEL FAMILY: {model_name}")
    print("=" * 100)

    logistic_pack = prepare_data_for_model(df, target_col, horizon.lower(), config["purge_gap_days"], config["subsampling_ratio"], scale_for_model=True)
    tree_pack = prepare_data_for_model(df, target_col, horizon.lower(), config["purge_gap_days"], config["subsampling_ratio"], scale_for_model=False)
    split_info = tree_pack.metadata.get("split_info", {})
    feature_cols = tree_pack.metadata.get("feature_columns", list(tree_pack.X_train.columns))
    class_weights = compute_class_weight_dict_from_meta(tree_pack.metadata)
    sample_weights_tree = compute_sample_weights(tree_pack.y_train, class_weights)
    sample_weights_log = compute_sample_weights(logistic_pack.y_train, compute_class_weight_dict_from_meta(logistic_pack.metadata))

    print_terminal_model_summary(config, split_info, len(feature_cols), tree_pack.metadata.get("recommended_positive_class_weight"))

    candidates: List[Dict[str, Any]] = []
    candidates.append(train_candidate(
        "LogisticRegression_baseline",
        "LogisticRegression",
        make_logistic_model(compute_class_weight_dict_from_meta(logistic_pack.metadata)),
        logistic_pack,
        None,
        primary_threshold,
        top_k_pct,
        reward_to_risk,
    ))

    for idx, params in enumerate(RF_PARAM_GRID, start=1):
        candidates.append(train_candidate(
            f"RandomForest_cfg{idx}",
            "RandomForest",
            make_random_forest_model(class_weights, params),
            tree_pack,
            None,
            primary_threshold,
            top_k_pct,
            reward_to_risk,
        ))

    for idx, params in enumerate(HGB_PARAM_GRID, start=1):
        candidates.append(train_candidate(
            f"HistGradientBoosting_cfg{idx}",
            "HistGradientBoosting",
            make_hgb_model(params),
            tree_pack,
            sample_weights_tree,
            primary_threshold,
            top_k_pct,
            reward_to_risk,
        ))

    best = max(candidates, key=lambda row: row["selection_score"])
    best_algorithm = best["algorithm_name"]
    best_candidate_name = best["candidate_name"]
    base_estimator = best["estimator"]
    pack = logistic_pack if best_algorithm == "LogisticRegression" else tree_pack

    print("\n" + "-" * 100)
    print(f"Best validation candidate for {model_name}: {best_candidate_name}")
    print(f"Winning algorithm family: {best_algorithm}")
    print("Calibrating winning estimator on val_calib...")

    calibrated_model = wrap_with_isotonic_calibration(base_estimator, pack.X_val_calib, pack.y_val_calib)

    val_select_prob = safe_predict_proba(calibrated_model, pack.X_val_select)
    val_calib_prob = safe_predict_proba(calibrated_model, pack.X_val_calib)
    test_prob = safe_predict_proba(calibrated_model, pack.X_test)
    top_k_threshold = threshold_from_top_k(val_select_prob, top_k_pct)

    val_select_metrics = evaluate_full_metrics(pack.y_val_select, val_select_prob, primary_threshold, top_k_pct, reward_to_risk)
    val_calib_metrics = evaluate_full_metrics(pack.y_val_calib, val_calib_prob, primary_threshold, top_k_pct, reward_to_risk)
    test_metrics = evaluate_full_metrics(pack.y_test, test_prob, primary_threshold, top_k_pct, reward_to_risk, compute_bootstrap_ci=True)

    distribution_shift = {
        "val_select_positive_rate": val_select_metrics["positive_rate"],
        "val_calib_positive_rate": val_calib_metrics["positive_rate"],
        "test_positive_rate": test_metrics["positive_rate"],
        "test_minus_val_select_positive_rate": test_metrics["positive_rate"] - val_select_metrics["positive_rate"],
        "test_minus_val_calib_positive_rate": test_metrics["positive_rate"] - val_calib_metrics["positive_rate"],
    }

    timestampless_name = sanitize_filename(model_name)
    artifact_path = os.path.join(MODEL_OUTPUT_DIR, f"{timestampless_name}.joblib")

    importance_df = compute_permutation_importance_safe(calibrated_model, pack.X_val_select, pack.y_val_select)

    artifact = {
        "model": calibrated_model,
        "base_estimator": base_estimator,
        "algorithm_family_name": ALGORITHM_FAMILY_NAME,
        "dataset_name": DATASET_NAME,
        "model_name": model_name,
        "description": config["description"],
        "target_col": target_col,
        "horizon": horizon,
        "winner_algorithm": best_algorithm,
        "winner_candidate_name": best_candidate_name,
        "primary_threshold": primary_threshold,
        "top_k_threshold": top_k_threshold,
        "top_k_pct": top_k_pct,
        "reward_to_risk": reward_to_risk,
        "atr_mult": config["atr_mult"],
        "look_forward": config["look_forward"],
        "feature_columns": list(pack.X_train.columns),
        "scaler": pack.metadata.get("scaler"),
        "winsor_bounds": pack.metadata.get("winsor_bounds"),
        "split_info": split_info,
        "metadata": pack.metadata,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(artifact, artifact_path)

    # Plots.
    save_class_distribution_plot(pack.y_train, pack.y_val_select, pack.y_val_calib, pack.y_test, f"{model_name} - Class Distribution", os.path.join(GRAPH_OUTPUT_DIR, f"{timestampless_name}_class_distribution.png"))
    comparison_df = pd.DataFrame([
        {"algorithm_name": c["algorithm_name"], "candidate_name": c["candidate_name"], "selection_score": c["selection_score"], **c["metrics"]}
        for c in candidates
    ]).sort_values("selection_score", ascending=False)
    save_model_comparison_plot(comparison_df, f"{model_name} - Val_Select Model Comparison", os.path.join(GRAPH_OUTPUT_DIR, f"{timestampless_name}_validation_model_comparison.png"))
    save_confusion_matrix_plot(pack.y_test, test_prob, primary_threshold, f"{model_name} - Test Confusion Matrix", os.path.join(GRAPH_OUTPUT_DIR, f"{timestampless_name}_test_confusion_matrix.png"))
    save_roc_plot(pack.y_test, test_prob, f"{model_name} - Test ROC Curve", os.path.join(GRAPH_OUTPUT_DIR, f"{timestampless_name}_test_roc_curve.png"))
    save_pr_plot(pack.y_test, test_prob, f"{model_name} - Test Precision-Recall Curve", os.path.join(GRAPH_OUTPUT_DIR, f"{timestampless_name}_test_pr_curve.png"))
    save_probability_histogram(pack.y_test, test_prob, primary_threshold, f"{model_name} - Test Probability Histogram", os.path.join(GRAPH_OUTPUT_DIR, f"{timestampless_name}_test_probability_histogram.png"))
    save_calibration_plot(pack.y_test, test_prob, f"{model_name} - Test Calibration Curve", os.path.join(GRAPH_OUTPUT_DIR, f"{timestampless_name}_test_calibration_curve.png"))
    save_feature_importance_plot(importance_df, f"{model_name} - Val_Select Feature Importance", os.path.join(GRAPH_OUTPUT_DIR, f"{timestampless_name}_validation_feature_importance.png"))

    print_tiered_test_summary(model_name, test_metrics, val_select_metrics, val_calib_metrics, distribution_shift, artifact_path)

    result = {
        "model_name": model_name,
        "description": config["description"],
        "target_col": target_col,
        "winner_algorithm": best_algorithm,
        "winner_candidate_name": best_candidate_name,
        "primary_threshold": primary_threshold,
        "top_k_threshold": top_k_threshold,
        "top_k_pct": top_k_pct,
        "reward_to_risk": reward_to_risk,
        "artifact_path": artifact_path,
        "recommended_positive_class_weight": pack.metadata.get("recommended_positive_class_weight"),
        "feature_columns": list(pack.X_train.columns),
        "val_select_metrics": val_select_metrics,
        "val_calib_metrics": val_calib_metrics,
        "final_test_metrics": test_metrics,
        "distribution_shift": distribution_shift,
        "split_info": split_info,
        "candidate_comparison": comparison_df.drop(columns=[], errors="ignore").to_dict(orient="records"),
    }
    return result


def make_report_text(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("PROJECT AEGIS - STOCKS EDITION (v4)")
    lines.append("MODEL TRAINING EVALUATION RESULTS")
    lines.append("=" * 100)
    lines.append(f"Algorithm Family: {ALGORITHM_FAMILY_NAME}")
    lines.append(f"Dataset Path: {DATASET_PATH}")
    lines.append("Training orchestrator uses ml_preprocessing_pipeline_v4.py")
    lines.append("")
    lines.append("v4 features: 4-way split, held-out calibration, winsorization, min-history filter, cost-sensitive thresholding.")

    for res in results:
        lines.append("\n" + "-" * 100)
        lines.append(f"SUB-MODEL: {res['model_name']}")
        lines.append(f"Target: {res['target_col']}")
        lines.append(f"Description: {res['description']}")
        lines.append(f"Winner Algorithm: {res['winner_algorithm']}")
        lines.append(f"Winner Candidate: {res['winner_candidate_name']}")
        lines.append(f"Primary Threshold: {res['primary_threshold']:.6f}")
        lines.append(f"Top-K Threshold: {res['top_k_threshold']:.6f}")
        lines.append(f"Top-K%: {res['top_k_pct']:.4f}")
        lines.append(f"Reward-to-Risk: {res['reward_to_risk']:.4f}")
        lines.append(f"Recommended Positive Class Weight: {res['recommended_positive_class_weight']}")
        lines.append(f"Saved Artifact: {res['artifact_path']}")
        lines.append("\nSplit Info:")
        for k, v in res["split_info"].items():
            lines.append(f"    - {k}: {v}")
        lines.append("\nVal_Select Metrics:")
        for k, v in res["val_select_metrics"].items():
            lines.append(format_report_value(k, v))
        lines.append("\nVal_Calib Metrics:")
        for k, v in res["val_calib_metrics"].items():
            lines.append(format_report_value(k, v))
        lines.append("\nFinal Test Metrics:")
        for k, v in res["final_test_metrics"].items():
            lines.append(format_report_value(k, v))
        lines.append("\nDistribution Shift:")
        for k, v in res["distribution_shift"].items():
            lines.append(format_report_value(k, v))
        lines.append("\nCandidate Comparison Table:")
        try:
            comp_df = pd.DataFrame(res["candidate_comparison"])
            lines.append(comp_df.to_string(index=False))
        except Exception:
            lines.append("Candidate comparison could not be rendered.")
    lines.append("\n" + "=" * 100)
    lines.append("END OF REPORT")
    lines.append("=" * 100)
    return "\n".join(lines)


def format_report_value(key: str, value: Any) -> str:
    if isinstance(value, float):
        return f"    - {key}: {value:.6f}" if not math.isnan(value) else f"    - {key}: NaN"
    if isinstance(value, tuple) and len(value) == 2:
        return f"    - {key}: [{value[0]:.6f}, {value[1]:.6f}]"
    return f"    - {key}: {value}"


def json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def main() -> None:
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print_section_separator()
    print(f"INITIALIZING {ALGORITHM_FAMILY_NAME} (v4)")
    print_section_separator()
    print(f"[*] Loading STOCKS dataset from explicit DATASET_PATH")
    print(f"    Path: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"STOCKS dataset was not found at: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"[*] Loaded shape: {df.shape}")
    print(f"[*] Columns: {list(df.columns)}")

    results: List[Dict[str, Any]] = []
    for horizon, config in HORIZON_CONFIGS.items():
        results.append(train_single_horizon_model(df, horizon, config))

    report_path = os.path.join(OUTPUT_BASE_DIR, f"Stocks_Model_Training_Evaluation_Results_v4_{timestamp}.txt")
    json_path = os.path.join(OUTPUT_BASE_DIR, f"Stocks_Model_Training_Summary_v4_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(make_report_text(results))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=json_default)

    print("\n" + "=" * 100)
    print("MODEL TRAINING COMPLETED")
    print("=" * 100)
    print(f"Report saved to: {report_path}")
    print(f"JSON summary saved to: {json_path}")
    print(f"Graphs saved to: {GRAPH_OUTPUT_DIR}")
    print(f"Model artifacts saved to: {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
