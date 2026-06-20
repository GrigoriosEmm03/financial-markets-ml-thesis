# Importing the Libraries
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# ==============================================================================
# PROJECT AEGIS - GLOBAL MODEL EVALUATION
#
# This script reads the JSON summary files produced by the v4 model-training
# scripts for CRYPTO, FOREX, INDICES, and STOCKS. It ranks all 12 trained
# sub-models using their final holdout test performance.
#
# Input:  Four JSON summary files, one per asset class.
# Output: A timestamped text report with a global model leaderboard.
# ============================================================================== 

JSON_INPUT_PATHS: Dict[str, str] = {
    "CRYPTO": (
        r"C:\Users\grego\OneDrive\Υπολογιστής\Π.Μ.Σ. ΣΤΗΝ ΑΝΑΛΥΤΙΚΗ ΤΩΝ ΕΠΙΧΕΙΡΗΣΕΩΝ & "
        r"ΣΤΗΝ ΕΠΙΣΤΗΜΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ\ΔΙΠΛΩΜΑΤΙΚΗ\Terminal Outputs\Model Training Evaluation Results - CRYPTO"
        r"\Crypto_Model_Training_Summary_v4_20260425_230236.json"
    ),
    "FOREX": (
        r"C:\Users\grego\OneDrive\Υπολογιστής\Π.Μ.Σ. ΣΤΗΝ ΑΝΑΛΥΤΙΚΗ ΤΩΝ ΕΠΙΧΕΙΡΗΣΕΩΝ & "
        r"ΣΤΗΝ ΕΠΙΣΤΗΜΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ\ΔΙΠΛΩΜΑΤΙΚΗ\Terminal Outputs\Model Training Evaluation Results - FOREX"
        r"\Forex_Model_Training_Summary_v4_20260425_235021.json"
    ),
    "INDICES": (
        r"C:\Users\grego\OneDrive\Υπολογιστής\Π.Μ.Σ. ΣΤΗΝ ΑΝΑΛΥΤΙΚΗ ΤΩΝ ΕΠΙΧΕΙΡΗΣΕΩΝ & "
        r"ΣΤΗΝ ΕΠΙΣΤΗΜΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ\ΔΙΠΛΩΜΑΤΙΚΗ\Terminal Outputs\Model Training Evaluation Results - INDICES"
        r"\Indices_Model_Training_Summary_v4_20260426_000014.json"
    ),
    "STOCKS": (
        r"C:\Users\grego\OneDrive\Υπολογιστής\Π.Μ.Σ. ΣΤΗΝ ΑΝΑΛΥΤΙΚΗ ΤΩΝ ΕΠΙΧΕΙΡΗΣΕΩΝ & "
        r"ΣΤΗΝ ΕΠΙΣΤΗΜΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ\ΔΙΠΛΩΜΑΤΙΚΗ\Terminal Outputs\Model Training Evaluation Results - STOCKS"
        r"\Stocks_Model_Training_Summary_v4_20260426_000837.json"
    ),
}

OUTPUT_DIR = (
    r"C:\Users\grego\OneDrive\Υπολογιστής\Π.Μ.Σ. ΣΤΗΝ ΑΝΑΛΥΤΙΚΗ ΤΩΝ ΕΠΙΧΕΙΡΗΣΕΩΝ & "
    r"ΣΤΗΝ ΕΠΙΣΤΗΜΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ\ΔΙΠΛΩΜΑΤΙΚΗ\Terminal Outputs\Evaluation of the Models"
)

# Ranking weights.
# The ranking prioritizes out-of-sample discrimination and ranking ability.
# Accuracy is intentionally excluded because these datasets are highly imbalanced.
RANKING_WEIGHTS: Dict[str, float] = {
    "average_precision": 0.40,
    "lift_at_top_k": 0.30,
    "roc_auc": 0.20,
    "precision_at_top_k": 0.10,
}

# Optional penalty weights.
# These penalties prevent models with unstable behavior from being ranked too high.
PENALTY_WEIGHTS: Dict[str, float] = {
    "negative_utility_penalty": 0.10,
    "poor_calibration_penalty": 0.05,
    "below_random_auc_penalty": 0.15,
}


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def ensure_output_dir() -> None:
    """Create the output directory if it does not already exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_float(value: Any, default: float = math.nan) -> float:
    """Convert a value to float safely."""
    try:
        if value is None:
            return default
        value = float(value)
        return value
    except Exception:
        return default


def is_nan(value: Any) -> bool:
    """Return True if a value is NaN-like."""
    try:
        return math.isnan(float(value))
    except Exception:
        return False


def clean_metric(value: Any, default: float = 0.0) -> float:
    """Convert metric values to numeric values suitable for scoring."""
    numeric = safe_float(value, default=default)
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def load_json_summary(asset_class: str, file_path: str) -> List[Dict[str, Any]]:
    """Load a model-training JSON summary file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON summary file not found for {asset_class}: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of model summaries in {file_path}.")

    return data


def infer_horizon(model_name: str, target_col: str) -> str:
    """Infer the horizon from the model name or target column."""
    text = f"{model_name} {target_col}".lower()
    if "day" in text:
        return "Day"
    if "swing" in text:
        return "Swing"
    if "long" in text:
        return "Long"
    return "Unknown"


def compute_global_score(metrics: Dict[str, Any]) -> float:
    """
    Compute a global score for ranking the trained sub-models.

    The score emphasizes metrics that are meaningful for imbalanced financial
    classification problems:
    - Average Precision / PR-AUC
    - Lift at Top-K
    - ROC-AUC
    - Precision at Top-K

    Penalties are applied for clearly undesirable properties:
    - Negative expected utility
    - Poor probability quality
    - ROC-AUC below random baseline
    """
    average_precision = clean_metric(metrics.get("average_precision"))
    lift_at_top_k = clean_metric(metrics.get("lift_at_top_k"))
    roc_auc = clean_metric(metrics.get("roc_auc"))
    precision_at_top_k = clean_metric(metrics.get("precision_at_top_k"))
    expected_utility = safe_float(metrics.get("expected_utility_per_signal"), default=math.nan)
    brier_score = clean_metric(metrics.get("brier_score"), default=0.0)

    raw_score = (
        RANKING_WEIGHTS["average_precision"] * average_precision
        + RANKING_WEIGHTS["lift_at_top_k"] * lift_at_top_k
        + RANKING_WEIGHTS["roc_auc"] * roc_auc
        + RANKING_WEIGHTS["precision_at_top_k"] * precision_at_top_k
    )

    penalty = 0.0

    if not math.isnan(expected_utility) and expected_utility < 0:
        penalty += PENALTY_WEIGHTS["negative_utility_penalty"] * abs(expected_utility)

    penalty += PENALTY_WEIGHTS["poor_calibration_penalty"] * brier_score

    if roc_auc < 0.5:
        penalty += PENALTY_WEIGHTS["below_random_auc_penalty"] * (0.5 - roc_auc)

    return raw_score - penalty


def classify_model_quality(row: Dict[str, Any]) -> str:
    """Assign a simple qualitative label to a model."""
    roc_auc = clean_metric(row.get("roc_auc"))
    average_precision = clean_metric(row.get("average_precision"))
    lift_at_top_k = clean_metric(row.get("lift_at_top_k"))
    global_score = clean_metric(row.get("global_score"))

    if roc_auc >= 0.65 and lift_at_top_k >= 1.50 and average_precision > 0.03:
        return "Strong relative performer"
    if roc_auc >= 0.58 and lift_at_top_k >= 1.00:
        return "Moderate but usable signal"
    if roc_auc >= 0.50 and global_score > 0:
        return "Weak signal"
    return "Unreliable or not useful"


def extract_model_rows(asset_class: str, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract one leaderboard row per trained sub-model."""
    rows: List[Dict[str, Any]] = []

    for model_summary in summaries:
        model_name = str(model_summary.get("model_name", "Unknown_Model"))
        target_col = str(model_summary.get("target_col", "Unknown_Target"))
        winner_algorithm = str(model_summary.get("winner_algorithm", "Unknown_Algorithm"))
        winner_candidate = str(model_summary.get("winner_candidate_name", "Unknown_Candidate"))
        horizon = infer_horizon(model_name, target_col)
        metrics = model_summary.get("final_test_metrics", {})

        if not isinstance(metrics, dict):
            raise ValueError(f"Missing or invalid final_test_metrics for {model_name}.")

        global_score = compute_global_score(metrics)

        row = {
            "asset_class": asset_class,
            "horizon": horizon,
            "model_name": model_name,
            "target_col": target_col,
            "winner_algorithm": winner_algorithm,
            "winner_candidate": winner_candidate,
            "global_score": global_score,
            "roc_auc": clean_metric(metrics.get("roc_auc")),
            "average_precision": clean_metric(metrics.get("average_precision")),
            "precision_at_top_k": clean_metric(metrics.get("precision_at_top_k")),
            "lift_at_top_k": clean_metric(metrics.get("lift_at_top_k")),
            "brier_score": clean_metric(metrics.get("brier_score")),
            "log_loss": clean_metric(metrics.get("log_loss")),
            "precision": clean_metric(metrics.get("precision")),
            "recall": clean_metric(metrics.get("recall")),
            "f1": clean_metric(metrics.get("f1")),
            "mcc": clean_metric(metrics.get("mcc")),
            "balanced_accuracy": clean_metric(metrics.get("balanced_accuracy")),
            "expected_utility_per_signal": safe_float(metrics.get("expected_utility_per_signal"), default=math.nan),
            "positive_rate": clean_metric(metrics.get("positive_rate")),
            "predicted_positive_rate": clean_metric(metrics.get("predicted_positive_rate")),
            "predicted_to_actual_ratio": clean_metric(metrics.get("predicted_to_actual_ratio")),
        }
        row["quality_label"] = classify_model_quality(row)
        rows.append(row)

    return rows


def make_leaderboard(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a sorted leaderboard dataframe."""
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No model rows were extracted. Please check the JSON input files.")

    df = df.sort_values(
        by=["global_score", "roc_auc", "average_precision", "lift_at_top_k"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def format_float(value: Any, decimals: int = 6) -> str:
    """Format float values for reports."""
    numeric = safe_float(value, default=math.nan)
    if math.isnan(numeric):
        return "NaN"
    return f"{numeric:.{decimals}f}"


def build_report(leaderboard: pd.DataFrame) -> str:
    """Build the final text report."""
    lines: List[str] = []

    lines.append("=" * 120)
    lines.append("PROJECT AEGIS - GLOBAL MODEL EVALUATION")
    lines.append("=" * 120)
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Input JSON files:")
    for asset_class, path in JSON_INPUT_PATHS.items():
        lines.append(f"  - {asset_class}: {path}")
    lines.append("")

    lines.append("Ranking methodology:")
    lines.append("  The ranking is based on final holdout test metrics from the JSON summary files.")
    lines.append("  The global score prioritizes metrics that are suitable for imbalanced financial classification problems.")
    lines.append("  Accuracy is intentionally excluded because the target classes are highly imbalanced.")
    lines.append("")
    lines.append("Global score formula:")
    lines.append("  score = 0.40 * Average Precision + 0.30 * Lift@TopK + 0.20 * ROC-AUC + 0.10 * Precision@TopK")
    lines.append("          - utility/calibration/below-random penalties")
    lines.append("")

    display_columns = [
        "rank",
        "model_name",
        "asset_class",
        "horizon",
        "winner_algorithm",
        "global_score",
        "roc_auc",
        "average_precision",
        "precision_at_top_k",
        "lift_at_top_k",
        "brier_score",
        "expected_utility_per_signal",
        "quality_label",
    ]

    table_df = leaderboard[display_columns].copy()
    float_columns = [
        "global_score",
        "roc_auc",
        "average_precision",
        "precision_at_top_k",
        "lift_at_top_k",
        "brier_score",
        "expected_utility_per_signal",
    ]
    for col in float_columns:
        table_df[col] = table_df[col].apply(lambda x: format_float(x, 6))

    lines.append("GLOBAL LEADERBOARD - ALL 12 TRAINED SUB-MODELS")
    lines.append("-" * 120)
    lines.append(table_df.to_string(index=False))
    lines.append("")

    lines.append("DETAILED MODEL METRICS")
    lines.append("-" * 120)
    for _, row in leaderboard.iterrows():
        lines.append(f"Rank {int(row['rank'])}: {row['model_name']}")
        lines.append(f"  Asset Class:                 {row['asset_class']}")
        lines.append(f"  Horizon:                     {row['horizon']}")
        lines.append(f"  Winner Algorithm:            {row['winner_algorithm']}")
        lines.append(f"  Winner Candidate:            {row['winner_candidate']}")
        lines.append(f"  Global Score:                {format_float(row['global_score'])}")
        lines.append(f"  ROC-AUC:                     {format_float(row['roc_auc'])}")
        lines.append(f"  Average Precision / PR-AUC:  {format_float(row['average_precision'])}")
        lines.append(f"  Precision@TopK:              {format_float(row['precision_at_top_k'])}")
        lines.append(f"  Lift@TopK:                   {format_float(row['lift_at_top_k'])}")
        lines.append(f"  Brier Score:                 {format_float(row['brier_score'])}")
        lines.append(f"  Log-Loss:                    {format_float(row['log_loss'])}")
        lines.append(f"  Precision at Primary Tau:    {format_float(row['precision'])}")
        lines.append(f"  Recall at Primary Tau:       {format_float(row['recall'])}")
        lines.append(f"  F1 at Primary Tau:           {format_float(row['f1'])}")
        lines.append(f"  MCC:                         {format_float(row['mcc'])}")
        lines.append(f"  Balanced Accuracy:           {format_float(row['balanced_accuracy'])}")
        lines.append(f"  Expected Utility per Signal: {format_float(row['expected_utility_per_signal'])}")
        lines.append(f"  Actual Positive Rate:        {format_float(row['positive_rate'])}")
        lines.append(f"  Predicted Positive Rate:     {format_float(row['predicted_positive_rate'])}")
        lines.append(f"  Predicted/Actual Ratio:      {format_float(row['predicted_to_actual_ratio'])}")
        lines.append(f"  Quality Label:               {row['quality_label']}")
        lines.append("")

    lines.append("SHORT INTERPRETATION")
    lines.append("-" * 120)

    best = leaderboard.iloc[0]
    worst = leaderboard.iloc[-1]
    lines.append(
        f"The best-ranked model is {best['model_name']} because it achieved the highest global score "
        f"({format_float(best['global_score'])}), supported by ROC-AUC={format_float(best['roc_auc'])}, "
        f"Average Precision={format_float(best['average_precision'])}, and Lift@TopK={format_float(best['lift_at_top_k'])}."
    )
    lines.append(
        f"The weakest-ranked model is {worst['model_name']} because it achieved the lowest global score "
        f"({format_float(worst['global_score'])}), with ROC-AUC={format_float(worst['roc_auc'])}, "
        f"Average Precision={format_float(worst['average_precision'])}, and Lift@TopK={format_float(worst['lift_at_top_k'])}."
    )
    lines.append("")

    lines.append("Important methodological note:")
    lines.append(
        "The ranking should be interpreted as a relative comparison of trained sub-models, not as proof that the highest-ranked model is automatically production-ready. "
        "For imbalanced financial datasets, ROC-AUC, PR-AUC, Precision@TopK, and Lift@TopK are more informative than accuracy. "
        "The final deployment decision should also consider model stability, calibration quality, and how the central routing algorithm will consume the model outputs."
    )
    lines.append("")
    lines.append("=" * 120)
    lines.append("END OF GLOBAL MODEL EVALUATION REPORT")
    lines.append("=" * 120)

    return "\n".join(lines)


def main() -> None:
    """Main execution function."""
    ensure_output_dir()

    print("=" * 120)
    print("PROJECT AEGIS - GLOBAL MODEL EVALUATION")
    print("=" * 120)
    print("Loading JSON summary files...")

    all_rows: List[Dict[str, Any]] = []
    for asset_class, path in JSON_INPUT_PATHS.items():
        print(f"Reading {asset_class} summary: {path}")
        summaries = load_json_summary(asset_class, path)
        rows = extract_model_rows(asset_class, summaries)
        all_rows.extend(rows)
        print(f"  Extracted {len(rows)} sub-models from {asset_class}.")

    leaderboard = make_leaderboard(all_rows)
    report_text = build_report(leaderboard)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"Evaluation_of_the_Models_{timestamp}.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    print("\n" + "=" * 120)
    print("GLOBAL MODEL LEADERBOARD")
    print("=" * 120)
    terminal_columns = [
        "rank",
        "model_name",
        "asset_class",
        "horizon",
        "winner_algorithm",
        "global_score",
        "roc_auc",
        "average_precision",
        "precision_at_top_k",
        "lift_at_top_k",
        "quality_label",
    ]
    print(leaderboard[terminal_columns].to_string(index=False))

    print("\n" + "=" * 120)
    print("EVALUATION COMPLETED")
    print("=" * 120)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
