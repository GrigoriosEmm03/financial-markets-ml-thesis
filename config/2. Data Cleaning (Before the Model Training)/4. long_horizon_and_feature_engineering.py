"""
long_horizon_and_feature_engineering.py

Step 5 of the data pipeline for the thesis project (v3 of the features).

Replaces long_horizon_features.py. In addition to the long-horizon rolling and
lag features it produced before, this script now also:

  - Drops the integer calendar features (Month, Day_of_Week) and replaces
    them with cyclic encodings (sin/cos), so that tree-based models can no
    longer memorize specific years/months as proxies for market regime.
  - Adds three regime-aware features (drawdown from 252d high, 60d realized
    volatility percentile rank, 30d cumulative return percentile rank), all
    scale-invariant, computed strictly per Ticker, and using only past
    observations.

All new features are stationary, computed strictly per Ticker (no cross-ticker
leakage) and use only past information (no forward leakage).

Input:
    CSV Files\\4. Static Transformation & Cleansing - Feature Transformation\\
        DATASET_*_ML_READY.csv

Output:
    CSV Files\\5. Long Horizon Features Addition\\
        DATASET_*_ML_READY_v3.csv

Environment: thesis_env (Anaconda), Python 3.x
"""

# Importing the libraries
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATH CONFIGURATION
# ============================================================

INPUT_DIR = (
    r"C:\Users\grego\OneDrive\Υπολογιστής\Π.Μ.Σ. ΣΤΗΝ ΑΝΑΛΥΤΙΚΗ ΤΩΝ ΕΠΙΧΕΙΡΗΣΕΩΝ & "
    r"ΣΤΗΝ ΕΠΙΣΤΗΜΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ\ΔΙΠΛΩΜΑΤΙΚΗ\CSV Files"
    r"\4. Static Transformation & Cleansing - Feature Transformation"
)

OUTPUT_DIR = (
    r"C:\Users\grego\OneDrive\Υπολογιστής\Π.Μ.Σ. ΣΤΗΝ ΑΝΑΛΥΤΙΚΗ ΤΩΝ ΕΠΙΧΕΙΡΗΣΕΩΝ & "
    r"ΣΤΗΝ ΕΠΙΣΤΗΜΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ\ΔΙΠΛΩΜΑΤΙΚΗ\CSV Files"
    r"\5. Long Horizon Features Addition"
)

DATASETS = [
    "DATASET_CRYPTO_ML_READY.csv",
    "DATASET_FOREX_ML_READY.csv",
    "DATASET_INDICES_ML_READY.csv",
    "DATASET_STOCKS_ML_READY.csv",
]

# Long-horizon rolling windows (aligned with the Long target horizon).
WINDOWS = [30, 60, 90]

# Long lags of daily return (useful for Swing and Long horizons).
LAG_STEPS = [5, 10, 21]

# Window for the regime-aware features (~ 1 year).
REGIME_WINDOW = 252


# ============================================================
# HELPERS
# ============================================================

def rolling_per_ticker(df, column, window, agg):
    """
    Per-Ticker rolling aggregation. agg in {'sum', 'std', 'mean'}.
    """
    grouped = df.groupby("Ticker")[column]
    if agg == "sum":
        return grouped.transform(lambda x: x.rolling(window, min_periods=window).sum())
    if agg == "std":
        return grouped.transform(lambda x: x.rolling(window, min_periods=window).std())
    if agg == "mean":
        return grouped.transform(lambda x: x.rolling(window, min_periods=window).mean())
    raise ValueError(f"Unsupported aggregation type: {agg}")


def lag_per_ticker(df, column, steps):
    """
    Per-Ticker lag with shift(steps). Prevents cross-ticker leakage at
    the boundary between two tickers.
    """
    return df.groupby("Ticker")[column].shift(steps)


# ============================================================
# CYCLIC CALENDAR ENCODING
# ============================================================

def add_cyclic_calendar_features(df):
    """
    Replaces integer Month (1..12) and Day_of_Week (0..6) with their cyclic
    sin/cos encodings. This prevents tree models from using Month as a
    year/regime proxy (a behaviour we observed in the v2 results, where
    the integer Month was the top permutation-importance feature in the
    Long sub-model).

    Returns
    -------
    df : pd.DataFrame
        Augmented dataframe (integer columns dropped at the end).
    new_cols : list of str
        Names of the newly added columns.
    """
    new_cols = []

    if "Month" in df.columns:
        df["month_sin"] = np.sin(2.0 * np.pi * df["Month"] / 12.0)
        df["month_cos"] = np.cos(2.0 * np.pi * df["Month"] / 12.0)
        new_cols.extend(["month_sin", "month_cos"])

    if "Day_of_Week" in df.columns:
        df["dow_sin"] = np.sin(2.0 * np.pi * df["Day_of_Week"] / 7.0)
        df["dow_cos"] = np.cos(2.0 * np.pi * df["Day_of_Week"] / 7.0)
        new_cols.extend(["dow_sin", "dow_cos"])

    drop_cols = [c for c in ("Month", "Day_of_Week") if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    return df, new_cols


# ============================================================
# REGIME-AWARE FEATURES
# ============================================================

def add_regime_features(df, return_col="Return_Close"):
    """
    Adds three per-Ticker regime-aware features, all stationary and
    bounded:

      1. drawdown_from_252d_high
         Reconstructs a synthetic price index per Ticker from the
         daily returns (cum-product of (1 + r)) and computes
            synthetic_price / rolling_max_252(synthetic_price) - 1
         Range: [-1, 0]. 0 means at all-time high in the window;
         -0.5 means 50% off the recent peak.

      2. vol_60d_pct_rank
         60-day rolling std of returns, then 252-day rolling
         percentile rank within the same Ticker. Range: [0, 1].

      3. return_30d_pct_rank
         30-day rolling sum of returns, then 252-day rolling
         percentile rank within the same Ticker. Range: [0, 1].

    Returns
    -------
    df : pd.DataFrame
    new_cols : list of str
    """
    new_cols = []

    # 1. Drawdown from the 252-day high.
    df["_synthetic_price"] = (
        df.groupby("Ticker")[return_col]
        .transform(lambda r: (1.0 + r).cumprod())
    )
    df["_rolling_max_252"] = (
        df.groupby("Ticker")["_synthetic_price"]
        .transform(lambda s: s.rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW).max())
    )
    df["drawdown_from_252d_high"] = df["_synthetic_price"] / df["_rolling_max_252"] - 1.0
    df.drop(columns=["_synthetic_price", "_rolling_max_252"], inplace=True)
    new_cols.append("drawdown_from_252d_high")

    # 2. 60-day realized vol, percentile-ranked over 252 days.
    df["_vol_60d"] = (
        df.groupby("Ticker")[return_col]
        .transform(lambda r: r.rolling(60, min_periods=60).std())
    )
    df["vol_60d_pct_rank"] = (
        df.groupby("Ticker")["_vol_60d"]
        .transform(lambda s: s.rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW).rank(pct=True))
    )
    df.drop(columns=["_vol_60d"], inplace=True)
    new_cols.append("vol_60d_pct_rank")

    # 3. 30-day cumulative return, percentile-ranked over 252 days.
    df["_cum_30d"] = (
        df.groupby("Ticker")[return_col]
        .transform(lambda r: r.rolling(30, min_periods=30).sum())
    )
    df["return_30d_pct_rank"] = (
        df.groupby("Ticker")["_cum_30d"]
        .transform(lambda s: s.rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW).rank(pct=True))
    )
    df.drop(columns=["_cum_30d"], inplace=True)
    new_cols.append("return_30d_pct_rank")

    return df, new_cols


# ============================================================
# MAIN FEATURE-ENGINEERING ROUTINE
# ============================================================

def add_long_horizon_and_engineered_features(df):
    """
    Runs the full Step-5 feature engineering pipeline:
        A. Cyclic calendar encodings (replace Month, Day_of_Week).
        B. Cumulative returns over 30 / 60 / 90 days.
        C. Realized volatility over 30 / 60 / 90 days.
        D. Smoothed RSI over 30 / 60 / 90 days.
        E. Smoothed ATR% over 30 / 60 / 90 days.
        F. Long lags of daily return (5 / 10 / 21 days).
        G. Regime-aware features (drawdown, vol pct rank, return pct rank).
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    new_columns = []

    # A. Cyclic calendar encodings (drops the integer Month / Day_of_Week).
    df, cyclic_new = add_cyclic_calendar_features(df)
    new_columns.extend(cyclic_new)

    # B. Cumulative returns.
    for w in WINDOWS:
        col = f"Return_Close_cum_{w}"
        df[col] = rolling_per_ticker(df, "Return_Close", w, "sum")
        new_columns.append(col)

    # C. Realized volatility.
    for w in WINDOWS:
        col = f"Return_Close_std_{w}"
        df[col] = rolling_per_ticker(df, "Return_Close", w, "std")
        new_columns.append(col)

    # D. Smoothed RSI (only if available).
    if "RSI" in df.columns:
        for w in WINDOWS:
            col = f"RSI_mean_{w}"
            df[col] = rolling_per_ticker(df, "RSI", w, "mean")
            new_columns.append(col)

    # E. Smoothed ATR% (only if available).
    if "ATR_pct" in df.columns:
        for w in WINDOWS:
            col = f"ATR_pct_mean_{w}"
            df[col] = rolling_per_ticker(df, "ATR_pct", w, "mean")
            new_columns.append(col)

    # F. Long lags of daily return.
    for k in LAG_STEPS:
        col = f"Return_Close_lag{k}"
        df[col] = lag_per_ticker(df, "Return_Close", k)
        new_columns.append(col)

    # G. Regime-aware features.
    df, regime_new = add_regime_features(df)
    new_columns.extend(regime_new)

    # Replace any potential infinities before returning.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df, new_columns


# ============================================================
# DATASET PROCESSING
# ============================================================

def process_dataset(filename):
    """
    Reads an ML-ready CSV (output of feature_transformation.py),
    applies the full Step-5 feature engineering, drops rows with NaN
    in any of the new features (warmup period), and saves a v3 CSV.
    """
    input_path = os.path.join(INPUT_DIR, filename)
    output_filename = filename.replace("_ML_READY.csv", "_ML_READY_v3.csv")
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    print("\n" + "-" * 80)
    print(f"Processing: {filename}")
    print("-" * 80)

    if not os.path.exists(input_path):
        print(f"    [WARNING] Input file not found. Skipping.")
        print(f"    Expected path: {input_path}")
        return

    df = pd.read_csv(input_path)
    rows_before = len(df)
    cols_before = df.shape[1]
    print(f"    Loaded rows:                {rows_before}")
    print(f"    Columns before:             {cols_before}")

    required_cols = ["Date", "Ticker", "Return_Close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"    [ERROR] Missing required columns: {missing}")
        return

    df_augmented, new_cols = add_long_horizon_and_engineered_features(df)

    before_drop = len(df_augmented)
    df_augmented = df_augmented.dropna(subset=new_cols).reset_index(drop=True)
    rows_dropped = before_drop - len(df_augmented)

    print(f"    New / replaced features:    {len(new_cols)}")
    print(f"    Columns after:              {df_augmented.shape[1]}")
    print(f"    Rows dropped (warmup):      {rows_dropped}")
    print(f"    Final rows:                 {len(df_augmented)}")

    df_augmented.to_csv(output_path, index=False)
    print(f"    Saved to:                   {output_path}")


def main():
    print("=" * 80)
    print("LONG HORIZON & FEATURE ENGINEERING - Step 5 of the Data Pipeline")
    print("=" * 80)
    print(f"Input  directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Rolling windows:  {WINDOWS}")
    print(f"Lag steps:        {LAG_STEPS}")
    print(f"Regime window:    {REGIME_WINDOW} days")
    print(f"Datasets:         {len(DATASETS)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in DATASETS:
        process_dataset(filename)

    print("\n" + "=" * 80)
    print("LONG HORIZON & FEATURE ENGINEERING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
