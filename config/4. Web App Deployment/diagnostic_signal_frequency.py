"""
diagnostic_signal_frequency.py — AegisTrader (analysis tool, not part of the app)
================================================================================

Answers one question: "How often would each model ACTUALLY fire a signal?"

For each (model, ticker) probe it scores the last ~250 COMPLETED daily bars and
reports the predicted-probability distribution plus the share of days that fall
in each tier (BUY / WATCH / NO_SIGNAL). This tells us whether a quiet day like
today is normal, or whether a model is silent essentially always.

Run on the machine that has models/ + internet:
    python diagnostic_signal_frequency.py
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
import live_inference as li


N_BARS = 250  # how many recent completed bars to score per probe

# All 12 sub-models, in the authoritative global-score ranking from
# Evaluation_of_the_Models_20260426_173430.txt (rank / global_score), each with a
# representative ticker. (^GSPC is reused for the three index horizons.)
PROBES = [
    ("AegisTrader_Crypto_Day",    "Crypto",  "Day",   "BTC-USD"),    # 1  0.9126
    ("AegisTrader_Stocks_Long",   "Stocks",  "Long",  "AAPL"),       # 2  0.8383
    ("AegisTrader_Forex_Day",     "Forex",   "Day",   "EURUSD=X"),   # 3  0.7620
    ("AegisTrader_Indices_Swing", "Indices", "Swing", "^GSPC"),      # 4  0.6776
    ("AegisTrader_Forex_Swing",   "Forex",   "Swing", "GBPUSD=X"),   # 5  0.6484
    ("AegisTrader_Indices_Day",   "Indices", "Day",   "^GSPC"),      # 6  0.6094
    ("AegisTrader_Stocks_Swing",  "Stocks",  "Swing", "MSFT"),       # 7  0.5652
    ("AegisTrader_Stocks_Day",    "Stocks",  "Day",   "AAPL"),       # 8  0.5351
    ("AegisTrader_Indices_Long",  "Indices", "Long",  "^GSPC"),      # 9  0.4661
    ("AegisTrader_Crypto_Swing",  "Crypto",  "Swing", "ETH-USD"),    # 10 0.3988
    ("AegisTrader_Forex_Long",    "Forex",   "Long",  "EURUSD=X"),   # 11 0.3102
    ("AegisTrader_Crypto_Long",   "Crypto",  "Long",  "BTC-USD"),    # 12 0.0820
]


def score_recent(model_id: str, ticker: str, n: int = N_BARS):
    """Predicted win-probabilities for the last n completed bars, or None."""
    ohlcv = li.download_ohlcv(ticker)
    if ohlcv is None:
        return None
    augmented, _ = li.build_features(ohlcv)
    if augmented is None:
        return None

    today = datetime.now(timezone.utc).date()
    augmented = augmented[augmented["Date"].dt.date < today]
    if augmented.empty:
        return None

    artifact = li.load_artifact(model_id)
    feature_cols = list(artifact["feature_columns"])

    tail = augmented.iloc[-n:].copy()
    X = tail[feature_cols].copy()
    valid = ~X.isna().any(axis=1)
    X = X[valid]
    if X.empty:
        return None

    for col, (lo, hi) in (artifact.get("winsor_bounds") or {}).items():
        if col in X.columns:
            X[col] = X[col].clip(lower=lo, upper=hi)
    scaler = artifact.get("scaler")
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)

    proba = artifact["model"].predict_proba(X)[:, 1]
    return pd.Series(proba)


def main() -> None:
    print("=" * 88)
    print(f"AegisTrader signal-frequency diagnostic — last ~{N_BARS} completed bars")
    print("=" * 88)
    for model_id, asset, horizon, ticker in PROBES:
        try:
            proba = score_recent(model_id, ticker)
        except Exception as exc:  # noqa: BLE001
            print(f"\n{model_id} [{ticker}] -> ERROR: {type(exc).__name__}: {exc}")
            continue
        if proba is None or proba.empty:
            print(f"\n{model_id} [{ticker}] -> no data")
            continue

        be = config.BUY_THRESHOLD[model_id]
        tk = li.load_artifact(model_id).get("top_k_threshold")
        tiers = Counter(li._tier(float(p), be, tk)[0] for p in proba)
        n = len(proba)
        buy, watch, no = tiers.get("BUY", 0), tiers.get("WATCH", 0), tiers.get("NO_SIGNAL", 0)
        tk_txt = f"{tk:.3f}" if tk is not None else "n/a"

        print(f"\n{model_id}  [{ticker}]  n={n}")
        print(f"  prob   : min={proba.min():.3f}  median={proba.median():.3f}  "
              f"mean={proba.mean():.3f}  p90={proba.quantile(.9):.3f}  max={proba.max():.3f}")
        print(f"  cutoffs: top-k={tk_txt}   break-even={be:.3f}")
        print(f"  tiers  : BUY {buy} ({buy/n:.0%})   "
              f"WATCH {watch} ({watch/n:.0%})   NO_SIGNAL {no} ({no/n:.0%})")
    print("\n" + "=" * 88)
    print("Reading: a healthy model fires WATCH/BUY on a meaningful share of days. "
          "If a model is ~100% NO_SIGNAL even here, its bracket is too hard; if it "
          "fires sometimes, today was simply a quiet day.")


if __name__ == "__main__":
    main()
