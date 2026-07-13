"""
config.py — AegisTrader Web App Deployment
============================================

Central configuration for the serving layer. This module defines CONSTANTS ONLY.
Importing it has no side effects (no file I/O, no model loading, no network).
Every other deployment file (`router.py`, `live_inference.py`, `explainer.py`,
`app.py`) imports values from here so there is a single source of truth.

Provenance of the values below:
  - Ticker universes: copied verbatim from the dataset-creation scripts
    (Code_for_{CRYPTO,FOREX,INDICES,STOCKS}_Dataset.py).
  - Artifact filenames: follow the training naming convention
    `AegisTrader_{AssetClass}_{Horizon}.joblib`.
  - Test ROC-AUC values: taken from the v4 training summaries
    (Model_Training_Summary_v4, dated 2026-04-25 / 2026-04-26).
  - Feature warm-up requirement (MIN_REQUIRED_BARS): derived from the rolling
    windows in long_horizon_and_feature_engineering.py.

Source of truth for trade-ticket parameters (IMPORTANT):
  - Loaded from the `.joblib` artifact at inference time: the fitted estimator
    (`model`), `scaler`, `winsor_bounds`, and `feature_columns`. These define
    the exact preprocessing and prediction, so they must come from the artifact.
  - Defined HERE instead of the artifact: the bracket geometry per sub-model
    (`look_forward`, `atr_mult`, `reward_to_risk`) and the derived BUY threshold.
    Reason: every training script hard-coded a single, uniform (crypto) geometry
    into the artifacts, but the targets the models actually learned were built
    in target_variable_engineering.py with DIFFERENT, per-asset geometry. The
    artifact values are therefore correct only for Crypto and wrong for Forex,
    Indices and Stocks. BARRIER_GEOMETRY below reproduces the true per-asset
    target geometry, and the trade ticket / BUY threshold read from it — NOT
    from the artifact. (This train/serve geometry mismatch is a known limitation
    of the frozen training code and should be documented for the defense.)
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# 1. PATHS
# =============================================================================
# config.py lives in ".../Code Files/4. Web App Deployment/".
# The 12 trained artifacts have been copied into the local "models/" subfolder,
# so paths are resolved RELATIVE to this file. This keeps the app self-contained
# and portable (independent of the absolute OneDrive path, which contains Greek
# characters and spaces that break easily on other machines).
BASE_DIR: Path = Path(__file__).resolve().parent
MODELS_DIR: Path = BASE_DIR / "models"

# =============================================================================
# 2. TAXONOMY (asset classes, horizons, the 12 sub-model IDs)
# =============================================================================
ASSET_CLASSES: list[str] = ["Crypto", "Forex", "Indices", "Stocks"]
HORIZONS: list[str] = ["Day", "Swing", "Long"]

# The 12 sub-model IDs, e.g. "AegisTrader_Crypto_Day".
MODEL_IDS: list[str] = [
    f"AegisTrader_{asset}_{horizon}"
    for asset in ASSET_CLASSES
    for horizon in HORIZONS
]

# Absolute path to each artifact on disk.
ARTIFACT_PATHS: dict[str, Path] = {
    model_id: MODELS_DIR / f"{model_id}.joblib" for model_id in MODEL_IDS
}

# =============================================================================
# 3. TICKER UNIVERSE (per asset class)
# =============================================================================
# These are the symbols attempted during dataset creation. At serving time,
# live_inference.py downloads each symbol via yfinance and SKIPS any symbol that
# fails to download (e.g. a coin/pair that has since been delisted or renamed),
# continuing with the rest. A failure here must never crash the app.
TICKERS: dict[str, list[str]] = {
    "Crypto": [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "AVAX-USD", "SHIB-USD", "DOT-USD",
        "TRX-USD", "LINK-USD", "MATIC-USD", "BCH-USD", "LTC-USD",
        "UNI-USD", "NEAR-USD", "ICP-USD", "ATOM-USD", "XMR-USD",
        "ETC-USD", "FIL-USD", "HBAR-USD", "APT-USD", "LDO-USD",
        "ARB-USD", "VET-USD", "QNT-USD", "MKR-USD", "AAVE-USD",
        "GRT-USD", "ALGO-USD", "STX-USD", "SAND-USD", "EOS-USD",
        "XTZ-USD", "THETA-USD", "IMX-USD", "EGLD-USD", "MANA-USD",
        "AXS-USD", "FLOW-USD", "KAVA-USD", "NEO-USD", "KLAY-USD",
        "FTM-USD", "SNX-USD", "CRV-USD", "GALA-USD", "CHZ-USD",
    ],
    "Forex": [
        "EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X",
        "EURJPY=X", "GBPJPY=X", "EURGBP=X", "USDCAD=X", "USDCHF=X",
        "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X", "AUDNZD=X",
        "EURAUD=X", "EURCAD=X", "EURCHF=X", "EURNZD=X", "GBPAUD=X",
        "GBPCAD=X", "GBPCHF=X", "GBPNZD=X", "AUDCAD=X", "AUDCHF=X",
        "CADCHF=X", "NZDCAD=X", "NZDCHF=X", "USDCNY=X", "USDHKD=X",
        "USDSGD=X", "USDINR=X", "USDMXN=X", "USDZAR=X", "USDBRL=X",
        "USDTRY=X", "USDKRW=X", "USDSEK=X", "USDNOK=X", "USDDKK=X",
        "USDPLN=X", "USDTWD=X", "USDTHB=X",
    ],
    "Indices": [
        "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX",
        "^FTSE", "^GDAXI", "^FCHI", "^STOXX50E", "^IBEX",
        "^N225", "^HSI", "^STI", "^KS11", "^BSESN",
        "^JKSE", "^BVSP", "^MXX", "^GSPTSE", "^AXJO",
        "^AORD",
    ],
    "Stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "BRK-B", "LLY", "V",
        "TSM", "UNH", "JNJ", "JPM", "XOM",
        "WMT", "MA", "PG", "AVGO", "HD",
        "CVX", "MRK", "ABBV", "KO", "PEP",
        "COST", "ORCL", "BAC", "ADBE", "MCD",
        "CSCO", "CRM", "ACN", "NFLX", "LIN",
        "AMD", "TMO", "ABT", "NKE", "DIS",
        "WFC", "TM", "DHR", "INTC", "QCOM",
        "TXN", "VZ", "PM", "INTU", "IBM",
    ],
}

# Reverse lookup: symbol -> asset class (used for validation in router.py).
TICKER_TO_ASSET_CLASS: dict[str, str] = {
    ticker: asset for asset, ticker_list in TICKERS.items() for ticker in ticker_list
}

# =============================================================================
# 4. MARKET DATA DOWNLOAD (yfinance)
# =============================================================================
YF_INTERVAL: str = "1d"  # all models were trained on daily bars.
YF_TIMEOUT_SECONDS: int = 10  # hard stop per Yahoo Finance request.

# How much history to download per symbol, in CALENDAR days.
# The last row is valid only after the FULL warm-up of the feature chain, which
# stacks additively:
#   - add_indicators  : SMA_200 drops the first ~200 trading bars.
#   - long_horizon    : vol_60d_pct_rank needs a further 60 + 252 - 1 = 311 bars.
#   => ~511 clean TRADING bars are needed before the most recent bar can be scored.
# For ~5/7 markets (stocks/forex/indices) that is ~745 calendar days at the very
# edge, so 750 was too tight once holidays are included (the symptom: a major
# stock returning "data unavailable"). 1200 calendar days (~3.3y) yields ~820
# trading bars for those markets — a comfortable margin — and 1200 bars for crypto.
LOOKBACK_CALENDAR_DAYS: int = 1200

# Regime-feature warm-up (informational; the binding single-feature constraint).
# Note the EFFECTIVE warm-up is larger (~511 bars) because SMA_200 is consumed
# first; see LOOKBACK_CALENDAR_DAYS above.
#   - drawdown_from_252d_high : 252 bars
#   - return_30d_pct_rank     : 30 + 252 - 1 = 281 bars
#   - vol_60d_pct_rank        : 60 + 252 - 1 = 311 bars  <-- binding (regime stage)
MIN_REQUIRED_BARS: int = 311

# =============================================================================
# 5. SIGNAL GEOMETRY (barriers, horizon, validity)
# =============================================================================
# Per-asset, per-horizon bracket geometry, copied from the ASSET_CONFIGS in
# target_variable_engineering.py. This is the geometry that DEFINES the target
# each model predicts, so the live trade ticket must use it (see docstring):
#   long-only bracket:  Stop-Loss   = Entry - atr_mult * ATR
#                       Take-Profit = Entry + atr_mult * ATR * rr
#   evaluated over `look_forward` TRADING bars.
BARRIER_GEOMETRY: dict[str, dict[str, dict[str, float]]] = {
    "Crypto": {
        "Day":   {"look_forward": 3,   "atr_mult": 1.5, "rr": 2.0},
        "Swing": {"look_forward": 14,  "atr_mult": 2.5, "rr": 3.0},
        "Long":  {"look_forward": 60,  "atr_mult": 4.0, "rr": 4.0},
    },
    "Forex": {
        "Day":   {"look_forward": 2,   "atr_mult": 1.0, "rr": 1.5},
        "Swing": {"look_forward": 10,  "atr_mult": 1.5, "rr": 2.0},
        "Long":  {"look_forward": 45,  "atr_mult": 2.5, "rr": 3.0},
    },
    "Indices": {
        "Day":   {"look_forward": 5,   "atr_mult": 1.2, "rr": 1.5},
        "Swing": {"look_forward": 20,  "atr_mult": 2.0, "rr": 2.0},
        "Long":  {"look_forward": 120, "atr_mult": 3.0, "rr": 3.0},
    },
    "Stocks": {
        "Day":   {"look_forward": 5,   "atr_mult": 1.5, "rr": 1.5},
        "Swing": {"look_forward": 21,  "atr_mult": 2.0, "rr": 2.5},
        "Long":  {"look_forward": 90,  "atr_mult": 3.0, "rr": 3.0},
    },
}

# Convenience view: look_forward in TRADING bars, per asset and horizon.
LOOK_FORWARD_BARS: dict[str, dict[str, int]] = {
    asset: {h: int(BARRIER_GEOMETRY[asset][h]["look_forward"]) for h in HORIZONS}
    for asset in ASSET_CLASSES
}

# `signal_valid_until` = today + N CALENDAR days, derived from look_forward.
#   - Crypto trades 24/7            -> 1 trading bar = 1 calendar day.
#   - Forex/Indices/Stocks (~5/7)   -> calendar = round(bars * 7 / 5).
_CALENDAR_DAYS_PER_TRADING_DAY: dict[str, float] = {
    "Crypto": 1.0, "Forex": 7 / 5, "Indices": 7 / 5, "Stocks": 7 / 5,
}
SIGNAL_VALIDITY_DAYS: dict[str, dict[str, int]] = {
    asset: {
        h: int(round(BARRIER_GEOMETRY[asset][h]["look_forward"]
                     * _CALENDAR_DAYS_PER_TRADING_DAY[asset]))
        for h in HORIZONS
    }
    for asset in ASSET_CLASSES
}

# =============================================================================
# 6. DECISION THRESHOLDS
# =============================================================================
# (a) BUY threshold — when is a signal actionable?
# Economic break-even per sub-model: tau* = 1 / (1 + rr), using the TRUE
# per-asset reward-to-risk from BARRIER_GEOMETRY (defense decision: option b).
# A signal is a BUY only if the model's calibrated win-probability >= tau*.
# This intentionally does NOT use the artifact's stored `primary_threshold`,
# which was computed from uniform (crypto) rr and is wrong for non-crypto.
BUY_THRESHOLD: dict[str, float] = {
    f"AegisTrader_{asset}_{horizon}": round(
        1.0 / (1.0 + BARRIER_GEOMETRY[asset][horizon]["rr"]), 4
    )
    for asset in ASSET_CLASSES
    for horizon in HORIZONS
}

# (b) Quality flag — should the signal carry a "low confidence" badge?
# Test-set ROC-AUC per sub-model, from the v4 training summaries.
# Used ONLY to flag low-confidence signals in the UI. The router ALWAYS returns
# the sub-model that matches the user's stated preferences; it never reroutes to
# a different asset class. If a selected model's ROC-AUC is below
# QUALITY_THRESHOLD, the app still shows the signal but attaches a visible
# "low confidence" badge/warning.
QUALITY_THRESHOLD: float = 0.50

MODEL_TEST_ROC_AUC: dict[str, float] = {
    "AegisTrader_Crypto_Day":    0.6863,
    "AegisTrader_Crypto_Swing":  0.6384,
    "AegisTrader_Crypto_Long":   0.4645,
    "AegisTrader_Forex_Day":     0.6133,
    "AegisTrader_Forex_Swing":   0.5696,
    "AegisTrader_Forex_Long":    0.6111,
    "AegisTrader_Indices_Day":   0.5549,
    "AegisTrader_Indices_Swing": 0.5633,
    "AegisTrader_Indices_Long":  0.5142,
    "AegisTrader_Stocks_Day":    0.5730,
    "AegisTrader_Stocks_Swing":  0.5810,
    "AegisTrader_Stocks_Long":   0.5349,
}

# Derived: the set of sub-models that should carry a "low confidence" badge.
# Auto-updates if MODEL_TEST_ROC_AUC or QUALITY_THRESHOLD is edited after a
# future retraining. With the current values, only AegisTrader_Crypto_Long
# (0.4645) is below 0.50.
LOW_CONFIDENCE_MODELS: set[str] = {
    model_id
    for model_id, auc in MODEL_TEST_ROC_AUC.items()
    if auc < QUALITY_THRESHOLD
}

# =============================================================================
# 7. MODEL DESCRIPTIONS (user-facing, English)
# =============================================================================
# Purely descriptive text shown in the dashboard. These deliberately make NO
# claim about model quality; the quality signal is handled dynamically via the
# low-confidence badge above. Keeping descriptions quality-neutral means they
# stay correct even after the underlying models are improved.
MODEL_DESCRIPTIONS: dict[str, str] = {
    "AegisTrader_Crypto_Day":
        "Short-term signals for the top-50 cryptocurrencies, aimed at moves over "
        "roughly the next 3 trading days with a 2:1 reward-to-risk target.",
    "AegisTrader_Crypto_Swing":
        "Medium-term signals for the top-50 cryptocurrencies, targeting swings "
        "over about 14 trading days at a 3:1 reward-to-risk target.",
    "AegisTrader_Crypto_Long":
        "Long-horizon signals for the top-50 cryptocurrencies, looking ahead "
        "around 60 trading days at a 4:1 reward-to-risk target.",
    "AegisTrader_Forex_Day":
        "Short-term signals across major, minor and exotic currency pairs, aimed "
        "at moves over roughly the next 2 trading days at a 1.5:1 reward-to-risk "
        "target.",
    "AegisTrader_Forex_Swing":
        "Medium-term signals across major, minor and exotic currency pairs, "
        "targeting swings over about 10 trading days at a 2:1 reward-to-risk "
        "target.",
    "AegisTrader_Forex_Long":
        "Long-horizon signals across major, minor and exotic currency pairs, "
        "looking ahead around 45 trading days at a 3:1 reward-to-risk target.",
    "AegisTrader_Indices_Day":
        "Short-term signals for major global equity indices, aimed at moves over "
        "roughly the next 5 trading days at a 1.5:1 reward-to-risk target.",
    "AegisTrader_Indices_Swing":
        "Medium-term signals for major global equity indices, targeting swings "
        "over about 20 trading days at a 2:1 reward-to-risk target.",
    "AegisTrader_Indices_Long":
        "Long-horizon signals for major global equity indices, looking ahead "
        "around 120 trading days at a 3:1 reward-to-risk target.",
    "AegisTrader_Stocks_Day":
        "Short-term signals for 50 large-cap global stocks, aimed at moves over "
        "roughly the next 5 trading days at a 1.5:1 reward-to-risk target.",
    "AegisTrader_Stocks_Swing":
        "Medium-term signals for 50 large-cap global stocks, targeting swings "
        "over about 21 trading days at a 2.5:1 reward-to-risk target.",
    "AegisTrader_Stocks_Long":
        "Long-horizon signals for 50 large-cap global stocks, looking ahead "
        "around 90 trading days at a 3:1 reward-to-risk target.",
}

# =============================================================================
# 8. SELF-CHECK (runs only when executed directly: `python config.py`)
# =============================================================================
# Importing this module never triggers the checks below. Run the file directly
# to validate internal consistency and to confirm that all 12 artifacts are
# present in the models/ folder on this machine.
if __name__ == "__main__":
    print("=" * 70)
    print("AegisTrader config.py — self-check")
    print("=" * 70)

    # Internal consistency.
    assert len(MODEL_IDS) == 12, f"Expected 12 model IDs, got {len(MODEL_IDS)}"
    for mapping_name, mapping in (
        ("MODEL_TEST_ROC_AUC", MODEL_TEST_ROC_AUC),
        ("MODEL_DESCRIPTIONS", MODEL_DESCRIPTIONS),
        ("ARTIFACT_PATHS", ARTIFACT_PATHS),
    ):
        missing = set(MODEL_IDS) - set(mapping)
        extra = set(mapping) - set(MODEL_IDS)
        assert not missing, f"{mapping_name} is missing: {sorted(missing)}"
        assert not extra, f"{mapping_name} has unexpected keys: {sorted(extra)}"

    expected_counts = {"Crypto": 50, "Forex": 43, "Indices": 21, "Stocks": 50}
    for asset, expected in expected_counts.items():
        actual = len(TICKERS[asset])
        assert actual == expected, f"{asset}: expected {expected} tickers, got {actual}"

    # Geometry / threshold consistency.
    for asset in ASSET_CLASSES:
        for h in HORIZONS:
            g = BARRIER_GEOMETRY[asset][h]
            assert set(g) == {"look_forward", "atr_mult", "rr"}, f"bad geometry keys for {asset}/{h}"
            assert g["look_forward"] > 0 and g["atr_mult"] > 0 and g["rr"] > 0
    assert set(BUY_THRESHOLD) == set(MODEL_IDS), "BUY_THRESHOLD keys must match MODEL_IDS"
    # Spot-check the option-(b) economic threshold for a non-crypto model.
    assert abs(BUY_THRESHOLD["AegisTrader_Stocks_Day"] - 1 / (1 + 1.5)) < 1e-9

    print("[OK] 12 sub-models; metric/description/path keys all aligned.")
    print(f"[OK] Ticker counts: "
          f"{', '.join(f'{a}={len(TICKERS[a])}' for a in ASSET_CLASSES)} "
          f"(total {sum(len(v) for v in TICKERS.values())}).")
    print(f"[OK] Low-confidence models (ROC-AUC < {QUALITY_THRESHOLD}): "
          f"{sorted(LOW_CONFIDENCE_MODELS) or 'none'}")

    print("-" * 70)
    print("Per-asset barrier geometry (look_forward / atr_mult / rr) "
          "| validity(cal.days) | BUY tau*:")
    for asset in ASSET_CLASSES:
        for h in HORIZONS:
            g = BARRIER_GEOMETRY[asset][h]
            mid = f"AegisTrader_{asset}_{h}"
            print(f"   {asset:<8} {h:<6} "
                  f"{int(g['look_forward']):>3}b / {g['atr_mult']:.1f}x / {g['rr']:.1f}rr "
                  f"| valid {SIGNAL_VALIDITY_DAYS[asset][h]:>3}d "
                  f"| tau* {BUY_THRESHOLD[mid]:.4f}")

    # Artifact presence (warning only — does not fail the check).
    print("-" * 70)
    print(f"Looking for artifacts in: {MODELS_DIR}")
    found, absent = [], []
    for model_id in MODEL_IDS:
        (found if ARTIFACT_PATHS[model_id].exists() else absent).append(model_id)
    print(f"[OK] Found {len(found)}/12 artifacts.")
    if absent:
        print(f"[WARNING] Missing {len(absent)} artifact(s): {absent}")
        print("          (Expected if you are not running on the machine that "
              "holds the models/ folder.)")
    print("=" * 70)
    print("Self-check complete.")
