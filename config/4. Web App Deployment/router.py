"""
router.py — AegisTrader Web App Deployment
==========================================

Deterministic model selection. Given the user's answers to the profiling
questionnaire, this module selects exactly one of the 12 trained sub-models.

There is NO machine learning and NO LLM here. The mapping from a finite set of
closed answers to a finite set of models is a lookup/decision problem, so it is
implemented as a transparent, deterministic rule cascade. This makes every
routing outcome reproducible and fully explainable for the thesis defense.

------------------------------------------------------------------------------
ROUTING LOGIC
------------------------------------------------------------------------------
Horizon  (always): Question 1 -> {Day, Swing, Long}.

Asset class (priority cascade, highest first):
  1. A valid ticker (Q3) -> asset class is taken from the ticker. Overrides Q2.
  2. Otherwise, a concrete Q2 choice -> that asset class.
  3. Otherwise (Q2 = "no preference") -> asset class from Q4 (risk) bins.

Question 5 (experience) does NOT affect routing. It is forwarded unchanged so
that explainer.py can use it to set the depth of the (optional) explanation.

model_id = "AegisTrader_{AssetClass}_{Horizon}"

A signal whose selected model is below the ROC-AUC quality threshold (see
config.LOW_CONFIDENCE_MODELS) is still returned, but flagged so the UI can show
a "low confidence" warning. The router never reroutes to a different model.

------------------------------------------------------------------------------
HOW app.py SHOULD CALL THIS MODULE
------------------------------------------------------------------------------
  1. If the user typed a ticker in Q3, call validate_ticker(raw_text).
       - is_blank  -> treat as "no specific ticker" (strongest signal).
       - is_valid  -> pass tv.canonical as `ticker` to route().
       - otherwise -> show tv.message and offer "retry" or "skip"
                      ("skip" == call route() with ticker=None).
  2. Call route(q1_key, q2_key, risk, ticker=<canonical or None>).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import config

# =============================================================================
# 1. QUESTIONNAIRE OPTION KEYS  (internal keys <-> English UI labels)
# =============================================================================
# app.py renders the *labels*; the user's choice is mapped back to a *key*,
# and the keys below are the only thing the routing logic depends on. This
# decouples the deterministic logic from the display wording.

# --- Question 1: time horizon -> model horizon ---
HORIZON_CHOICES: dict[str, str] = {
    "within_1_week": "Day",
    "1_week_to_1_month": "Swing",
    "1_month_to_3_months_plus": "Long",
}
HORIZON_LABELS: dict[str, str] = {
    "within_1_week": "Within 1 week (1-7 days)",
    "1_week_to_1_month": "1 week to 1 month",
    "1_month_to_3_months_plus": "1 month to 3 months or more",
}

# --- Question 2: asset class preference ---
# Value None marks "no preference", which defers the asset choice to Q4.
ASSET_CHOICES: dict[str, Optional[str]] = {
    "crypto": "Crypto",
    "stocks": "Stocks",
    "indices": "Indices",
    "forex": "Forex",
    "no_preference": None,
}
ASSET_LABELS: dict[str, str] = {
    "crypto": "Cryptocurrencies (Crypto)",
    "stocks": "Stocks",
    "indices": "Indices",
    "forex": "Foreign Exchange (Forex)",
    "no_preference": "I don't have a particular preference",
}

# --- Question 5: investment experience (NOT used for routing) ---
# Forwarded to explainer.py to control explanation depth.
EXPERIENCE_CHOICES: dict[str, str] = {
    "experienced": "Yes, I have invested many times in the past",
    "beginner": "No, I'm a beginner",
    "long_term_single": "I have invested only once, in a fixed asset, with a long-term horizon",
    "periodic": "I have invested during specific periods, with breaks",
}

# Question 4 (risk, 1-10 slider) -> asset class, used ONLY when Q2 = no_preference.
# Bins are contiguous and non-overlapping over the integer range 1..10.
RISK_BINS: list[tuple[int, int, str]] = [
    (1, 3, "Indices"),
    (4, 5, "Stocks"),
    (6, 8, "Forex"),
    (9, 10, "Crypto"),
]

# =============================================================================
# 2. RETURN TYPES
# =============================================================================
@dataclass
class TickerValidation:
    """Result of validating the optional free-text ticker from Question 3."""
    raw: str
    is_valid: bool
    is_blank: bool
    canonical: Optional[str]       # canonical symbol, e.g. "BTC-USD"
    asset_class: Optional[str]     # "Crypto" | "Forex" | "Indices" | "Stocks"
    message: Optional[str]         # English message to show on invalid input


@dataclass
class RouteResult:
    """Everything app.py and live_inference.py need after model selection."""
    model_id: str
    asset_class: str
    horizon: str                   # "Day" | "Swing" | "Long"
    specific_ticker: Optional[str] # canonical symbol, or None => strongest signal
    ticker_mode: str               # "specific" | "strongest"
    asset_source: str              # "ticker" | "q2_preference" | "q4_risk_fallback"
    low_confidence: bool
    model_description: str
    signal_validity_days: int
    artifact_path: str
    notes: list[str] = field(default_factory=list)


# =============================================================================
# 3. TICKER VALIDATION (Question 3)
# =============================================================================
# Case-insensitive, whitespace-tolerant lookup back to the canonical symbol.
_CANONICAL_BY_UPPER: dict[str, str] = {
    ticker.upper(): ticker
    for ticker_list in config.TICKERS.values()
    for ticker in ticker_list
}

_INVALID_TICKER_MESSAGE = (
    'The code "{raw}" was not recognized. Please type a code exactly as it '
    "appears in the lists (for example BTC-USD, EURUSD=X, ^GSPC, or AAPL), "
    "or skip this question to let the system pick the strongest signal."
)


def validate_ticker(raw: Optional[str]) -> TickerValidation:
    """Validate the optional Q3 ticker. Blank is valid intent (not an error)."""
    cleaned = (raw or "").strip()
    if cleaned == "":
        return TickerValidation(
            raw=raw or "", is_valid=False, is_blank=True,
            canonical=None, asset_class=None, message=None,
        )
    canonical = _CANONICAL_BY_UPPER.get(cleaned.upper())
    if canonical is None:
        return TickerValidation(
            raw=raw or "", is_valid=False, is_blank=False,
            canonical=None, asset_class=None,
            message=_INVALID_TICKER_MESSAGE.format(raw=cleaned),
        )
    return TickerValidation(
        raw=raw or "", is_valid=True, is_blank=False,
        canonical=canonical,
        asset_class=config.TICKER_TO_ASSET_CLASS[canonical],
        message=None,
    )


# =============================================================================
# 4. CORE RESOLUTION HELPERS
# =============================================================================
def _risk_to_asset(risk: int) -> str:
    """Map the Q4 risk score (1-10) to an asset class."""
    if not isinstance(risk, int) or not (1 <= risk <= 10):
        raise ValueError(f"risk must be an integer in 1..10, got {risk!r}")
    for low, high, asset in RISK_BINS:
        if low <= risk <= high:
            return asset
    raise RuntimeError(f"risk {risk} fell outside all bins (should be impossible)")


def _resolve_asset(
    q2_key: str, risk: int, ticker_asset: Optional[str]
) -> tuple[str, str, list[str]]:
    """Apply the asset-class priority cascade. Returns (asset, source, notes)."""
    notes: list[str] = []

    # 1. A valid ticker wins outright.
    if ticker_asset is not None:
        q2_asset = ASSET_CHOICES.get(q2_key)
        if q2_asset is not None and q2_asset != ticker_asset:
            notes.append(
                f"Your ticker is a {ticker_asset} asset, so it overrides your "
                f"Question 2 selection ({q2_asset})."
            )
        return ticker_asset, "ticker", notes

    # 2. A concrete Q2 preference.
    q2_asset = ASSET_CHOICES[q2_key]
    if q2_asset is not None:
        return q2_asset, "q2_preference", notes

    # 3. No preference -> fall back to the risk score.
    asset = _risk_to_asset(risk)
    notes.append(
        f"No asset preference was given, so the asset class was derived from "
        f"your risk tolerance ({risk}/10) -> {asset}."
    )
    return asset, "q4_risk_fallback", notes


# =============================================================================
# 5. PUBLIC ENTRY POINT
# =============================================================================
def route(
    q1_key: str,
    q2_key: str,
    risk: int,
    ticker: Optional[str] = None,
) -> RouteResult:
    """
    Select one of the 12 sub-models.

    Parameters
    ----------
    q1_key : one of HORIZON_CHOICES
    q2_key : one of ASSET_CHOICES
    risk   : integer 1..10 (Q4). Only consulted when q2_key == "no_preference".
    ticker : None for "no specific ticker", otherwise a CANONICAL symbol that
             has already passed validate_ticker(). Passing an unvalidated or
             unknown symbol raises ValueError on purpose.
    """
    if q1_key not in HORIZON_CHOICES:
        raise ValueError(f"Unknown horizon key {q1_key!r}; expected one of {list(HORIZON_CHOICES)}")
    if q2_key not in ASSET_CHOICES:
        raise ValueError(f"Unknown asset key {q2_key!r}; expected one of {list(ASSET_CHOICES)}")

    horizon = HORIZON_CHOICES[q1_key]

    ticker_asset: Optional[str] = None
    specific_ticker: Optional[str] = None
    if ticker is not None:
        tv = validate_ticker(ticker)
        if not tv.is_valid:
            raise ValueError(
                f"route() received the non-canonical ticker {ticker!r}. Validate "
                "with validate_ticker() and resolve retry/skip before calling route()."
            )
        ticker_asset = tv.asset_class
        specific_ticker = tv.canonical

    asset, asset_source, notes = _resolve_asset(q2_key, risk, ticker_asset)

    model_id = f"AegisTrader_{asset}_{horizon}"
    if model_id not in config.MODEL_IDS:
        raise RuntimeError(f"Constructed unknown model_id {model_id!r}")

    low_confidence = model_id in config.LOW_CONFIDENCE_MODELS
    if low_confidence:
        notes.append(
            "This model scored below the ROC-AUC quality threshold on the test "
            "set, so the signal is marked low confidence."
        )

    return RouteResult(
        model_id=model_id,
        asset_class=asset,
        horizon=horizon,
        specific_ticker=specific_ticker,
        ticker_mode="specific" if specific_ticker else "strongest",
        asset_source=asset_source,
        low_confidence=low_confidence,
        model_description=config.MODEL_DESCRIPTIONS[model_id],
        signal_validity_days=config.SIGNAL_VALIDITY_DAYS[asset][horizon],
        artifact_path=str(config.ARTIFACT_PATHS[model_id]),
        notes=notes,
    )


# =============================================================================
# 6. SELF-CHECK  (runs only via `python router.py`)
# =============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("AegisTrader router.py — self-check")
    print("=" * 78)

    # ---- Risk bins: full coverage, no overlap, correct asset ----
    expected_risk = {1: "Indices", 2: "Indices", 3: "Indices",
                     4: "Stocks", 5: "Stocks",
                     6: "Forex", 7: "Forex", 8: "Forex",
                     9: "Crypto", 10: "Crypto"}
    for r, exp in expected_risk.items():
        got = _risk_to_asset(r)
        assert got == exp, f"risk {r}: expected {exp}, got {got}"
    print("[OK] Q4 risk bins cover 1..10 with no overlap.")

    # ---- Ticker validation ----
    tv = validate_ticker("btc-usd")          # case-insensitive
    assert tv.is_valid and tv.canonical == "BTC-USD" and tv.asset_class == "Crypto"
    tv = validate_ticker("  AAPL ")          # whitespace-tolerant
    assert tv.is_valid and tv.canonical == "AAPL" and tv.asset_class == "Stocks"
    tv = validate_ticker("^GSPC")
    assert tv.is_valid and tv.asset_class == "Indices"
    tv = validate_ticker("EURUSD=X")
    assert tv.is_valid and tv.asset_class == "Forex"
    tv = validate_ticker("")                 # blank is intent, not error
    assert (not tv.is_valid) and tv.is_blank and tv.message is None
    tv = validate_ticker("BTCUSD")           # invalid -> message
    assert (not tv.is_valid) and (not tv.is_blank) and tv.message
    print("[OK] Ticker validation: canonical, blank, and invalid cases.")

    # ---- Representative routing scenarios ----
    Scenario = tuple  # (label, q1, q2, risk, ticker, expected_model, expected_source)
    scenarios = [
        ("Concrete asset + horizon",
         "within_1_week", "stocks", 5, None, "AegisTrader_Stocks_Day", "q2_preference"),
        ("No preference, low risk -> Indices",
         "1_month_to_3_months_plus", "no_preference", 2, None,
         "AegisTrader_Indices_Long", "q4_risk_fallback"),
        ("No preference, high risk -> Crypto",
         "1_week_to_1_month", "no_preference", 10, None,
         "AegisTrader_Crypto_Swing", "q4_risk_fallback"),
        ("Ticker overrides Q2 (AAPL beats Crypto)",
         "1_week_to_1_month", "crypto", 5, "AAPL",
         "AegisTrader_Stocks_Swing", "ticker"),
        ("Low-confidence model is flagged (Crypto_Long)",
         "1_month_to_3_months_plus", "crypto", 5, None,
         "AegisTrader_Crypto_Long", "q2_preference"),
    ]

    print("-" * 78)
    header = f"{'scenario':<42}{'model_id':<28}{'src':<8}"
    print(header)
    print("-" * 78)
    for label, q1, q2, risk, tk, exp_model, exp_src in scenarios:
        res = route(q1, q2, risk, ticker=tk)
        assert res.model_id == exp_model, f"{label}: expected {exp_model}, got {res.model_id}"
        assert res.asset_source == exp_src, f"{label}: expected {exp_src}, got {res.asset_source}"
        src_short = {"q2_preference": "Q2", "q4_risk_fallback": "Q4", "ticker": "ticker"}[res.asset_source]
        print(f"{label:<42}{res.model_id:<28}{src_short:<8}")

    # Specific assertions on the override and low-confidence cases.
    override = route("1_week_to_1_month", "crypto", 5, ticker="AAPL")
    assert override.specific_ticker == "AAPL" and override.ticker_mode == "specific"
    assert any("overrides" in n for n in override.notes)

    lowconf = route("1_month_to_3_months_plus", "crypto", 5)
    assert lowconf.low_confidence is True
    assert lowconf.ticker_mode == "strongest" and lowconf.specific_ticker is None

    print("-" * 78)
    print("[OK] 5/5 routing scenarios produced the expected model and source.")

    # ---- Defensive input validation ----
    for bad in (
        lambda: route("bad_key", "stocks", 5),
        lambda: route("within_1_week", "bad_asset", 5),
        lambda: route("within_1_week", "no_preference", 0),   # risk out of range
        lambda: route("within_1_week", "crypto", 5, ticker="NOPE"),  # unvalidated ticker
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError was not raised")
    print("[OK] Invalid inputs raise ValueError as expected.")

    # ---- Exhaustive: every (horizon, concrete asset) maps to a real model ----
    count = 0
    for q1 in HORIZON_CHOICES:
        for q2 in ("crypto", "stocks", "indices", "forex"):
            res = route(q1, q2, 5)
            assert res.model_id in config.MODEL_IDS
            count += 1
    assert count == 12
    print(f"[OK] All {count} concrete (horizon x asset) combinations resolve to valid models.")

    print("=" * 78)
    print("Self-check complete.")
