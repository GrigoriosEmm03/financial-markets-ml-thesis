"""
live_inference.py — AegisTrader Web App Deployment
==================================================

Turns a selected sub-model (from router.py) into a concrete, copy-paste trade
signal. For one ticker (or the strongest ticker in an asset class) it:

    1. Downloads recent daily OHLCV from Yahoo Finance (yfinance).
    2. Rebuilds the EXACT training features by reusing the three frozen pipeline
       files verbatim:
         - add_technical_indicators                 (add_indicators.py)
         - process_static_transformations           (feature_transformation.py)
         - add_long_horizon_and_engineered_features (long_horizon_and_feature_engineering.py)
       The target-engineering and ml-preprocessing scripts are NEVER run here.
    3. Aligns to the artifact's feature_columns, applies the artifact's
       winsor_bounds (clip) and scaler (only if present), and predicts with the
       calibrated model.
    4. Builds the trade ticket using the per-asset bracket geometry from
       config.BARRIER_GEOMETRY and the economic threshold config.BUY_THRESHOLD.

Train/serve correctness notes:
    * Geometry (atr_mult / rr / look_forward) and the BUY threshold come from
      config, NOT from the artifact (the artifacts stored uniform crypto values
      that are wrong for non-crypto — see config.py docstring).
    * The winsorization here is the same clip as apply_winsorization() in the
      preprocessing pipeline, but driven by the artifact's stored bounds; the
      preprocessing module itself is never imported.

SETUP REQUIREMENT:
    Copy the three frozen pipeline files into a "feature_pipeline/" subfolder of
    this deployment folder (any leading numbering in the filenames is fine; they
    are located by pattern). Change FEATURE_PIPELINE_DIR below to point elsewhere
    if you prefer.
"""

from __future__ import annotations

import contextlib
import io
import os
import importlib.util
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import yfinance as yf

import config

# =============================================================================
# 1. LOAD THE THREE FROZEN PIPELINE MODULES (verbatim, by file path)
# =============================================================================
FEATURE_PIPELINE_DIR: Path = config.BASE_DIR / "feature_pipeline"

# The three frozen pipeline files are looked up in these locations, in order:
# a "feature_pipeline/" subfolder (recommended) or right next to this file.
_PIPELINE_SEARCH_DIRS: list[Path] = [FEATURE_PIPELINE_DIR, config.BASE_DIR]

# User-facing singular labels for the ticket header.
ASSET_SINGULAR: dict[str, str] = {
    "Crypto": "Crypto", "Forex": "Forex", "Indices": "Index", "Stocks": "Stock",
}

# Data-sanity: reject a ticker whose most recent COMPLETED bar is older than this
# many calendar days. The last completed bar is normally 1 (crypto, 7/7) to a few
# (stocks/indices/forex around holidays) days old, so 14 days is a safe ceiling
# that only catches broken or stale feeds -- e.g. a delisted/renamed symbol that
# yfinance still answers with months-old prices (the APT-USD 0.0001 case).
MAX_DATA_STALENESS_DAYS: int = 14


@contextlib.contextmanager
def _silent_in_tempdir():
    """Run a block with stdout suppressed and cwd in an empty temp dir.

    Needed when importing add_indicators.py, whose top-level loop is not guarded
    by `if __name__ == "__main__"`. Running it from an empty directory means its
    file loop finds nothing and has no effect; stdout is suppressed for cleanliness.
    """
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                yield
        finally:
            os.chdir(cwd)


def _find_pipeline_file(pattern: str) -> Path:
    for directory in _PIPELINE_SEARCH_DIRS:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    searched = ", ".join(str(d) for d in _PIPELINE_SEARCH_DIRS)
    raise FileNotFoundError(
        f"Could not find a file matching '{pattern}'. Searched: {searched}. "
        "Copy the three frozen pipeline files (add_indicators, feature_transformation, "
        "long_horizon_and_feature_engineering) into a 'feature_pipeline' subfolder of "
        "the deployment folder."
    )


def _load_module(name: str, path: Path, silent_import: bool = False):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    if silent_import:
        with _silent_in_tempdir():
            spec.loader.exec_module(module)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
    return module


_PIPELINE_FUNCTIONS: Optional[tuple] = None


def _ensure_numba_cache_dir() -> None:
    """Keep pandas-ta/numba import fast by using a writable temp cache."""
    cache_dir = Path(tempfile.gettempdir()) / "aegistrader_numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))


def pipeline_functions():
    """Load the frozen feature pipeline lazily on the first actual signal request."""
    global _PIPELINE_FUNCTIONS
    if _PIPELINE_FUNCTIONS is None:
        _ensure_numba_cache_dir()
        add_ind_mod = _load_module(
            "_aegis_add_indicators", _find_pipeline_file("*add_indicators*.py"),
            silent_import=True,
        )
        ft_mod = _load_module(
            "_aegis_feature_transformation", _find_pipeline_file("*feature_transformation*.py")
        )
        lh_mod = _load_module(
            "_aegis_long_horizon", _find_pipeline_file("*long_horizon*.py")
        )
        _PIPELINE_FUNCTIONS = (
            add_ind_mod.add_technical_indicators,
            ft_mod.process_static_transformations,
            lh_mod.add_long_horizon_and_engineered_features,
        )
    return _PIPELINE_FUNCTIONS

# Columns the dataset-creation scripts produced (long format), in their order.
_OHLCV_COLS = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]

# Existing human-readable asset mapping used by the app's "Asset codes" dialog.
_ASSET_MAPPING_PATH: Path = config.BASE_DIR / "AegisTrader_Asset_Mapping.xlsx"
_DISPLAY_NAME_BY_TICKER: Optional[dict[str, str]] = None


def _display_name_lookup() -> dict[str, str]:
    """Lazy {canonical ticker -> public display name} lookup from the mapping XLSX."""
    global _DISPLAY_NAME_BY_TICKER
    if _DISPLAY_NAME_BY_TICKER is not None:
        return _DISPLAY_NAME_BY_TICKER

    lookup: dict[str, str] = {}
    try:
        mapping = pd.read_excel(
            _ASSET_MAPPING_PATH,
            sheet_name="Asset_Mapping",
            usecols=["Display Name", "Ticker (yfinance)"],
        )
    except Exception as exc:  # noqa: BLE001 - names are nice-to-have, never blocking
        print(f"[warn] Could not load asset display names: {exc}", file=sys.stderr)
        _DISPLAY_NAME_BY_TICKER = lookup
        return lookup

    for record in mapping.to_dict("records"):
        ticker = record.get("Ticker (yfinance)")
        display_name = record.get("Display Name")
        if pd.isna(ticker) or pd.isna(display_name):
            continue
        ticker_text = str(ticker).strip()
        name_text = str(display_name).strip()
        if ticker_text and name_text:
            lookup[ticker_text.upper()] = name_text

    _DISPLAY_NAME_BY_TICKER = lookup
    return lookup


def get_asset_display_name(ticker: Optional[str]) -> Optional[str]:
    """Return the public name for a ticker, e.g. AAPL -> Apple Inc."""
    if not ticker:
        return None
    return _display_name_lookup().get(ticker.upper())


def format_asset_label(ticker: Optional[str], display_name: Optional[str] = None) -> str:
    """Display ticker plus the public asset name when the mapping is available."""
    if not ticker:
        return "-"
    name = (display_name or get_asset_display_name(ticker) or "").strip()
    if name and name.upper() != ticker.upper():
        return f"{ticker} - {name}"
    return ticker


# =============================================================================
# 2. RESULT TYPES
# =============================================================================
@dataclass
class TradeTicket:
    ok: bool
    status: str                      # "BUY" | "WATCH" | "NO_SIGNAL" | "UNAVAILABLE"
    model_id: str
    asset_class: str
    horizon: str
    ticker: Optional[str]
    low_confidence: bool
    display_name: Optional[str] = None
    probability: Optional[float] = None
    threshold: Optional[float] = None
    top_k_cutoff: Optional[float] = None
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reward_to_risk: Optional[float] = None
    as_of_date: Optional[str] = None
    valid_until: Optional[str] = None
    valid_days: Optional[int] = None
    generated_at: Optional[str] = None
    scored: int = 0
    skipped: int = 0
    notes: list[str] = field(default_factory=list)
    text: str = ""                   # copy-paste block for the broker


# =============================================================================
# 3. ARTIFACT CACHE
# =============================================================================
_ARTIFACT_CACHE: dict[str, dict] = {}


def load_artifact(model_id: str) -> dict:
    if model_id not in _ARTIFACT_CACHE:
        path = config.ARTIFACT_PATHS[model_id]
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found for {model_id}: {path}")
        _ARTIFACT_CACHE[model_id] = joblib.load(path)
    return _ARTIFACT_CACHE[model_id]


# =============================================================================
# 4. DATA + FEATURE PIPELINE
# =============================================================================
def download_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    """Download recent daily OHLCV in the same long format used for training.

    Returns None on any failure (delisted/renamed symbol, empty response), so the
    caller can simply skip the ticker.
    """
    end = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=config.LOOKBACK_CALENDAR_DAYS)
    try:
        raw = yf.download(
            ticker, start=start.date(), end=end.date(),
            interval=config.YF_INTERVAL, auto_adjust=True,
            progress=False, threads=False, timeout=config.YF_TIMEOUT_SECONDS,
        )
    except Exception:
        return None
    if raw is None or raw.empty:
        return None

    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    # The reset index column is the date (named "Date" or "Datetime").
    if "Date" not in raw.columns:
        first = raw.columns[0]
        raw = raw.rename(columns={first: "Date"})
    raw["Ticker"] = ticker
    for col in _OHLCV_COLS:
        if col not in raw.columns:
            # Volume can legitimately be absent for some FX feeds; fill with 0
            # so feature_transformation's Forex path handles it as it did in training.
            if col == "Volume":
                raw["Volume"] = 0.0
            else:
                return None
    return raw[_OHLCV_COLS].copy()


def build_features(ohlcv: pd.DataFrame) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Rebuild training features. Returns (feature_frame, price_ref) or (None, None).

    price_ref carries Date/Close/ATR (absolute), captured before
    feature_transformation drops them, for the trade-ticket geometry.
    """
    # Stage 1 — technical indicators. live_inference always processes a single
    # ticker, so calling the frozen function directly on the one-ticker frame is
    # identical to the training-time `groupby("Ticker").apply(...)` (one group),
    # and avoids a pandas-version difference where groupby/apply can drop the
    # grouping column. Then drop the indicator warm-up rows, as training did.
    (
        add_technical_indicators,
        process_static_transformations,
        add_long_horizon_and_engineered_features,
    ) = pipeline_functions()

    with contextlib.redirect_stdout(io.StringIO()):
        df = add_technical_indicators(ohlcv.copy())
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        return None, None

    price_ref = df[["Date", "Close", "ATR"]].copy()
    price_ref["Date"] = pd.to_datetime(price_ref["Date"])

    # Stage 2 — static transformation & cleansing (frozen, driven via temp CSV so
    # the EXACT same code runs as in training). The "_FINAL" name is what the
    # function expects; no target columns are needed.
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "inference_FINAL.csv")
        df.to_csv(in_path, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            process_static_transformations(in_path, td)
        out_path = os.path.join(td, "inference_ML_READY.csv")
        if not os.path.exists(out_path):
            return None, None
        ml_ready = pd.read_csv(out_path)

    # Stage 3 — long-horizon & engineered features (frozen), then drop warmup NaN
    # exactly as process_dataset() does (subset=new_cols).
    with contextlib.redirect_stdout(io.StringIO()):
        augmented, new_cols = add_long_horizon_and_engineered_features(ml_ready)
    augmented = augmented.dropna(subset=new_cols).reset_index(drop=True)
    if augmented.empty:
        return None, None

    augmented["Date"] = pd.to_datetime(augmented["Date"])
    return augmented, price_ref


def score_last_bar(model_id: str, augmented: pd.DataFrame) -> Optional[tuple[float, pd.Timestamp]]:
    """Align to artifact feature_columns, winsorize, scale-if-needed, predict.

    Returns (probability, last_bar_date) or None if the latest row is not clean.
    """
    artifact = load_artifact(model_id)
    feature_cols = list(artifact["feature_columns"])

    missing = [c for c in feature_cols if c not in augmented.columns]
    if missing:
        raise RuntimeError(
            f"{model_id}: rebuilt features are missing {len(missing)} column(s) "
            f"required by the artifact, e.g. {missing[:6]}. The feature pipeline "
            "and the artifact are out of sync."
        )

    # Use the last COMPLETED daily bar. The current UTC day's bar is still
    # forming (intraday), so dropping any bar dated today-or-later avoids
    # predicting on a partial OHLC. For crypto run on a weekend this drops the
    # in-progress day; for stocks the last bar is already an earlier closed day,
    # so nothing is dropped.
    today_utc_date = datetime.now(timezone.utc).date()
    completed = augmented[augmented["Date"].dt.date < today_utc_date]
    if completed.empty:
        return None
    last = completed.iloc[[-1]].copy()
    X = last[feature_cols].copy()
    if X.isna().any().any():
        return None  # insufficiently warmed-up last bar -> skip

    # Winsorization: clip to stored train bounds (identical to apply_winsorization).
    bounds = artifact.get("winsor_bounds") or {}
    for col, (lo, hi) in bounds.items():
        if col in X.columns:
            X[col] = X[col].clip(lower=lo, upper=hi)

    # Scaling only when the winning model needed it (LogisticRegression).
    scaler = artifact.get("scaler")
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)

    proba = float(artifact["model"].predict_proba(X)[:, 1][0])
    last_date = pd.to_datetime(last.iloc[0]["Date"])
    return proba, last_date


# =============================================================================
# 5. TICKET CONSTRUCTION
# =============================================================================
def _fmt_price(p: float) -> str:
    ap = abs(p)
    decimals = 2 if ap >= 100 else 4 if ap >= 1 else 6 if ap >= 0.01 else 8
    return f"{p:,.{decimals}f}"


def _generated_at_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _tier(proba: float, tau_be: float, tau_tk: Optional[float]) -> tuple[str, str]:
    """Classify a probability into a transparent tier (Plan C).

    Returns (status, human_line).
      BUY       : proba >= break-even           -> positive expected value
      WATCH     : top_k <= proba < break-even    -> top-ranked, but not yet EV-positive
      NO_SIGNAL : proba < top_k                  -> below the model's top tier
    If top_k is unavailable or >= break-even, the WATCH band collapses and only
    BUY / NO_SIGNAL remain.
    """
    if proba >= tau_be:
        return "BUY", "BUY — positive expected value (probability at or above break-even)."
    if tau_tk is not None and tau_tk < tau_be and proba >= tau_tk:
        return "WATCH", ("WATCH — among the model's top-ranked setups, but below the "
                         "break-even threshold (not yet positive expected value).")
    return "NO_SIGNAL", "NO SIGNAL — below the model's top-ranked tier."


def _build_ticket(
    model_id: str, asset: str, horizon: str, ticker: str,
    proba: float, entry: float, atr: float, last_date: pd.Timestamp,
    scored: int, skipped: int, extra_notes: list[str],
) -> TradeTicket:
    g = config.BARRIER_GEOMETRY[asset][horizon]
    sl_dist = g["atr_mult"] * atr
    stop_loss = entry - sl_dist
    take_profit = entry + sl_dist * g["rr"]
    tau_be = config.BUY_THRESHOLD[model_id]
    tau_tk = load_artifact(model_id).get("top_k_threshold")
    status, status_line = _tier(proba, tau_be, tau_tk)
    low_conf = model_id in config.LOW_CONFIDENCE_MODELS
    valid_days = config.SIGNAL_VALIDITY_DAYS[asset][horizon]
    valid_until = (last_date.date() + timedelta(days=valid_days)).isoformat()
    generated_at = _generated_at_text()

    sl_pct = (stop_loss - entry) / entry * 100.0
    tp_pct = (take_profit - entry) / entry * 100.0
    display_name = get_asset_display_name(ticker)
    asset_label = format_asset_label(ticker, display_name)
    header = (f"AegisTrader Signal — {asset_label} "
              f"({ASSET_SINGULAR[asset]} / {horizon} / {model_id})")
    tk_txt = f"{tau_tk:.2f}" if tau_tk is not None else "n/a"
    lines = [
        header,
        f"Signal:      {status_line}",
        f"Entry:       {_fmt_price(entry)}",
        f"Stop-Loss:   {_fmt_price(stop_loss)}   ({sl_pct:+.2f}%)",
        f"Take-Profit: {_fmt_price(take_profit)}   ({tp_pct:+.2f}%)",
        f"Reward/Risk: {g['rr']:.1f} : 1",
        f"Confidence:  {proba:.3f}   (top-k {tk_txt} · break-even {tau_be:.2f})",
        f"As of:       {last_date.date().isoformat()}",
        f"Generated:   {generated_at}",
        f"Valid until: {valid_until}   (~{valid_days} calendar days)",
    ]
    if low_conf:
        lines.append("! Low confidence (test ROC-AUC below 0.50) — treat with caution.")

    return TradeTicket(
        ok=True,
        status=status,
        model_id=model_id, asset_class=asset, horizon=horizon, ticker=ticker,
        low_confidence=low_conf, display_name=display_name,
        probability=round(proba, 4), threshold=tau_be, top_k_cutoff=tau_tk,
        entry=entry, stop_loss=stop_loss, take_profit=take_profit,
        reward_to_risk=g["rr"], as_of_date=last_date.date().isoformat(),
        valid_until=valid_until, valid_days=valid_days, generated_at=generated_at,
        scored=scored, skipped=skipped, notes=list(extra_notes),
        text="\n".join(lines),
    )


def _score_ticker(model_id: str, ticker: str) -> Optional[tuple[float, float, float, pd.Timestamp]]:
    """Full per-ticker pipeline. Returns (proba, entry, atr, last_date) or None.

    A single ticker must never abort a whole scan. Any failure below -- a bad
    download, empty features, STALE data, or a feature/artifact mismatch (e.g. an
    index whose Yahoo data has no Volume, so the frozen pipeline drops
    Return_Volume that this model requires) -- is treated as "skip this ticker"
    and returns None. Reasons are printed to stderr so scans stay debuggable
    without crashing the UI.
    """
    ohlcv = download_ohlcv(ticker)
    if ohlcv is None:
        return None
    try:
        augmented, price_ref = build_features(ohlcv)
        if augmented is None or len(augmented) < 1:
            return None
        scored = score_last_bar(model_id, augmented)
        if scored is None:
            return None
        proba, last_date = scored

        # Data-sanity: never surface a signal built on stale data. A feed that is
        # many days behind (delisted/renamed symbol returning months-old prices)
        # is skipped rather than shown as a live opportunity.
        stale_days = (datetime.now(timezone.utc).date() - last_date.date()).days
        if stale_days > MAX_DATA_STALENESS_DAYS:
            print(f"[skip] {model_id} / {ticker}: stale data, last completed bar "
                  f"{last_date.date()} is {stale_days}d old (> {MAX_DATA_STALENESS_DAYS})",
                  file=sys.stderr)
            return None

        ld = last_date.date()
        ref_row = price_ref.loc[price_ref["Date"].dt.date == ld]
        if ref_row.empty:
            # Fall back to the latest completed bar at or before last_date (never
            # the dropped current-day bar).
            earlier = price_ref[price_ref["Date"].dt.date <= ld]
            if earlier.empty:
                return None
            ref_row = earlier.iloc[[-1]]
        entry = float(ref_row.iloc[-1]["Close"])
        atr = float(ref_row.iloc[-1]["ATR"])
        if not np.isfinite(entry) or not np.isfinite(atr) or entry <= 0 or atr <= 0:
            return None
        return proba, entry, atr, last_date
    except Exception as exc:  # noqa: BLE001 - skip the ticker, keep the scan alive
        print(f"[skip] {model_id} / {ticker}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None


# =============================================================================
# 6. PUBLIC ENTRY POINT
# =============================================================================
def generate_signal(
    model_id: str,
    asset_class: str,
    horizon: str,
    specific_ticker: Optional[str] = None,
    candidate_tickers: Optional[list[str]] = None,
) -> TradeTicket:
    """
    Build a trade ticket for a routed model.

    specific_ticker  : if given, score only that ticker.
    candidate_tickers: override the ticker universe (testing); defaults to
                       config.TICKERS[asset_class] in strongest-signal mode.
    """
    low_conf = model_id in config.LOW_CONFIDENCE_MODELS

    # ---- specific ticker ----
    if specific_ticker is not None:
        res = _score_ticker(model_id, specific_ticker)
        if res is None:
            display_name = get_asset_display_name(specific_ticker)
            asset_label = format_asset_label(specific_ticker, display_name)
            generated_at = _generated_at_text()
            return TradeTicket(
                ok=False, status="UNAVAILABLE", model_id=model_id,
                asset_class=asset_class, horizon=horizon, ticker=specific_ticker,
                low_confidence=low_conf, display_name=display_name,
                generated_at=generated_at,
                notes=[
                    "Yahoo Finance did not return enough recent usable daily data "
                    "for this asset and horizon.",
                    "Try another ticker, leave the asset code blank to let the system "
                    "scan the asset class, or retry later if Yahoo Finance is slow.",
                ],
                text=f"AegisTrader Signal — {asset_label}: data unavailable, no signal.",
            )
        proba, entry, atr, last_date = res
        return _build_ticket(model_id, asset_class, horizon, specific_ticker,
                             proba, entry, atr, last_date, scored=1, skipped=0,
                             extra_notes=[])

    # ---- strongest signal across the asset class ----
    universe = candidate_tickers if candidate_tickers is not None else config.TICKERS[asset_class]
    best = None  # (proba, ticker, entry, atr, last_date)
    scored = skipped = 0
    for tk in universe:
        res = _score_ticker(model_id, tk)
        if res is None:
            skipped += 1
            continue
        scored += 1
        proba, entry, atr, last_date = res
        if best is None or proba > best[0]:
            best = (proba, tk, entry, atr, last_date)

    if best is None:
        generated_at = _generated_at_text()
        return TradeTicket(
            ok=False, status="UNAVAILABLE", model_id=model_id,
            asset_class=asset_class, horizon=horizon, ticker=None,
            low_confidence=low_conf, scored=scored, skipped=skipped,
            generated_at=generated_at,
            notes=[
                "No ticker in this asset class produced a usable live signal.",
                "Yahoo Finance may be slow/unavailable, or the available tickers did "
                "not have enough recent clean daily history. Try a narrower scope, "
                "a specific ticker, or retry later.",
            ],
            text=f"AegisTrader Signal — {asset_class}/{horizon}: no usable data, no signal.",
        )

    proba, tk, entry, atr, last_date = best
    note = f"Strongest of {scored} scored ticker(s) ({skipped} skipped)."
    return _build_ticket(model_id, asset_class, horizon, tk, proba, entry, atr,
                         last_date, scored=scored, skipped=skipped, extra_notes=[note])


# =============================================================================
# 7. SELF-CHECK  (runs on a machine WITH the artifacts + internet)
# =============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("AegisTrader live_inference.py — self-check (needs models/ + internet)")
    print("=" * 78)
    print(f"Feature pipeline search dirs: {[str(d) for d in _PIPELINE_SEARCH_DIRS]}")
    print("Frozen modules loaded:",
          "add_technical_indicators,",
          "process_static_transformations,",
          "add_long_horizon_and_engineered_features")

    checks = [
        ("AegisTrader_Stocks_Swing", "Stocks", "Swing", "AAPL"),   # specific stock
        ("AegisTrader_Crypto_Day",   "Crypto", "Day",   "BTC-USD"),  # specific crypto
    ]
    for model_id, asset, horizon, ticker in checks:
        print("\n" + "-" * 78)
        print(f"Scenario: specific ticker {ticker} -> {model_id}")
        try:
            ticket = generate_signal(model_id, asset, horizon, specific_ticker=ticker)
            print(ticket.text)
            print(f"[status={ticket.status}, ok={ticket.ok}]")
        except Exception as exc:  # noqa: BLE001 - self-check should report, not crash
            print(f"[ERROR] {type(exc).__name__}: {exc}")

    # Strongest-signal over a wider crypto subset, so a top-tier (WATCH/BUY)
    # setup has a realistic chance to appear (depends on the day's market).
    print("\n" + "-" * 78)
    print("Scenario: strongest signal, Crypto/Day over a 12-ticker subset")
    try:
        ticket = generate_signal(
            "AegisTrader_Crypto_Day", "Crypto", "Day", specific_ticker=None,
            candidate_tickers=["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
                               "ADA-USD", "AVAX-USD", "DOGE-USD", "LINK-USD", "DOT-USD",
                               "LTC-USD", "BCH-USD"],
        )
        print(ticket.text)
        print(f"[picked={ticket.ticker}, status={ticket.status}, "
              f"scored={ticket.scored}, skipped={ticket.skipped}]")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {type(exc).__name__}: {exc}")

    print("=" * 78)
    print("Self-check complete.")
