"""
explainer.py -- AegisTrader Web App Deployment
==============================================

Static, template-based explanations for the 12 AegisTrader sub-models and for
their live BUY / WATCH / NO SIGNAL outcomes.

This module is intentionally deterministic: it does not call an LLM and it does
not generate free-form text from hidden state. All wording is driven by
`config.py` plus the frozen quality labels from the final thesis evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import config


@dataclass(frozen=True)
class SubModelProfile:
    """Static metadata for one routed sub-model."""

    model_name: str
    asset_class: str
    horizon: str
    description: str
    roc_auc: float
    quality_label: str


# Final quality labels from:
# Terminal Outputs/Evaluation of the Models/Evaluation_of_the_Models_20260426_173430.txt
_QUALITY_LABELS: dict[str, str] = {
    "AegisTrader_Crypto_Day": "Strong relative performer",
    "AegisTrader_Crypto_Swing": "Moderate but usable signal",
    "AegisTrader_Crypto_Long": "Unreliable or not useful",
    "AegisTrader_Forex_Day": "Moderate but usable signal",
    "AegisTrader_Forex_Swing": "Weak signal",
    "AegisTrader_Forex_Long": "Weak signal",
    "AegisTrader_Indices_Day": "Weak signal",
    "AegisTrader_Indices_Swing": "Weak signal",
    "AegisTrader_Indices_Long": "Weak signal",
    "AegisTrader_Stocks_Day": "Weak signal",
    "AegisTrader_Stocks_Swing": "Moderate but usable signal",
    "AegisTrader_Stocks_Long": "Weak signal",
}


def _parse_model_name(model_name: str) -> tuple[str, str]:
    """Return (asset_class, horizon) from AegisTrader_<Asset>_<Horizon>."""

    if model_name not in config.MODEL_IDS:
        raise KeyError(f"Unknown model_name {model_name!r}")

    parts = model_name.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Expected model name in the form 'AegisTrader_<Asset>_<Horizon>', got {model_name!r}"
        )
    return parts[1], parts[2]


SUB_MODEL_PROFILES: dict[str, SubModelProfile] = {
    model_id: SubModelProfile(
        model_name=model_id,
        asset_class=_parse_model_name(model_id)[0],
        horizon=_parse_model_name(model_id)[1],
        description=config.MODEL_DESCRIPTIONS[model_id],
        roc_auc=config.MODEL_TEST_ROC_AUC[model_id],
        quality_label=_QUALITY_LABELS[model_id],
    )
    for model_id in config.MODEL_IDS
}


def build_model_card(model_name: str) -> str:
    """Return the static dashboard card for one sub-model."""

    profile = SUB_MODEL_PROFILES[model_name]
    return (
        f"### {profile.model_name}\n\n"
        f"**Asset class:** {profile.asset_class}  \n"
        f"**Horizon:** {profile.horizon}  \n"
        f"**Backtested quality:** {profile.quality_label}  \n"
        f"**Test ROC-AUC:** {profile.roc_auc:.4f}\n\n"
        f"{profile.description}"
    )


# =====================================================================
# SIGNAL PRESENTATION (dynamic) -- extends the static model card above.
# Shares SUB_MODEL_PROFILES as the single source of truth for quality,
# so the signal card and the model card can never contradict each other.
# =====================================================================

# Plan C tier constants -- must match the tier logic used upstream.
TIER_BUY = "BUY"
TIER_WATCH = "WATCH"
TIER_NO_SIGNAL = "NO_SIGNAL"


def _derive_tier(
    probability: float,
    break_even: float,
    top_k_cutoff: Optional[float],
) -> str:
    """
    Recompute the Plan C tier from raw numbers.

    This is not the canonical source of the tier; that lives upstream in the
    inference/routing layer. It exists only as a defensive consistency check so
    the explainer can detect an upstream mismatch instead of silently
    describing a wrong tier.
    """

    if probability >= break_even:
        return TIER_BUY
    if top_k_cutoff is not None and top_k_cutoff < break_even and probability >= top_k_cutoff:
        return TIER_WATCH
    return TIER_NO_SIGNAL


def _signal_quality_caveat(tier: str, quality_label: str) -> str:
    """
    Option-1 consistency caveat.

    The false-confidence risk exists only for BUY, so BUY carries a
    quality-scaled reservation; WATCH/NO_SIGNAL stay factual.
    """

    if tier == TIER_BUY:
        if quality_label == "Unreliable or not useful":
            return (
                "Reliability warning: the live probability crosses the buy threshold, "
                "but in backtesting this model ranked below random and did not provide "
                "a usable signal. This BUY should not be treated as actionable; it is "
                "shown only for transparency."
            )
        if quality_label == "Weak signal":
            return (
                "Note: this model produced a comparatively weak backtested signal. "
                "Treat this as a low-conviction candidate and require strong "
                "independent confirmation before acting."
            )
        if quality_label == "Moderate but usable signal":
            return (
                "This model showed a moderate backtested signal. Use it as one input "
                "among several, not as a standalone decision."
            )
        return (
            "This is the strongest-ranked model in the lineup. Even so, every signal "
            "is probabilistic, so size positions responsibly."
        )

    if tier == TIER_WATCH:
        return (
            "This candidate is on the watchlist: its probability is elevated but has "
            "not cleared the buy threshold. No action is recommended yet."
        )

    return (
        "No signal today: the model's probability is below the watchlist threshold "
        "for this asset and horizon."
    )


def build_signal_presentation(
    model_name: str,
    probability: float,
    break_even: float,
    top_k_cutoff: Optional[float],
    tier: str,
) -> str:
    """
    Return a static, honest, template-based explanation of a live signal.

    All numeric inputs (probability, break_even, top_k_cutoff, tier) come from
    the inference/routing layer at runtime, so there is no train/serve risk and
    no stale threshold. `tier` is the canonical tier decided upstream; this
    function only explains it and cross-checks it.
    """

    profile = SUB_MODEL_PROFILES[model_name]

    # Defensive check: does the passed tier match what the numbers imply?
    derived = _derive_tier(probability, break_even, top_k_cutoff)
    if derived != tier:
        warnings.warn(
            f"[explainer] tier mismatch for {model_name}: upstream='{tier}', "
            f"numbers imply='{derived}' (P={probability:.4f}, "
            f"break_even={break_even:.4f}, top_k={top_k_cutoff!r})"
        )

    band_note = ""
    if top_k_cutoff is None:
        band_note = (
            "\n\n*(This model does not expose a separate watchlist threshold, so only "
            "BUY / NO SIGNAL are available.)*"
        )
    elif top_k_cutoff >= break_even:
        band_note = (
            "\n\n*(For this model the watchlist and buy thresholds coincide, so the "
            "WATCH band is empty by construction.)*"
        )

    caveat = _signal_quality_caveat(tier, profile.quality_label)

    # The signal card already shows the tier, probability, and both thresholds, so
    # this returns ONLY the parts that add new information: the honest tier/quality
    # caveat and, where relevant, the collapsed-WATCH-band note.
    return f"{caveat}{band_note}"


if __name__ == "__main__":
    print("All is ok.")
