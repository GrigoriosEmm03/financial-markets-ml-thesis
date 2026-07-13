"""
app.py -- AegisTrader Web App (Streamlit entry point)
=====================================================

Ties the deterministic layers together into a single UI:

    router.py          -> maps the 5-question form to one of the 12 sub-models
    live_inference.py  -> downloads data, rebuilds frozen features, scores, tickets
    explainer.py       -> static, honest, template-based explanations
    config.py          -> constants, tickers, thresholds

Run locally (Windows, thesis_env active):

    python -m streamlit run app.py

The app opens at http://localhost:8501. Stop with Ctrl+C.

Design notes:
- No st.form is used for the questionnaire, so the "Browse asset codes" button
  can sit next to Question 3 and open an st.dialog (buttons other than a submit
  button are not allowed inside st.form).
- All heavy work (data download + scoring) happens behind buttons and is cached,
  so a plain page load never touches the network or the 12 model artifacts.
- Question 5 (experience) does not affect routing or the model explanation; it
  only drives a short app-level tip, so nothing here overstates its role.
"""

from __future__ import annotations

import base64
import math
import time
from html import escape
import pandas as pd
import streamlit as st

import config
import router
import live_inference as li
import explainer

# =============================================================================
# 0. CONSTANTS
# =============================================================================
CACHE_TTL_SECONDS = 6 * 60 * 60  # refresh cached scans at most every 6 hours
SIGNAL_DISPLAY_VERSION = 3        # bump when the cached TradeTicket display changes
FEATURE_PIPELINE_CACHE_VERSION = 1
MAPPING_PATH = config.BASE_DIR / "AegisTrader_Asset_Mapping.xlsx"
LOGO_PATH = config.BASE_DIR / "assets" / "aegis_logo.png"
ML_IMAGE_PATH = config.BASE_DIR / "assets" / "ML Image.png"

# Traffic-light palette for the model quality tiers (dashboard charts).
QUALITY_COLORS: dict[str, str] = {
    "Strong relative performer": "#2ECC71",   # vivid green
    "Moderate but usable signal": "#F1C40F",  # yellow
    "Weak signal": "#E67E22",                 # orange
    "Unreliable or not useful": "#E74C3C",    # red
}
# Short labels shown in charts.
QUALITY_SHORT: dict[str, str] = {
    "Strong relative performer": "Strong",
    "Moderate but usable signal": "Moderate",
    "Weak signal": "Weak",
    "Unreliable or not useful": "Unreliable",
}


def inject_theme_css() -> None:
    """Neon-futuristic styling applied directly in the app (independent of
    config.toml, so it always renders)."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

        .stApp {
            background:
                radial-gradient(1100px 550px at 15% -10%, rgba(34,211,238,0.10), transparent 60%),
                radial-gradient(900px 500px at 95% 5%, rgba(0,255,163,0.08), transparent 60%),
                #0A0E17;
        }
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        h1, h2, h3, h4 {
            font-family: 'Orbitron', 'Inter', sans-serif !important;
            letter-spacing: 0;
            text-shadow: 0 0 14px rgba(34,211,238,0.30);
        }
        code, pre, kbd { font-family: 'JetBrains Mono', monospace !important; }

        /* Neon buttons */
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(90deg, #22D3EE, #00FFA3) !important;
            color: #06121A !important; border: none !important;
            font-weight: 600 !important; border-radius: 10px !important;
            box-shadow: 0 0 16px rgba(34,211,238,0.35) !important;
            transition: box-shadow .2s ease, transform .05s ease;
        }
        .stButton > button:hover { box-shadow: 0 0 26px rgba(0,255,163,0.55) !important; }
        .stButton > button:active { transform: translateY(1px); }

        /* Glowing glass metric cards */
        [data-testid="stMetric"] {
            background: rgba(18,26,42,0.55);
            border: 1px solid rgba(34,211,238,0.30);
            border-radius: 14px; padding: 14px 18px;
            box-shadow: 0 0 18px rgba(34,211,238,0.12);
            backdrop-filter: blur(6px);
        }
        [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace; color: #22D3EE;
        }

        /* Tabs + inputs accents */
        .stTabs [aria-selected="true"] { color: #22D3EE !important; }
        .stTabs [data-baseweb="tab-highlight"] { background-color: #22D3EE !important; }

        /* Animated neon wave banner */
        .aegis-wave-wrap { width: 100%; height: 64px; overflow: hidden; margin: 2px 0 10px; }
        .aegis-wave { width: 200%; height: 100%; animation: aegisflow 9s linear infinite; }
        @keyframes aegisflow { from { transform: translateX(0); } to { transform: translateX(-50%); } }

        /* Top hero: logo + candles centered above title and wave. */
        .aegis-hero {
            position: relative;
            min-height: 390px;
            padding: 64px 0 14px;
            overflow: visible;
            text-align: center;
        }
        .aegis-hero-copy {
            max-width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .aegis-hero-mark {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 30px;
            width: 100%;
            margin: 0 auto 24px;
        }
        .aegis-hero-image {
            flex: 0 0 auto;
            width: 220px;
            text-align: center;
        }
        .aegis-hero-image img {
            width: 100%;
            height: auto;
            display: block;
            filter: drop-shadow(0 0 22px rgba(0,255,163,0.18));
        }
        .aegis-hero-ml-image img {
            width: 84%;
            margin: 0 auto;
        }
        .aegis-hero-candles {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            width: min(760px, 54vw);
            height: 170px;
            margin: 0;
        }
        .aegis-hero-candles .aegis-candle {
            position: relative;
            width: 11px;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: aegisbob 3.2s ease-in-out infinite;
            animation-delay: calc(var(--i) * -0.11s);
        }
        .aegis-hero-candles .aegis-candle.up { color: #2ECC71; }
        .aegis-hero-candles .aegis-candle.down { color: #E74C3C; }
        .aegis-hero-candles .aegis-wick {
            position: absolute;
            width: 2px;
            background: currentColor;
            opacity: 0.75;
        }
        .aegis-hero-candles .aegis-body {
            width: 100%;
            background: currentColor;
            border-radius: 2px;
            box-shadow: 0 0 12px currentColor;
        }
        .aegis-hero h1 {
            font-family: 'Orbitron', 'Inter', sans-serif;
            font-size: 60px;
            line-height: 1.05;
            letter-spacing: 0;
            margin: 0 auto 18px;
            color: #F4FDFF;
            text-shadow: 0 0 18px rgba(34,211,238,0.36);
            white-space: nowrap;
        }
        .aegis-hero p {
            margin: 0 auto;
            max-width: 1120px;
            color: rgba(230,237,243,0.72);
            font-weight: 600;
        }
        @media (max-width: 1200px) {
            .aegis-hero { min-height: 370px; }
            .aegis-hero-mark { gap: 22px; }
            .aegis-hero-image { width: 190px; }
            .aegis-hero-candles { width: min(620px, 52vw); }
            .aegis-hero h1 { font-size: 46px; }
        }
        @media (max-width: 820px) {
            .aegis-hero {
                min-height: auto;
                padding-top: 28px;
            }
            .aegis-hero-mark {
                flex-direction: column;
                gap: 10px;
                margin-bottom: 18px;
            }
            .aegis-hero-image { width: 180px; }
            .aegis-hero-candles {
                width: 100%;
                height: 150px;
                gap: 5px;
            }
            .aegis-hero-candles .aegis-candle { width: 8px; }
            .aegis-hero h1 { font-size: 34px; line-height: 1.14; white-space: normal; }
        }
        @keyframes aegisbob {
            0%, 100% { transform: translateY(13px); }
            50% { transform: translateY(-15px); }
        }

        /* Fixed-height dashboard SVG panels. */
        .aegis-dashboard-chart {
            height: 390px;
            padding: 12px 14px 10px;
            margin: 0 0 22px;
            background: rgba(8, 13, 23, 0.45);
            border: 1px solid rgba(34, 211, 238, 0.16);
            border-radius: 8px;
            box-shadow: 0 0 18px rgba(34, 211, 238, 0.06);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .aegis-chart-title {
            color: #F4FDFF;
            font-weight: 800;
            font-size: 16px;
            line-height: 1.25;
            margin: 0;
        }
        .aegis-chart-subtitle {
            color: rgba(230, 237, 243, 0.78);
            font-size: 13px;
            font-style: italic;
            font-weight: 600;
            margin: 4px 0 6px;
        }
        .aegis-chart-body {
            flex: 1;
            min-height: 0;
        }
        .aegis-chart-body svg {
            width: 100%;
            height: 100%;
            display: block;
        }
        .aegis-chart-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 8px 14px;
            align-items: center;
            justify-content: center;
            color: rgba(230, 237, 243, 0.82);
            font-size: 12px;
            font-weight: 600;
            padding-top: 4px;
        }
        .aegis-legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
            vertical-align: -1px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_wave() -> None:
    """A seamless, CSS-animated neon wave banner (pure SVG + CSS, no JS)."""
    st.markdown(
        """
        <div class="aegis-wave-wrap">
          <svg class="aegis-wave" viewBox="0 0 2880 120" preserveAspectRatio="none">
            <defs>
              <linearGradient id="aegisGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0" stop-color="#22D3EE"/>
                <stop offset="0.5" stop-color="#00FFA3"/>
                <stop offset="1" stop-color="#22D3EE"/>
              </linearGradient>
            </defs>
            <path d="M0,60 Q180,10 360,60 T720,60 T1080,60 T1440,60 T1800,60 T2160,60 T2520,60 T2880,60"
                  fill="none" stroke="url(#aegisGrad)" stroke-width="3" opacity="0.75"/>
            <path d="M0,72 Q180,32 360,72 T720,72 T1080,72 T1440,72 T1800,72 T2160,72 T2520,72 T2880,72"
                  fill="none" stroke="url(#aegisGrad)" stroke-width="2" opacity="0.35"/>
          </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


def candles_html(
    n: int = 34,
    body_scale: float = 2.4,
    wick_base: float = 14,
    wick_wave: float = 10,
) -> str:
    """Return deterministic CSS-animated candle HTML."""
    import math

    prices = [50 + 20 * math.sin(i * 0.42) + 9 * math.sin(i * 0.15 + 1.3)
              for i in range(n + 1)]
    candles = []
    for i in range(n):
        o, c = prices[i], prices[i + 1]
        up = c >= o
        body_h = max(7, abs(c - o) * body_scale)
        wick_h = body_h + wick_base + wick_wave * abs(math.sin(i * 0.9))
        candles.append(
            f'<div class="aegis-candle {"up" if up else "down"}" style="--i:{i}">'
            f'<span class="aegis-wick" style="height:{wick_h:.0f}px"></span>'
            f'<span class="aegis-body" style="height:{body_h:.0f}px"></span>'
            f'</div>'
        )
    return "".join(candles)


def render_candles(n: int = 34) -> None:
    """A decorative, CSS-animated candlestick strip (pure HTML/CSS, no JS).

    A deterministic pseudo-price series colours each candle green (up) / red
    (down); a phase-shifted 'bob' animation makes them rise and fall like a wave.
    """
    st.markdown(
        """
        <style>
        .aegis-chart { display:flex; align-items:center; justify-content:center;
            gap:6px; height:150px; margin:6px 0 2px; }
        .aegis-candle { position:relative; width:9px; display:flex; align-items:center;
            justify-content:center; animation: aegisbob 3.2s ease-in-out infinite;
            animation-delay: calc(var(--i) * -0.11s); }
        .aegis-candle.up { color:#2ECC71; } .aegis-candle.down { color:#E74C3C; }
        .aegis-wick { position:absolute; width:2px; background:currentColor; opacity:0.75; }
        .aegis-body { width:100%; background:currentColor; border-radius:2px;
            box-shadow:0 0 9px currentColor; }
        @keyframes aegisbob { 0%,100%{ transform:translateY(11px);} 50%{ transform:translateY(-13px);} }
        </style>
        <div class="aegis-chart">""" + candles_html(n) + """</div>
        """,
        unsafe_allow_html=True,
    )


def image_data_uri(path) -> str:
    """Embed the local logo in custom HTML without relying on external serving."""
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_hero(has_logo: bool, has_ml_image: bool) -> None:
    logo_html = ""
    if has_logo:
        logo_html = (
            '<div class="aegis-hero-image">'
            f'<img src="{image_data_uri(LOGO_PATH)}" alt="AegisTrader logo">'
            '</div>'
        )
    ml_image_html = ""
    if has_ml_image:
        ml_image_html = (
            '<div class="aegis-hero-image aegis-hero-ml-image">'
            f'<img src="{image_data_uri(ML_IMAGE_PATH)}" alt="Machine learning illustration">'
            '</div>'
        )

    st.markdown(
        f"""
        <section class="aegis-hero">
          <div class="aegis-hero-copy">
            <div class="aegis-hero-mark">
              {logo_html}
              <div class="aegis-hero-candles">
                {candles_html(36, body_scale=3.1, wick_base=22, wick_wave=15)}
              </div>
              {ml_image_html}
            </div>
            <h1>AegisTrader &mdash; ML Trading Signals</h1>
            <p>
              Academic thesis prototype. Educational use only &mdash; not investment advice.
              Signals are probabilistic and past performance does not guarantee future results.
            </p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


TIER_BADGE: dict[str, tuple[str, str]] = {
    "BUY": ("\U0001F7E2", "BUY - high conviction"),
    "WATCH": ("\U0001F7E1", "WATCH - moderate conviction"),
    "NO_SIGNAL": ("\u26AA", "Low conviction - best available"),
    "UNAVAILABLE": ("\u26A0\uFE0F", "Signal unavailable"),
}

TIER_RANK: dict[str, int] = {"BUY": 2, "WATCH": 1, "NO_SIGNAL": 0, "UNAVAILABLE": -1}

# Option A: only models with genuine predictive power headline Tab 2. Reliability
# is judged by the frozen quality label (driven by test ROC-AUC + lift), NOT by
# global_score, which is inflated by base rate and would rank degenerate models
# (e.g. Stocks_Long, ROC-AUC 0.535) misleadingly high.
RELIABLE_QUALITIES: set[str] = {
    "Strong relative performer", "Moderate but usable signal",
}


def signal_strength_key(ticket: li.TradeTicket) -> tuple[int, float]:
    """Within one reliability group, rank by tier (BUY > WATCH > NO_SIGNAL) then
    live probability. Both are comparable inside the group, so this matches
    intuition without letting a degenerate model's inflated BUY win."""
    return (TIER_RANK.get(ticket.status, -1), ticket.probability or 0.0)


def ranking_rows(tickets: list[li.TradeTicket]) -> "pd.DataFrame":
    """Build the display table (with the honest reliability score: test ROC-AUC)."""
    rows = []
    for t in tickets:
        profile = explainer.SUB_MODEL_PROFILES[t.model_id]
        rows.append({
            "Model": t.model_id,
            "Quality": profile.quality_label,
            "Test ROC-AUC": f"{profile.roc_auc:.3f}",
            "Ticker": t.ticker,
            "Tier": t.status.replace("_", " "),
            "Probability": fmt_pct(t.probability),
        })
    return pd.DataFrame(rows)


def filter_table(df: "pd.DataFrame", query: str) -> "pd.DataFrame":
    """Keep rows whose text (any column) contains the query (case-insensitive)."""
    if not query:
        return df
    q = query.lower()
    mask = df.apply(
        lambda row: q in " ".join(str(v).lower() for v in row.to_numpy()), axis=1
    )
    return df[mask]


def show_ranking_table(tickets: list[li.TradeTicket], query: str) -> None:
    """Render one ranking table, filtered by the shared search query."""
    table = filter_table(ranking_rows(tickets), query)
    if table.empty:
        st.caption("No models match the filter.")
    else:
        st.dataframe(table, hide_index=True, width="stretch")


# ---- Models Dashboard data + charts (static model metadata, no live data) -------
def models_dataframe() -> "pd.DataFrame":
    """One row per sub-model, from config + explainer profiles (all verified)."""
    rows = []
    for model_id in config.MODEL_IDS:
        profile = explainer.SUB_MODEL_PROFILES[model_id]
        rows.append({
            "Model": model_id,
            "Name": model_id.replace("AegisTrader_", ""),
            "Asset class": profile.asset_class,
            "Horizon": profile.horizon,
            "Quality": profile.quality_label,
            "ROC-AUC": profile.roc_auc,
            "Break-even": config.BUY_THRESHOLD[model_id],
        })
    return pd.DataFrame(rows)


QUALITY_SHORT_COLORS: dict[str, str] = {
    short: QUALITY_COLORS[long] for long, short in QUALITY_SHORT.items()
}

ASSET_COLORS: dict[str, str] = {
    "Crypto": "#00FFA3",
    "Forex": "#22D3EE",
    "Indices": "#FFB020",
    "Stocks": "#FF2E97",
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    clean = hex_color.lstrip("#")
    return tuple(int(clean[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _lerp_color(left: str, right: str, t: float) -> str:
    t = _clamp(t, 0.0, 1.0)
    l_rgb = _hex_to_rgb(left)
    r_rgb = _hex_to_rgb(right)
    return _rgb_to_hex(tuple(
        int(round(l_rgb[i] + (r_rgb[i] - l_rgb[i]) * t)) for i in range(3)
    ))


def _auc_color(value: float) -> str:
    stops = [
        (0.40, "#B91C1C"),
        (0.465, "#EF4444"),
        (0.50, "#A7B0B8"),
        (0.60, "#34D399"),
        (0.75, "#00FF7A"),
    ]
    if value <= stops[0][0]:
        return stops[0][1]
    for (left_v, left_c), (right_v, right_c) in zip(stops, stops[1:]):
        if value <= right_v:
            return _lerp_color(left_c, right_c, (value - left_v) / (right_v - left_v))
    return stops[-1][1]


def _chart_panel(title: str, subtitle: str, chart_svg: str,
                 legend_html: str = "") -> str:
    legend = f'<div class="aegis-chart-legend">{legend_html}</div>' if legend_html else ""
    return f"""
    <div class="aegis-dashboard-chart">
      <p class="aegis-chart-title">{escape(title)}</p>
      <p class="aegis-chart-subtitle">{escape(subtitle)}</p>
      <div class="aegis-chart-body">{chart_svg}</div>
      {legend}
    </div>
    """


def _legend(items: list[tuple[str, str]]) -> str:
    return "".join(
        f'<span><span class="aegis-legend-dot" style="background:{color}"></span>'
        f'{escape(label)}</span>'
        for label, color in items
    )


def _model_power_chart(fdf: "pd.DataFrame") -> str:
    bar_df = fdf.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    left, right, top, bottom = 188, 32, 20, 32
    width, height = 760, 286
    chart_w = width - left - right
    chart_h = height - top - bottom
    domain_min, domain_max = 0.40, 0.75

    def x_pos(value: float) -> float:
        pct = (value - domain_min) / (domain_max - domain_min)
        return left + _clamp(pct, 0.0, 1.0) * chart_w

    rows = []
    count = max(len(bar_df), 1)
    row_h = chart_h / count
    bar_h = min(18, max(8, row_h * 0.52))
    for i, row in bar_df.iterrows():
        y = top + i * row_h + row_h / 2
        auc = float(row["ROC-AUC"])
        x = x_pos(auc)
        color = _auc_color(auc)
        label = escape(str(row["Name"]))
        rows.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#D9E2EA" font-size="12" font-weight="650">{label}</text>'
            f'<rect x="{left}" y="{y - bar_h / 2:.1f}" width="{max(2, x - left):.1f}" '
            f'height="{bar_h:.1f}" rx="3" fill="{color}" opacity="0.92"/>'
            f'<text x="{min(x + 8, width - 38):.1f}" y="{y + 4:.1f}" '
            f'fill="#E6EDF3" font-size="12" font-weight="700">{auc:.3f}</text>'
        )

    grid = []
    for tick in [0.40, 0.50, 0.60, 0.70]:
        x = x_pos(tick)
        dash = ' stroke-dasharray="5 5"' if tick == 0.50 else ""
        grid.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_h}" '
            f'stroke="#95A5A6" stroke-opacity="{0.55 if tick == 0.50 else 0.18}"{dash}/>'
            f'<text x="{x:.1f}" y="{height - 8}" text-anchor="middle" '
            f'fill="#AAB6C2" font-size="11">{tick:.2f}</text>'
        )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Predictive Power by Model">'
        f'<rect width="{width}" height="{height}" fill="transparent"/>'
        + "".join(grid)
        + "".join(rows)
        + f'<text x="{left + chart_w / 2:.1f}" y="14" text-anchor="middle" '
        f'fill="#AAB6C2" font-size="11">Test ROC-AUC</text>'
        + '</svg>'
    )
    return _chart_panel(
        "Predictive Power by Model",
        "Which models rank meaningfully above random? (grey = 0.50)",
        svg,
        _legend([("Low ROC-AUC", "#B91C1C"), ("Random 0.50", "#A7B0B8"), ("High ROC-AUC", "#00FF7A")]),
    )


def _pie_slice_path(cx: float, cy: float, r: float,
                    start_deg: float, end_deg: float) -> str:
    start = math.radians(start_deg - 90)
    end = math.radians(end_deg - 90)
    x1, y1 = cx + r * math.cos(start), cy + r * math.sin(start)
    x2, y2 = cx + r * math.cos(end), cy + r * math.sin(end)
    large_arc = 1 if end_deg - start_deg > 180 else 0
    return (
        f'M {cx:.1f} {cy:.1f} L {x1:.1f} {y1:.1f} '
        f'A {r:.1f} {r:.1f} 0 {large_arc} 1 {x2:.1f} {y2:.1f} Z'
    )


def _reliability_pie_chart(fdf: "pd.DataFrame") -> str:
    qmix = (
        fdf["Quality (short)"].value_counts()
        .reindex(list(QUALITY_SHORT.values()), fill_value=0)
        .rename_axis("Quality").reset_index(name="Models")
    )
    qmix = qmix[qmix["Models"] > 0]
    total = int(qmix["Models"].sum())
    cx, cy, radius = 380, 139, 106
    width, height = 760, 286

    pieces = []
    start = 0.0
    if len(qmix) == 1:
        quality = str(qmix.iloc[0]["Quality"])
        count = int(qmix.iloc[0]["Models"])
        color = QUALITY_SHORT_COLORS[quality]
        pieces.append(
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{color}" opacity="0.92"/>'
            f'<text x="{cx}" y="{cy + 8}" text-anchor="middle" fill="#06121A" '
            f'font-size="42" font-weight="900">{count}</text>'
        )
    else:
        for _, row in qmix.iterrows():
            quality = str(row["Quality"])
            count = int(row["Models"])
            angle = 360.0 * count / total
            end = start + angle
            color = QUALITY_SHORT_COLORS[quality]
            pieces.append(
                f'<path d="{_pie_slice_path(cx, cy, radius, start, end)}" '
                f'fill="{color}" stroke="#0A0E17" stroke-width="3" opacity="0.94"/>'
            )
            mid = math.radians(start + angle / 2 - 90)
            tx = cx + radius * 0.58 * math.cos(mid)
            ty = cy + radius * 0.58 * math.sin(mid)
            pieces.append(
                f'<text x="{tx:.1f}" y="{ty + 7:.1f}" text-anchor="middle" '
                f'fill="#06121A" font-size="28" font-weight="900">{count}</text>'
            )
            start = end

    svg = (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Reliability pie chart">'
        f'<rect width="{width}" height="{height}" fill="transparent"/>'
        + "".join(pieces)
        + f'<text x="{cx}" y="{cy + radius + 30}" text-anchor="middle" '
        f'fill="#D9E2EA" font-size="13" font-weight="700">{total} selected models</text>'
        + '</svg>'
    )
    legend_items = [(quality, QUALITY_SHORT_COLORS[quality]) for quality in qmix["Quality"]]
    return _chart_panel(
        "Reliability Mix of Selected Models",
        "Models per quality tier. Counts are printed on the pie.",
        svg,
        _legend(legend_items),
    )


def _heatmap_chart(fdf: "pd.DataFrame") -> str:
    assets = [a for a in config.ASSET_CLASSES if a in set(fdf["Asset class"])]
    horizons = [h for h in config.HORIZONS if h in set(fdf["Horizon"])]
    values = {
        (str(row["Asset class"]), str(row["Horizon"])): float(row["ROC-AUC"])
        for _, row in fdf.iterrows()
    }
    width, height = 760, 286
    left, top = 128, 36
    chart_w, chart_h = 570, 194
    col_w = chart_w / max(len(horizons), 1)
    row_h = chart_h / max(len(assets), 1)
    cells = []

    for j, horizon in enumerate(horizons):
        x = left + j * col_w + col_w / 2
        cells.append(
            f'<text x="{x:.1f}" y="24" text-anchor="middle" fill="#D9E2EA" '
            f'font-size="13" font-weight="800">{escape(horizon)}</text>'
        )
    for i, asset in enumerate(assets):
        y = top + i * row_h + row_h / 2
        cells.append(
            f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" fill="#D9E2EA" '
            f'font-size="13" font-weight="800">{escape(asset)}</text>'
        )
        for j, horizon in enumerate(horizons):
            value = values.get((asset, horizon))
            if value is None:
                continue
            x = left + j * col_w
            y0 = top + i * row_h
            color = _auc_color(value)
            text_color = "#06121A" if value >= 0.50 else "#F4FDFF"
            cells.append(
                f'<rect x="{x + 4:.1f}" y="{y0 + 4:.1f}" width="{col_w - 8:.1f}" '
                f'height="{row_h - 8:.1f}" rx="6" fill="{color}" opacity="0.92"/>'
                f'<text x="{x + col_w / 2:.1f}" y="{y0 + row_h / 2 + 6:.1f}" '
                f'text-anchor="middle" fill="{text_color}" font-size="18" '
                f'font-weight="900">{value:.2f}</text>'
            )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Heatmap">'
        f'<rect width="{width}" height="{height}" fill="transparent"/>'
        + "".join(cells)
        + '</svg>'
    )
    return _chart_panel(
        "Predictive Power by Asset Class and Horizon",
        "Where signal concentrates across markets and horizons.",
        svg,
        _legend([("Below 0.50", "#C0392B"), ("Around 0.50", "#95A5A6"), ("Above 0.50", "#27AE60")]),
    )


def _line_chart(fdf: "pd.DataFrame") -> str:
    width, height = 760, 286
    left, right, top, bottom = 72, 34, 22, 48
    chart_w, chart_h = width - left - right, height - top - bottom
    domain_min, domain_max = 0.40, 0.75
    horizons = [h for h in config.HORIZONS if h in set(fdf["Horizon"])]
    assets = [a for a in config.ASSET_CLASSES if a in set(fdf["Asset class"])]

    def x_pos(horizon: str) -> float:
        if len(horizons) == 1:
            return left + chart_w / 2
        return left + horizons.index(horizon) * (chart_w / (len(horizons) - 1))

    def y_pos(value: float) -> float:
        pct = (value - domain_min) / (domain_max - domain_min)
        return top + (1 - _clamp(pct, 0.0, 1.0)) * chart_h

    parts = []
    for tick in [0.40, 0.50, 0.60, 0.70]:
        y = y_pos(tick)
        dash = ' stroke-dasharray="5 5"' if tick == 0.50 else ""
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" '
            f'stroke="#95A5A6" stroke-opacity="{0.50 if tick == 0.50 else 0.16}"{dash}/>'
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" fill="#AAB6C2" '
            f'font-size="11">{tick:.2f}</text>'
        )
    for horizon in horizons:
        x = x_pos(horizon)
        parts.append(
            f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle" fill="#D9E2EA" '
            f'font-size="12" font-weight="800">{escape(horizon)}</text>'
        )

    for asset in assets:
        adf = fdf[fdf["Asset class"] == asset].copy()
        adf["Horizon"] = pd.Categorical(adf["Horizon"], categories=config.HORIZONS, ordered=True)
        adf = adf.sort_values("Horizon")
        points = [
            (x_pos(str(row["Horizon"])), y_pos(float(row["ROC-AUC"])), float(row["ROC-AUC"]))
            for _, row in adf.iterrows()
            if str(row["Horizon"]) in horizons
        ]
        if not points:
            continue
        color = ASSET_COLORS[asset]
        point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        if len(points) > 1:
            parts.append(
                f'<polyline points="{point_str}" fill="none" stroke="{color}" '
                f'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        for x, y, auc in points:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#0A0E17" '
                f'stroke-width="2"/>'
                f'<text x="{x:.1f}" y="{y - 11:.1f}" text-anchor="middle" fill="#E6EDF3" '
                f'font-size="11" font-weight="700">{auc:.2f}</text>'
            )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Line chart">'
        f'<rect width="{width}" height="{height}" fill="transparent"/>'
        + "".join(parts)
        + '</svg>'
    )
    return _chart_panel(
        "Predictive Power Across Horizons",
        "Does the signal decay at longer horizons?",
        svg,
        _legend([(asset, ASSET_COLORS[asset]) for asset in assets]),
    )


def _slicer(label: str, options: list, key: str) -> list:
    """Dropdown multiselect. Starts empty (placeholder); an empty selection means
    'all'. Uses Streamlit's built-in 'Select all' -- no custom sentinel, so there
    is only one select-all and picked values leave the dropdown list."""
    chosen = st.multiselect(label, list(options), default=[], key=key,
                            placeholder="All (choose to filter)")
    return chosen if chosen else list(options)


def render_models_dashboard() -> None:
    st.markdown("### Models Dashboard")
    st.caption(
        "A visual overview of AegisTrader's 12 sub-models and the 164 assets they "
        "cover.  **Which Asset Classes and Horizons Carry Genuine Predictive Power "
        "-- and How Much of the Market Do We Actually Cover?**"
    )

    df = models_dataframe()
    df["Quality (short)"] = df["Quality"].map(QUALITY_SHORT)
    total_assets = sum(len(v) for v in config.TICKERS.values())
    reliable_n = int(df["Quality"].isin(RELIABLE_QUALITIES).sum())
    best = df.loc[df["ROC-AUC"].idxmax()]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sub-models", len(config.MODEL_IDS))
    k2.metric("Tradable assets", total_assets)
    k3.metric("Asset classes", len(config.ASSET_CLASSES))
    k4.metric("Reliable models", reliable_n, help="Strong or Moderate quality")
    k5.metric("Best ROC-AUC", f"{best['ROC-AUC']:.3f}", help=str(best["Model"]))

    st.write("")
    s1, s2, s3 = st.columns(3)
    with s1:
        pick_asset = _slicer("Asset Class", config.ASSET_CLASSES, "slc_asset")
    with s2:
        pick_hz = _slicer("Horizon", config.HORIZONS, "slc_hz")
    with s3:
        pick_q = _slicer("Quality", list(QUALITY_SHORT.values()), "slc_q")

    fdf = df[
        df["Asset class"].isin(pick_asset)
        & df["Horizon"].isin(pick_hz)
        & df["Quality (short)"].isin(pick_q)
    ]
    if fdf.empty:
        st.info("No models match the current slicer selection.")
        return

    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown(_model_power_chart(fdf), unsafe_allow_html=True)
    with top_right:
        st.markdown(_reliability_pie_chart(fdf), unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.markdown(_heatmap_chart(fdf), unsafe_allow_html=True)
    with bottom_right:
        st.markdown(_line_chart(fdf), unsafe_allow_html=True)

    with st.expander("Show the model table"):
        model_query = st.text_input(
            "Search the model table", key="model_table_filter",
            placeholder="e.g. Crypto, Long, Weak, Forex_Day  (press Enter to apply)",
        ).strip()
        model_tbl = (
            fdf.assign(**{"ROC-AUC": fdf["ROC-AUC"].map("{:.3f}".format),
                          "Break-even": fdf["Break-even"].map("{:.1%}".format)})
               .drop(columns=["Name", "Quality (short)"])
        )
        model_tbl = filter_table(model_tbl, model_query)
        if model_tbl.empty:
            st.caption("No models match the filter.")
        else:
            table_height = 36 * (len(model_tbl) + 1) + 6
            st.dataframe(
                model_tbl, hide_index=True, width="stretch", height=table_height
            )


EXPERIENCE_TIP: dict[str, str] = {
    "experienced": "You have traded before - still, size each position to a risk "
                   "you can absorb if the stop is hit.",
    "beginner": "As a beginner, consider paper-trading this signal first and never "
                "risk money you cannot afford to lose.",
    "long_term_single": "You favour long-term, single-asset positions - short and "
                        "swing horizons here move faster than you may be used to.",
    "periodic": "You trade in bursts - remember each signal has a limited validity "
                "window shown on the card.",
}


# =============================================================================
# 1. CACHED DATA / SCAN HELPERS
# =============================================================================
# Deduplicate yfinance downloads across every scan and horizon. Streamlit reruns
# the script top-to-bottom on each interaction but does NOT re-import modules, so
# this wraps live_inference.download_ohlcv exactly once per process.
if not getattr(li.download_ohlcv, "_aegis_cached", False):
    _raw_download_ohlcv = li.download_ohlcv

    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def _cached_download_ohlcv(ticker: str):
        return _raw_download_ohlcv(ticker)

    _cached_download_ohlcv._aegis_cached = True  # type: ignore[attr-defined]
    li.download_ohlcv = _cached_download_ohlcv  # type: ignore[assignment]


# Deduplicate feature rebuilding across Day/Swing/Long models. The OHLCV
# download is already cached above; this avoids recomputing the same stationary
# feature frame for the same ticker during market scans.
if not getattr(li.build_features, "_aegis_cached", False):
    _raw_build_features = li.build_features

    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def _cached_build_features(
        ohlcv: "pd.DataFrame",
        cache_version: int = FEATURE_PIPELINE_CACHE_VERSION,
    ):
        _ = cache_version
        return _raw_build_features(ohlcv)

    _cached_build_features._aegis_cached = True  # type: ignore[attr-defined]
    li.build_features = _cached_build_features  # type: ignore[assignment]


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_asset_mapping() -> pd.DataFrame:
    """Human-readable name <-> yfinance ticker table for Question 3."""
    df = pd.read_excel(MAPPING_PATH, sheet_name="Asset_Mapping")
    return df[["Display Name", "Ticker (yfinance)", "Asset Class", "Category / Note"]]


# CoinGecko lists a few of our tickers under their post-rebrand symbols.
_CRYPTO_SYMBOL_ALIASES: dict[str, str] = {"MATIC": "POL", "KLAY": "KAIA", "FTM": "S"}


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_crypto_ranks() -> dict[str, int]:
    """{symbol (upper, e.g. 'BTC') -> market-cap rank} from CoinGecko's keyless
    public API. One cached batch call per day (top 250 by market cap). Returns an
    empty dict on any failure so the asset-codes dialog degrades gracefully and
    never crashes on a network problem."""
    import json
    import urllib.request

    url = ("https://api.coingecko.com/api/v3/coins/markets"
           "?vs_currency=usd&order=market_cap_desc&per_page=250&page=1")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AegisTrader-thesis/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
    except Exception:  # noqa: BLE001 - network/rate-limit/parse: fall back to no ranks
        return {}

    ranks: dict[str, int] = {}
    for coin in data:
        symbol = str(coin.get("symbol", "")).upper()
        rank = coin.get("market_cap_rank")
        # Keep the highest-cap coin for each symbol (first in market_cap_desc order).
        if symbol and rank and symbol not in ranks:
            ranks[symbol] = int(rank)
    return ranks


def enriched_asset_mapping() -> pd.DataFrame:
    """Asset mapping with the crypto rows' 'Category / Note' filled by live
    market-cap rank (CoinGecko). Non-crypto rows and the fallback path are left
    untouched."""
    mapping = load_asset_mapping().copy()
    ranks = fetch_crypto_ranks()
    if not ranks:
        return mapping

    def _note(ticker: str) -> str:
        symbol = str(ticker).upper().replace("-USD", "")
        symbol = _CRYPTO_SYMBOL_ALIASES.get(symbol, symbol)
        rank = ranks.get(symbol)
        return f"Market cap rank #{rank}" if rank else "Market cap rank unavailable"

    is_crypto = mapping["Asset Class"] == "Crypto"
    mapping.loc[is_crypto, "Category / Note"] = (
        mapping.loc[is_crypto, "Ticker (yfinance)"].map(_note)
    )
    return mapping


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def scan_model(model_id: str, asset_class: str, horizon: str,
               specific_ticker: str | None,
               display_version: int = SIGNAL_DISPLAY_VERSION) -> li.TradeTicket:
    """One routed result: a specific ticker, or the strongest over the universe."""
    _ = display_version
    if specific_ticker:
        return li.generate_signal(model_id, asset_class, horizon,
                                  specific_ticker=specific_ticker)
    return li.generate_signal(model_id, asset_class, horizon,
                              candidate_tickers=config.TICKERS[asset_class])


def scan_all_markets(asset_scope: str | None = None) -> list[li.TradeTicket]:
    """Best ticket per model, with a progress bar.

    Per-model results are cached (scan_model), so only the first run in a 6-hour
    window is slow; the download cache also removes duplicate fetches across the
    three horizons of each asset class, and the feature cache avoids rebuilding
    the same ticker features for Day/Swing/Long.
    """
    model_ids = [
        model_id for model_id in config.MODEL_IDS
        if asset_scope is None or model_id.split("_")[1] == asset_scope
    ]
    tickets: list[li.TradeTicket] = []
    label = "all markets" if asset_scope is None else asset_scope
    progress = st.progress(0.0, text=f"Scanning {label}...")
    for i, model_id in enumerate(model_ids):
        _, asset_class, horizon = model_id.split("_")
        tickets.append(scan_model(model_id, asset_class, horizon, None))
        progress.progress((i + 1) / len(model_ids),
                          text=f"Scanned {model_id}")
    progress.empty()
    return tickets


# =============================================================================
# 2. SMALL FORMAT HELPERS
# =============================================================================
def fmt_price(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}" if abs(value) >= 100 else f"{value:.4f}"


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rem = divmod(int(round(seconds)), 60)
    return f"{minutes}m {rem:02d}s"


# =============================================================================
# 3. ASSET-CODE DIALOG (Question 3 helper)
# =============================================================================
@st.dialog("Asset codes", width="large")
def asset_codes_dialog() -> None:
    st.write(
        "Type in the search box to filter across all columns, then **click a row** "
        "to fill Question 3 automatically and close this window."
    )
    try:
        mapping = enriched_asset_mapping()
    except FileNotFoundError:
        st.error(
            "AegisTrader_Asset_Mapping.xlsx was not found next to app.py. "
            "Place a copy in the deployment folder."
        )
        return

    # Visible, always-on search box. Streamlit's text_input applies on Enter or
    # when focus leaves the box (true per-keystroke input is not natively
    # supported). It matches the typed text against every column.
    query = st.text_input(
        "Search assets", key="asset_code_filter",
        placeholder="e.g. Apple, ETH, Crypto, rank 1  (press Enter to apply)",
    ).strip()
    if query:
        q = query.lower()
        mask = mapping.apply(
            lambda row: q in " ".join(str(v).lower() for v in row.to_numpy()), axis=1
        )
        mapping = mapping[mask]

    view = mapping.reset_index(drop=True)  # positions align with the displayed rows
    if view.empty:
        st.info("No assets match your search.")
        st.caption("Tickers look like ETH-USD, EURUSD=X, ^GSPC, AAPL.")
        return

    event = st.dataframe(
        view, key="asset_codes_df", hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-row",
    )
    rows = event.selection.rows
    if rows:
        # Positions index the (filtered) dataframe we passed -> deterministic.
        # Store the pick in a NON-widget key; it is written into the Q3 text_input's
        # state before that widget is created next run, then st.rerun() closes this.
        ticker = str(view.iloc[rows[0]]["Ticker (yfinance)"])
        st.session_state["q3_pick"] = ticker
        st.rerun()

    st.caption("Tickers look like ETH-USD, EURUSD=X, ^GSPC, AAPL.")


# =============================================================================
# 4. SIGNAL CARD (shared by both tabs)
# =============================================================================
def render_signal_card(ticket: li.TradeTicket, experience_key: str | None = None) -> None:
    if not ticket.ok or ticket.status == "UNAVAILABLE":
        asset_label = li.format_asset_label(ticket.ticker, getattr(ticket, "display_name", None))
        generated_at = getattr(ticket, "generated_at", None)
        if ticket.ticker:
            unavailable = (
                f"Live data was unavailable or insufficient for **{asset_label}**. "
                "Try another ticker, leave the asset code blank to scan the asset class, "
                "or retry later if Yahoo Finance is slow."
            )
        else:
            unavailable = (
                "No usable live signal could be produced for this scan scope. "
                "Try a narrower scope, a specific ticker, or retry later if Yahoo Finance is slow."
            )
        if generated_at:
            unavailable += f" Generated at {generated_at}."
        st.warning(unavailable)
        for note in ticket.notes:
            st.caption(note)
        return

    emoji, label = TIER_BADGE[ticket.status]
    st.subheader(f"{emoji}  {label}")
    asset_label = li.format_asset_label(ticket.ticker, getattr(ticket, "display_name", None))
    st.markdown(
        f"**{asset_label}** &nbsp; | &nbsp; {ticket.asset_class} / {ticket.horizon} "
        f"&nbsp; | &nbsp; `{ticket.model_id}`"
    )

    if ticket.low_confidence:
        st.warning("Low-confidence model (test ROC-AUC below 0.50). "
                   "Treat this signal with extra caution.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Entry", fmt_price(ticket.entry))
    col2.metric("Stop-Loss", fmt_price(ticket.stop_loss))
    col3.metric("Take-Profit", fmt_price(ticket.take_profit))
    col4.metric("Probability", fmt_pct(ticket.probability))

    generated_at = getattr(ticket, "generated_at", None)
    generated_part = f"  |  Generated at {generated_at}" if generated_at else ""
    st.caption(
        f"Reward/Risk {ticket.reward_to_risk:.1f} : 1  |  "
        f"Thresholds: top-k {fmt_pct(ticket.top_k_cutoff)} , "
        f"break-even {fmt_pct(ticket.threshold)}  |  "
        f"As of {ticket.as_of_date}  |  Valid until {ticket.valid_until} "
        f"(~{ticket.valid_days} days){generated_part}"
    )

    # Copy blocks: each key field is its own st.code (line-by-line copy button),
    # plus the full broker-ready ticket in an expander.
    with st.expander("Copy values"):
        st.caption("Entry")
        st.code(fmt_price(ticket.entry), language=None)
        st.caption("Stop-Loss")
        st.code(fmt_price(ticket.stop_loss), language=None)
        st.caption("Take-Profit")
        st.code(fmt_price(ticket.take_profit), language=None)
        st.caption("Probability")
        st.code(fmt_pct(ticket.probability), language=None)
        st.caption("Full signal")
        st.code(ticket.text, language=None)

    # Honest tier/quality caveat only -- the numbers are already on the card above.
    caveat = explainer.build_signal_presentation(
        model_name=ticket.model_id,
        probability=ticket.probability,
        break_even=ticket.threshold,
        top_k_cutoff=ticket.top_k_cutoff,
        tier=ticket.status,
    )
    if caveat.strip():
        st.caption(caveat)

    if experience_key and experience_key in EXPERIENCE_TIP:
        st.info(EXPERIENCE_TIP[experience_key])


# =============================================================================
# 5. PAGE SHELL
# =============================================================================
_HAS_LOGO = LOGO_PATH.exists()
_HAS_ML_IMAGE = ML_IMAGE_PATH.exists()
st.set_page_config(
    page_title="AegisTrader",
    page_icon=(str(LOGO_PATH) if _HAS_LOGO else "\U0001F4C8"),
    layout="wide",
)
inject_theme_css()

render_hero(_HAS_LOGO, _HAS_ML_IMAGE)
render_wave()

tab_signal, tab_models, tab_dashboard, tab_market = st.tabs(
    ["Get a signal", "Get to know our models", "Models Dashboard",
     "Strongest Across All Markets"]
)


# =============================================================================
# 6. TAB 1 -- GET A SIGNAL (routed flow)
# =============================================================================
with tab_signal:
    st.markdown("### Answer a few questions")

    q1_key = st.radio(
        "**Question 1 \u2014 Time horizon.** When do you seek to get a return on "
        "your investment?",
        options=list(router.HORIZON_LABELS.keys()),
        format_func=lambda k: router.HORIZON_LABELS[k],
        key="q1",
    )

    q2_key = st.radio(
        "**Question 2 \u2014 Asset category.** Do you have a preference for a "
        "specific investment class?",
        options=list(router.ASSET_LABELS.keys()),
        format_func=lambda k: router.ASSET_LABELS[k],
        key="q2",
    )

    st.markdown(
        "**Question 3 \u2014 Specific asset (optional).** If you have a specific "
        "asset in mind, write its code. Leave blank to let the system pick the "
        "strongest available signal."
    )
    # If the user picked a row in the asset-codes dialog, write it into the Q3
    # widget's state BEFORE the widget is created (Streamlit forbids setting a
    # widget's state after it is instantiated).
    if "q3_pick" in st.session_state:
        st.session_state["q3"] = st.session_state.pop("q3_pick")
    q3_col, btn_col = st.columns([3, 1])
    q3_raw = q3_col.text_input(
        "Asset code", key="q3", label_visibility="collapsed",
        placeholder="e.g. BTC-USD, EURUSD=X, ^GSPC, AAPL (or leave blank)",
    )
    if btn_col.button("\U0001F4CB Browse asset codes", width="stretch"):
        asset_codes_dialog()

    risk = st.slider(
        "**Question 4 \u2014 Risk tolerance.** 1 = protect capital at all costs, "
        "10 = willing to take high risk for higher return. "
        "(Used only when Question 2 is 'no preference'.)",
        min_value=1, max_value=10, value=5, key="q4",
    )

    q5_key = st.radio(
        "**Question 5 \u2014 Investment experience.** (Does not change the signal; "
        "used only for a short tailored tip.)",
        options=list(router.EXPERIENCE_CHOICES.keys()),
        format_func=lambda k: router.EXPERIENCE_CHOICES[k],
        key="q5",
    )

    st.divider()

    if st.button("Get my signal", type="primary"):
        validation = router.validate_ticker(q3_raw)
        if (not validation.is_blank) and (not validation.is_valid):
            st.session_state["invalid_ticker_msg"] = validation.message
            st.session_state.pop("route_request", None)
        else:
            st.session_state["route_request"] = {
                "q1": q1_key, "q2": q2_key, "risk": risk, "q5": q5_key,
                "ticker": validation.canonical if validation.is_valid else None,
            }
            st.session_state.pop("invalid_ticker_msg", None)

    # Invalid ticker -> retry or skip.
    if "invalid_ticker_msg" in st.session_state:
        st.error(st.session_state["invalid_ticker_msg"])
        retry_col, skip_col = st.columns(2)
        if retry_col.button("Try a different code"):
            st.session_state.pop("invalid_ticker_msg", None)
            st.rerun()
        if skip_col.button("Skip and use the strongest available signal"):
            st.session_state["route_request"] = {
                "q1": q1_key, "q2": q2_key, "risk": risk, "q5": q5_key, "ticker": None,
            }
            st.session_state.pop("invalid_ticker_msg", None)
            st.rerun()

    # Render the routed signal.
    if "route_request" in st.session_state:
        req = st.session_state["route_request"]
        result = router.route(req["q1"], req["q2"], req["risk"], ticker=req["ticker"])

        source_text = {
            "ticker": "your specific asset code (Question 3)",
            "q2_preference": "your asset preference (Question 2)",
            "q4_risk_fallback": "your risk tolerance (Question 4)",
        }.get(result.asset_source, result.asset_source)
        st.markdown(
            f"**Selected model:** `{result.model_id}` &nbsp; "
            f"(chosen from {source_text})"
        )

        with st.spinner("Scoring live market data..."):
            ticket = scan_model(
                result.model_id, result.asset_class, result.horizon,
                result.specific_ticker,
            )
        render_signal_card(ticket, experience_key=req["q5"])


# =============================================================================
# 7. TAB 2 -- STRONGEST ACROSS ALL MARKETS (quality-aware, lazy)
# =============================================================================
with tab_market:
    st.markdown("### Strongest signal across all markets")
    st.caption(
        "Scans all 12 models across their full universes. The headline is drawn "
        "only from models with genuine predictive power (Strong / Moderate quality); "
        "weaker models are listed separately for transparency, so a degenerate "
        "high-probability reading never masquerades as the best signal. This is the "
        "heavy scan -- it runs only when you ask and is cached for 6 hours."
    )
    st.warning(
        "This is the heaviest live-data workflow in the app. The first uncached run "
        "downloads Yahoo Finance data and rebuilds features for many tickers; it can "
        "take a while on Streamlit Cloud. Choose one asset class for a faster scan, "
        "or use All markets when you really want the full sweep."
    )

    market_scope = st.selectbox(
        "Scan scope",
        options=["All markets", *config.ASSET_CLASSES],
        key="market_scan_scope_select",
        help=(
            "All markets scans every asset class. A single asset class is much "
            "lighter and still uses live Yahoo Finance data."
        ),
    )
    scope_asset = None if market_scope == "All markets" else market_scope
    scope_tickers = sum(len(v) for v in config.TICKERS.values()) if scope_asset is None else len(config.TICKERS[scope_asset])
    scope_models = len(config.MODEL_IDS) if scope_asset is None else len(config.HORIZONS)
    st.caption(
        f"Selected scope: {scope_models} model(s), up to {scope_tickers} ticker(s). "
        "Downloads, feature frames, and model results are cached for 6 hours."
    )

    if st.button("\U0001F50D Scan selected scope", type="primary"):
        st.session_state["run_market_scan"] = True
        st.session_state["market_scan_scope"] = market_scope

    if st.session_state.get("run_market_scan"):
        active_scope = st.session_state.get("market_scan_scope", "All markets")
        active_asset = None if active_scope == "All markets" else active_scope
        scan_started = time.perf_counter()
        all_tickets = scan_all_markets(active_asset)
        scan_seconds = time.perf_counter() - scan_started
        ok_tickets = [t for t in all_tickets if t.ok]
        total_scored = sum(t.scored for t in all_tickets)
        total_skipped = sum(t.skipped for t in all_tickets)
        st.success(
            f"Scan completed in {fmt_duration(scan_seconds)}. "
            f"{len(ok_tickets)}/{len(all_tickets)} model(s) returned usable tickets; "
            f"{total_scored} ticker attempt(s) scored, {total_skipped} skipped. "
            "Cached runs may complete much faster."
        )

        def _is_reliable(t: li.TradeTicket) -> bool:
            return explainer.SUB_MODEL_PROFILES[t.model_id].quality_label in RELIABLE_QUALITIES

        reliable = sorted([t for t in ok_tickets if _is_reliable(t)],
                          key=signal_strength_key, reverse=True)
        low_rel = sorted([t for t in ok_tickets if not _is_reliable(t)],
                         key=signal_strength_key, reverse=True)

        if reliable:
            scope_title = "all markets" if active_asset is None else active_asset
            st.markdown(f"#### \U0001F3C6 Today's strongest signal ({scope_title}, reliable models only)")
            render_signal_card(reliable[0])
        else:
            st.warning("No reliable model could be scored right now. Try again later.")

        # Shared search box for both ranking tables (applies on Enter / focus-out).
        table_query = st.text_input(
            "Filter the ranking tables",
            key="market_table_filter",
            placeholder="e.g. Crypto, BUY, ETH, Forex_Day  (press Enter to apply)",
        ).strip()

        if reliable:
            st.markdown("#### Reliable models")
            st.caption("Strong / Moderate quality. Ranked by tier, then live probability.")
            show_ranking_table(reliable, table_query)

        if low_rel:
            st.markdown("#### Lower-reliability models")
            st.caption(
                "Weak / Unreliable quality (test ROC-AUC near 0.50). Shown for "
                "transparency only -- **not recommended for action**. A high "
                "probability here reflects a degenerate model, not a strong opportunity."
            )
            show_ranking_table(low_rel, table_query)

        skipped_models = [t.model_id for t in all_tickets if not t.ok]
        if skipped_models:
            st.caption("No live data right now for: " + ", ".join(skipped_models))


# =============================================================================
# 8. TAB 2 -- GET TO KNOW OUR MODELS (static reference cards)
# =============================================================================
with tab_models:
    st.markdown("### The 12 AegisTrader sub-models")
    st.caption("Static reference cards from the frozen thesis evaluation.")
    for model_id in config.MODEL_IDS:
        st.markdown(explainer.build_model_card(model_id))
        st.divider()


# =============================================================================
# 9. TAB 3 -- MODELS DASHBOARD (KPIs, slicers, charts)
# =============================================================================
with tab_dashboard:
    render_models_dashboard()
