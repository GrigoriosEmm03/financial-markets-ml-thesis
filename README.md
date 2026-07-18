# Financial Markets ML Thesis — AegisTrader

Codebase developed for my MSc thesis in **Business Analytics & Data Science**.
The thesis builds an end-to-end machine-learning workflow that turns raw
financial time-series into supervised classification datasets, engineers
technical and regime-aware features, trains and calibrates a family of binary
classifiers, and serves the resulting signals through an interactive web
application called **AegisTrader**.

The repository is intentionally public in the spirit of academic transparency.
It is organised as a **code archive**: every script that produced the thesis
results is included, grouped into the four stages of the pipeline. The raw
datasets and the trained model artifacts are **not** committed (see
[What is *not* in this repository](#what-is-not-in-this-repository)).

> **Naming.** The repository is named `financial-markets-ml-thesis`. The
> system built inside it — the 12 sub-models plus the serving layer and web
> app — is referred to throughout the code as **AegisTrader** (internally
> "Project Aegis").

---

## Table of contents

- [Overview](#overview)
- [System at a glance](#system-at-a-glance)
- [Repository structure](#repository-structure)
- [The pipeline, stage by stage](#the-pipeline-stage-by-stage)
- [Modelling methodology](#modelling-methodology)
- [The AegisTrader web application](#the-aegistrader-web-application)
- [Results](#results)
- [What is *not* in this repository](#what-is-not-in-this-repository)
- [Technologies](#technologies)
- [Running the code locally](#running-the-code-locally)
- [Known limitations](#known-limitations)
- [Author and academic context](#author-and-academic-context)
- [License](#license)

---

## Overview

The system is organised around a simple taxonomy: **four asset classes**
(Crypto, Forex, Indices, Stocks) crossed with **three trading horizons**
(Day, Swing, Long), giving **12 independent binary classifiers**. Each
classifier answers a single, well-defined question for its market and horizon:

> *Given today's market state, would a long-only bracket trade opened at the
> next bar reach its take-profit before its stop-loss, within the horizon's
> look-forward window?*

The target is therefore a triple-barrier-style, outcome-based label rather than
a raw directional return, and the bracket geometry (stop distance in ATR
multiples, reward-to-risk ratio, look-forward length) is defined per asset
class and horizon.

The four stages of the pipeline correspond one-to-one with the four numbered
folders in the repository:

1. **Data acquisition** — download daily OHLCV data for each asset universe.
2. **Feature and target engineering** — technical indicators, stationary
   transformations, regime-aware long-horizon features, and the barrier-based
   targets.
3. **Model training and evaluation** — a shared preprocessing pipeline, one
   training orchestrator per asset class, and a global evaluation report.
4. **Web-app deployment** — the deterministic router, the live-inference
   engine, the rule-based explainer, and the Streamlit application.

---

## System at a glance

| Property | Value |
|---|---|
| Asset classes | Crypto, Forex, Indices, Stocks |
| Trading horizons | Day, Swing, Long |
| Sub-models | 12 (4 asset classes × 3 horizons) |
| Asset universe | 164 symbols (50 Crypto, 43 Forex, 21 Indices, 50 Stocks) |
| Data frequency | Daily bars (`1d`) from Yahoo Finance |
| Task | Binary classification (bracket-trade win / not-win) |
| Learners considered | HistGradientBoosting, Random Forest, Logistic Regression |
| Probability calibration | Isotonic, on a dedicated held-out fold |
| Serving | Streamlit web app with a deterministic model router |

---

## Repository structure

```text
financial-markets-ml-thesis/
│
├── README.md
└── config/
    ├── 1. Code to get the data/
    │   ├── Code for CRYPTO Dataset.py
    │   ├── Code for FOREX Dataset.py
    │   ├── Code for INDICES Dataset.py
    │   └── Code for STOCKS Dataset.py
    │
    ├── 2. Data Cleaning (Before the Model Training)/
    │   ├── 1. add_indicators.py
    │   ├── 2. target_variable_engineering.py
    │   ├── 3. feature_transformation.py
    │   └── 4. long_horizon_and_feature_engineering.py
    │
    ├── 3. Model Training/
    │   ├── ml_preprocessing_pipeline_v4.py
    │   ├── model_training_crypto_v4.py
    │   ├── model_training_forex_v4.py
    │   ├── model_training_Indices_v4.py
    │   ├── model_training_stocks_v4.py
    │   └── evaluation_of_the_models.py
    │
    └── 4. Web App Deployment/
        ├── config.py
        ├── router.py
        ├── live_inference.py
        ├── explainer.py
        ├── app.py
        ├── diagnostic_signal_frequency.py
        └── validate_deployment.py
```

The numbered folders and file prefixes encode the intended execution order.
The `config/` folder is a container for all source code; it is not a folder of
configuration files.

---

## The pipeline, stage by stage

### Stage 1 — Data acquisition (`1. Code to get the data/`)

One script per asset class downloads daily OHLCV history from Yahoo Finance
(`yfinance`) for its ticker universe and writes a per-class CSV
(`DATASET_CRYPTO.csv`, `DATASET_FOREX.csv`, `DATASET_INDICES.csv`,
`DATASET_STOCKS.csv`). The crypto script, for example, pulls history from
1 January 2019 to the run date across 50 coins. Symbols that fail to download
are skipped so that a single delisted or renamed ticker never aborts the run.

### Stage 2 — Feature and target engineering (`2. Data Cleaning …/`)

Four scripts run in sequence:

1. **`add_indicators.py`** — computes technical indicators per ticker with
   `pandas_ta`: RSI(14), MACD(12, 26, 9), SMA(50), SMA(200), Bollinger
   Bands(20, 2σ) and ATR(14). Indicators are calculated strictly per ticker to
   avoid cross-asset contamination.

2. **`target_variable_engineering.py`** — builds the supervised targets. For
   each horizon it simulates a long-only bracket from the next bar
   (stop-loss = entry − `atr_mult` × ATR, take-profit = entry + `atr_mult` ×
   ATR × `rr`) over a `look_forward` window and labels the outcome as a win or
   not. Geometry is defined per asset class and horizon in `ASSET_CONFIGS`,
   producing `Target_Day_Win`, `Target_Swing_Win` and `Target_Long_Win`.

3. **`feature_transformation.py`** — enforces stationarity and scale
   invariance: percentage returns, lagged features, calendar features, and
   division of absolute indicators (e.g. MACD, ATR) by Close, followed by
   dropping the remaining absolute, non-stationary columns. All time-dependent
   operations are applied strictly per ticker after sorting by (Ticker, Date).

4. **`long_horizon_and_feature_engineering.py`** — adds long-horizon rolling
   and lag features, replaces integer calendar fields with cyclic (sin/cos)
   encodings, and adds three scale-invariant, regime-aware features: drawdown
   from the trailing 252-day high, the 60-day realised-volatility percentile
   rank, and the 30-day cumulative-return percentile rank. Every feature is
   stationary, computed per ticker, and uses past information only.

### Stage 3 — Model training and evaluation (`3. Model Training/`)

- **`ml_preprocessing_pipeline_v4.py`** — the shared preprocessing layer used
  by all four training scripts. It applies a per-ticker minimum-history filter,
  train-only winsorization at the 1% / 99% percentiles (bounds are stored so
  the identical clip can be re-applied at inference), and a **four-way
  chronological split with purge gaps**: `train → val_select → val_calib →
  test`. `val_select` drives model selection; `val_calib` is reserved solely
  for probability calibration, which avoids reusing one validation fold for
  both purposes.

- **`model_training_{crypto,forex,indices,stocks}_v4.py`** — one orchestrator
  per asset class, each training the three horizon models for that class. See
  [Modelling methodology](#modelling-methodology) for what happens inside.
  Each script serialises its trained artifacts to `.joblib` and writes a JSON
  training summary plus a human-readable evaluation report.

- **`evaluation_of_the_models.py`** — reads the four JSON summaries and
  produces a single global leaderboard that ranks all 12 sub-models by their
  held-out test performance.

### Stage 4 — Web-app deployment (`4. Web App Deployment/`)

The serving layer that turns the trained models into a usable application. See
[The AegisTrader web application](#the-aegistrader-web-application).

---

## Modelling methodology

Each of the 12 sub-models is trained as a calibrated binary classifier with the
following design choices:

- **Learners.** For every (asset class, horizon) the training script fits and
  compares several candidates — `HistGradientBoostingClassifier` (multiple
  configurations), `RandomForestClassifier`, and a `LogisticRegression`
  baseline — and selects the best-performing one on the `val_select` fold.

- **Class imbalance.** Wins are the minority outcome, so training uses
  class-derived sample weights (a recommended positive-class weight computed
  from the training distribution) rather than resampling.

- **Probability calibration.** The selected estimator is wrapped with isotonic
  calibration (`CalibratedClassifierCV`, `method="isotonic"`) fitted on the
  held-out `val_calib` fold, so the predicted probabilities can be read as
  meaningful win-likelihoods rather than raw scores.

- **Evaluation.** Models are assessed on the final `test` fold with a broad set
  of metrics, including ROC-AUC, average precision (PR-AUC), the Brier score,
  Matthews correlation coefficient, balanced accuracy and F1.

- **Leakage controls.** Splits are strictly chronological with purge gaps
  between folds; every feature is computed per ticker using past information
  only; winsorization bounds and any scaler are fitted on training data alone
  and stored in the artifact for reuse at inference.

---

## The AegisTrader web application

The web app is deliberately **deterministic and transparent** end to end: there
is no LLM and no hidden state in the serving path, so every outcome can be
traced and explained during the thesis defence.

- **`config.py`** — a constants-only single source of truth (ticker universes,
  bracket geometry, decision thresholds, per-model test ROC-AUC, and
  user-facing descriptions). Importing it has no side effects.

- **`router.py`** — maps the answers of a short profiling questionnaire to
  exactly one of the 12 sub-models through a deterministic rule cascade. The
  horizon comes from one question; the asset class is resolved by a priority
  cascade (an explicit ticker overrides an explicit asset-class choice, which
  overrides a risk-based fallback). The router never reroutes to a different
  model.

- **`live_inference.py`** — turns a selected sub-model into a concrete trade
  ticket. It downloads recent daily data from `yfinance`, rebuilds the training
  features by **reusing the frozen Stage-2 feature scripts verbatim**, aligns to
  the artifact's stored feature columns, applies the stored winsorization bounds
  (and scaler, if present), and predicts with the calibrated model. The
  target-engineering and preprocessing scripts are never executed at inference,
  which keeps training and serving consistent.

- **`explainer.py`** — produces static, template-based explanations of each
  model and of its BUY / WATCH / NO-SIGNAL outcome, with wording driven by
  `config.py` and by the frozen quality labels from the final evaluation.

- **`app.py`** — the Streamlit entry point that ties the layers together. Heavy
  work (data download and scoring) runs only behind cached buttons, so a plain
  page load never touches the network or the model artifacts.

- **`diagnostic_signal_frequency.py`** — an offline analysis tool (not part of
  the app) that scores recent bars per model to report how often each model
  would actually fire a signal.

- **`validate_deployment.py`** — offline pre-deployment checks for packaging,
  artifacts, mappings and configuration, run before deploying to Streamlit
  Community Cloud.

---

## Results

Held-out **test-set ROC-AUC** for the 12 sub-models, taken from the v4 training
summaries (25–26 April 2026):

| Asset class | Day | Swing | Long |
|---|---|---|---|
| Crypto | 0.686 | 0.638 | 0.465 |
| Forex | 0.613 | 0.570 | 0.611 |
| Indices | 0.555 | 0.563 | 0.514 |
| Stocks | 0.573 | 0.581 | 0.535 |

Predicting daily bracket-trade outcomes on liquid markets is a genuinely hard
problem, and the numbers reflect that: most models sit modestly above the 0.50
baseline, and one (Crypto Long, 0.465) falls below it. The application treats
this honestly rather than hiding it — any model whose test ROC-AUC is under
`0.50` is served with a visible **low-confidence** flag in the UI, and the
model descriptions make no quality claims of their own.

---

## What is *not* in this repository

By design, the repository contains **source code only**. The following are
excluded and must be regenerated or supplied locally:

- **Raw and processed datasets** — the per-class CSVs and all intermediate
  files. Run Stage 1, then Stage 2, to recreate them.
- **Trained model artifacts** — the 12 `.joblib` files produced by Stage 3.
  They are required by the web app but are not committed.
- **Training outputs** — the JSON summaries and evaluation reports.

Reproducibility caveat: several Stage-2 and Stage-3 scripts contain hard-coded
absolute Windows paths from the development machine. Anyone re-running the
pipeline elsewhere must adjust those paths to their own environment.

---

## Technologies

Verified from the imports actually used in the code:

- **Language:** Python 3
- **Data acquisition:** `yfinance`; `urllib` for a keyless CoinGecko REST call
  (market-cap context in the app)
- **Feature engineering:** `pandas`, `numpy`, `pandas_ta`
- **Machine learning:** `scikit-learn` (`HistGradientBoostingClassifier`,
  `RandomForestClassifier`, `LogisticRegression`, `CalibratedClassifierCV`,
  `MinMaxScaler`, model-selection and evaluation metrics), `joblib` for
  artifact serialisation
- **Visualisation (training):** `matplotlib`
- **Web application:** `streamlit`
- **Tooling:** Anaconda (`thesis_env`), VS Code

---

## Running the code locally

The repository does not ship a `requirements.txt`; install the libraries listed
above into a clean environment (Python 3, Anaconda recommended). A typical
end-to-end run is:

```bash
# Stage 1 — download the data (one script per asset class)
python "config/1. Code to get the data/Code for CRYPTO Dataset.py"

# Stage 2 — indicators, targets, transformations, long-horizon features
python "config/2. Data Cleaning (Before the Model Training)/1. add_indicators.py"
# ... then scripts 2, 3, 4 in order

# Stage 3 — preprocessing + per-class training + global evaluation
python "config/3. Model Training/model_training_crypto_v4.py"
# ... repeat for forex, indices, stocks, then evaluation_of_the_models.py

# Stage 4 — run the web app (requires the trained .joblib artifacts locally)
python -m streamlit run "config/4. Web App Deployment/app.py"
```

Before running Stage 2 and Stage 3, update the hard-coded input/output paths in
those scripts to match your machine. The web app expects the 12 trained
artifacts to be available locally in the location resolved by
`config.py`.

---

## Known limitations

- **Modest predictive power.** Test ROC-AUC values are close to the 0.50
  baseline for most models, which is expected for daily directional/outcome
  prediction on efficient, liquid markets. Signals should be read as
  probabilistic and low-confidence, not as reliable forecasts.
- **Train/serve geometry mismatch.** Each training script hard-coded a single,
  uniform (crypto) bracket geometry into the saved artifacts, whereas the
  targets were built with per-asset geometry. The serving layer corrects for
  this by taking the true per-asset geometry and BUY threshold from `config.py`
  rather than from the artifact. This is documented explicitly in the code.
- **Environment coupling.** Absolute local paths in some scripts reduce
  out-of-the-box portability.
- **Not investment advice.** This is an academic research project. Nothing here
  is a recommendation to trade.

---

## Author and academic context

**Gregory Emmanouilidis**
MSc in Business Analytics & Data Science — master's thesis project.

---

## License

<!-- No license file is currently included in the repository. Until one is added,
     the default is "all rights reserved". Consider adding a LICENSE file
     (e.g. MIT for permissive reuse, or CC BY-NC for non-commercial academic use)
     and updating this section accordingly. -->

This code is shared for academic and demonstration purposes as part of an MSc
thesis. No formal open-source license is currently attached; see the note above.
