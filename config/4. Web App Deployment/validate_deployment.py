"""
validate_deployment.py -- AegisTrader pre-deploy checks
=======================================================

Run this before deploying the Streamlit app:

    python validate_deployment.py

The checks are intentionally offline: no Yahoo/yfinance calls and no live
scoring. The goal is to catch packaging, artifact, mapping, and configuration
problems before the app reaches Streamlit Community Cloud.
"""

from __future__ import annotations

import argparse
import ast
import gc
import importlib.metadata as metadata
import math
import py_compile
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REQUIREMENTS_PATH = BASE_DIR / "requirements.txt"
ASSET_MAPPING_PATH = BASE_DIR / "AegisTrader_Asset_Mapping.xlsx"
TARGET_ENGINEERING_PATH = (
    BASE_DIR.parent
    / "2. Data Cleaning BEFORE TRAINING OF THE MODELS"
    / "2. target_variable_engineering.py"
)

_TARGET_FILE_TO_ASSET_CLASS = {
    "DATASET_CRYPTO_READY.csv": "Crypto",
    "DATASET_FOREX_READY.csv": "Forex",
    "DATASET_INDICES_READY.csv": "Indices",
    "DATASET_STOCKS_READY.csv": "Stocks",
}


class CheckFailed(RuntimeError):
    """Raised when one validation check fails."""


class Reporter:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"[FAIL] {message}")

    def run(self, name: str, fn) -> None:
        print(f"\n-- {name}")
        try:
            fn()
        except CheckFailed as exc:
            self.fail(str(exc))
        except Exception as exc:  # noqa: BLE001 - report every pre-deploy failure
            self.fail(f"{type(exc).__name__}: {exc}")

    def finish(self) -> int:
        print("\n" + "=" * 72)
        if self.warnings:
            print(f"Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  - {warning}")
        if self.failures:
            print(f"Deployment validation FAILED ({len(self.failures)} issue(s)).")
            for failure in self.failures:
                print(f"  - {failure}")
            return 1
        print("Deployment validation PASSED.")
        return 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def _parse_pinned_requirements() -> dict[str, str]:
    _require(REQUIREMENTS_PATH.exists(), f"Missing {REQUIREMENTS_PATH}")
    pins: dict[str, str] = {}
    for raw_line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise CheckFailed(f"Requirement is not pinned with ==: {raw_line!r}")
        package, version = [part.strip() for part in line.split("==", 1)]
        _require(package and version, f"Invalid requirement line: {raw_line!r}")
        pins[package] = version
    return pins


def _load_target_engineering_geometry() -> dict[str, dict[str, dict[str, float]]]:
    """Read ASSET_CONFIGS from the target-engineering source without importing it."""
    _require(
        TARGET_ENGINEERING_PATH.exists(),
        f"Missing target-engineering source: {TARGET_ENGINEERING_PATH}",
    )
    tree = ast.parse(TARGET_ENGINEERING_PATH.read_text(encoding="utf-8"))
    asset_configs = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "ASSET_CONFIGS" for target in node.targets):
            asset_configs = ast.literal_eval(node.value)
            break

    _require(asset_configs is not None, "ASSET_CONFIGS not found in target_variable_engineering.py.")
    _require(isinstance(asset_configs, dict), "ASSET_CONFIGS is not a dictionary literal.")
    _require(
        set(asset_configs) == set(_TARGET_FILE_TO_ASSET_CLASS),
        "ASSET_CONFIGS dataset keys do not match expected training datasets.",
    )

    geometry: dict[str, dict[str, dict[str, float]]] = {}
    for filename, asset_class in _TARGET_FILE_TO_ASSET_CLASS.items():
        horizons = asset_configs[filename]
        _require(isinstance(horizons, dict), f"ASSET_CONFIGS[{filename!r}] is not a dictionary.")
        geometry[asset_class] = {}
        for horizon, values in horizons.items():
            _require(isinstance(values, dict), f"ASSET_CONFIGS[{filename!r}][{horizon!r}] is not a dictionary.")
            _require(
                set(values) == {"look_forward", "atr_mult", "rr"},
                f"Bad ASSET_CONFIGS keys for {filename}/{horizon}: {sorted(values)}",
            )
            geometry[asset_class][horizon] = {
                "look_forward": float(values["look_forward"]),
                "atr_mult": float(values["atr_mult"]),
                "rr": float(values["rr"]),
            }
    return geometry


def check_requirements(reporter: Reporter) -> None:
    pins = _parse_pinned_requirements()
    required = {
        "streamlit",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "pandas-ta",
        "numba",
        "yfinance",
        "openpyxl",
    }
    missing_pins = sorted(required - set(pins))
    _require(not missing_pins, f"requirements.txt is missing pins: {missing_pins}")

    mismatches: list[str] = []
    for package, expected in sorted(pins.items()):
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            mismatches.append(f"{package}: not installed (expected {expected})")
            continue
        if installed != expected:
            mismatches.append(f"{package}: installed {installed}, expected {expected}")

    _require(not mismatches, "Installed package mismatch: " + "; ".join(mismatches))
    reporter.ok(f"{len(pins)} pinned packages match the active Python environment.")


def check_python_files(reporter: Reporter) -> None:
    py_files = [
        path for path in BASE_DIR.rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    _require(py_files, "No Python files found to compile.")
    for path in py_files:
        py_compile.compile(str(path), doraise=True)
    reporter.ok(f"{len(py_files)} Python files compile.")


def check_core_config(reporter: Reporter) -> None:
    sys.path.insert(0, str(BASE_DIR))
    import config  # noqa: PLC0415
    import explainer  # noqa: PLC0415
    import router  # noqa: PLC0415

    expected_model_ids = {
        f"AegisTrader_{asset}_{horizon}"
        for asset in config.ASSET_CLASSES
        for horizon in config.HORIZONS
    }
    _require(len(config.MODEL_IDS) == 12, f"Expected 12 model IDs, got {len(config.MODEL_IDS)}")
    _require(set(config.MODEL_IDS) == expected_model_ids, "MODEL_IDS do not match asset x horizon grid.")

    keyed_mappings = {
        "ARTIFACT_PATHS": config.ARTIFACT_PATHS,
        "MODEL_TEST_ROC_AUC": config.MODEL_TEST_ROC_AUC,
        "MODEL_DESCRIPTIONS": config.MODEL_DESCRIPTIONS,
        "BUY_THRESHOLD": config.BUY_THRESHOLD,
        "SUB_MODEL_PROFILES": explainer.SUB_MODEL_PROFILES,
    }
    for name, mapping in keyed_mappings.items():
        _require(set(mapping) == set(config.MODEL_IDS), f"{name} keys do not match MODEL_IDS.")

    _require(set(config.TICKERS) == set(config.ASSET_CLASSES), "TICKERS keys do not match ASSET_CLASSES.")
    _require(config.YF_TIMEOUT_SECONDS > 0, "YF_TIMEOUT_SECONDS must be positive.")
    _require(config.LOOKBACK_CALENDAR_DAYS > 0, "LOOKBACK_CALENDAR_DAYS must be positive.")
    all_tickers = [ticker for tickers in config.TICKERS.values() for ticker in tickers]
    _require(len(all_tickers) == len(set(all_tickers)), "Duplicate ticker found across universes.")
    _require(
        set(config.TICKER_TO_ASSET_CLASS) == set(all_tickers),
        "TICKER_TO_ASSET_CLASS does not cover exactly the configured tickers.",
    )

    for asset in config.ASSET_CLASSES:
        _require(asset in config.BARRIER_GEOMETRY, f"Missing geometry for {asset}.")
        _require(asset in config.SIGNAL_VALIDITY_DAYS, f"Missing validity days for {asset}.")
        for horizon in config.HORIZONS:
            geometry = config.BARRIER_GEOMETRY[asset][horizon]
            _require(
                set(geometry) == {"look_forward", "atr_mult", "rr"},
                f"Bad geometry keys for {asset}/{horizon}: {sorted(geometry)}",
            )
            for field, value in geometry.items():
                _require(value > 0, f"Geometry value {asset}/{horizon}/{field} must be positive.")
            model_id = f"AegisTrader_{asset}_{horizon}"
            expected_tau = round(1.0 / (1.0 + geometry["rr"]), 4)
            _require(
                math.isclose(config.BUY_THRESHOLD[model_id], expected_tau, abs_tol=1e-12),
                f"BUY_THRESHOLD[{model_id}] should be {expected_tau}, got {config.BUY_THRESHOLD[model_id]}",
            )
            _require(
                config.SIGNAL_VALIDITY_DAYS[asset][horizon] > 0,
                f"Validity days for {asset}/{horizon} must be positive.",
            )

    target_geometry = _load_target_engineering_geometry()
    _require(
        set(target_geometry) == set(config.BARRIER_GEOMETRY),
        "Target-engineering asset classes do not match BARRIER_GEOMETRY.",
    )
    for asset in config.ASSET_CLASSES:
        _require(
            set(target_geometry[asset]) == set(config.BARRIER_GEOMETRY[asset]),
            f"Target-engineering horizons do not match BARRIER_GEOMETRY for {asset}.",
        )
        for horizon in config.HORIZONS:
            for field in ("look_forward", "atr_mult", "rr"):
                serving_value = float(config.BARRIER_GEOMETRY[asset][horizon][field])
                target_value = float(target_geometry[asset][horizon][field])
                _require(
                    math.isclose(serving_value, target_value, rel_tol=0.0, abs_tol=1e-12),
                    (
                        f"BARRIER_GEOMETRY[{asset}][{horizon}][{field}]={serving_value} "
                        f"does not match ASSET_CONFIGS target geometry ({target_value})."
                    ),
                )

    expected_low_conf = {
        model_id for model_id, auc in config.MODEL_TEST_ROC_AUC.items()
        if auc < config.QUALITY_THRESHOLD
    }
    _require(config.LOW_CONFIDENCE_MODELS == expected_low_conf, "LOW_CONFIDENCE_MODELS is out of sync.")

    # Router smoke checks: no network, no model loading.
    assert router.validate_ticker("btc-usd").canonical == "BTC-USD"
    assert router.validate_ticker("  AAPL ").canonical == "AAPL"
    assert router.validate_ticker("").is_blank
    route = router.route("1_week_to_1_month", "crypto", 5, ticker="AAPL")
    _require(route.model_id == "AegisTrader_Stocks_Swing", "Ticker override route check failed.")

    reporter.ok("Config, target geometry, explainer, and router mappings are internally consistent.")


def check_asset_mapping(reporter: Reporter) -> None:
    sys.path.insert(0, str(BASE_DIR))
    import config  # noqa: PLC0415

    _require(ASSET_MAPPING_PATH.exists(), f"Missing {ASSET_MAPPING_PATH.name}")
    mapping = pd.read_excel(ASSET_MAPPING_PATH, sheet_name="Asset_Mapping")
    required_columns = {"Asset Class", "Display Name", "Ticker (yfinance)", "Base Symbol", "Category / Note"}
    missing_columns = sorted(required_columns - set(mapping.columns))
    _require(not missing_columns, f"Asset mapping missing columns: {missing_columns}")

    mapping_tickers = set(mapping["Ticker (yfinance)"].dropna().astype(str))
    configured_tickers = {
        ticker for ticker_list in config.TICKERS.values() for ticker in ticker_list
    }
    missing_from_mapping = sorted(configured_tickers - mapping_tickers)
    extra_in_mapping = sorted(mapping_tickers - configured_tickers)
    _require(not missing_from_mapping, f"Configured tickers missing from asset mapping: {missing_from_mapping[:10]}")
    _require(not extra_in_mapping, f"Asset mapping has tickers not in config: {extra_in_mapping[:10]}")
    _require(mapping["Display Name"].notna().all(), "Asset mapping contains blank Display Name values.")
    reporter.ok(f"Asset mapping covers all {len(configured_tickers)} configured tickers.")


def check_artifacts(reporter: Reporter, load_artifacts: bool) -> None:
    sys.path.insert(0, str(BASE_DIR))
    import config  # noqa: PLC0415

    missing = [model_id for model_id, path in config.ARTIFACT_PATHS.items() if not path.exists()]
    _require(not missing, f"Missing model artifacts: {missing}")
    reporter.ok("All 12 model artifact files are present.")

    if not load_artifacts:
        reporter.warn("Skipped artifact unpickle/key checks (--skip-artifact-load).")
        return

    import joblib  # noqa: PLC0415

    required_keys = {"model", "feature_columns", "winsor_bounds"}
    for model_id, path in config.ARTIFACT_PATHS.items():
        artifact = joblib.load(path)
        missing_keys = sorted(required_keys - set(artifact))
        _require(not missing_keys, f"{model_id} artifact missing keys: {missing_keys}")
        _require(artifact.get("feature_columns"), f"{model_id} has no feature_columns.")
        del artifact
        gc.collect()
    reporter.ok("All artifacts unpickle and expose required serving keys.")


def check_core_imports(reporter: Reporter, timeout_seconds: int) -> None:
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(BASE_DIR)!r})\n"
        "import config, router, explainer, live_inference\n"
        "print('imports-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BASE_DIR,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise CheckFailed(f"Core import subprocess failed: {details}")
    if "imports-ok" not in result.stdout:
        raise CheckFailed("Core import subprocess did not reach completion marker.")
    reporter.ok("config/router/explainer/live_inference import in a clean subprocess.")


def check_feature_pipeline_import(reporter: Reporter, timeout_seconds: int) -> None:
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(BASE_DIR)!r})\n"
        "import live_inference\n"
        "live_inference.pipeline_functions()\n"
        "print('pipeline-imports-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BASE_DIR,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise CheckFailed(f"Feature pipeline import subprocess failed: {details}")
    if "pipeline-imports-ok" not in result.stdout:
        raise CheckFailed("Feature pipeline import subprocess did not reach completion marker.")
    reporter.ok("Frozen feature pipeline imports in a clean subprocess.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AegisTrader deployment readiness.")
    parser.add_argument(
        "--skip-artifact-load",
        action="store_true",
        help="Only check that .joblib files exist; do not unpickle them.",
    )
    parser.add_argument(
        "--skip-heavy-import",
        action="store_true",
        help="Skip the live_inference import subprocess check.",
    )
    parser.add_argument(
        "--skip-pipeline-load",
        action="store_true",
        help="Skip loading the frozen feature pipeline modules.",
    )
    parser.add_argument(
        "--import-timeout",
        type=int,
        default=180,
        help="Timeout in seconds for the clean import subprocess.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reporter = Reporter()

    print("=" * 72)
    print("AegisTrader deployment validation")
    print(f"Deployment folder: {BASE_DIR}")
    print(f"Python executable: {sys.executable}")

    reporter.run("Pinned dependency versions", lambda: check_requirements(reporter))
    reporter.run("Python syntax", lambda: check_python_files(reporter))
    reporter.run("Config/router/explainer consistency", lambda: check_core_config(reporter))
    reporter.run("Asset mapping", lambda: check_asset_mapping(reporter))
    reporter.run("Model artifacts", lambda: check_artifacts(reporter, not args.skip_artifact_load))
    if args.skip_heavy_import:
        reporter.warn("Skipped clean live_inference import check (--skip-heavy-import).")
    else:
        reporter.run("Clean core imports", lambda: check_core_imports(reporter, args.import_timeout))
    if args.skip_pipeline_load:
        reporter.warn("Skipped frozen feature pipeline import check (--skip-pipeline-load).")
    else:
        reporter.run(
            "Frozen feature pipeline imports",
            lambda: check_feature_pipeline_import(reporter, args.import_timeout),
        )

    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
