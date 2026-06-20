# Importing the libraries
import pandas as pd
import numpy as np
import os

# PATH CONFIGURATION
# Input directory: Where the _FINAL.csv files (with Targets) are located
INPUT_DIR = r"C:\Users\grego\OneDrive\Υπολογιστής\Π.Μ.Σ. ΣΤΗΝ ΑΝΑΛΥΤΙΚΗ ΤΩΝ ΕΠΙΧΕΙΡΗΣΕΩΝ & ΣΤΗΝ ΕΠΙΣΤΗΜΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ\ΔΙΠΛΩΜΑΤΙΚΗ\CSV Files\3. CSVs με την προσθήκη Target Variable Engineering"

# Output directory: Where the _ML_READY.csv files will be saved
OUTPUT_DIR = r"C:\Users\grego\OneDrive\Υπολογιστής\Π.Μ.Σ. ΣΤΗΝ ΑΝΑΛΥΤΙΚΗ ΤΩΝ ΕΠΙΧΕΙΡΗΣΕΩΝ & ΣΤΗΝ ΕΠΙΣΤΗΜΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ\ΔΙΠΛΩΜΑΤΙΚΗ\CSV Files\4. Static Transformation & Cleansing - Feature Transformation"

# List of files to process
FILES = [
    "DATASET_CRYPTO_FINAL.csv",
    "DATASET_STOCKS_FINAL.csv",
    "DATASET_FOREX_FINAL.csv",
    "DATASET_INDICES_FINAL.csv"
]

def process_static_transformations(filepath, output_dir):
    """
    Executes the static transformations: Returns, Lags, Calendar Features,
    and drops all absolute/non-stationary columns. Includes a safety mechanism
    for datasets without Volume data (like Forex).
    """
    filename = os.path.basename(filepath)
    print(f"\n⚙️ Processing {filename}...")
    
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Ensure Data is sorted by Ticker and Date strictly before any shift/pct_change operations
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    # ---------------------------------------------------------
    # STEP 1: Stationarity & Returns
    # ---------------------------------------------------------
    # Calculate daily percentage returns strictly per Ticker
    df['Return_Close'] = df.groupby('Ticker')['Close'].pct_change()
    df['Return_Volume'] = df.groupby('Ticker')['Volume'].pct_change()
    
    # CRITICAL FIX: Make absolute indicators Scale-Invariant (Percentages)
    # MACD and ATR are absolute values. We must divide them by Close to make them comparable 
    # across assets with vastly different prices (e.g., BTC vs SHIB).
    df['ATR_pct'] = df['ATR'] / df['Close']
    
    if 'MACD_12_26_9' in df.columns:
        df['MACD_pct'] = df['MACD_12_26_9'] / df['Close']
    
    if 'MACDh_12_26_9' in df.columns:
        df['MACDh_pct'] = df['MACDh_12_26_9'] / df['Close']

    # ---------------------------------------------------------
    # STEP 2: Calendar Features
    # ---------------------------------------------------------
    # Extract cyclical time features
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

    # ---------------------------------------------------------
    # STEP 3: Tabular Lagged Features
    # ---------------------------------------------------------
    # Select only stationary features to lag
    features_to_lag = ['Return_Close', 'Return_Volume', 'RSI_14', 'ATR_pct', 'MACD_pct', 'MACDh_pct']
    
    # Check if RSI exists as 'RSI' instead of 'RSI_14' just in case
    if 'RSI' in df.columns and 'RSI_14' not in df.columns:
        features_to_lag[2] = 'RSI'
        
    actual_features_to_lag = [col for col in features_to_lag if col in df.columns]
    
    # Also add Bollinger Bandwidth (BBB) and %B (BBP) which are inherently stationary
    bb_stationary_cols = [col for col in df.columns if col.startswith('BBB_') or col.startswith('BBP_')]
    actual_features_to_lag.extend(bb_stationary_cols)

    # Apply lag 1 and lag 2 strictly per Ticker
    for feature in actual_features_to_lag:
        df[f'{feature}_lag1'] = df.groupby('Ticker')[feature].shift(1)
        df[f'{feature}_lag2'] = df.groupby('Ticker')[feature].shift(2)

    # ---------------------------------------------------------
    # STEP 4: Feature Selection (Drop absolute values)
    # ---------------------------------------------------------
    # Drop raw OHLCV and any absolute technical indicators
    cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    
    # Drop absolute technical indicators
    abs_tech_cols = ['SMA_50', 'SMA_200', 'ATR', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
    bb_absolute_cols = [col for col in df.columns if col.startswith('BBL_') or col.startswith('BBM_') or col.startswith('BBU_')]
    
    cols_to_drop.extend(abs_tech_cols)
    cols_to_drop.extend(bb_absolute_cols)
    
    # Safely drop columns if they exist
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df.drop(columns=existing_cols_to_drop, inplace=True)

    # ---------------------------------------------------------
    # STEP 5: Post-Lag NaN Handling & FOREX FIX
    # ---------------------------------------------------------
    # Volume returns might produce infinity if previous volume was 0. 
    # Replace inf with NaN so dropna() can catch it.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # NEW FIX: If the asset has no volume data (like Forex), Return_Volume will be entirely NaN.
    # Instead of dropping all rows, we drop the useless Volume columns.
    if 'Return_Volume' in df.columns and df['Return_Volume'].isna().all():
        print("   ⚠️ Detected entirely missing Volume data (common in Forex). Dropping Volume columns instead of rows.")
        vol_cols_to_remove = ['Return_Volume', 'Return_Volume_lag1', 'Return_Volume_lag2']
        existing_vol_cols = [c for c in vol_cols_to_remove if c in df.columns]
        df.drop(columns=existing_vol_cols, inplace=True)
    
    # Drop all rows with NaNs (created by lags and percentage changes)
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    print(f"   🧹 Dropped {initial_rows - final_rows} rows due to NaN values from Lags/Returns.")

    # ---------------------------------------------------------
    # SAVE THE ML-READY DATASET
    # ---------------------------------------------------------
    output_filename = filename.replace("_FINAL.csv", "_ML_READY.csv")
    output_path = os.path.join(output_dir, output_filename)
    
    df.to_csv(output_path, index=False)
    print(f"   ✅ Saved {output_filename} successfully! (Total Clean Rows: {final_rows})")


def main():
    print("🚀 Starting Phase 1: Static Transformation & Cleansing Pipeline...")
    
    # Automatically create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for file in FILES:
        filepath = os.path.join(INPUT_DIR, file)
        if os.path.exists(filepath):
            process_static_transformations(filepath, OUTPUT_DIR)
        else:
            print(f"\n⚠️ WARNING: File not found at {filepath}")

    print("\n🏁 Phase 1 Completed. Datasets are completely agnostic and ML-Ready!")

if __name__ == "__main__":
    main()