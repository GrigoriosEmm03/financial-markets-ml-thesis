# importing the libraries
import pandas as pd
import pandas_ta as ta
import glob
import os

# list with the files that we created before
files = [
    "DATASET_CRYPTO.csv",
    "DATASET_STOCKS.csv",
    "DATASET_FOREX.csv",
    "DATASET_INDICES.csv"
]

def add_technical_indicators(df):
    """
    Here we calculate the indicators.
    ATTENTION: Calculations are done per Ticker to avoid mixing the data!
    """
    # 1. RSI (14 periods)
    df['RSI'] = df.ta.rsi(length=14)

    # 2. MACD (Trend)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1) # Add the MACD columns to the dataframe

    # 3. Simple Moving Averages (SMA)
    df['SMA_50'] = df.ta.sma(length=50)  # Medium-term trend
    df['SMA_200'] = df.ta.sma(length=200) # Long-term trend

    # 4. Bollinger Bands (Volatility)
    bbands = df.ta.bbands(length=20, std=2)
    df = pd.concat([df, bbands], axis=1)

    # 5. ATR (For Stop Loss calculation)
    df['ATR'] = df.ta.atr(length=14)

    return df

print("🚀 Starting indicator calculation...\n")

for file in files:
    if os.path.exists(file):
        print(f"📂 Processing: {file}...", end=" ")
        
        # Load the CSV
        df = pd.read_csv(file)
        
        # Group by Ticker
        # VERY IMPORTANT: We don't want Bitcoin's RSI to be affected by Ethereum's.
        # That's why we calculate the indicators separately for each Ticker.
        df_grouped = df.groupby('Ticker', group_keys=False).apply(add_technical_indicators)
        
        # Clean NaN values (Indicators create empty values in the first few days)
        df_clean = df_grouped.dropna()
        
        # Save to a new "_READY" file
        output_filename = file.replace(".csv", "_READY.csv")
        df_clean.to_csv(output_filename, index=False)
        
        print(f"✅ Done! ({len(df_clean)} rows)")
        print(f"   💾 Saved as: {output_filename}")
    else:
        print(f"⚠️ File {file} not found.")

print("\n🏁 Process completed. Data is ready for training!")