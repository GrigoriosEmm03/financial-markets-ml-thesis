import yfinance as yf
import pandas as pd
import datetime

# CONFIGURATION
# Indices have a long history. We start from 2000 to capture different market cycles (Dot-com bubble, 2008 crisis, Covid crash, etc.) which is GREAT for AI training.
START_DATE = "2000-01-01"
END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')

# List of Major Global Indices
# Yahoo Finance uses '^' prefix for indices
INDICES_TICKERS = [
    # US Markets
    "^GSPC",   # S&P 500
    "^DJI",    # Dow Jones Industrial Average
    "^IXIC",   # NASDAQ Composite
    "^RUT",    # Russell 2000
    "^VIX",    # CBOE Volatility Index (Fear Index - Very important feature!)
    
    # European Markets
    "^FTSE",   # FTSE 100 (UK)
    "^GDAXI",  # DAX Performance Index (Germany)
    "^FCHI",   # CAC 40 (France)
    "^STOXX50E", # EURO STOXX 50
    "^IBEX",   # IBEX 35 (Spain)
    
    # Asian Markets
    "^N225",   # Nikkei 225 (Japan)
    "^HSI",    # Hang Seng Index (Hong Kong)
    "^STI",    # Straits Times Index (Singapore)
    "^KS11",   # KOSPI Composite Index (South Korea)
    "^BSESN",  # S&P BSE SENSEX (India)
    "^JKSE",   # Jakarta Composite Index (Indonesia)
    
    # Others (Americas/Pacific)
    "^BVSP",   # IBOVESPA (Brazil)
    "^MXX",    # IPC (Mexico)
    "^GSPTSE", # S&P/TSX Composite (Canada)
    "^AXJO",   # S&P/ASX 200 (Australia)
    "^AORD"    # All Ordinaries (Australia)
]

OUTPUT_FILENAME = "DATASET_INDICES.csv"

# EXECUTION
def download_indices_data():
    print(f"🚀 Starting download for {len(INDICES_TICKERS)} Global Indices...")
    print(f"📅 Timeframe: {START_DATE} to {END_DATE}")
    print("-" * 50)

    all_data_frames = []
    success_count = 0

    for ticker in INDICES_TICKERS:
        try:
            print(f"⏳ Downloading: {ticker}...", end=" ")
            
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

            if len(df) > 0:
                df['Ticker'] = ticker
                df.reset_index(inplace=True)

                # Flatten MultiIndex if present
                if isinstance(df.columns, pd.MultiIndex):
                     df.columns = df.columns.get_level_values(0)

                # Keep relevant columns
                cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']
                
                # Check and filter
                existing_cols = [c for c in cols_to_keep if c in df.columns]
                df = df[existing_cols]

                all_data_frames.append(df)
                print(f"✅ ({len(df)} rows)")
                success_count += 1
            else:
                print("⚠️ Empty")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Combine and Save
    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)
        
        # Save to CSV
        final_df.to_csv(OUTPUT_FILENAME, index=False)
        
        print("-" * 50)
        print(f"🎉 INDICES DATA SAVED: {OUTPUT_FILENAME}")
        print(f"📊 Total Rows: {len(final_df)}")
        print(f"🌍 Total Indices: {success_count}")
    else:
        print("❌ Failed to download data.")

if __name__ == "__main__":
    download_indices_data()