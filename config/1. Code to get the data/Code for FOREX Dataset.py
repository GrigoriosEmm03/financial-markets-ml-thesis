import yfinance as yf
import pandas as pd
import datetime

# CONFIGURATION
# Forex markets are open 24/5 (approx 260 trading days per year).
# To reach ~100k rows with ~45 pairs, we need about 8-9 years of history.
START_DATE = "2015-01-01"
END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')

# List of Major, Minor, and Exotic Forex Pairs
# Yahoo Finance syntax for Forex usually ends with "=X"
FOREX_TICKERS = [
    # Majors (Most Liquid)
    "EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", 
    "EURJPY=X", "GBPJPY=X", "EURGBP=X", "USDCAD=X", "USDCHF=X",
    
    # Minors & Crosses
    "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X", "AUDNZD=X",
    "EURAUD=X", "EURCAD=X", "EURCHF=X", "EURNZD=X", "GBPAUD=X",
    "GBPCAD=X", "GBPCHF=X", "GBPNZD=X", "AUDCAD=X", "AUDCHF=X",
    "CADCHF=X", "NZDCAD=X", "NZDCHF=X",
    
    # Exotics & Others (Good for training volatility)
    "USDCNY=X", "USDHKD=X", "USDSGD=X", "USDINR=X", "USDMXN=X",
    "USDZAR=X", "USDBRL=X", "USDTRY=X", "USDKRW=X", "USDSEK=X",
    "USDNOK=X", "USDDKK=X", "USDPLN=X", "USDTWD=X", "USDTHB=X"
]

# Note: "JPY=X" is often used for USD/JPY in Yahoo, but we check specifically.

OUTPUT_FILENAME = "DATASET_FOREX.csv"

# EXECUTION
def download_forex_data():
    print(f"🚀 Starting download for {len(FOREX_TICKERS)} Forex pairs...")
    print(f"📅 Timeframe: {START_DATE} to {END_DATE}")
    print("-" * 50)

    all_data_frames = []
    success_count = 0

    for ticker in FOREX_TICKERS:
        try:
            print(f"⏳ Downloading: {ticker}...", end=" ")
            
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

            if len(df) > 0:
                # Yahoo sometimes uses "USDJPY=X" or just "JPY=X". 
                # We standardize the ticker name in our dataset.
                df['Ticker'] = ticker
                
                df.reset_index(inplace=True)

                # Flatten MultiIndex if present
                if isinstance(df.columns, pd.MultiIndex):
                     df.columns = df.columns.get_level_values(0)

                # Keep relevant columns
                # Note: Volume in Forex is often 0 on Yahoo (Tick volume vs Real volume)
                # We keep it for consistency with other datasets
                cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']
                
                # Check availability of columns
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
        print(f"🎉 FOREX DATA SAVED: {OUTPUT_FILENAME}")
        print(f"📊 Total Rows: {len(final_df)}")
        print(f"🌍 Total Pairs: {success_count}")
    else:
        print("❌ Failed to download data.")

if __name__ == "__main__":
    download_forex_data()