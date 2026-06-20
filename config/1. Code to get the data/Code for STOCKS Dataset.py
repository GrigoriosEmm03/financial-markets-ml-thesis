import yfinance as yf
import pandas as pd
import datetime

# ==========================================
# CONFIGURATION
# ==========================================
START_DATE = "2018-01-01" # Stocks need a bit more history to reach 100k rows due to weekends
END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')

# List of Top 50 Global Stocks (Tech, Finance, Health, Energy, Consumer)
STOCK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "LLY", "V",
    "TSM", "UNH", "JNJ", "JPM", "XOM", "WMT", "MA", "PG", "AVGO", "HD",
    "CVX", "MRK", "ABBV", "KO", "PEP", "COST", "ORCL", "BAC", "ADBE", "MCD",
    "CSCO", "CRM", "ACN", "NFLX", "LIN", "AMD", "TMO", "ABT", "NKE", "DIS",
    "WFC", "TM", "DHR", "INTC", "QCOM", "TXN", "VZ", "PM", "INTU", "IBM"
]

OUTPUT_FILENAME = "DATASET_STOCKS.csv"

# ==========================================
# EXECUTION
# ==========================================
def download_stock_data():
    print(f"🚀 Starting download for {len(STOCK_TICKERS)} stocks...")
    print("-" * 50)

    all_data_frames = []
    
    for ticker in STOCK_TICKERS:
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
                # 'Adj Close' is often better for stocks (accounts for dividends/splits)
                cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']
                
                # Check if all columns exist before filtering
                existing_cols = [c for c in cols_to_keep if c in df.columns]
                df = df[existing_cols]

                all_data_frames.append(df)
                print(f"✅ ({len(df)} rows)")
            else:
                print("⚠️ Empty")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Combine and Save
    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)
        final_df.to_csv(OUTPUT_FILENAME, index=False)
        print("-" * 50)
        print(f"🎉 STOCKS SAVED: {OUTPUT_FILENAME}")
        print(f"📊 Total Rows: {len(final_df)}")
    else:
        print("❌ Failed to download data.")

if __name__ == "__main__":
    download_stock_data()