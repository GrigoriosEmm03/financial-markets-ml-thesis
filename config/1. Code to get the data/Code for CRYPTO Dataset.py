import yfinance as yf
import pandas as pd
import datetime

# ==========================================
# CONFIGURATION
# ==========================================
# We want ~100k rows. 
# Calculation: 50 assets * 365 days * 5.5 years ≈ 100,000+ rows
START_DATE = "2019-01-01"
END_DATE = datetime.datetime.now().strftime('%Y-%m-%d') # Today

# List of Top 50 Cryptocurrencies (Tickers on Yahoo Finance usually end with -USD)
CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "SHIB-USD", "DOT-USD",
    "TRX-USD", "LINK-USD", "MATIC-USD", "BCH-USD", "LTC-USD",
    "UNI-USD", "NEAR-USD", "ICP-USD", "ATOM-USD", "XMR-USD",
    "ETC-USD", "FIL-USD", "HBAR-USD", "APT-USD", "LDO-USD",
    "ARB-USD", "VET-USD", "QNT-USD", "MKR-USD", "AAVE-USD",
    "GRT-USD", "ALGO-USD", "STX-USD", "SAND-USD", "EOS-USD",
    "XTZ-USD", "THETA-USD", "IMX-USD", "EGLD-USD", "MANA-USD",
    "AXS-USD", "FLOW-USD", "KAVA-USD", "NEO-USD", "KLAY-USD",
    "FTM-USD", "SNX-USD", "CRV-USD", "GALA-USD", "CHZ-USD"
]

OUTPUT_FILENAME = "DATASET_CRYPTO.csv"

# ==========================================
# EXECUTION
# ==========================================
def download_crypto_data():
    print(f"🚀 Starting download for {len(CRYPTO_TICKERS)} crypto assets...")
    print(f"📅 Timeframe: {START_DATE} to {END_DATE}")
    print("-" * 50)

    all_data_frames = []
    success_count = 0

    for ticker in CRYPTO_TICKERS:
        try:
            print(f"⏳ Downloading: {ticker}...", end=" ")
            
            # Download data from Yahoo Finance
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

            if len(df) > 0:
                # Add a column to identify the asset
                df['Ticker'] = ticker
                
                # Reset index to make Date a column
                df.reset_index(inplace=True)
                
                # Keep only relevant columns and ensure consistency
                # Yahoo often returns MultiIndex columns, we flatten them if necessary
                if isinstance(df.columns, pd.MultiIndex):
                     df.columns = df.columns.get_level_values(0)

                # Select specific columns to keep file clean
                cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
                df = df[cols_to_keep]

                all_data_frames.append(df)
                print(f"✅ Success ({len(df)} rows)")
                success_count += 1
            else:
                print("⚠️ Empty data (Skipping)")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("-" * 50)
    
    # Combine all individual dataframes into one huge dataset
    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)
        
        # Save to CSV
        final_df.to_csv(OUTPUT_FILENAME, index=False)
        
        print(f"🎉 MISSION ACCOMPLISHED!")
        print(f"💾 Data saved to: {OUTPUT_FILENAME}")
        print(f"📊 Total Rows: {len(final_df)}")
        print(f"📈 Total Assets: {success_count}")
    else:
        print("❌ No data was downloaded.")

# Run the function
if __name__ == "__main__":
    download_crypto_data()