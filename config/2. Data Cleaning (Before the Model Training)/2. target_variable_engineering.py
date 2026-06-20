# Importing the libraries
import pandas as pd
import numpy as np
import os

# PATH CONFIGURATION (Dynamic Path Management)

# Input directory (Where the ready CSVs with technical indicators are located)
INPUT_DIR = r"C:\Users\grego\OneDrive\Υπολογιστής\Π.Μ.Σ. ΣΤΗΝ ΑΝΑΛΥΤΙΚΗ ΤΩΝ ΕΠΙΧΕΙΡΗΣΕΩΝ & ΣΤΗΝ ΕΠΙΣΤΗΜΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ\ΔΙΠΛΩΜΑΤΙΚΗ\CSV Files\CSVs με την προσθήκη δεικτών"

# Output directory (Where the final CSVs with target variables will be saved)
OUTPUT_DIR = r"C:\Users\grego\OneDrive\Υπολογιστής\Π.Μ.Σ. ΣΤΗΝ ΑΝΑΛΥΤΙΚΗ ΤΩΝ ΕΠΙΧΕΙΡΗΣΕΩΝ & ΣΤΗΝ ΕΠΙΣΤΗΜΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ\ΔΙΠΛΩΜΑΤΙΚΗ\CSV Files\Τελικά CSVs με την προσθήκη Target Variables"


# BUSINESS LOGIC & CONFIGURATION
ASSET_CONFIGS = {
    "DATASET_CRYPTO_READY.csv": {
        "Day":   {"look_forward": 3,  "atr_mult": 1.5, "rr": 2.0},
        "Swing": {"look_forward": 14, "atr_mult": 2.5, "rr": 3.0},
        "Long":  {"look_forward": 60, "atr_mult": 4.0, "rr": 4.0}
    },
    "DATASET_STOCKS_READY.csv": {
        "Day":   {"look_forward": 5,  "atr_mult": 1.5, "rr": 1.5},
        "Swing": {"look_forward": 21, "atr_mult": 2.0, "rr": 2.5},
        "Long":  {"look_forward": 90, "atr_mult": 3.0, "rr": 3.0}
    },
    "DATASET_FOREX_READY.csv": {
        "Day":   {"look_forward": 2,  "atr_mult": 1.0, "rr": 1.5},
        "Swing": {"look_forward": 10, "atr_mult": 1.5, "rr": 2.0},
        "Long":  {"look_forward": 45, "atr_mult": 2.5, "rr": 3.0}
    },
    "DATASET_INDICES_READY.csv": {
        "Day":   {"look_forward": 5,  "atr_mult": 1.2, "rr": 1.5},
        "Swing": {"look_forward": 20, "atr_mult": 2.0, "rr": 2.0},
        "Long":  {"look_forward": 120,"atr_mult": 3.0, "rr": 3.0}
    }
}

def simulate_multi_horizon_targets(df, file_key):
    """
    Simulates trades for Day, Swing, and Long profiles simultaneously.
    Utilizes NumPy arrays for drastic execution speedup.
    """
    config = ASSET_CONFIGS.get(file_key)
    if not config:
        return df

    # Extract columns to NumPy arrays for O(1) memory access speed
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    atrs = df['ATR'].values
    total_rows = len(df)
    
    # Initialize dictionary to hold results before assigning them to the DataFrame
    results = {}
    for profile in config.keys():
        results[f'Target_{profile}_Win'] = np.full(total_rows, np.nan)
        results[f'Target_{profile}_Fast'] = np.full(total_rows, np.nan)
        results[f'Target_{profile}_LowStress'] = np.full(total_rows, np.nan)
        results[f'Target_{profile}_MFE'] = np.full(total_rows, np.nan) 

    # Iterate through the dataset
    for i in range(total_rows):
        entry_price = closes[i]
        current_atr = atrs[i]
        
        # Skip if ATR is NaN (e.g., at the beginning of the dataset)
        if np.isnan(current_atr):
            continue

        for profile, params in config.items():
            lf_days = params['look_forward']
            
            # Check if we have enough future data to evaluate the trade
            if i + lf_days >= total_rows:
                continue
                
            sl_dist = params['atr_mult'] * current_atr
            stop_loss = entry_price - sl_dist
            take_profit = entry_price + (sl_dist * params['rr'])
            
            trade_won = 0
            days_to_complete = 0
            max_adverse_drop = entry_price
            max_favorable_excursion = entry_price
            
            # Scan the future days using the predefined arrays
            for j in range(i + 1, i + 1 + lf_days):
                days_to_complete += 1
                day_open = opens[j]
                day_high = highs[j]
                day_low = lows[j]
                
                # Update extreme price points for this specific trade
                if day_low < max_adverse_drop:
                    max_adverse_drop = day_low
                if day_high > max_favorable_excursion:
                    max_favorable_excursion = day_high
                
                # 1. GAP ANALYSIS: Check if market gapped beyond our levels at the open
                if day_open <= stop_loss:
                    trade_won = 0
                    break
                elif day_open >= take_profit:
                    trade_won = 1
                    break
                    
                # 2. INTRA-DAY ANALYSIS: Pessimistic approach (assume low hits before high)
                if day_low <= stop_loss:
                    trade_won = 0
                    break
                elif day_high >= take_profit:
                    trade_won = 1
                    break
        
            # Record base outcome
            results[f'Target_{profile}_Win'][i] = trade_won
            
            # Record strict targets
            if trade_won == 1:
                results[f'Target_{profile}_Fast'][i] = 1 if days_to_complete <= (lf_days / 2) else 0
                stress_threshold = entry_price - (sl_dist * 0.5)
                results[f'Target_{profile}_LowStress'][i] = 1 if max_adverse_drop >= stress_threshold else 0
            else:
                results[f'Target_{profile}_Fast'][i] = 0
                results[f'Target_{profile}_LowStress'][i] = 0
                
            # Record Continuous Target (MFE as a percentage gain from entry price)
            results[f'Target_{profile}_MFE'][i] = ((max_favorable_excursion - entry_price) / entry_price) * 100

    # Assign the calculated arrays back to the DataFrame at once (highly efficient)
    for col_name, array_data in results.items():
        df[col_name] = array_data

    return df

print("🎯 Starting advanced multi-horizon trade simulation...\n")

# Ensure the output directory exists; if not, create it automatically
os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in ASSET_CONFIGS.keys():
    # Construct the full path for the input file
    input_path = os.path.join(INPUT_DIR, filename)
    
    if os.path.exists(input_path):
        print(f"⚙️ Processing {filename}...")
        
        df = pd.read_csv(input_path)
        
        # Apply the simulation per Ticker to avoid data leakage between different assets
        df_labeled = df.groupby('Ticker', group_keys=False).apply(
            lambda x: simulate_multi_horizon_targets(x, filename)
        )
        
        # Drop rows with NaN in the long-term target (end of dataset where the future is unknown)
        df_final = df_labeled.dropna(subset=['Target_Long_Win'])
        
        # Define the new filename and save it to the output directory
        output_filename = filename.replace("_READY.csv", "_FINAL.csv")
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        df_final.to_csv(output_path, index=False)
        
        print(f"   ✅ Done! Generated Day, Swing, and Long targets.")
        print(f"   💾 Saved as: {output_filename} in the target directory.\n")
    else:
        print(f"⚠️ File {filename} not found in directory: {INPUT_DIR}\n")

print("🏁 Process completed. Complex target variables successfully created!")