import os
import pandas as pd
import numpy as np
import warnings
import importlib.util

warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration & Import
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "stores_created")

# Dynamically import the WorldQuant engine
wq_path = os.path.join(BASE_DIR, "5_test_worldquant_alphas.py")
spec = importlib.util.spec_from_file_location("wq_module", wq_path)
wq_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wq_module)
WorldQuantAlphas = wq_module.WorldQuantAlphas

# EXCLUDED: 'Industrials' and 'Miscellaneous' for further research
best_sector_alphas = {
    'Technology': 'alpha_016', 
    'Health Care': 'alpha_035',
    'Telecommunications': 'alpha_067', 
    'Consumer Discretionary': 'alpha_006',
    'Real Estate': 'alpha_029', 
    'Finance': 'alpha_075',
    'Utilities': 'alpha_001', 
    'Energy': 'alpha_058',
    'Consumer Staples': 'alpha_101', 
    'Basic Materials': 'alpha_035'
}

# ==========================================
# 2. Math Helpers
# ==========================================
def rank_neutralize_scale(signal_df, universe_bool, sector_mapping):
    """Ranks, sector-neutralizes, and scales to exactly 1.0 Gross Exposure"""
    signal = signal_df.where(universe_bool, np.nan)
    signal = signal.rank(axis=1, pct=True, ascending=True)
    
    aligned_sectors = sector_mapping.reindex(signal.columns).fillna('Unknown')
    
    # Modern Pandas Fix: Transpose (.T), Groupby, Transform, Transpose Back
    sector_means = signal.T.groupby(aligned_sectors).transform('mean').T
    signal = signal.sub(sector_means)
    
    portfolio = signal.fillna(0.0)
    return portfolio.div(portfolio.abs().sum(axis=1), axis=0).fillna(0.0)

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    print("Loading Data...")
    pv = pd.read_parquet(os.path.join(BASE_DIR, "all_prices_5000_tickers.parquet"), engine="pyarrow")
    universe = pd.read_parquet(os.path.join(DATA_DIR, "universe_5m.parquet"))
    sector_mapping = pd.read_csv(os.path.join(BASE_DIR, "top_5000_us_by_marketcap.csv")).set_index("symbol")["sector"]
    
    # We only need to generate targets for the absolute latest day available
    latest_date = universe.index[-1]
    active_universe = universe.loc[[latest_date]].astype(bool)

    # ----------------------------------------------------
    # PRE-FLIGHT DATE CHECK
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("      PRE-FLIGHT DATE VERIFICATION")
    print("="*50)
    print(f"-> Latest Market Close Data Found: {latest_date.date() if hasattr(latest_date, 'date') else latest_date}")
    print("-> Status: OK. Generating targets for the next US market open.")
    print("="*50 + "\n")

    # ----------------------------------------------------
    # STEP A: Generate Live Alphas
    # ----------------------------------------------------
    print("Initializing WorldQuant Engine...")
    df_volume = pv['Volume']
    df_vwap = (pv['High'] + pv['Low'] + pv['Adj Close']) / 3
    
    wq_engine = WorldQuantAlphas(pv, pv['Adj Close'].pct_change(), df_volume, df_vwap, sector_mapping)
    
    master_alpha = pd.DataFrame(0.0, index=active_universe.index, columns=active_universe.columns)
    
    for sector, best_alpha in best_sector_alphas.items():
        if not hasattr(wq_engine, best_alpha):
            print(f"Skipping {best_alpha} for {sector}: Method not found in WorldQuantAlphas class.")
            continue
            
        try:
            print(f"Computing {best_alpha} for {sector}...")
            raw_signal = getattr(wq_engine, best_alpha)()
            
            if isinstance(raw_signal, np.ndarray):
                raw_signal = pd.DataFrame(raw_signal, index=pv.index, columns=pv.columns)
                
            # Grab the absolute last row by position to bypass date string type mismatches
            today_signal_row = raw_signal.iloc[[-1]] 
            
            # Extract today's row, rank, neutralize, and invert globally
            today_signal = today_signal_row.reindex(columns=active_universe.columns).where(active_universe, np.nan)
            today_ranked = today_signal.rank(axis=1, pct=True, ascending=True)
            aligned_sectors = sector_mapping.reindex(today_ranked.columns).fillna('Unknown')
            
            # Modern Pandas Fix: Transpose (.T), Groupby, Transform, Transpose Back
            sector_means = today_ranked.T.groupby(aligned_sectors).transform('mean').T
            today_neutral = today_ranked.sub(sector_means)
            
            # WQ Inversion
            today_inverted = -1.0 * today_neutral
            
            # Map strictly to the target sector
            sector_mask = (sector_mapping == sector).reindex(active_universe.columns).fillna(False)
            sector_tickers = sector_mapping[sector_mask].index.intersection(master_alpha.columns)
            
            master_alpha.loc[latest_date, sector_tickers] = today_inverted[sector_tickers].fillna(0.0).values[0]
            
        except Exception as e:
            print(f"Warning: Failed to process {best_alpha} for {sector}: {e}")

    # ----------------------------------------------------
    # STEP B: Final Scaling & Upload Prep
    # ----------------------------------------------------
    print("\nFormatting Final Targets...")
    final_portfolio = rank_neutralize_scale(master_alpha, active_universe, sector_mapping)
    
    # Scale to optimized QRT 2M Long / 2M Short Book ($4M Gross Notional)
    BOOK_SIZE = 4_000_000
    target_notional = final_portfolio.loc[latest_date] * (BOOK_SIZE / 2)
    final_targets = target_notional[target_notional != 0].dropna()

    targets_df = pd.DataFrame({
        "internal_code": final_targets.index,
        "target_notional": final_targets.values,
        "currency": "USD"
    })
    
    output_dir = os.path.join(BASE_DIR, "targets")
    os.makedirs(output_dir, exist_ok=True)
    targets_path = os.path.join(output_dir, "targets.csv")
    targets_df.to_csv(targets_path, index=False)
    
    print(f"SUCCESS: Generated {len(targets_df)} Pure Alpha targets.")
    print(f"Targets saved to {targets_path} ready for QSEC Upload.")
