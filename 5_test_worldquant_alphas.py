import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import warnings
from utils import backtest_portfolio

warnings.filterwarnings("ignore")

# ==========================================
# 1. WorldQuant Alpha Engine
# ==========================================
class WorldQuantAlphas:
    def __init__(self, df_prices, df_returns, df_volume, df_vwap, df_adv20):
        self.open = df_prices['Open']
        self.close = df_prices['Adj Close'] # Using Adj Close for PnL accuracy
        self.high = df_prices['High']
        self.low = df_prices['Low']
        self.volume = df_volume
        self.returns = df_returns
        self.vwap = df_vwap
        self.adv20 = df_adv20
        
    def rank(self, x): return x.rank(axis=1, pct=True)
    def delay(self, x, d): return x.shift(d)
    def correlation(self, x, y, d): return x.rolling(window=d).corr(y)
    def delta(self, x, d): return x.diff(d)
    def signedpower(self, x, a): return np.sign(x) * (x.abs() ** a)
    def ts_min(self, x, d): return x.rolling(window=d).min()
    def ts_max(self, x, d): return x.rolling(window=d).max()
    def ts_argmax(self, x, d): return x.rolling(window=d).apply(np.argmax, raw=True) + 1
    
    def ts_rank(self, x, d):
        def rank_last(slice_):
            if np.isnan(slice_).any(): return np.nan
            return rankdata(slice_)[-1]
        return x.rolling(window=d).apply(rank_last, raw=True)

    def alpha_001(self):
        cond = self.returns < 0
        std_ret = self.returns.rolling(20).std()
        val = std_ret.where(cond, self.close)
        return self.rank(self.ts_argmax(self.signedpower(val, 2), 5)) - 0.5

    def alpha_002(self):
        # Add 1 to volume to prevent log(0) -inf errors
        term1 = self.rank(self.delta(np.log(self.volume + 1), 2))
        term2 = self.rank((self.close - self.open) / self.open)
        return -1 * self.correlation(term1, term2, 6)

    def alpha_003(self):
        return -1 * self.correlation(self.rank(self.open), self.rank(self.volume), 10)

    def alpha_004(self):
        return -1 * self.ts_rank(self.rank(self.low), 9)

    def alpha_005(self):
        term1 = self.rank(self.open - (self.vwap.rolling(10).sum() / 10))
        term2 = -1 * np.abs(self.rank(self.close - self.vwap))
        return term1 * term2

    def alpha_006(self):
        return -1 * self.correlation(self.open, self.volume, 10)

    def alpha_008(self):
        sum_open = self.open.rolling(5).sum()
        sum_ret = self.returns.rolling(5).sum()
        prod = sum_open * sum_ret
        return -1 * self.rank(prod - self.delay(prod, 10))

    def generate_all(self):
        print("Generating Alpha Features...")
        alpha_dict = {}
        methods = [m for m in dir(self) if m.startswith('alpha_')]
        for m in methods:
            alpha_dict[m] = getattr(self, m)()
        
        # Combine into a MultiIndex DataFrame [feature, ticker]
        features_df = pd.concat(alpha_dict, axis=1)
        features_df.columns.names = ["feature", "ticker"]
        return features_df

# ==========================================
# 2. Sector-Specific Portfolio Generator
# ==========================================
def generate_sector_portfolio_vectorized(
    entire_features: pd.DataFrame, universe: pd.DataFrame, 
    start_date: str, end_date: str, signal_column: str, 
    target_sector: str, sector_mapping: pd.Series
):
    sector_mask = (sector_mapping == target_sector).reindex(universe.columns, fill_value=False)
    universe_boolean = universe.loc[start_date:end_date].astype(bool) & sector_mask
    
    # Extract feature using MultiIndex selection
    signal1 = entire_features.xs(signal_column, axis=1, level=0).loc[start_date:end_date].shift(5)
    signal1 = signal1.where(universe_boolean, np.nan)
    
    # Cross-sectional rank within the sector
    signal1 = signal1.rank(axis=1, method="min", ascending=True)
    
    # Neutralize by sector (demean across the row)
    signal1 = signal1.sub(signal1.mean(axis=1), axis=0)
    signal1 = signal1.div(signal1.abs().sum(axis=1), axis=0)
    
    portfolio = -1 * signal1.fillna(0)
    portfolio = portfolio.div(portfolio.abs().sum(axis=1), axis=0).fillna(0)
    
    # QRT Safety Check: Ensure no single weight exceeds 5% constraint
    # Calculate the max absolute weight for each day
    max_weights = portfolio.abs().max(axis=1)
    
    # If any weight exceeds 0.049, create a scale-down factor for that day
    scale_down_factors = np.where(max_weights > 0.049, 0.049 / max_weights, 1.0)
    
    # Multiply the portfolio by the scale-down factors along the time axis
    portfolio = portfolio.mul(scale_down_factors, axis=0)
    
    return portfolio.fillna(0)

# ==========================================
# 3. Main Execution Block
# ==========================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "stores_created")

    print("Loading Data...")
    pv = pd.read_parquet(os.path.join(BASE_DIR, "all_prices_5000_tickers.parquet"), engine="pyarrow")
    universe = pd.read_parquet(os.path.join(DATA_DIR, "universe_5m.parquet"))
    returns = pd.read_parquet(os.path.join(DATA_DIR, "returns.parquet"))
    sector_mapping = pd.read_csv(os.path.join(BASE_DIR, "top_5000_us_by_marketcap.csv")).set_index("symbol")["sector"]

    df_volume = pv['Volume']
    df_vwap = (pv['High'] + pv['Low'] + pv['Adj Close']) / 3
    df_adv20 = df_volume.rolling(20).mean()

    # Generate Features
    wq_engine = WorldQuantAlphas(pv, returns, df_volume, df_vwap, df_adv20)
    features = wq_engine.generate_all()

    # Define Backtest Parameters
    start_date = "2025-12-01"
    end_date = "2026-06-01"
    sectors = sector_mapping.dropna().unique()
    features_list = features.columns.get_level_values(0).unique()

    print(f"\nRunning 6-Month Sector Backtest ({start_date} to {end_date})...")
    results = []

    for sector in sectors:
        for feature in features_list:
            portfolio = generate_sector_portfolio_vectorized(
                features, universe, start_date, end_date, feature, sector, sector_mapping
            )
            
            if portfolio.abs().sum().sum() == 0:
                continue
                
            # Add , *_ to absorb any additional return values
            sr, pnl, *_ = backtest_portfolio(
                portfolio, returns.loc[start_date:end_date], 
                universe.loc[start_date:end_date], False, False
            )
            
            results.append({"sector": sector, "feature": feature, "sharpe_ratio": sr})

    results_df = pd.DataFrame(results)
    best_alphas_idx = results_df.groupby("sector")["sharpe_ratio"].idxmax()
    best_alphas = results_df.loc[best_alphas_idx].sort_values(by="sharpe_ratio", ascending=False)

    print("\n=== Top Alpha Formulas by Sector ===")
    print(best_alphas.to_string(index=False))
