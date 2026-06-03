import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import warnings
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ==========================================
# 1. Complete WorldQuant 101 Alpha Engine
# ==========================================
class WorldQuantAlphas:
    def __init__(self, df_prices, df_returns, df_volume, df_vwap, sector_mapping):
        self.open = df_prices['Open']
        self.close = df_prices['Adj Close'] 
        self.high = df_prices['High']
        self.low = df_prices['Low']
        self.volume = df_volume
        self.returns = df_returns
        self.vwap = df_vwap
        self.sector_mapping = sector_mapping.reindex(self.close.columns).fillna('Unknown')
        
    # --- Dynamic ADV Helper ---
    def adv(self, d):
        return self.volume.rolling(window=d).mean()

    # --- Core Operators ---
    def rank(self, x): return x.rank(axis=1, pct=True)
    def delay(self, x, d): return x.shift(d)
    def correlation(self, x, y, d): return x.rolling(window=int(d)).corr(y)
    def covariance(self, x, y, d): return x.rolling(window=int(d)).cov(y)
    def scale(self, x, a=1): return x.div(x.abs().sum(axis=1), axis=0) * a
    def delta(self, x, d): return x.diff(int(d))
    def signedpower(self, x, a): return np.sign(x) * (x.abs() ** a)
    def ts_min(self, x, d): return x.rolling(window=int(d)).min()
    def ts_max(self, x, d): return x.rolling(window=int(d)).max()
    def ts_argmax(self, x, d): return x.rolling(window=int(d)).apply(np.argmax, raw=True) + 1
    def ts_argmin(self, x, d): return x.rolling(window=int(d)).apply(np.argmin, raw=True) + 1
    
    def ts_rank(self, x, d):
        def rank_last(slice_):
            if np.isnan(slice_).any(): return np.nan
            return rankdata(slice_)[-1]
        return x.rolling(window=int(d)).apply(rank_last, raw=True)

    def decay_linear(self, x, d):
        d = int(np.floor(d))
        weights = np.arange(d, 0, -1)
        weights = weights / weights.sum()
        def apply_weights(slice_):
            if np.isnan(slice_).any(): return np.nan
            return np.dot(slice_, weights)
        return x.rolling(d).apply(apply_weights, raw=True)

    def indneutralize(self, x):
        return x.sub(x.groupby(self.sector_mapping, axis=1).transform('mean'), axis=0)

    # --- Alpha Formulations 1 to 101 ---
    def alpha_001(self): return self.rank(self.ts_argmax(self.signedpower(self.returns.rolling(20).std().where(self.returns < 0, self.close), 2), 5)) - 0.5
    def alpha_002(self): return -1 * self.correlation(self.rank(self.delta(np.log(self.volume + 1), 2)), self.rank((self.close - self.open) / self.open), 6)
    def alpha_003(self): return -1 * self.correlation(self.rank(self.open), self.rank(self.volume), 10)
    def alpha_004(self): return -1 * self.ts_rank(self.rank(self.low), 9)
    def alpha_005(self): return self.rank(self.open - (self.vwap.rolling(10).sum() / 10)) * (-1 * np.abs(self.rank(self.close - self.vwap)))
    def alpha_006(self): return -1 * self.correlation(self.open, self.volume, 10)
    def alpha_007(self): return (-1 * self.ts_rank(np.abs(self.delta(self.close, 7)), 60) * np.sign(self.delta(self.close, 7))).where(self.adv(20) < self.volume, -1)
    def alpha_008(self): return -1 * self.rank((self.open.rolling(5).sum() * self.returns.rolling(5).sum()) - self.delay((self.open.rolling(5).sum() * self.returns.rolling(5).sum()), 10))
    def alpha_009(self): delta_c = self.delta(self.close, 1); return delta_c.where(self.ts_min(delta_c, 5) > 0, delta_c.where(self.ts_max(delta_c, 5) < 0, -delta_c))
    def alpha_010(self): delta_c = self.delta(self.close, 1); return self.rank(delta_c.where(self.ts_min(delta_c, 4) > 0, delta_c.where(self.ts_max(delta_c, 4) < 0, -delta_c)))
    def alpha_011(self): return (self.rank(self.ts_max(self.vwap - self.close, 3)) + self.rank(self.ts_min(self.vwap - self.close, 3))) * self.rank(self.delta(self.volume, 3))
    def alpha_012(self): return np.sign(self.delta(self.volume, 1)) * (-1 * self.delta(self.close, 1))
    def alpha_013(self): return -1 * self.rank(self.covariance(self.rank(self.close), self.rank(self.volume), 5))
    def alpha_014(self): return -1 * self.rank(self.delta(self.returns, 3)) * self.correlation(self.open, self.volume, 10)
    def alpha_015(self): return -1 * self.correlation(self.rank(self.high), self.rank(self.volume), 3).rolling(3).sum()
    def alpha_016(self): return -1 * self.rank(self.covariance(self.rank(self.high), self.rank(self.volume), 5))
    def alpha_017(self): return -1 * self.rank(self.ts_rank(self.close, 10)) * self.rank(self.delta(self.delta(self.close, 1), 1)) * self.rank(self.ts_rank((self.volume / self.adv(20)), 5))
    def alpha_018(self): return -1 * self.rank((self.close.diff(1).abs().rolling(5).std() + (self.close - self.open)) + self.correlation(self.close, self.open, 10))
    def alpha_019(self): return -1 * np.sign((self.close - self.delay(self.close, 7)) + self.delta(self.close, 7)) * (1 + self.rank(1 + self.returns.rolling(250).sum()))
    def alpha_020(self): return -1 * self.rank(self.open - self.delay(self.high, 1)) * self.rank(self.open - self.delay(self.close, 1)) * self.rank(self.open - self.delay(self.low, 1))
    def alpha_021(self): return np.where((self.close.rolling(8).sum() / 8 + self.close.rolling(8).std()) < (self.close.rolling(2).sum() / 2), -1, np.where((self.close.rolling(2).sum() / 2) < (self.close.rolling(8).sum() / 8 - self.close.rolling(8).std()), 1, np.where(self.volume / self.adv(20) >= 1, 1, -1)))
    def alpha_022(self): return -1 * (self.delta(self.correlation(self.high, self.volume, 5), 5) * self.rank(self.close.rolling(20).std()))
    def alpha_023(self): return (-1 * self.delta(self.high, 2)).where((self.high.rolling(20).sum() / 20) < self.high, 0)
    def alpha_024(self): cond = (self.delta((self.close.rolling(100).sum() / 100), 100) / self.delay(self.close, 100)) <= 0.05; return (-1 * (self.close - self.ts_min(self.close, 100))).where(cond, -1 * self.delta(self.close, 3))
    def alpha_025(self): return self.rank(((-1 * self.returns) * self.adv(20) * self.vwap) * (self.high - self.close))
    def alpha_026(self): return -1 * self.ts_max(self.correlation(self.ts_rank(self.volume, 5), self.ts_rank(self.high, 5), 5), 3)
    def alpha_027(self): return pd.DataFrame(np.where(self.rank(self.correlation(self.rank(self.volume), self.rank(self.vwap), 6).rolling(2).sum() / 2.0) > 0.5, -1, 1), index=self.close.index, columns=self.close.columns)
    def alpha_028(self): return self.scale(self.correlation(self.adv(20), self.low, 5) + ((self.high + self.low) / 2) - self.close)
    def alpha_029(self): return self.ts_min(self.rank(self.rank(self.scale(np.log(self.ts_min(self.rank(self.rank(-1 * self.rank(self.delta(self.close - 1, 5)))), 2).rolling(1).sum() + 1)))), 5) + self.ts_rank(self.delay(-1 * self.returns, 6), 5)
    def alpha_030(self): return (1.0 - self.rank(np.sign(self.close - self.delay(self.close, 1)) + np.sign(self.delay(self.close, 1) - self.delay(self.close, 2)) + np.sign(self.delay(self.close, 2) - self.delay(self.close, 3)))) * self.volume.rolling(5).sum() / self.volume.rolling(20).sum()
    def alpha_031(self): return self.rank(self.rank(self.rank(self.decay_linear(-1 * self.rank(self.rank(self.delta(self.close, 10))), 10)))) + self.rank(-1 * self.delta(self.close, 3)) + np.sign(self.scale(self.correlation(self.adv(20), self.low, 12)))
    def alpha_032(self): return self.scale((self.close.rolling(7).sum() / 7) - self.close) + (20 * self.scale(self.correlation(self.vwap, self.delay(self.close, 5), 230)))
    def alpha_033(self): return self.rank(-1 * ((1 - (self.open / self.close)) ** 1))
    def alpha_034(self): return self.rank((1 - self.rank(self.returns.rolling(2).std() / self.returns.rolling(5).std())) + (1 - self.rank(self.delta(self.close, 1))))
    def alpha_035(self): return (self.ts_rank(self.volume, 32) * (1 - self.ts_rank((self.close + self.high) - self.low, 16))) * (1 - self.ts_rank(self.returns, 32))
    def alpha_036(self): return 2.21 * self.rank(self.correlation(self.close - self.open, self.delay(self.volume, 1), 15)) + 0.7 * self.rank(self.open - self.close) + 0.73 * self.rank(self.ts_rank(self.delay(-1 * self.returns, 6), 5)) + self.rank(np.abs(self.correlation(self.vwap, self.adv(20), 6))) + 0.6 * self.rank((self.close.rolling(200).sum() / 200 - self.open) * (self.close - self.open))
    def alpha_037(self): return self.rank(self.correlation(self.delay(self.open - self.close, 1), self.close, 200)) + self.rank(self.open - self.close)
    def alpha_038(self): return -1 * self.rank(self.ts_rank(self.close, 10)) * self.rank(self.close / self.open)
    def alpha_039(self): return -1 * self.rank(self.delta(self.close, 7) * (1 - self.rank(self.decay_linear(self.volume / self.adv(20), 9)))) * (1 + self.rank(self.returns.rolling(250).sum()))
    def alpha_040(self): return -1 * self.rank(self.high.rolling(10).std()) * self.correlation(self.high, self.volume, 10)
    def alpha_041(self): return ((self.high * self.low)**0.5) - self.vwap
    def alpha_042(self): return self.rank((self.vwap - self.close)) / self.rank((self.vwap + self.close))
    def alpha_043(self): return self.ts_rank(self.volume / self.adv(20), 20) * self.ts_rank(-1 * self.delta(self.close, 7), 8)
    def alpha_044(self): return -1 * self.correlation(self.high, self.rank(self.volume), 5)
    def alpha_045(self): return -1 * (self.rank(self.delay(self.close, 5).rolling(20).sum() / 20) * self.correlation(self.close, self.volume, 2) * self.rank(self.correlation(self.close.rolling(5).sum(), self.close.rolling(20).sum(), 2)))
    def alpha_046(self): cond = ((self.delay(self.close, 20) - self.delay(self.close, 10)) / 10) - ((self.delay(self.close, 10) - self.close) / 10); return pd.DataFrame(np.where(cond > 0.25, -1, np.where(cond < 0, 1, -1 * (self.close - self.delay(self.close, 1)))), index=self.close.index, columns=self.close.columns)
    def alpha_047(self): return (((self.rank(1 / self.close) * self.volume) / self.adv(20)) * ((self.high * self.rank(self.high - self.close)) / (self.high.rolling(5).sum() / 5))) - self.rank(self.vwap - self.delay(self.vwap, 5))
    def alpha_049(self): cond = ((self.delay(self.close, 20) - self.delay(self.close, 10)) / 10) - ((self.delay(self.close, 10) - self.close) / 10); return pd.DataFrame(np.where(cond < -0.1, 1, -1 * (self.close - self.delay(self.close, 1))), index=self.close.index, columns=self.close.columns)
    def alpha_050(self): return -1 * self.ts_max(self.rank(self.correlation(self.rank(self.volume), self.rank(self.vwap), 5)), 5)
    def alpha_051(self): cond = ((self.delay(self.close, 20) - self.delay(self.close, 10)) / 10) - ((self.delay(self.close, 10) - self.close) / 10); return pd.DataFrame(np.where(cond < -0.05, 1, -1 * (self.close - self.delay(self.close, 1))), index=self.close.index, columns=self.close.columns)
    def alpha_052(self): return ((-1 * self.ts_min(self.low, 5) + self.delay(self.ts_min(self.low, 5), 5)) * self.rank((self.returns.rolling(240).sum() - self.returns.rolling(20).sum()) / 220)) * self.ts_rank(self.volume, 5)
    def alpha_053(self): return -1 * self.delta(((self.close - self.low) - (self.high - self.close)) / (self.close - self.low), 9)
    def alpha_054(self): return (-1 * ((self.low - self.close) * (self.open ** 5))) / (((self.low - self.high) * (self.close ** 5)) + 1e-6)
    def alpha_055(self): return -1 * self.correlation(self.rank((self.close - self.ts_min(self.low, 12)) / ((self.ts_max(self.high, 12) - self.ts_min(self.low, 12)) + 1e-6)), self.rank(self.volume), 6)
    def alpha_060(self): return -(1 * ((2 * self.scale(self.rank((((self.close - self.low) - (self.high - self.close)) / ((self.high - self.low) + 1e-6)) * self.volume))) - self.scale(self.rank(self.ts_argmax(self.close, 10)))))
    def alpha_061(self): return (self.rank(self.vwap - self.ts_min(self.vwap, 16)) < self.rank(self.correlation(self.vwap, self.adv(180), 18))).astype(int)
    def alpha_062(self): return -1 * ((self.rank(self.correlation(self.vwap, self.adv(20).rolling(22).sum(), 10)) < self.rank((self.rank(self.open) + self.rank(self.open)) < (self.rank((self.high + self.low) / 2) + self.rank(self.high))))).astype(int)
    def alpha_064(self): return -1 * ((self.rank(self.correlation(self.open.rolling(13).sum() * 0.178 + self.low.rolling(13).sum() * (1-0.178), self.adv(120).rolling(13).sum(), 17)) < self.rank(self.delta(((self.high + self.low) / 2) * 0.178 + self.vwap * (1-0.178), 4)))).astype(int)
    def alpha_065(self): return -1 * ((self.rank(self.correlation((self.open * 0.008 + self.vwap * (1-0.008)), self.adv(60).rolling(9).sum(), 6)) < self.rank(self.open - self.ts_min(self.open, 14)))).astype(int)
    def alpha_074(self): return -1 * ((self.rank(self.correlation(self.close, self.adv(30).rolling(37).sum(), 15)) < self.rank(self.correlation(self.rank(self.high * 0.026 + self.vwap * (1-0.026)), self.rank(self.volume), 11)))).astype(int)
    def alpha_075(self): return (self.rank(self.correlation(self.vwap, self.volume, 4)) < self.rank(self.correlation(self.rank(self.low), self.rank(self.adv(50)), 12))).astype(int)
    def alpha_086(self): return -1 * ((self.ts_rank(self.correlation(self.close, self.adv(20).rolling(15).sum(), 6), 20) < self.rank((self.open + self.close) - (self.vwap + self.open)))).astype(int)
    def alpha_101(self): return (self.close - self.open) / ((self.high - self.low) + 0.001)

    # --- IndNeutralize Dependent Alphas ---
    def alpha_048(self): return self.indneutralize((self.correlation(self.delta(self.close, 1), self.delta(self.delay(self.close, 1), 1), 250) * self.delta(self.close, 1)) / self.close) / (self.delta(self.close, 1) / self.delay(self.close, 1)).pow(2).rolling(250).sum()
    def alpha_058(self): return -1 * self.ts_rank(self.decay_linear(self.correlation(self.indneutralize(self.vwap), self.volume, 4), 8), 6)
    def alpha_059(self): return -1 * self.ts_rank(self.decay_linear(self.correlation(self.indneutralize((self.vwap * 0.728) + (self.vwap * (1-0.728))), self.volume, 4), 16), 8)
    def alpha_063(self): return -1 * (self.rank(self.decay_linear(self.delta(self.indneutralize(self.close), 2), 8)) - self.rank(self.decay_linear(self.correlation((self.vwap * 0.318) + (self.open * (1-0.318)), self.adv(180).rolling(37).sum(), 14), 12)))
    def alpha_067(self): return -1 * (self.rank(self.high - self.ts_min(self.high, 2)) ** self.rank(self.correlation(self.indneutralize(self.vwap), self.indneutralize(self.adv(20)), 6)))
    def alpha_070(self): return -1 * (self.rank(self.delta(self.vwap, 1)) ** self.ts_rank(self.correlation(self.indneutralize(self.close), self.adv(50), 18), 18))
    def alpha_079(self): return (self.rank(self.delta(self.indneutralize((self.close * 0.607) + (self.open * (1-0.607))), 1)) < self.rank(self.correlation(self.ts_rank(self.vwap, 4), self.ts_rank(self.adv(150), 9), 15))).astype(int)
    def alpha_080(self): return -1 * (self.rank(np.sign(self.delta(self.indneutralize((self.open * 0.868) + (self.high * (1-0.868))), 4))) ** self.ts_rank(self.correlation(self.high, self.adv(10), 5), 6))
    def alpha_087(self): return -1 * np.maximum(self.rank(self.decay_linear(self.delta((self.close * 0.37) + (self.vwap * (1-0.37)), 2), 3)), self.ts_rank(self.decay_linear(np.abs(self.correlation(self.indneutralize(self.adv(81)), self.close, 13)), 5), 14))
    def alpha_089(self): return self.ts_rank(self.decay_linear(self.correlation((self.low * 0.967) + (self.low * (1-0.967)), self.adv(10), 7), 6), 4) - self.ts_rank(self.decay_linear(self.delta(self.indneutralize(self.vwap), 3), 10), 15)
    def alpha_090(self): return -1 * (self.rank(self.close - self.ts_max(self.close, 5)) ** self.ts_rank(self.correlation(self.indneutralize(self.adv(40)), self.low, 5), 3))
    def alpha_091(self): return -1 * (self.ts_rank(self.decay_linear(self.decay_linear(self.correlation(self.indneutralize(self.close), self.volume, 10), 16), 4), 5) - self.rank(self.decay_linear(self.correlation(self.vwap, self.adv(30), 4), 3)))
    def alpha_093(self): return self.ts_rank(self.decay_linear(self.correlation(self.indneutralize(self.vwap), self.adv(81), 17), 20), 8) / self.rank(self.decay_linear(self.delta((self.close * 0.524) + (self.vwap * (1-0.524)), 3), 16))
    def alpha_097(self): return -1 * (self.rank(self.decay_linear(self.delta(self.indneutralize((self.low * 0.721) + (self.vwap * (1-0.721))), 3), 20)) - self.ts_rank(self.decay_linear(self.ts_rank(self.correlation(self.ts_rank(self.low, 8), self.ts_rank(self.adv(60), 17), 5), 19), 16), 7))
    def alpha_100(self): return -1 * (1.5 * self.scale(self.indneutralize(self.indneutralize(self.rank((((self.close - self.low) - (self.high - self.close)) / ((self.high - self.low)+1e-6)) * self.volume)))) - self.scale(self.indneutralize((self.correlation(self.close, self.rank(self.adv(20)), 5) - self.rank(self.ts_argmin(self.close, 30)))))) * (self.volume / self.adv(20))

    def generate_all(self):
        print("Generating Alpha Features (1 to 101)...")
        alpha_dict = {}
        methods = [m for m in dir(self) if m.startswith('alpha_')]
        for m in methods:
            try:
                result = getattr(self, m)()
                
                # FIX: Automatically convert any raw numpy arrays back to Pandas DataFrames
                if isinstance(result, np.ndarray):
                    result = pd.DataFrame(result, index=self.close.index, columns=self.close.columns)
                    
                alpha_dict[m] = result
            except Exception as e:
                # Upgraded print statement to show the exact error if an alpha fails
                print(f"Skipping {m} due to error: {e}")
        
        features_df = pd.concat(alpha_dict, axis=1)
        features_df.columns.names = ["feature", "ticker"]
        return features_df

# ==========================================
# 2. Global Sector-Neutral Portfolio Generator
# ==========================================
def generate_global_sector_neutral_portfolio(
    entire_features: pd.DataFrame, universe: pd.DataFrame, 
    start_date: str, end_date: str, signal_column: str, 
    sector_mapping: pd.Series
):
    universe_boolean = universe.loc[start_date:end_date].astype(bool)
    signal1 = entire_features.xs(signal_column, axis=1, level=0).loc[start_date:end_date].shift(5)
    signal1 = signal1.where(universe_boolean, np.nan)
    signal1 = signal1.rank(axis=1, method="min", ascending=True)
    
    aligned_sectors = sector_mapping.reindex(signal1.columns).fillna('Unknown')
    signal1 = signal1.sub(signal1.groupby(aligned_sectors, axis=1).transform('mean'), axis=0)
    
    portfolio = -1 * signal1.fillna(0)
    return portfolio.div(portfolio.abs().sum(axis=1), axis=0).fillna(0)

# ==========================================
# 3. Parallelized Main Execution
# ==========================================
def process_feature(feature, features, universe, start_date, end_date, sector_mapping, returns_subset, sectors):
    """Worker function to process a single feature across all sectors"""
    portfolio = generate_global_sector_neutral_portfolio(
        features, universe, start_date, end_date, feature, sector_mapping
    )
    daily_stock_pnl = portfolio.shift(1) * returns_subset
    
    feature_results = []
    for sector in sectors:
        sector_tickers = sector_mapping[sector_mapping == sector].index.intersection(daily_stock_pnl.columns)
        if len(sector_tickers) == 0: continue
            
        sector_daily_pnl = daily_stock_pnl[sector_tickers].sum(axis=1)
        mean_pnl = sector_daily_pnl.mean()
        std_pnl = sector_daily_pnl.std()
        
        sharpe = np.sqrt(252) * (mean_pnl / std_pnl) if std_pnl != 0 and not np.isnan(std_pnl) else 0.0
        feature_results.append({"sector": sector, "feature": feature, "sharpe_ratio": sharpe})
        
    return feature_results

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
    
    wq_engine = WorldQuantAlphas(pv, returns, df_volume, df_vwap, sector_mapping)
    features = wq_engine.generate_all()

    start_date = "2025-12-01"
    end_date = "2026-06-01"
    sectors = sector_mapping.dropna().unique()
    features_list = features.columns.get_level_values(0).unique()
    returns_subset = returns.loc[start_date:end_date]

    print(f"\nRunning Parallel Sector-Attribution Backtest ({start_date} to {end_date})...")
    
    # Run backtests in parallel
    all_results = Parallel(n_jobs=-1)(
        delayed(process_feature)(
            feature, features, universe, start_date, end_date, sector_mapping, returns_subset, sectors
        ) for feature in features_list
    )
    
    results = [item for sublist in all_results for item in sublist]
    results_df = pd.DataFrame(results)
    
    # ----------------------------------------------------
    # EXPORT FULL RESULTS TO CSV
    # ----------------------------------------------------
    csv_path = "all_alpha_sector_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"SUCCESS: Full results matrix saved to {csv_path}")

    # Print Top Summaries to Terminal
    best_alphas_idx = results_df.groupby("sector")["sharpe_ratio"].idxmax()
    best_alphas = results_df.loc[best_alphas_idx].sort_values(by="sharpe_ratio", ascending=False)

    print("\n=== Top Alpha Formulas by Sector ===")
    print(best_alphas.to_string(index=False))
