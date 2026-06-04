import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration & Weights
# ==========================================
coeff_dict = {
    "relative_strength_index": -1.0, "williams_r": -1.0, "rsi": -1.0,
    "volatility_20": 1.0, "volatility_60": 1.0, "trend_1_3": 1.0,
    "trend_5_20": 1.0, "trend_20_60": -1.0, "average_true_range": -1.0,
    "macd": 1.0, "macd_signal": 1.0, "macd_histogram": -1.0,
    "trix": 1.0, "commodity_channel_index": 7.3160,
    "chande_momentum_oscillator": -1.0, "ichimoku_conversion": -1.0,
    "ichimoku_base": -1.0, "ichimoku_leading_a": -1.0,
    "ichimoku_leading_b": -1.0, "know_sure_thing": 1.0,
    "ultimate_oscillator": -1.0, "aroon_up": -1.0, "aroon_down": 1.0,
    "aroon_oscillator": -1.0, "stochastic_k": -1.0, "stochastic_d": -1.0,
    "on_balance_volume": 1.0, "ease_of_movement": 1.0,
    "chaikin_money_flow": 1.1280, "accumulation_distribution_index": -1.0,
    "volume": -8.2977
}

best_sector_alphas = {
    'Technology': 'alpha_016', 'Health Care': 'alpha_035',
    'Telecommunications': 'alpha_067', 'Consumer Discretionary': 'alpha_006',
    'Real Estate': 'alpha_029', 'Finance': 'alpha_075',
    'Utilities': 'alpha_001', 'Energy': 'alpha_058',
    'Consumer Staples': 'alpha_101', 'Industrials': 'alpha_100',
    'Basic Materials': 'alpha_035', 'Miscellaneous': 'alpha_023'
}

start_date = "2025-12-01"
end_date = "2026-06-01"

# ==========================================
# 2. Math Helpers
# ==========================================
def rank_neutralize_scale(signal_df, universe_bool, sector_mapping):
    """Ranks, sector-neutralizes, and scales to unit capital over a time series"""
    signal = signal_df.where(universe_bool, np.nan)
    signal = signal.rank(axis=1, method="min", ascending=True)
    
    aligned_sectors = sector_mapping.reindex(signal.columns).fillna('Unknown')
    signal = signal.sub(signal.groupby(aligned_sectors, axis=1).transform('mean'), axis=0)
    
    portfolio = -1 * signal.fillna(0)
    return portfolio.div(portfolio.abs().sum(axis=1), axis=0).fillna(0)

def orthogonalize_timeseries(alpha_df, beta_df):
    """Cross-Sectional OLS applied row-by-row to prevent look-ahead bias"""
    pure_alpha = pd.DataFrame(np.nan, index=alpha_df.index, columns=alpha_df.columns)
    
    for date in alpha_df.index:
        x = beta_df.loc[date].values
        y = alpha_df.loc[date].values
        
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() > 2:
            cov_matrix = np.cov(x[mask], y[mask])
            beta_coef = cov_matrix[0, 1] / cov_matrix[0, 0] if cov_matrix[0, 0] > 1e-8 else 0.0
            pure_alpha.loc[date] = y - (beta_coef * x)
        else:
            pure_alpha.loc[date] = y
            
    return pure_alpha

def calculate_metrics(daily_pnl_series):
    total_pnl = daily_pnl_series.sum()
    variance = daily_pnl_series.var()
    mean_pnl = daily_pnl_series.mean()
    std_pnl = daily_pnl_series.std()
    gross_sharpe = np.sqrt(252) * (mean_pnl / std_pnl) if std_pnl != 0 and not np.isnan(std_pnl) else 0.0
    
    cumulative_pnl = daily_pnl_series.cumsum()
    rolling_max = cumulative_pnl.cummax()
    max_drawdown = (cumulative_pnl - rolling_max).min()
    
    return total_pnl, variance, gross_sharpe, max_drawdown

# ==========================================
# 3. Execution
# ==========================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "stores_created")
    
    print("Loading Data (Vectorized)...")
    universe = pd.read_parquet(os.path.join(DATA_DIR, "universe_5m.parquet"))
    returns = pd.read_parquet(os.path.join(DATA_DIR, "returns.parquet"))
    sector_mapping = pd.read_csv(os.path.join(BASE_DIR, "top_5000_us_by_marketcap.csv")).set_index("symbol")["sector"]
    
    technical_indicators = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))
    wq_features = pd.read_parquet(os.path.join(DATA_DIR, "wq_features.parquet"))
    
    universe_window = universe.loc[start_date:end_date].astype(bool)
    returns_window = returns.loc[start_date:end_date]

    # --- Construct Beta ---
    print("Constructing Beta Signal...")
    master_beta = pd.DataFrame(0.0, index=universe_window.index, columns=universe_window.columns)
    
    for feature_name, weight in coeff_dict.items():
        if feature_name in technical_indicators.columns.get_level_values(0):
            indicator_data = technical_indicators.xs(feature_name, axis=1, level=0).loc[start_date:end_date].shift(5)
            ranked_indicator = indicator_data.rank(axis=1, pct=True) * weight
            master_beta = master_beta.add(ranked_indicator, fill_value=0)

    beta_portfolio = rank_neutralize_scale(master_beta, universe_window, sector_mapping)

    # --- Construct Alpha ---
    print("Stitching WorldQuant Alphas...")
    master_alpha = pd.DataFrame(0.0, index=universe_window.index, columns=universe_window.columns)
    
    for sector, best_alpha in best_sector_alphas.items():
        if best_alpha in wq_features.columns.get_level_values(0):
            alpha_data = wq_features.xs(best_alpha, axis=1, level=0).loc[start_date:end_date].shift(5)
            sector_mask = (sector_mapping == sector).reindex(universe_window.columns).fillna(False)
            master_alpha = master_alpha.add(alpha_data.where(sector_mask, 0.0), fill_value=0)

    alpha_portfolio = rank_neutralize_scale(master_alpha, universe_window, sector_mapping)

    # --- Orthogonalize and Blend ---
    print("Orthogonalizing Signals (Cross-Sectional OLS)...")
    pure_alpha = orthogonalize_timeseries(alpha_portfolio, beta_portfolio)
    pure_alpha_portfolio = rank_neutralize_scale(pure_alpha, universe_window, sector_mapping)
    
    print("Blending Signals (50/50 Scaled)...")
    # Equal 50/50 split optimally maximizes Sharpe when correlation is strictly 0
    combined_signal = (0.50 * beta_portfolio) + (0.50 * pure_alpha_portfolio)
    final_portfolio = rank_neutralize_scale(combined_signal, universe_window, sector_mapping)
    
    daily_stock_pnl = final_portfolio.shift(1) * returns_window
    
    results = []
    sectors = sector_mapping.dropna().unique()
    
    for sector in sectors:
        sector_tickers = sector_mapping[sector_mapping == sector].index.intersection(daily_stock_pnl.columns)
        if len(sector_tickers) == 0: continue
            
        sector_daily_pnl = daily_stock_pnl[sector_tickers].sum(axis=1)
        tot_pnl, var, sharpe, max_dd = calculate_metrics(sector_daily_pnl)
        results.append({"Entity": sector, "Total PnL": tot_pnl, "Variance": var, "Max Drawdown": max_dd, "Gross Sharpe": sharpe})
        
    overall_daily_pnl = daily_stock_pnl.sum(axis=1)
    tot_pnl, var, sharpe, max_dd = calculate_metrics(overall_daily_pnl)
    results.append({"Entity": "OVERALL PORTFOLIO", "Total PnL": tot_pnl, "Variance": var, "Max Drawdown": max_dd, "Gross Sharpe": sharpe})
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print(" "*25 + "ORTHOGONALIZED BLEND RISK ATTRIBUTION")
    print("="*80)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("="*80)

    import plotly.express as px
    results_df.to_csv("blended_strategy_stats.csv", index=False)
    
    cumulative_pnl = overall_daily_pnl.cumsum()
    cumulative_pnl.name = "Cumulative PnL"
    fig = px.line(cumulative_pnl, title="Orthogonal Blended Strategy (Alpha ⊥ Beta) Cumulative PnL", template="plotly_dark")
    fig.write_html("blended_cumulative_pnl.html")
    
    print("\nSUCCESS: Backtest completed instantly.")
