import os
import pandas as pd
import numpy as np
import warnings

# Import the robust engine we built in the previous step
from 5_test_worldquant_alphas import WorldQuantAlphas 

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
    'Technology': 'alpha_016',
    'Health Care': 'alpha_035',
    'Telecommunications': 'alpha_067',
    'Consumer Discretionary': 'alpha_006',
    'Real Estate': 'alpha_029',
    'Finance': 'alpha_075',
    'Utilities': 'alpha_001',
    'Energy': 'alpha_058',
    'Consumer Staples': 'alpha_101',
    'Industrials': 'alpha_075',
    'Basic Materials': 'alpha_035',
    'Miscellaneous': 'alpha_023'
}

BETA_WEIGHT = 0.40
ALPHA_WEIGHT = 0.60

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

def calculate_metrics(daily_pnl_series):
    """Calculates risk and return metrics for a daily PnL stream"""
    total_pnl = daily_pnl_series.sum()
    variance = daily_pnl_series.var()
    
    # Gross Sharpe (No transaction costs assumed)
    mean_pnl = daily_pnl_series.mean()
    std_pnl = daily_pnl_series.std()
    gross_sharpe = np.sqrt(252) * (mean_pnl / std_pnl) if std_pnl != 0 and not np.isnan(std_pnl) else 0.0
    
    # Max Drawdown (Additive logic for Long/Short market neutral)
    cumulative_pnl = daily_pnl_series.cumsum()
    rolling_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - rolling_max
    max_drawdown = drawdown.min()
    
    return total_pnl, variance, gross_sharpe, max_drawdown

# ==========================================
# 3. Execution
# ==========================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "stores_created")
    
    print("Loading Data...")
    pv = pd.read_parquet(os.path.join(BASE_DIR, "all_prices_5000_tickers.parquet"), engine="pyarrow")
    universe = pd.read_parquet(os.path.join(DATA_DIR, "universe_5m.parquet"))
    returns = pd.read_parquet(os.path.join(DATA_DIR, "returns.parquet"))
    sector_mapping = pd.read_csv(os.path.join(BASE_DIR, "top_5000_us_by_marketcap.csv")).set_index("symbol")["sector"]
    
    technical_indicators = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))
    
    # Restrict to backtest window
    universe_window = universe.loc[start_date:end_date].astype(bool)
    returns_window = returns.loc[start_date:end_date]

    # --- Construct Beta (T-5 lagged for execution delay) ---
    print("Constructing Historical Beta Signal...")
    master_beta = pd.DataFrame(0.0, index=universe_window.index, columns=universe_window.columns)
    
    for feature_name, weight in coeff_dict.items():
        try:
            indicator_data = technical_indicators.xs(feature_name, axis=1, level=0).loc[start_date:end_date].shift(5)
            ranked_indicator = indicator_data.rank(axis=1, pct=True) * weight
            master_beta = master_beta.add(ranked_indicator, fill_value=0)
        except KeyError:
            continue

    beta_portfolio = rank_neutralize_scale(master_beta, universe_window, sector_mapping)

    # --- Construct Alpha (T-5 lagged automatically inside engine structure) ---
    print("Generating Historical WorldQuant Alphas...")
    df_volume = pv['Volume']
    df_vwap = (pv['High'] + pv['Low'] + pv['Adj Close']) / 3
    
    wq_engine = WorldQuantAlphas(pv, returns, df_volume, df_vwap, sector_mapping)
    wq_features = wq_engine.generate_all()
    
    print("Stitching Sector-Specific Alphas...")
    master_alpha = pd.DataFrame(0.0, index=universe_window.index, columns=universe_window.columns)
    
    for sector, best_alpha in best_sector_alphas.items():
        alpha_data = wq_features.xs(best_alpha, axis=1, level=0).loc[start_date:end_date].shift(5)
        sector_mask = (sector_mapping == sector).reindex(universe_window.columns).fillna(False)
        master_alpha = master_alpha.add(alpha_data.where(sector_mask, 0.0), fill_value=0)

    alpha_portfolio = rank_neutralize_scale(master_alpha, universe_window, sector_mapping)

    # --- Blend and Test ---
    print("Blending Signals and Running Risk Attribution...")
    combined_signal = (BETA_WEIGHT * beta_portfolio) + (ALPHA_WEIGHT * alpha_portfolio)
    final_portfolio = rank_neutralize_scale(combined_signal, universe_window, sector_mapping)
    
    # Daily Stock PnL matrix (Assume 1.0 Total Book size)
    daily_stock_pnl = final_portfolio.shift(1) * returns_window
    
    results = []
    sectors = sector_mapping.dropna().unique()
    
    # Sector Attribution
    for sector in sectors:
        sector_tickers = sector_mapping[sector_mapping == sector].index.intersection(daily_stock_pnl.columns)
        if len(sector_tickers) == 0: continue
            
        sector_daily_pnl = daily_stock_pnl[sector_tickers].sum(axis=1)
        
        tot_pnl, var, sharpe, max_dd = calculate_metrics(sector_daily_pnl)
        results.append({
            "Entity": sector,
            "Total PnL": tot_pnl,
            "Variance": var,
            "Max Drawdown": max_dd,
            "Gross Sharpe": sharpe
        })
        
  # Overall Portfolio Attribution
    overall_daily_pnl = daily_stock_pnl.sum(axis=1)
    tot_pnl, var, sharpe, max_dd = calculate_metrics(overall_daily_pnl)
    results.append({
        "Entity": "OVERALL PORTFOLIO",
        "Total PnL": tot_pnl,
        "Variance": var,
        "Max Drawdown": max_dd,
        "Gross Sharpe": sharpe
    })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print(" "*25 + "BLENDED STRATEGY RISK ATTRIBUTION")
    print("="*80)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("="*80)

    # ----------------------------------------------------
    # EXPORT RESULTS & GENERATE GRAPH
    # ----------------------------------------------------
    import plotly.express as px
    
    # Save the Stats Matrix
    csv_path = "blended_strategy_stats.csv"
    results_df.to_csv(csv_path, index=False)
    
    # Calculate Cumulative PnL and Generate Interactive Graph
    cumulative_pnl = overall_daily_pnl.cumsum()
    cumulative_pnl.name = "Cumulative PnL"
    
    fig = px.line(
        cumulative_pnl, 
        title="Blended Strategy (Alpha + Beta) Cumulative PnL",
        labels={'value': 'Total Notional PnL', 'index': 'Date'},
        template="plotly_dark"
    )
    
    # Save the graph as an interactive HTML file
    html_path = "blended_cumulative_pnl.html"
    fig.write_html(html_path)
    
    print(f"\nSUCCESS: Stats saved to {csv_path}")
    print(f"SUCCESS: Interactive PnL Graph saved to {html_path}")
