import os
import pandas as pd
import warnings
import importlib.util

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # or '/kaggle/working/qrt-p2-automated-pipeline'
DATA_DIR = os.path.join(BASE_DIR, "stores_created")

# Dynamically import the engine
wq_path = os.path.join(BASE_DIR, "5_test_worldquant_alphas.py")
spec = importlib.util.spec_from_file_location("wq_module", wq_path)
wq_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wq_module)
WorldQuantAlphas = wq_module.WorldQuantAlphas

print("Loading Pricing Data...")
pv = pd.read_parquet(os.path.join(BASE_DIR, "all_prices_5000_tickers.parquet"), engine="pyarrow")
returns = pd.read_parquet(os.path.join(DATA_DIR, "returns.parquet"))
sector_mapping = pd.read_csv(os.path.join(BASE_DIR, "top_5000_us_by_marketcap.csv")).set_index("symbol")["sector"]

print("Generating WorldQuant Alphas (This will take a few minutes)...")
df_volume = pv['Volume']
df_vwap = (pv['High'] + pv['Low'] + pv['Adj Close']) / 3

wq_engine = WorldQuantAlphas(pv, returns, df_volume, df_vwap, sector_mapping)
wq_features = wq_engine.generate_all()

print("Saving WQ Features to disk...")
wq_features = wq_features.astype("float32") # Downcast for memory
wq_features.to_parquet(os.path.join(DATA_DIR, "wq_features.parquet"), compression="zstd")
print("SUCCESS: wq_features.parquet saved!")
