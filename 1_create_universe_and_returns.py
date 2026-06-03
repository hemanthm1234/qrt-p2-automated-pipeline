# %% [markdown]
# Notebook to use Yahoo data for creating universe and returns

# %%
# Importing libraries
import pandas as pd
import requests
from io import BytesIO
import os

# %%
# Create stores folder if doens't exist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "stores_created")

os.makedirs(DATA_DIR, exist_ok=True)

# %%
pv = pd.read_parquet(os.path.join(BASE_DIR, "all_prices_5000_tickers.parquet"), engine="pyarrow")

# Think hard: need to generate for last 6 months, but need padding for rolling windows
last_date = pv.index.max()
six_months_ago = last_date - pd.DateOffset(months=6)
padding_date = six_months_ago - pd.DateOffset(months=2) # 2 months padding is > 20 trading days
pv = pv.loc[pv.index >= padding_date]

# %%
#Calcuate Average Daily Volume for trailing 20 days
df_daily_volume = pv['Close'].mul(pv['Volume']).fillna(0)
df_average_daily_volume = df_daily_volume.rolling(window=20, min_periods=1).mean()

# %%
#Creating 5M+ Average Daily Volume Universe
df_universe_5m = (df_average_daily_volume  > 5_000_000).astype(int)
# df_universe_5m.sum(axis=1).plot()

# %%
# Saving 5M universe to stores folder
df_universe_5m = df_universe_5m.loc[df_universe_5m.index >= six_months_ago]
df_universe_5m.to_parquet(os.path.join(DATA_DIR, "universe_5m.parquet"),engine="pyarrow")

# %%
#Creating 1M+ Average Daily Volume Universe
df_universe_1m = (df_average_daily_volume  > 1_000_000).astype(int)
# df_universe_1m.sum(axis=1).plot()

# %%
#Saving 1M universe to parquet file
df_universe_1m = df_universe_1m.loc[df_universe_1m.index >= six_months_ago]
df_universe_1m.to_parquet(os.path.join(DATA_DIR, "universe_1m.parquet"),engine="pyarrow")

# %%
#Calculating returns for each ticker every day
returns = pv['Adj Close'].pct_change(fill_method=None).fillna(0)

# %%
#Saving returns to parquet file
returns = returns.loc[returns.index >= six_months_ago]
returns.to_parquet(os.path.join(DATA_DIR, "returns.parquet"),engine="pyarrow")


