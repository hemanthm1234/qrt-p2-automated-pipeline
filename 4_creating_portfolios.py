# %%
import pandas as pd
import numpy as np
import os

# %%
BASE_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(BASE_DIR, "stores")

# %%
### CODE FOR WORKSHOP ONLY ####
url_selected_signals = "https://github.com/yszanwar/phase2_qrt_challenge/releases/download/signals/selected_signals.parquet"
selected_signals = pd.read_parquet(url_selected_signals, engine="pyarrow")

# %%
#Loading selected signals
selected_signals = pd.read_parquet(os.path.join(DATA_DIR, "selected_signals.parquet"))

# %%
selected_signals.corr()

# %%
# Sharpe for selected signals
sharpe_ratios = selected_signals.mean() / selected_signals.std() * (252 ** 0.5)
sharpe_ratios.sort_values(ascending=False)

# %%
# Creating equal weight portfolio based on a single signal, with optional sector neutralization
equal_weight_portfolio = selected_signals.sum(axis=1)

# %%
#Equal weight sharpe
print("Equal Weight Portfolio Sharpe Ratio:", np.round(equal_weight_portfolio.mean() / equal_weight_portfolio.std() * (252 ** 0.5), 2))
# equal_weight_portfolio.cumsum().plot(title="Equal Weight Portfolio Cumulative Returns")

# %%
#Equal Vol Portfolio
vols = selected_signals.std()
inv_vols = 1 / vols
weights = inv_vols / inv_vols.sum()
equal_vol_portfolio = (selected_signals * weights).sum(axis=1)

# %%
#Equal vol sharpe
print("Equal Vol Portfolio Sharpe Ratio:", np.round(equal_vol_portfolio.mean() / equal_vol_portfolio.std() * (252 ** 0.5), 2))
# equal_vol_portfolio.cumsum().plot(title="Equal Vol Portfolio Cumulative Returns")

# %%
cov = selected_signals.cov()
inv_cov = np.linalg.inv(cov)
mu = selected_signals.mean()

w = inv_cov @ mu

# project to long-only ## Pay Atttention 
w = np.clip(w, 0, None)
mv_weights = w / w.sum()

# %%
mv_weights

# %%
mv_portfolio = (selected_signals * mv_weights).sum(axis=1)

# %%
#Mean Variance Optimized Sharpe
print("Mean Variance Portfolio Sharpe Ratio:", np.round(mv_portfolio.mean() / mv_portfolio.std() * (252 ** 0.5), 2))
# mv_portfolio.cumsum().plot(title="Mean Variance Portfolio Cumulative Returns")

# %%



