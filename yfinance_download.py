import argparse
from pathlib import Path
import time

import pandas as pd
import yfinance as yf

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download all price data (Open, High, Low, Close, Adj Close, Volume) for US tickers from Yahoo Finance."
    )
    # Changed default to look in the current working directory for ease of use
    parser.add_argument(
        "--input",
        type=str,
        default="top_5000_us_by_marketcap.csv",
        help="Path to CSV containing ticker symbols.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stores_created/all_prices_5000_tickers.parquet",
        help="Output Parquet file path.",
    )
    import datetime
    default_start = (datetime.date.today() - datetime.timedelta(days=545)).strftime('%Y-%m-%d')
    
    parser.add_argument(
        "--start",
        type=str,
        default=default_start,
        help="Start date for download (YYYY-MM-DD). Defaults to ~1.5 years ago.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of tickers to download per batch.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between batches to reduce rate limiting.",
    )
    return parser.parse_args()

def load_tickers(csv_path: str):
    path_obj = Path(csv_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Ticker CSV not found: {path_obj.resolve()}")
    
    # 1. Read the entire CSV (don't restrict usecols yet)
    df = pd.read_csv(path_obj)
    
    # 2. Normalize all column headers (strip whitespace and make lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    # 3. Find the right column name dynamically
    if 'symbol' in df.columns:
        ticker_col = 'symbol'
    elif 'ticker' in df.columns:
        ticker_col = 'ticker'
    else:
        # If it STILL fails, this will print the actual columns so you can debug the action logs
        raise ValueError(f"Could not find a ticker/symbol column. Available columns are: {df.columns.tolist()}")
    
    # 4. Clean the tickers: replace / and . with - for Yahoo Finance compatibility
    tickers = (
        df[ticker_col].astype(str)
        .str.strip()
        .replace("", pd.NA)
        .str.replace("/", "-", regex=False)  # Fixes BRK/A -> BRK-A
        .str.replace(".", "-", regex=False)  # Fixes BF.B -> BF-B
        .dropna()
        .unique()
        .tolist()
    )
    return tickers

def download_all_prices(tickers, start_date, batch_size, sleep_seconds):
    all_data = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"Downloading batch {i + 1}-{i + len(batch)} of {len(tickers)} tickers")

        try:
            data = yf.download(
                batch,
                start=start_date,
                auto_adjust=False,
                progress=False,
                threads=True,
            )

            if data.empty:
                print("Warning: No data returned for this batch. Skipping batch.")
                continue

            all_data.append(data)

        except Exception as exc:
            print(f"Batch failed: {exc}")

        time.sleep(sleep_seconds)

    if not all_data:
        raise RuntimeError("No price data was downloaded. Check your ticker list and Yahoo Finance access.")

    final_df = pd.concat(all_data, axis=1)
    # Remove any duplicate columns that may arise from yfinance issues
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    return final_df

def main():
    args = parse_args()
    
    tickers = load_tickers(args.input)
    print(f"Loaded {len(tickers)} tickers from {args.input}")

    final_df = download_all_prices(
        tickers,
        start_date=args.start,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_df = final_df.sort_index(axis=0)
    final_df.to_parquet(args.output)
    print(f"Saved all price data to {args.output}")

if __name__ == "__main__":
    main()
