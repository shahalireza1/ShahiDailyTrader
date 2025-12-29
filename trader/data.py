import yfinance as yf

def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise RuntimeError("No data downloaded")
    return df

