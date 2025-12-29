def run_backtest(df):
    df["ret"] = df["Close"].pct_change()
    df["strategy"] = df["signal"].shift(1) * df["ret"]
    equity = (1 + df["strategy"].fillna(0)).cumprod()
    return equity
