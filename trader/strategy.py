def sma_strategy(df, fast=20, slow=50):
    df = df.copy()
    df["fast"] = df["Close"].rolling(fast).mean()
    df["slow"] = df["Close"].rolling(slow).mean()
    df["signal"] = (df["fast"] > df["slow"]).astype(int)
    return df
