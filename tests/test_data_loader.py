import pandas as pd

import yfinance as yf

from trader.data.loaders import DataLoader, DataRequest


def test_fetch_flattens_multiindex_columns(monkeypatch, tmp_path):
    loader = DataLoader(cache_dir=tmp_path)
    request = DataRequest(symbol="SPY", start="2023-01-01", end="2023-01-10")

    def fake_download(symbol, start, end, progress, auto_adjust):
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        columns = pd.MultiIndex.from_product(
            [[symbol], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        )
        data = pd.DataFrame(
            [
                [1.0, 2.0, 0.5, 1.5, 1.4, 1_000],
                [1.2, 2.2, 0.7, 1.7, 1.6, 1_100],
            ],
            index=dates,
            columns=columns,
        )
        return data

    monkeypatch.setattr(yf, "download", fake_download)

    df = loader.fetch(request)

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df["Close"].iloc[0] == 1.4
    assert not df.isna().any().any()
