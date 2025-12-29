from trader.data import load_data
from trader.strategy import sma_strategy
from trader.backtest import run_backtest

def main():
    df = load_data("SPY", "2020-01-01", "2024-01-01")
    df = sma_strategy(df)
    equity = run_backtest(df)
    print("Final equity:", equity.iloc[-1])

if __name__ == "__main__":
    main()
