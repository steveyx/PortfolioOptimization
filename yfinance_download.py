import pandas as pd
import yfinance as yf


def get_nastaq_symbols(n=100):
    # get top companies (stock symbols) from nasdaq data
    year = 2015
    df_symbols = pd.read_csv("data/nasdaq_symbols.csv")
    filter1 = df_symbols["IPO Year"] <= year
    filter2 = df_symbols["Symbol"].str.find("^") < 0
    df_symbols = df_symbols[filter1 & filter2].sort_values(by="Market Cap", ascending=False)
    print("total number of stocks with IPO earlier than year {}: {}".format(year, len(df_symbols)))
    return df_symbols['Symbol'].iloc[:n].tolist()


if __name__ == "__main__":
    # to more stock symbol from nasdaq
    # tickers = get_nastaq_symbols(n=100)
    # select a few symbols manually
    tickers = ['MSFT', 'AAPL', 'TSLA', 'FB', 'BABA', 'NIO', 'GOOG', 'AMZN', 'NFLX']
    for t in tickers:
        ticker = yf.Ticker(t)
        h = ticker.history(period="10y")
        h.to_csv("data/{}.csv".format(t))
