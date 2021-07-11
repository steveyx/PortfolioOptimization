import yfinance as yf

tickers = ['MSFT', 'AAPL', 'TSLA', 'FB', 'BABA', 'NIO', 'GOOG', 'AMZN', 'NFLX']
for t in tickers:
    ticker = yf.Ticker(t)
    h = ticker.history(period="10y")
    h.to_csv("data/{}.csv".format(t))
