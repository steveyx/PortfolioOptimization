import pandas as pd
from portfolio_optimize import PortfolioOptimization
from portfolio_visualize import PortfolioVisualize
import numpy as np
import time
from yfinance_download import get_nastaq_symbols
np.random.seed(2)


if __name__ == "__main__":
    # list of stocks in portfolio
    stocks = ['AAPL', 'TSLA', 'AMZN', 'MSFT', 'FB', 'GOOG']
    # or to download top stock symbols as belows
    # stocks = get_nastaq_symbols(n=100)

    # convert daily stock prices into daily returns
    data = PortfolioOptimization.load_stock_data(stocks)
    # set number of runs of random portfolio weights
    num_portfolios = 1000
    t0 = time.time()
    results, initial_weights, best_indices = PortfolioOptimization.optimize_portfolio_by_simulation(
        df_stocks=data, n_portfolios=num_portfolios)
    t1 = time.time()
    g_results = PortfolioOptimization.optimize_portfolio_gradient_descent(
        data, initial_weights, delta=0.05)
    t2 = time.time()
    sr_sci, weights_sci = PortfolioOptimization.optimize_portfolio_scipy(data)
    t3 = time.time()
    print("simulation cpu time {}".format(t1 - t0))
    print("gradient descent cpu time {}".format(t2 - t1))
    print("scipy cpu time {}".format(t3 - t2))
    print("best sharpe ratio by gradient descent: {}, weights {}".format(g_results[-1, 2], g_results[-1, 3:]))
    print("best sharpe ratio by scipy: {}, weights {}".format(sr_sci, weights_sci))

    # convert results array to Pandas DataFrame
    cols = ['ret', 'stdev', 'sharpe'] + stocks
    results_frame = pd.DataFrame(results.T, columns=cols)
    # locate position of portfolio with highest Sharpe Ratio
    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
    print("best sharpe ratio by simulation: ", max_sharpe_port["sharpe"])
    # locate position of portfolio with minimum standard deviation
    min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
    PortfolioVisualize.visualize(results_frame, best_indices, g_results, max_sharpe_port, stocks)
    PortfolioVisualize.visualize_simulation(results_frame, max_sharpe_port, min_vol_port)
    PortfolioVisualize.plot_benchmark_table()
