import pandas as pd
from portfolio_optimize import PortfolioOptimization
from portfolio_animation import portfolio_optimization_benchmark_animation
import time


if __name__ == "__main__":
    # list of stocks in portfolio
    stocks = ['AAPL', 'TSLA', 'AMZN', 'MSFT', 'FB', 'GOOG']  #
    # convert daily stock prices into daily returns
    data = PortfolioOptimization.load_stock_data(stocks)
    # set number of runs of random portfolio weights
    num_portfolios = 200
    t0 = time.time()
    results, initial_weights, best_indices = PortfolioOptimization.optimize_portfolio_by_simulation(
        df_stocks=data, n_portfolios=num_portfolios)
    t1 = time.time()
    g_results = PortfolioOptimization.optimize_portfolio_gradient_descent(
        data, initial_weights, delta=0.05)
    t2 = time.time()
    print("simulation cal time {}".format(t1 - t0))
    print("gradient descent cal time {}".format(t2 - t1))
    print(g_results[-1, :])

    # convert results array to Pandas DataFrame
    cols = ['ret', 'stdev', 'sharpe'] + stocks
    results_frame = pd.DataFrame(results.T, columns=cols)
    # locate position of portfolio with highest Sharpe Ratio
    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
    print(max_sharpe_port)
    # locate position of portfolio with minimum standard deviation
    min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
    # portfolio_optimization_animation(results_frame, max_sharpe_port, min_vol_port, stocks, update_points=10)
    portfolio_optimization_benchmark_animation(results_frame, best_indices, g_results,
                                               max_sharpe_port, min_vol_port, stocks,
                                               update_points=1, file_format="gif")
