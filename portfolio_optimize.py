import numpy as np
import pandas as pd
from porfolio_other_optimizations import maximize_sharpe_ratio_scipy
from porfolio_other_optimizations import maximize_sharpe_ratio_valentyn

np.random.seed(2)


class PortfolioOptimization:

    @staticmethod
    def load_stock_data(symbols):
        _df = pd.DataFrame()
        for i, stk in enumerate(symbols):
            _df_stk = pd.read_csv("data/{}.csv".format(stk), usecols=["Date", "Close"])
            _df_stk.rename(columns={"Close": stk}, inplace=True)
            if _df.empty:
                _df = _df_stk
            else:
                _df = _df.merge(_df_stk, on="Date", how="left")
        _df["Date"] = pd.to_datetime(_df["Date"])
        _df.set_index("Date", drop=True, inplace=True)
        return _df

    @staticmethod
    def optimize_portfolio_by_simulation(df_stocks, n_portfolios=2500):
        returns = df_stocks.pct_change()
        # calculate mean daily return and covariance of daily returns
        mean_daily_returns = returns.mean()
        cov_matrix = returns.cov()
        # set up array to hold results elements
        # 0: return, 1 volatility, 2 sharpe ratio, 3+ stock weights
        _n_stocks = len(df_stocks.columns)
        _results = np.zeros((3 + _n_stocks, n_portfolios))
        # store the initial weights generated for future visualization
        _initial_w = np.array(np.random.random(_n_stocks))
        # re-balance weights to sum to 1
        _initial_w /= np.sum(_initial_w)
        _best_weights = _initial_w.reshape((1, -1))
        _best_sharpe_ratio = 0
        # store the index of the latest best portfolio found during optimization
        _best_indices = [0]
        for i in range(n_portfolios):
            if i == 0:
                weights = _initial_w
            else:
                # select random weights for portfolio holdings
                weights = np.array(np.random.random(_n_stocks))
                # re-balance weights to sum to 1
                weights /= np.sum(weights)
            # calculate portfolio return and volatility
            portfolio_return = np.sum(mean_daily_returns * weights) * 252
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            # store results in results array
            _results[0, i] = portfolio_return
            _results[1, i] = portfolio_std_dev
            # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
            _results[2, i] = _results[0, i] / _results[1, i]
            # iterate through the weight vector and add data to results array
            for j in range(len(weights)):
                _results[j + 3, i] = weights[j]
            if i == 0:
                _best_weights = _results[:, i].reshape((1, -1))
                _best_sharpe_ratio = _results[2, i]
            elif _results[2, i] > _best_sharpe_ratio:
                _best_sharpe_ratio = _results[2, i]
                _best_weights = np.concatenate([_best_weights, _results[:, i].reshape((1, -1))])
                _best_indices.append(i)
        return _results, _initial_w, _best_indices

    @staticmethod
    def optimize_portfolio_gradient_descent(df_stocks, weights_init, delta=0.02):
        def cal_sharpe_ratio(mean_return, cov_mat, stock_weights):
            _r = np.sum(mean_return * stock_weights) * 252
            _std = np.sqrt(np.dot(stock_weights.T, np.dot(cov_mat, stock_weights))) * np.sqrt(252)
            _sha = _r / _std
            return _r, _std, _sha

        returns = df_stocks.pct_change()
        mean_ret = returns.mean()
        cov = returns.cov()
        n_stocks = len(df_stocks.columns)
        weights = weights_init
        _ret, _std_dev, _sharpe = cal_sharpe_ratio(mean_ret, cov, weights)
        _d = delta
        _cur_sharpe, _cur_weights = _sharpe, weights
        _results = [[_ret, _std_dev, _sharpe] + list(_cur_weights)]
        # set max search iterations 1000
        for iter_i in range(1000):
            _res = []
            for i in range(n_stocks):
                _w = _cur_weights.copy()
                _w[i] += _d
                _w /= np.sum(_w)
                _ret_i, _std_dev_i, _sharpe_i = cal_sharpe_ratio(mean_ret, cov, _w)
                _res.append([_ret_i, _std_dev_i, _sharpe_i])
            _res_sharpe_delta = np.array(_res)[:, 2] - _cur_sharpe
            _max_delta = _res_sharpe_delta.max()
            if _max_delta < 0.0001:
                # stop search if no improvement
                break
            _max = np.abs(_max_delta).max()
            _d_weights = _res_sharpe_delta / _max * _d if _max < _d else _res_sharpe_delta
            new_weights = _cur_weights + _d_weights
            new_weights[new_weights < 0] = 0.0
            new_weights /= np.sum(new_weights)
            _ret_i, _std_dev_i, _sharpe_i = cal_sharpe_ratio(mean_ret, cov, new_weights)
            if _sharpe_i > _cur_sharpe:
                _cur_sharpe, _cur_weights = _sharpe_i, new_weights
                _r_i = [_ret_i, _std_dev_i, _sharpe_i] + _cur_weights.tolist()
                _results.append(_r_i)
            else:
                # use smaller delta and search again
                _d = _d * 0.5
                if _d < delta * 0.1:
                    # stop search if increase is less than 10% of delta
                    break
        return np.array(_results)

    @staticmethod
    def optimize_portfolio_scipy(df_stocks):
        returns = df_stocks.pct_change()
        mean_ret = returns.mean()
        cov = returns.cov()
        n_stocks = len(df_stocks.columns)
        r = maximize_sharpe_ratio_scipy(mean_ret, cov, 0, n_stocks)
        return r


if __name__ == "__main__":
    # list of stocks in portfolio
    stocks = ['AAPL', 'TSLA', 'AMZN', 'MSFT', 'FB', 'GOOG']  #
    # convert daily stock prices into daily returns
    data = PortfolioOptimization.load_stock_data(stocks)
    # set number of runs of random portfolio weights
    num_portfolios = 36
    import time

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
    # portfolio_optimization_benchmark_animation(results_frame, best_indices, g_results,
    #                                            max_sharpe_port, min_vol_port, stocks,
    #                                            update_points=1, file_format="gif")
