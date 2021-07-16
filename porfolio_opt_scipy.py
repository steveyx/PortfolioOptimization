# function to undertake Sharpe Ratio maximization subject to
# basic constraints of the portfolio

# Solver uses optimize library function from scipy that employs
# SLSQP  (Sequential Leas Squares Quadratic Programming) method to solve the
# non-linear single objective constrained optimization problem

# dependencies
import numpy as np
from scipy import optimize


def maximize_sharpe_ratio_scipy(mean_returns, covar_returns, risk_free_rate, n_portfolios):
    # define maximization of Sharpe Ratio using principle of duality
    def f_sr(x, mean_ret, cov_ret, risk_free):
        _std_dev = np.sqrt(np.matmul(np.matmul(x, cov_ret), x.T))
        _ret = np.matmul(np.array(mean_ret), x.T) - risk_free
        func = - np.sqrt(252) * (_ret / _std_dev)
        return func

    mean_returns = mean_returns.values
    covar_returns = covar_returns.values

    # define equality constraint representing fully invested portfolio
    def constraint_equation(x):
        array_a = np.ones(x.shape)
        b = 1
        _v = np.matmul(array_a, x.T) - b
        return _v

    # define bounds and other parameters
    x_init = np.repeat(1/n_portfolios, n_portfolios)
    cons = ({'type': 'eq', 'fun': constraint_equation})
    lb, ub = 0.0, 1.0
    _bounds = tuple([(lb, ub) for _ in range(n_portfolios)])

    # invoke minimize solver
    opt = optimize.minimize(f_sr, x0=x_init, args=(mean_returns, covar_returns, risk_free_rate),
                            method='SLSQP', bounds=_bounds, constraints=cons, tol=1.e-16)
    return opt


def maximize_sharpe_ratio_valentyn(mean_returns, covar_returns, risk_free_rate, n_portfolios):
    # define maximization of Sharpe Ratio using principle of duality
    def f_sr(x, mean_ret, cov_ret, risk_free):
        _std_dev = np.sqrt(np.matmul(np.matmul(x, cov_ret), x.T))
        _ret = np.matmul(np.array(mean_ret), x.T) - risk_free
        func = - np.sqrt(252) * (_ret / _std_dev)
        return func

    def f_r_t(x, mean_ret, cov_ret, risk_free):
        _co_var = np.matmul(np.matmul(x, cov_ret), x.T)
        _ret = np.matmul(np.array(mean_ret), x.T) - risk_free
        _rt = 2 * (_co_var / _ret)
        return _rt

    def mu_i(i, x, rt, mean_ret, cov_ret):
        v = mean_ret[i] - (2 / rt) * np.matmul(x.reshape(1, -1), cov_ret[i, :].reshape(-1, 1))
        v_m = mean_ret - (2 / rt) * np.matmul(x.reshape(1, -1), cov_ret).flatten()
        _i_add, _i_sub = np.argmax(v_m), np.argmin(v_m)
        _mu_add, _mu_sub = v_m[_i_add], v_m[_i_sub]
        _delta = _mu_add - _mu_sub
        _dw0 = _delta / 2 / np.abs(-cov_ret[_i_add, _i_sub])
        _dw1, _dw2 = 1 - x[_i_add], x[_i_sub] - 0
        _dw = min(_dw0, _dw1, _dw2)
        x[_i_add] += _dw
        x[_i_sub] -= _dw
        return _delta

    mean_returns = mean_returns.values
    covar_returns = covar_returns.values

    # define equality constraint representing fully invested portfolio
    def constraint_equation(x):
        array_a = np.ones(x.shape)
        b = 1
        _v = np.matmul(array_a, x.T) - b
        return _v

    # define bounds and other parameters
    x_init = np.repeat(1/n_portfolios, n_portfolios)
    cons = ({'type': 'eq', 'fun': constraint_equation})
    lb, ub = 0.0, 1.0
    _bounds = tuple([(lb, ub) for _ in range(n_portfolios)])

    # invoke minimize solver
    opt = optimize.minimize(f_sr, x0=x_init, args=(mean_returns, covar_returns, risk_free_rate),
                            method='SLSQP', bounds=_bounds, constraints=cons, tol=1.e-16)
    return opt