# function to undertake Sharpe Ratio maximization subject to
# basic constraints of the portfolio

# Solver uses optimize library function from scipy that employs
# SLSQP  (Sequential Leas Squares Quadratic Programming) method to solve the
# non-linear single objective constrained optimization problem

import numpy as np
from scipy import optimize


def maximize_sharpe_ratio_scipy(mean_returns, covar_returns, risk_free_rate, n_portfolios):
    # define maximization of Sharpe Ratio using principle of duality
    def f_sr(w, mean_ret, cov_ret, risk_free):
        _std_dev = np.sqrt(np.matmul(np.matmul(w, cov_ret), w.T))
        _ret = np.matmul(np.array(mean_ret), w.T) - risk_free
        func = - np.sqrt(252) * (_ret / _std_dev)
        return func

    mean_returns = mean_returns.values
    covar_returns = covar_returns.values

    # define equality constraint representing fully invested portfolio
    def constraint_equation(w):
        array_a = np.ones(w.shape)
        b = 1
        _v = np.matmul(array_a, w.T) - b
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
    def f_sr(w, ret, cov, risk_free):
        _std_dev = np.sqrt(np.matmul(np.matmul(w, cov), w.T))
        _ret = np.matmul(np.array(ret), w.T) - risk_free
        _sr = np.sqrt(252) * (_ret / _std_dev)
        return _sr

    def f_rt(w, ret, cov, risk_free):
        # calculate risk tolerance rt
        _co_var = np.matmul(np.matmul(w, cov), w.T)
        _ret = np.matmul(np.array(ret), w.T) - risk_free
        _rt = 2 * _co_var / _ret
        return _ret, _co_var, _rt,

    mean_ret = mean_returns.values
    cov_ret = covar_returns.values
    # x = np.repeat(1/n_portfolios, n_portfolios)
    x = np.random.random(n_portfolios)
    x = x/sum(x)
    for k in range(100):
        r, c, rt = f_rt(x, mean_ret, cov_ret, risk_free_rate)
        # calculate marginal utilities, partial derivatives of the objective function
        mu = mean_ret - (2 / rt) * np.matmul(cov_ret, x.reshape(-1, 1)).flatten()
        _i_add, _mu_add = np.argmax(mu), np.max(mu)
        _mu_p = mu[mu > 0]
        if len(_mu_p) > 0:
            _mu_sub = np.min(_mu_p)
            _i_sub = np.argwhere(mu == _mu_sub)[0, 0]
        else:
            break
        _delta = _mu_add - _mu_sub
        print("iteration {}: rt {:.6f}, delta mu {:.6f}, mu_add {:.6f}, mu_sub {:.6f}, weights {}".format(
            k, rt, _mu_add, _mu_sub, _delta, x))
        if _delta < 0.00001:
            break
        _dw0 = _delta / (2 * np.abs(rt * (cov_ret[_i_add, _i_add] + cov_ret[_i_sub, _i_sub] - cov_ret[_i_add, _i_sub])))
        _dw1, _dw2 = 1 - x[_i_add], x[_i_sub] - 0
        _dw = min(_dw0, _dw1, _dw2)
        x[_i_add] += _dw
        x[_i_sub] -= _dw
    sr = f_sr(x, mean_ret, cov_ret, risk_free_rate)
    print("valentyn method: SR {} \nweights {}".format(sr, x))
    return x