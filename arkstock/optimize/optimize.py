'''
This is a module that defines functions that can be used to optimize the weights in a diversified portfolio
'''

import scipy.optimize as spo
from typing import Tuple, Dict, List
import numpy as np
_ndarray = np.ndarray
'''aliased type for code clarity'''
_nested_ndarray = _ndarray
'''aliased type for code clarity'''

from ..exceptions import exception_wrapper, PortfolioOptimizationException

import unittest

@exception_wrapper(PortfolioOptimizationException, 'Portfolio optimization failed')
def optimize_portfolio(cov_mat: _nested_ndarray, constraints: Tuple[List[Dict]]) -> _nested_ndarray:
    '''
    Computes the minimum variance portfolio.

    Args:

    * `cov_mat` - a covariance matrix of asset returns
    * `constraints` - a set of constraints that must be met when optimizing the portfolio

    Returns:

    > Optimal asset weights
    '''

    if not isinstance(cov_mat, np.ndarray):
        raise ValueError("Covariance matrix is not a numpy array")

    cov_matrix = np.copy(cov_mat)
    num_stocks = len(cov_matrix)

    def minimize_covariance(weights):
        weights = weights.reshape(1, -1)
        tmp = np.dot(cov_matrix, weights.T)
        tmp = np.dot(weights, tmp)
        return tmp

    initial_guess = np.ones((1, num_stocks)) / num_stocks

    res = spo.minimize(fun=minimize_covariance,
                       x0=initial_guess,
                       constraints=constraints,
                       bounds=[(0., 1.) for x in range(num_stocks)])

    return res.x

def build_constraints(allow_short: bool =False,
    expected_ret: _ndarray =None, target_ret: float =None) -> Tuple[List[Dict]]:
    '''
    If allow_short is true then it allows the portfolio to hold short positions.
    If only allow_short is provided then the min variance portfolio is generated

    If expected_ret and target_ret is provided then the markowitz portfolio is 
    generated

    Note: As the variance is not invariant with respect
    to leverage, it is not possible to construct non-trivial
    market neutral minimum variance portfolios. This is because
    the variance approaches zero with decreasing leverage,
    i.e. the market neutral portfolio with minimum variance
    is not invested at all.

    Args:

    * `allow_short` - If False construct a long-only portfolio. If True, then allow shorting (i.e.
        negative weights)
    * `expected_ret` - The expected returns for each stock, usually based on historical returns.
    * `target_ret` - If the `target_ret` is 1, then the portfolio does not make or loose any money

    Returns:

    > A list of constraints that should be passed into `optimize_portfolio`
    '''

    constraints = []
    if allow_short:
        constraints.append({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
    else:
        constraints.append({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
        constraints.append({'type': 'ineq',
                            'fun': lambda W: -1 if np.any(W[W < 0]) else 1})

    if expected_ret is not None and target_ret is not None:

        def match_target_ret(weight):
            eret = expected_ret.reshape(1, -1)
            w = weight.reshape(-1, 1)
            val = np.dot(eret, w) - target_ret
            return val[0][0]

        constraints.append({'type': 'ineq', 'fun': lambda W: match_target_ret(W)})

    return tuple(constraints)


class _OptimizeTest(unittest.TestCase):
    def test_min_var_portfolio(self):
        prices = np.random.rand(45, 100) * 40
        returns = (prices[:, :-1] - prices[:, 1:]) / prices[:, 1:]
        covariance = np.cov(returns)

        constraints = build_constraints()
        weights = optimize_portfolio(covariance, constraints)
        w = np.round(weights, decimals=8) * 100

        self.assertAlmostEqual(sum(w), 100, places=2)
        self.assertFalse(np.any(w[w<0]))

    def test_markowitz_portfolio(self):
        prices = np.random.rand(45, 100) * 40
        returns = (prices[:, :-1] - prices[:, 1:]) / prices[:, 1:]
        covariance = np.cov(returns)
        exp_ret = np.mean(returns, axis=1)
        target_ret = 0.1

        constraints = build_constraints(expected_ret=exp_ret, target_ret=target_ret)
        weights = optimize_portfolio(covariance, constraints)
        w = np.round(weights, decimals=8) * 100

        self.assertAlmostEqual(sum(w), 100, places=2)
        self.assertTrue(np.dot(w, exp_ret) > target_ret)

if __name__ == '__main__':
    unittest.main()
