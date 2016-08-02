'''
This is a module that defines functions that can be used to diversify a portfolio given a set of
required stocks and the much larger set of viable stocks.
'''

from collections import namedtuple
from typing import List, Dict, Tuple
import numpy as np
_ndarray = np.ndarray
'''type alias for improved code clarity'''
_nested_ndarray = _ndarray
'''type alias for improved code clarity'''

from . import helper

import unittest

StockInfo = namedtuple('StockInfo', ['name', 'hist_prices', 'hist_returns'])
'''immutable container for information regarding a stock that could later be used to calculate weights'''

def diversify_portfolio(all_stock_names: List[str], all_stock_prices: _nested_ndarray,
    required_stocks: List[str], min_portfolio_size: int =30) -> List[StockInfo]:
    '''
    Given a a large data set containing all the required stock information, and a set of required stocks
    that must be in portfolio, this method will return a list of other stocks that, together with the
    required stocks, form a well diversified portfolio.

    Args:

    * `all_stock_names` - the ticker symbols of all the available stocks to choose from
    * `all_stock_prices` - the historical prices for all the available stocks to choose from. The latest
        price is at index 0.
    * `required_stocks` - the ticker symbols of the required stocks that must be in the portfolio
    * `min_portfolio_size` - the minimum size of the resultant portfolio

    Returns:

    > A list of stocks that would work well together in a portfolio.
    '''

    valid_names, valid_prices, valid_returns, valid_self_corr = get_valid_data(all_stock_names,
        all_stock_prices)

    stocks_in_port = get_stocks_in_portfolio(valid_names, valid_prices, valid_self_corr,
        required_stocks, min_portfolio_size)

    portfolio_stock_indexes = [valid_names.index(x) for x in stocks_in_port]
    price_data = np.array([valid_prices[i] for i in portfolio_stock_indexes])
    return_data = np.array([valid_returns[i] for i in portfolio_stock_indexes])

    port_info = []
    
    for stock_name, historical_prices, historical_returns in zip(stocks_in_port, price_data, return_data):
        port_info.append(StockInfo(stock_name, historical_prices, historical_returns))

    return port_info

def get_valid_data(all_stock_names: List[str],
    all_stock_prices: _nested_ndarray) -> Tuple[List[str], _nested_ndarray, _nested_ndarray, _nested_ndarray]:
    '''
    Given a large set of data, return only a subset that is valid.

    Args:

    * `all_stock_names` - the ticker symbols of all the available stocks to choose from
    * `all_stock_prices` - the historical prices for all the available stocks to choose from. The latest
        price is at index 0.

    Returns:

    > `valid_names`, `valid_prices`, `valid_returns`, and `valid_self_corr`
    '''

    all_stock_returns = helper.get_returns(all_stock_prices)

    valid_names = []
    valid_prices = []
    valid_returns = []

    for name, price, ret in zip(all_stock_names, all_stock_prices, all_stock_returns):
        if np.all(np.isfinite(ret)):
            valid_names.append(name)
            valid_prices.append(price)
            valid_returns.append(ret)

    return valid_names, valid_prices, valid_returns, helper.get_self_correlation(np.array(valid_returns))

def get_stocks_in_portfolio(valid_stock_names: List[str], valid_stock_prices: _nested_ndarray,
    valid_self_corr: _nested_ndarray, required_stocks: List[str], min_portfolio_size: int) -> List[str]:
    '''
    Given a a large data set containing all valid stock information, and a set of required stocks
    that must be in portfolio, this method will return a list of other stocks that, together with the
    required stocks, form a well diversified portfolio.

    Args:

    * `valid_stock_names` - the ticker symbols of all the available stocks to choose from
    * `valid_stock_prices` - the historical prices for all the available stocks to choose from. The latest
        price is at index 0.
    * `valid_self_corr` - the self correlation of the returns calculated from `valid_stock_prices`.
    * `required_stocks` - the ticker symbols of the required stocks that must be in the portfolio
    * `min_portfolio_size` - the minimum size of the resultant portfolio

    Returns:

    > A list of stock ticker symbols that would work well together in a portfolio.
    '''

    indexes = [valid_stock_names.index(x) for x in required_stocks]

    _corr = np.copy(valid_self_corr)

    for index in indexes:
        _corr = helper.perturb_correlations(_corr, index)

    _pca = helper.do_pca(_corr, min_portfolio_size / 2)
    _pca_res = helper.group_pca_result(_pca, valid_stock_names)

    _extract_stocks = helper.extract_stocks_from_group(_pca_res, required_stocks, min_portfolio_size)

    return list(_extract_stocks.keys())

class _DiversifyTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_diversify_portfolio(self):
        all_stock_names = [str(x) for x in range(10)]
        all_stock_prices = np.random.rand(10, 100)
        required_stocks = ['1', '5']

        portfolio = diversify_portfolio(all_stock_names, all_stock_prices, required_stocks, 6)
        self.assertTrue(len(portfolio) >= 6)

        stock_names = [stock.name for stock in portfolio]

        for req_stock in required_stocks:
            self.assertIn(req_stock, stock_names)

if __name__ == '__main__':
    unittest.main()