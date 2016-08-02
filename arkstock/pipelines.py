from typing import Callable, Any, Dict, List, Tuple
from functools import lru_cache
from collections import namedtuple
import numpy as np
_ndarray = np.ndarray
'''type alias for improved code clarity'''
_nested_ndarray = _ndarray
'''type alias for improved code clarity'''

from .data import (AbstractStockDatabaseClient, AbstractStatefulStockDatabaseClient,
    YahooFinanceStatefulClient)
from .diversify import diversify
from .optimize import optimize

StockInfo = diversify.StockInfo
'''type alias for improved code clarity'''
StockPortfolio = namedtuple('StockPortfolio', ['stocks', 'weights'])
'''immutable container for information regarding an optimized portfolio'''

def default_cached_db_client(db_client: AbstractStockDatabaseClient) -> AbstractStockDatabaseClient:
    '''
    This method adds an lru_cache wrapper onto the `db_client`'s #get_historical_data_for_stocks_in_exchange
    function handle.

    Args:

    * `db_client` - an instance of an AbstractStockDatabaseClient

    Returns:

    > A cached AbstractStockDatabaseClient instance
    '''
    cache = lru_cache(maxsize=50)
    db_client.get_historical_data_for_stocks_in_exchange = cache(
        db_client.get_historical_data_for_stocks_in_exchange)
    return db_client


def default_stateful_stock_exchange_data_source(database_client: AbstractStatefulStockDatabaseClient,
    exchange_filter: Callable[[str], bool]) -> Callable[[], Dict[str, List[float]]]:
    '''
    Create a data source that provides data for a stock exchange based on the
    AbstractStatefulStockDatabaseClient

    Args:

    * `database_client` - an implementation of an AbstractStockDatabaseClient
    * `exchange_filter` - the list of exchanges for which you would like to return data

    Returns:

    > A no-arg function that returns a dictionary containing all the requested data
    '''

    def get_data():

        exchange_list = filter(exchange_filter, database_client.get_available_exchanges())

        all_data = [database_client.get_historical_data_for_stocks_in_exchange(x) for x in exchange_list]

        return { k: v for d in all_data for k, v in d.items() }

    return get_data

def default_data_transformator(
    data_generator: Callable[[], Dict[str, List[float]]]) -> Callable[[], Tuple[List[str], _nested_ndarray]]:
    '''
    Given a data source that provides data, this method returns a function that wraps the result from the
    data source and returns a more useful view of the data

    Args:

    * `data_generator` - a callable that returns data

    Returns:

    > A no-arg function that returns a more useful view of the data
    '''
    
    def data_transformer():
        data = data_generator()
        stock_names = list(data.keys())
        stock_prices = np.array(list(data.values()))
        return stock_names, stock_prices

    return data_transformer

def default_diversifier(
    data_gen: Callable[[], Tuple[List[str], _nested_ndarray]]) -> Callable[[List[str]], List[StockInfo]]:
    '''
    Given a data source that provides data, this method returns a function that allows one to diversify
    the result from the datasource

    Args:

    * `data_gen` - a callable that returns data

    Returns:

    > A  function that returns a list of stocks that should be a part of the diversified portfolio, given a
    list of ticker symbols that should necessairly be in the portfolio
    '''

    def diversifier(required_stocks: List[str]) -> List[StockInfo]:
        stock_names, stock_prices = data_gen()
        return diversify.diversify_portfolio(stock_names, stock_prices, required_stocks)

    return diversifier

def default_optimizer(
    portfolio_gen: Callable[[List[str]], List[StockInfo]]) -> Callable[[List[str]], StockPortfolio]:
    '''
    Given a portfolio generator that generates diversified stocks, this method returns a MVO optimized
    portfolio holding the generated stocks

    Args:

    * `portfolio_gen` - a callable that returns diversified stocks given a list of ticker symbols that
        must absolutely be a part of the portfolio

    Returns:

    > A  function that returns a list of stocks that should be a part of the diversified portfolio, given a
    list of ticker symbols that should necessairly be in the portfolio
    '''

    def weighted_port_gen(required_stocks: List[str]) -> StockPortfolio:
        stocks_in_port = portfolio_gen(required_stocks)
        returns = np.array([x.hist_returns for x in stocks_in_port])

        constraints = optimize.build_constraints()
        weights = optimize.optimize_portfolio(np.cov(returns), constraints)
        rounded_weights = [x for x in np.round(weights, decimals=8)]

        return StockPortfolio(stocks_in_port, rounded_weights)

    return weighted_port_gen

class PortfolioGeneratorFactory(object):
    '''
    This class encapsulates the logic to create a data-pipeline that returns a diversified and optimized
    portfolio.
    '''

    @staticmethod
    def get_default(exchange_filter: Callable[[str], bool] =lambda x: True):
        '''
        This returns the default pipeline that can be used to diversify and optimize portfolios

        Args:

        * `exchange_filter` - a function that returns True or False depending on whether a stock exchange
            should be used to diversify/optimize a portfolio.

        Returns:

        > A function that allows one to pass in a list of preferred stocks and output a diversified portfolio

        Note:

        > Simple ad-hoc testing shows that the first call to the output pipeline takes 60 seconds, whereas
        subsequent calls only take 0.5 seconds.
        '''

        db_client = YahooFinanceStatefulClient()
        db_client = default_cached_db_client(db_client)

        stock_exchange_data_source = default_stateful_stock_exchange_data_source(db_client, exchange_filter)

        transformed_stock_exch_data = default_data_transformator(stock_exchange_data_source)

        diversifier = default_diversifier(transformed_stock_exch_data)

        optimizer = default_optimizer(diversifier)

        return optimizer