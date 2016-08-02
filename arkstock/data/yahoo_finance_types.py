from itertools import zip_longest
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Dict, Iterable
from datetime import date, timedelta

from .abstract_types import (AbstractStockDatabaseClient, AbstractStatefulStockDatabaseClient,
    AbstractStockMetadataClient, TimeInterval)
from .yahoo_finance_metadata import YahooFinanceMetadataClient
from ..exceptions import IllegalArgument
from . import yahoo_finance_gateway
YahooTimeInterval = yahoo_finance_gateway.YahooTimeInterval
'''type aliasing for code clarity'''

import unittest

class HelperUtils(object):

    @staticmethod
    def chunker(iterable: Iterable[Any], chunksize: int, fillvalue: Any =None) -> Iterable[Any]:
        '''
        Collect data into fixed-length chunks or blocks

        Args:
        
        * `iterable` - a data type is iterable
        * `chunksize` - the size of the blocks you wish to extract from the iterable
        * `fillvalue` - the default value to fill for the last block if there arent enough elements

        Returns:

        > A generator/iterable wherein each element is a block of size `chunksize`
        '''
    
        # chunker('ABCDEFG', 3, 'x') --> [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]
        args = [iter(iterable)] * chunksize
        return zip_longest(*args, fillvalue=fillvalue)

    @staticmethod
    def is_valid_ticker(ticker: str) -> bool:
        '''
        Check to ensure that a given ticker doesn't contain any weird symbols

        Args:

        * `ticker` - the string repr of the ticker symbol to check

        Returns:
        > True if the ticker is valid, false otherwise
        '''

        if '@' in ticker:
            return False
        if '&' in ticker:
            return False
        return True

    @staticmethod
    def convert_TimeInterval_to_YahooTimeInterval(time_interval: TimeInterval) -> YahooTimeInterval:
        '''
        A utility method to convert from the generic TimeInterval to the specific YahooTimeInterval
        '''

        mapping = {
            TimeInterval.weeks : YahooTimeInterval.weeks
        }

        return mapping.get(time_interval, time_interval)

class YahooFinanceClient(AbstractStockDatabaseClient):
    '''
    An concrete stock database client that implementes the methods that need to be implemented for
    interoperability with the rest of the code. This client is based on the yahoo finance API.

    This client is essentially stateless and is hence threadsafe.
    '''

    def __init__(self,
        yfgateway =yahoo_finance_gateway,
        yfmetadata: AbstractStockMetadataClient =None,
        global_exchange_list: List[str] =['ASE', 'NCM', 'NGM', 'NMS', 'NYQ', 'OBB', 'TOR', 'VAN']):

        self.yfgateway = yfgateway
        '''A pointer to the YahooFinanceGateway module'''

        if yfmetadata is None:
            yfmetadata = YahooFinanceMetadataClient()
        
        self.yfmetadata = yfmetadata
        '''A yahoo finance metadata client'''

        self.global_exchange_list = global_exchange_list
        '''Stock exchanges for which data is available'''

    def filter_based_on_mcap_limit(self, symbols: Iterable[str],
        mcap_limit_in_millions: int) -> List[str]:
        '''
        Given a list of stocks, only return stocks that have a market capitalization over the provided
        threshold

        Args:

        * `symbols` - the ticker symbols for which you wish to retrieve data
        * `mcap_limit_in_millions` - the mcap threshold
        
        Returns:

        > A list of symbols above the threshold
        '''

        if mcap_limit_in_millions == 0:
            return symbols

        valid_symbols = []

        chunked_stocks = HelperUtils.chunker(symbols, 500)

        for chunk in chunked_stocks:
            _stocks = [x for x in chunk if x is not None]

            _mcap = self.yfgateway.get_mcap(_stocks)

            _stocks_above_threshold = [
                symbol
                for symbol, mcap in _mcap if self.yfgateway.is_mcap_over_limit(mcap)
            ]
            valid_symbols += _stocks_above_threshold

        return valid_symbols

    @lru_cache(maxsize=50)
    def _cached_get_exchange_rate(self, curr: str,
        start_date: date, end_date: date, time_interval: YahooTimeInterval) -> List[float]:
        '''
        This is a cached wrapper around `yfgateway.get_historical_exch_rate_to_USD`

        > See `yfgateway.get_historical_exch_rate_to_USD`
        '''

        return self.yfgateway.get_historical_exch_rate_to_USD(curr,
            start_date, end_date, time_interval)

    def get_historical_stock_prices(self, symbol: str,
        start_date: date, end_date: date, time_interval: TimeInterval) -> List[float]:

        if not HelperUtils.is_valid_ticker(symbol):
            raise IllegalArgument('%s is not a valid ticker symbol' % symbol)

        yahoo_time_interval = HelperUtils.convert_TimeInterval_to_YahooTimeInterval(time_interval)

        stock_native_curr = self.yfgateway.get_native_currency_for_ticker(symbol)
        exch_rate = self._cached_get_exchange_rate(stock_native_curr,
            start_date, end_date, yahoo_time_interval)

        historical_prices = self.yfgateway.get_historical_stock_prices(symbol,
            start_date, end_date, yahoo_time_interval)

        return [
            rate * price
            for rate, price in zip(exch_rate, historical_prices)
        ]

    def get_historical_data_for_stocks(self, symbols: List[str],
        start_date: date, end_date: date, time_interval: TimeInterval) -> Dict[str, List[float]]:

        symbols = [x for x in symbols if HelperUtils.is_valid_ticker(x)]

        historical_prices = {}
        args = [(symbol, start_date, end_date, time_interval) for symbol in symbols]
        
        with ThreadPoolExecutor() as executor:
            future_to_stock = {
                executor.submit(self.get_historical_stock_prices, *arg): arg[0]
                for arg in args
            }

            for future in as_completed(future_to_stock):

                symbol = future_to_stock[future]

                try:
                    price_in_usd = future.result()

                except Exception as exc:
                    pass

                else:
                    historical_prices[symbol] = price_in_usd

        return historical_prices

    def get_historical_data_for_stocks_in_exchange(self, exchange_name: str,
        start_date: date, end_date: date, time_interval: TimeInterval,
        threshold_mcap: bool =True, mcap_limit_in_millions: int =300) -> Dict[str, List[float]]:

        stocks_in_exch = [
            x for x in self.yfmetadata.get_stocks_in_exchange(exchange_name)
            if HelperUtils.is_valid_ticker(x)
        ]

        if threshold_mcap:
            stocks_in_exch = self.filter_based_on_mcap_limit(stocks_in_exch, mcap_limit_in_millions)

        yahoo_time_interval = HelperUtils.convert_TimeInterval_to_YahooTimeInterval(time_interval)

        if len(stocks_in_exch) == 0:
            return {}

        stock_exch_native_currency = self.yfgateway.get_native_currency_for_ticker(stocks_in_exch[0])
        exchange_rate = self._cached_get_exchange_rate(stock_exch_native_currency,
            start_date, end_date, yahoo_time_interval)

        historical_prices = {}
        args = [(symbol, start_date, end_date, yahoo_time_interval) for symbol in stocks_in_exch]

        with ThreadPoolExecutor() as executor:
            future_to_stock = {
                executor.submit(self.yfgateway.get_historical_stock_prices, *arg): arg[0]
                for arg in args
            }

            for future in as_completed(future_to_stock):

                symbol = future_to_stock[future]

                try:
                    price_in_native_currency = future.result()

                except Exception as exc:
                    pass

                else:
                    historical_prices[symbol] = [
                        rate * price
                        for rate, price in zip(exchange_rate, price_in_native_currency)
                    ]

        return historical_prices

    def get_available_exchanges(self) -> List[str]:
        return self.global_exchange_list

class YahooFinanceStatefulClient(AbstractStatefulStockDatabaseClient):
    '''
    An concrete stock database client that implementes the methods that need to be implemented for
    interoperability with the rest of the code. This client is based on the yahoo finance API.

    This client is stateful and is as such not threadsafe.
    '''

    def __init__(self):
        self.yfclient = YahooFinanceClient()
        time_interval = TimeInterval.weeks
        end_date = date.today() - timedelta(days=date.today().weekday())
        start_date = end_date - timedelta(weeks=77)
        self.set_state(start_date, end_date, time_interval)

    def set_state(self, start_date: date, end_date: date, time_interval: TimeInterval):
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval

    def get_historical_stock_prices(self, symbol: str) -> List[float]:
        
        return self.yfclient.get_historical_stock_prices(symbol,
            self.start_date, self.end_date, self.time_interval)

    def get_historical_data_for_stocks(self, symbols: List[str]) -> Dict[str, List[float]]:
        
        return self.yfclient.get_historical_data_for_stocks(symbols,
            self.start_date, self.end_date, self.time_interval)

    def get_historical_data_for_stocks_in_exchange(self, exchange_name: str,
        threshold_mcap: bool =True, mcap_limit_in_millions: int =300) -> Dict[str, List[float]]:

        return self.yfclient.get_historical_data_for_stocks_in_exchange(exchange_name,
            self.start_date, self.end_date, self.time_interval, threshold_mcap=threshold_mcap,
            mcap_limit_in_millions=mcap_limit_in_millions)

    def get_available_exchanges(self) -> List[str]:
        return self.yfclient.get_available_exchanges()

class _YahooFinanceClientTest(unittest.TestCase):
    def setUp(self):
        self.yf_client = YahooFinanceClient()
        self.time_interval = TimeInterval.weeks
        self.end_date = date.today() - timedelta(days=date.today().weekday())
        self.start_date = self.end_date - timedelta(weeks=103)

    def test_get_historical_stock_prices(self):
        stock = 'GOOGL'
        google_prices = self.yf_client.get_historical_stock_prices(stock,
            self.start_date, self.end_date, self.time_interval)

        google_prices_without_conversion = self.yf_client.yfgateway.get_historical_stock_prices(
            stock, self.start_date, self.end_date, YahooTimeInterval.weeks)

        for x, y in zip(google_prices, google_prices_without_conversion):
            self.assertAlmostEqual(x, y, places=2)

        stock = 'SCC.TO'
        scc_prices = self.yf_client.get_historical_stock_prices(stock,
            self.start_date, self.end_date, self.time_interval)

        scc_prices_without_conversion = self.yf_client.yfgateway.get_historical_stock_prices(
            stock, self.start_date, self.end_date, YahooTimeInterval.weeks)

        for x, y in zip(scc_prices, scc_prices_without_conversion):
            self.assertNotAlmostEqual(x, y, places=0)

    def test_get_historical_data_for_stocks(self):
        stocks = ['GOOGL', 'SCC.TO']
        
        google_prices = self.yf_client.get_historical_stock_prices(stocks[0],
            self.start_date, self.end_date, self.time_interval)

        scc_prices = self.yf_client.get_historical_stock_prices(stocks[1],
            self.start_date, self.end_date, self.time_interval)

        stock_prices = self.yf_client.get_historical_data_for_stocks(stocks,
            self.start_date, self.end_date, self.time_interval)

        for x, y in zip(google_prices, stock_prices['GOOGL']):
            self.assertAlmostEqual(x, y, places=2)

        for x, y in zip(scc_prices, stock_prices['SCC.TO']):
            self.assertAlmostEqual(x, y, places=2)

    def test_get_historical_data_for_stocks_in_exchange(self):
        stocks = ['GOOGL', 'SCC.TO']

        class FakeMetadataClient:
            @classmethod
            def get_stocks_in_exchange(cls, exchange_name):
                return stocks

        self.yf_client = YahooFinanceClient(yfmetadata=FakeMetadataClient())

        google_prices = self.yf_client.get_historical_stock_prices(stocks[0],
            self.start_date, self.end_date, self.time_interval)

        scc_prices = self.yf_client.get_historical_stock_prices(stocks[1],
            self.start_date, self.end_date, self.time_interval)

        stock_prices = self.yf_client.get_historical_data_for_stocks_in_exchange('some_exchange',
            self.start_date, self.end_date, self.time_interval, threshold_mcap=False)

        for x, y in zip(google_prices, stock_prices['GOOGL']):
            self.assertAlmostEqual(x, y, places=2)

        for x, y in zip(scc_prices, stock_prices['SCC.TO']):
            # we're assuming that SCC.TO is in USD, which it isn't, hence we cannot expect them to equal
            self.assertNotAlmostEqual(x, y, places=2)

class _HelperUtilsTest(unittest.TestCase):

    def test_chunker(self):
        self.assertListEqual(list(HelperUtils.chunker('ABCDEFG', 3, 'x')),
            [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')])

    def test_is_valid_ticker(self):
        self.assertTrue(HelperUtils.is_valid_ticker('asdfasdf.assdfasdDASgasdf123'))
        self.assertFalse(HelperUtils.is_valid_ticker('asdf@adsf'))
        self.assertFalse(HelperUtils.is_valid_ticker('asdf&adsf'))

    def test_time_interval_conversion(self):
        self.assertEqual(YahooTimeInterval.weeks,
            HelperUtils.convert_TimeInterval_to_YahooTimeInterval(TimeInterval.weeks))
