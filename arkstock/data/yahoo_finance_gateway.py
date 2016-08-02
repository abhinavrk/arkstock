'''
A module that defines a set of method used to retrieve data from yahoo finance.

Due to the singleton nature of modules in python, all methods must inherently be functional.
'''

from datetime import date, timedelta
from enum import Enum
from typing import Union, List
from collections.abc import Iterable
from urllib.request import urlopen
from urllib.parse import quote

from ..exceptions import ImproperlyConfiguredRequest, IncompleteDataRetrieved, RequestFailedException

import unittest

base_url = 'http://ichart.yahoo.com/table.csv?'
'''Yahoo Finance's base url used to make requests to retrieve data'''

class YahooTimeInterval(Enum):
    '''
    An enum to encapsulate yahoo finance specific time interval
    '''

    weeks = ('w', 7)
    '''TimeInterval enum value to denote a time interval of 1 week i.e. 7 days'''

    def __init__(self, symbol, days):
        self.symbol = symbol # string specific to yahoo finance that represents that time range
        self.days = days # interval length in days

def num_intervals(start_date: date, end_date: date, time_interval: YahooTimeInterval) -> int:
    '''
    This method takes in the start_date, end_date, and time_interval and returns the number of complete
    time_intervals within that time range

    Args:

    * `start_date` - the start of the time range for which you would like to get data
    * `end_date` - the end of the time range for which you would like to get data
    * `time_interval` - the time interval at which you would like to get data for the given time range

    Returns:

    > Number of complete time intervals within that time range
    '''

    return int((end_date - start_date).days / time_interval.days)

def build_url_suffix(start_date: date, end_date: date, time_interval: YahooTimeInterval) -> str:
    '''
    This method takes in the start_date, end_date, and time_interval and creates the generic url suffix
    that is used in almost all requests to yahoo finance

    Args:

    * `start_date` - the start of the time range for which you would like to get data
    * `end_date` - the end of the time range for which you would like to get data
    * `time_interval` - the time interval at which you would like to get data for the given time range

    Returns:

    > The generic url suffix that can be used to query yahoo finance with the given context
    '''

    if start_date is None:
        raise ImproperlyConfiguredRequest('start_date is not specified')
    if end_date is None:
        raise ImproperlyConfiguredRequest('end_date is not specified')
    if time_interval is None:
        raise ImproperlyConfiguredRequest('time_interval is not specified')

    start_month = 'a=%s&' % str(start_date.month - 1)
    start_day = 'b=%s&' % str(start_date.day)
    start_year = 'c=%s&' % str(start_date.year)

    start_date_url_specifier = start_month + start_day + start_year

    end_month = 'd=%s&' % str(end_date.month - 1)
    end_day = 'e=%s&' % str(end_date.day)
    end_year = 'f=%s&' % str(end_date.year)

    end_date_url_specifier = end_month + end_day + end_year

    time_interval_url_specifier = 'g=%s&' % time_interval.symbol

    suffix = 'ignore=.csv'
    return start_date_url_specifier + end_date_url_specifier + time_interval_url_specifier + suffix

def bytes_to_string(bytes_or_bytes_array: Union[bytes, List[bytes]]) -> Union[str, List[str]]:
    '''
    Sometimes data from yahoo finance is in the form of bytes (or a list of bytes) and needs to be
    converted back into a string (or a list of strings). This method performs that transformation
    if and only if it is needed.

    Args:
    
    * `bytes_or_bytes_array` - bytes or an array of bytes

    Returns:
    
    > A string or a list of strings
    '''

    if isinstance(bytes_or_bytes_array, Iterable) and not isinstance(
            bytes_or_bytes_array, bytes):

        result = []
        for byte_str in bytes_or_bytes_array:
            if isinstance(byte_str, bytes):
                result.append(''.join(list(map(chr, byte_str))))
            else:
                result.append(byte_str)
        return result
    else:
        if isinstance(bytes_or_bytes_array, bytes):
            return ''.join(list(map(chr, bytes_or_bytes_array)))
        else:
            return bytes_or_bytes_array

def is_mcap_over_limit(mcap_string: str, limit_in_millions: int =300) -> bool:
    '''
    Given a market capitalization string, this method checks to see if a given mcap_string
    exceeds the given limit_in_millions

    If the string is not correctly formed or cannot be parsed, this method returns False

    This method exposes yahoo finance specific details on the input and is such is 'private'.
    It should be chained to the output of #get_mcap and should not be used otherwise unless you know
    what you're doing

    Args:
        
    * `mcap_string` - the market capitalization string retreived from yahoo finance
    * `limit_in_millions` - the testing threshold in millions

    Returns:
    
    > Whether or not the provided market cap string meets a given threshold

    Note:

    > mcap comparisons are done based on the mcap in the stock's native currency
    '''

    value = None
    try:
        if mcap_string[-1] == 'B':
            value = float(mcap_string[:-1]) * 1000
        elif mcap_string[-1] == 'T':
            value = float(mcap_string[:-1]) * 10e6
        elif mcap_string[-1] == 'M':
            value = float(mcap_string[:-1])
        else:
            value = 0.
    except Exception as exc:
        return False

    return value > limit_in_millions if value is not None else False

def get_mcap(symbols : List[str], code: str ='j1') -> List[List[str]]:
    '''
    Retrieve current market capitalization for a given list of stock symbols.

    Args:

    * `symbols` - a collection of strings that represent the stock symbols to be queried
    * `code` - the code (specific to yahoo finance) that represents the market cap data

    Returns:

    > A list containing the retrieved data, where each retrieved data object is itself a list
    of two elements, the first being the symbol, and the second being the market cap string
    for the symbol.

    Note:

    > The return for this method is not in terms of USD, but in terms of the 
    individual stock's native currency
    '''

    try:
        mcap_url = 'http://download.finance.yahoo.com/d/quotes.csv?'
        suffix = '&e=.csv'
        symbol_str = 's=' + quote(','.join(symbols), ',')
        code_str = '&f=' + code

        # Retrieve and split the file
        mcap_data = urlopen(mcap_url + symbol_str + code_str + suffix).read(
        ).splitlines()
        mcap_data_as_strings = bytes_to_string(mcap_data)
    
    except Exception as exc:
        raise RequestFailedException('Could not retrieve mcap for requested stocks') from exc
    
    else:
        try:
            assert (len(symbols) == len(mcap_data_as_strings))
        except AssertionError as exc:
            raise IncompleteDataRetrieved(
                'Only retrieved mcap data for some of the requested stocks') from exc
        return list(zip(symbols, mcap_data_as_strings))

def get_native_currency_for_ticker(symbol : str, code: str ='c4') -> str:
    '''
    This method returns the native currency for a given symbol

    Args:
    
    * `symbol` - a string repr for the ticker symbol for which native currency information must
        be retrieved
    * `code` - the code (specific to yahoo finance) that reprs currency data

    Returns:

    > A string repr of the native currency for the ticker
    '''

    currency_url = 'http://download.finance.yahoo.com/d/quotes.csv?'
    suffix = '&e=.csv'
    symbol_str = 's=' + quote(symbol)
    code_str = '&f=' + '' + code

    raw_file = urlopen(currency_url + symbol_str + code_str + suffix).read(
    ).splitlines()

    currency = bytes_to_string(raw_file[0][1:-1])

    return currency

def get_historical_exch_rate_to_USD(currency: str,
    start_date: date, end_date: date, time_interval: YahooTimeInterval) -> List[float]:
    '''
    Get exchange rate for currency to USD.

    Args:
        
    * `currency` - a string repr of the currency to be converted to USD
    * `start_date` - the start of the time range for which you would like to get data
    * `end_date` - the end of the time range for which you would like to get data
    * `time_interval` - the time interval at which you would like to get data for the given time
        range

    Returns:
    
    > A list of the historical exchange rate from the requested currency to USD

    Note:

    > The latest rate appears at index 0
    '''

    number_of_intervals = num_intervals(start_date, end_date, time_interval)

    if currency == 'USD' or currency.strip() == '/':
        return [1. for x in range(number_of_intervals)]

    symbol = '{0}=X'.format(currency)
    
    try:
        reverse_exch_rate = get_historical_stock_prices(symbol, start_date, end_date, time_interval)
    
    except RequestFailedException as exc:
        raise RequestFailedException(
            ('The request to retrieve historical exchange rates from yahoo finance'
            'for %s failed') % (currency)) from exc    

    except IncompleteDataRetrieved as exc:
        raise IncompleteDataRetrieved(
                ('Currency data could not be retrieved for days from %s to %s at an '
                'interval of %s') % (start_date, end_date, time_interval)) from exc

    return [1./ x for x in reverse_exch_rate]

def get_historical_stock_prices(symbol: str,
    start_date: date, end_date: date, time_interval: YahooTimeInterval) -> List[float]:
    '''
    This method retrieves historical price data for an ticker symbol. 
    Note: this returns data in the native currency and not in USD.

    Args:

    * `symbol` - a string repr of the ticker symbol for which you wish to retrieve 
        data
    * `start_date` - the start of the time range for which you would like to get data
    * `end_date` - the end of the time range for which you would like to get data
    * `time_interval` - the time interval at which you would like to get data for the given time
        range

    Returns:

    > A list of historical stock price data in the native currency

    Note:
    
    > The latest stock price appears at index 0
    '''

    number_of_intervals = num_intervals(start_date, end_date, time_interval)

    symbolic_query_uri = 's=%s&' % symbol
    total_url = base_url + symbolic_query_uri + build_url_suffix(start_date, end_date, time_interval)
    
    try:
        raw_file = urlopen(total_url).read().splitlines()
        raw_file = bytes_to_string(raw_file)
        l = []
        for line in raw_file[1:]:
            l.append(line.split(',')[-1])
    
    except Exception as exc:
        raise RequestFailedException(
            'The request to retrieve historical prices from yahoo finance for %s failed'
            % (symbol)) from exc
    
    else:
        if len(l) >= number_of_intervals:
            l = [float(x) for x in l]
            return l[:number_of_intervals]

        else:
            raise IncompleteDataRetrieved(
                ('Historical price data could not be retrieved for %s for '
                'days from %s to %s at an interval of %s') %
                (symbol, start_date, end_date, time_interval))

def get_sp500_historical_data(start_date: date, end_date: date,
    time_interval: YahooTimeInterval) -> List[float]:
    '''
    A special utility method to retrieve data for the S&P500 index

    Args:
    
    * `start_date` - the start of the time range for which you would like to get data
    * `end_date` - the end of the time range for which you would like to get data
    * `time_interval` - the time interval at which you would like to get data for the given time
        range

    Returns:
    
    > A list of historical price data for the S&P500 in USD

    Note:
    
    > The latest price appears at index 0
    '''
    symbol ='%5EGSPC'
    return get_historical_stock_prices(symbol, start_date, end_date, time_interval)


class _YahooFinanceGatewayTest(unittest.TestCase):
    def setUp(self):
        self.time_interval = YahooTimeInterval.weeks
        self.end_date = date.today() - timedelta(days=date.today().weekday())
        self.start_date = self.end_date - timedelta(weeks=103)

    def test_get_mcap(self):
        stock = 'GOOGL'
        mcap = get_mcap([stock])[0]
        self.assertIsInstance(mcap[0], str)
        self.assertIsInstance(mcap[1], str)
        self.assertIsInstance(float(mcap[1][:-1]), float)

    def test_is_mcap_over_limit(self):
        stock = 'GOOGL'
        mcap = get_mcap([stock])[0]
        self.assertTrue(is_mcap_over_limit(mcap[1]))
        self.assertFalse(is_mcap_over_limit(mcap[1], 10e9))

    def test_get_native_currency_for_ticker(self):
        stock = 'GOOGL'
        native_curr = get_native_currency_for_ticker(stock)
        self.assertEqual('USD', native_curr)
        stock = 'SCC.TO'
        native_curr = get_native_currency_for_ticker(stock)
        self.assertEqual('CAD', native_curr)

    def test_get_historical_exch_rate(self):
        curr = 'CAD'
        exch_rate = get_historical_exch_rate_to_USD(curr,
            self.start_date, self.end_date, self.time_interval)
        
        self.assertEqual(num_intervals(self.start_date, self.end_date, self.time_interval),
            len(exch_rate))
        
        curr = 'CADDD'
        self.assertRaises(RequestFailedException,
                          get_historical_exch_rate_to_USD,
                          curr, self.start_date, self.end_date, self.time_interval)

    def test_get_historical_stock_prices(self):
        stock = 'GOOGL'
        historical_prices = get_historical_stock_prices(stock,
            self.start_date, self.end_date, self.time_interval)

        self.assertEqual(num_intervals(self.start_date, self.end_date, self.time_interval),
            len(historical_prices))

        self.assertRaises(RequestFailedException, get_historical_stock_prices,
            'GOOGLER', self.start_date, self.end_date, self.time_interval)

        self.start_date = date.today() - timedelta(weeks=20 * 52)
        self.assertRaises(IncompleteDataRetrieved, get_historical_stock_prices,
            stock, self.start_date, self.end_date, self.time_interval)

if __name__ == '__main__':
    unittest.main()
