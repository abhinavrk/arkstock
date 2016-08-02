from abc import ABCMeta, abstractmethod
from typing import List, Dict
from datetime import date, timedelta
from enum import Enum

class TimeInterval(Enum):
    '''
    An enum to encapsulate package specific time intervals
    '''

    weeks = ('weeks', 7)
    '''TimeInterval enum value to denote a time interval of 1 week i.e. 7 days'''

    def __init__(self, symbol, days):
        self.symbol = symbol # string specific to yahoo finance that represents that time range
        self.days = days # interval length in days

class AbstractStockDatabaseClient(metaclass=ABCMeta):
    '''
    An abstract stock database client that defines the methods that need to be implemented for
    interoperability with the rest of the code
    '''

    @abstractmethod
    def get_historical_stock_prices(self, symbol: str,
        start_date: date, end_date: date, time_interval: TimeInterval) -> List[float]:
        '''
        This method retrieves historical price data for a ticker symbol in USD.

        Args:

        * `symbol` - the ticker symbol for which you wish to retrieve data
        * `start_date` - the start of the time range for which you would like to get data
        * `end_date` - the end of the time range for which you would like to get data
        * `time_interval` - the time interval at which you would like to get data for the given time range

        Returns:

        > A list of the historical stock price data in USD. The most recent price should be at index 0.

        Note:

        > In theory, the default currency does not need to be USD, it only needs to be consistent
        within the application. In practice, USD is by far the most commonly used and hence is preferred
        '''
        
        pass

    @abstractmethod
    def get_historical_data_for_stocks(self, symbols: List[str],
        start_date: date, end_date: date, time_interval: TimeInterval) -> Dict[str, List[float]]:
        '''
        This method returns historical data for all the provided ticker symbols in USD

        Args:

        * `symbols` - the ticker symbols for which you wish to retrieve data
        * `start_date` - the start of the time range for which you would like to get data
        * `end_date` - the end of the time range for which you would like to get data
        * `time_interval` - the time interval at which you would like to get data for the given time range

        Returns:

        > A dict mapping a ticker symbol to a list of historical stock price data in USD.
        The most recent price should be at index 0.

        Note:

        > This method will skip all ticker symbols that are invalid for some reason or another
        '''
        
        pass

    @abstractmethod
    def get_historical_data_for_stocks_in_exchange(self, exchange_name: str,
        start_date: date, end_date: date, time_interval: TimeInterval,
        threshold_mcap: bool =True, mcap_limit_in_millions: int =300) -> Dict[str, List[float]]:
        '''
        This method returns historical data for all ticker symbols in an exchange in USD that meet a 
        given mcap_limit

        Args:

        * `exchange_name` - string repr of the exchange for which you wish to retrieve data
        * `start_date` - the start of the time range for which you would like to get data
        * `end_date` - the end of the time range for which you would like to get data
        * `time_interval` - the time interval at which you would like to get data for the given time range
        * `threshold_mcap` - whether or not the stocks in the exchange should be filtered based on market
            capitalization
        * `mcap_limit_in_millions` - the market cap threshold

        Returns:

        > A dict mapping a ticker symbol to a list of historical stock price data in USD.
        The most recent price should be at index 0.

        Note:

        > The returned data is for the ticker symbols found in the provided exchange.
        '''

        pass

    @abstractmethod
    def get_available_exchanges(self) -> List[str]:
        '''
        This method returns a list of stock exchanges for which this client can return data
        '''

        pass

class AbstractStockMetadataClient(metaclass=ABCMeta):
    '''
    An abstract stock metadata client that defines the methods that need to be implemented for
    interoperability with the rest of the code
    '''

    @abstractmethod
    def get_stocks_in_exchange(self, exchange_name: str) -> List[str]:
        '''
        This method returns a list of ticker symbols that correspond to a provided `exchange_name`

        Args:

        * `exchange_name` - string repr of the exchange for which you wish to retrieve data

        Returns:

        > A list of ticker symbols found in that exchange
        '''

        pass

class AbstractStatefulStockDatabaseClient(AbstractStockDatabaseClient):
    '''
    Basically a stateful proxy of AbstractStockDatabaseClient
    '''

    @abstractmethod
    def set_state(self, start_date: date, end_date: date, time_interval: TimeInterval):
        '''
        This method sets the class wide execution state for stateful database clients

        Args:

        * `start_date` - the start of the time range for which you would like to get data
        * `end_date` - the end of the time range for which you would like to get data
        * `time_interval` - the time interval at which you would like to get data for the given time range

        Returns:

        > None
        '''

        pass