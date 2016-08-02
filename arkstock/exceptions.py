from functools import wraps
import warnings

# Common exceptions and methods

class ArkstockException(Exception):
    '''
    Base exception class from which all other package exceptions inherit. This allows the end user to simply
    catch ArkstockException and handle exceptions from the arkstock package in a consistent way.
    '''
    pass

class exception_wrapper(object):
    '''
    Wraps code to be executed in a try-except block and throws the right exception when needed
    '''

    def __init__(self, exception_class, exception_msg):
        self.exception_class = exception_class
        self.exception_msg = exception_msg

    def __call__(self, func_to_wrap):
        @wraps(func_to_wrap)
        def execute(*args, **kwargs):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                try:
                    return func_to_wrap(*args, **kwargs)
                except Exception as exc:
                    raise self.exception_class(self.exception_msg) from exc

        return execute

# Data exceptions

class DataRetrievalException(ArkstockException): pass

class ImproperlyConfiguredRequest(DataRetrievalException): pass

class IncompleteDataRetrieved(DataRetrievalException): pass

class RequestFailedException(DataRetrievalException): pass

class IllegalArgument(DataRetrievalException): pass

# Diversification exceptions

class DiversificationException(ArkstockException): pass

class StockCorrelationException(DiversificationException): pass

class HistoricalReturnsException(DiversificationException): pass

class PcaAccuracyException(DiversificationException): pass

# PortfolioOptimization exceptions

class PortfolioOptimizationException(ArkstockException): pass