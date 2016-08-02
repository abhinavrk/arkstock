from typing import List, Dict, Callable, Any, Iterable
import csv
import os
import unittest
from .abstract_types import AbstractStockMetadataClient

script_dir = os.path.dirname(__file__)
rel_path = 'metadata.csv'
abs_file_path = os.path.join(script_dir, rel_path)

class YahooFinanceMetadataClient(AbstractStockMetadataClient):
    """
    A StockMetadataClient that retrieves metadata for most stocks and exchanges in YahooFinance
    """
    file_name = abs_file_path
    '''The absolute path to the file where the metadata is stored'''

    def _get_all_stock_metadata(self) -> List[Dict[str, str]]:
        '''
        Return all stock metadata mappings in the classes file path
        '''
        
        stock_md = []
        
        with open(self.file_name, 'r') as metadata_file:
            reader = csv.DictReader(metadata_file)
            for row in reader:
                stock_md.append(row)

        return stock_md

    def _retrieve_stock_metadata(self, filter_func: Callable[[Dict[str, str]], bool],
        transformator: Callable[[Dict[str, str]], Any]) -> Iterable[Any]:
        """
        Return a generator that applies a transformation to every valid stock metadata object

        Args:

        * `filter_func` - the function that determines if a given row in a data file 
            should be used
        * `transformator` - the transformation that should be applied to every valid stock
            metadata object

        Returns:

        > A generator with the transformed stock metadata object

        Note:

        > Throws IOException
        """
        filtered_stock_md = [md_obj for md_obj in self._get_all_stock_metadata() if filter_func(md_obj)]
        return [transformator(md_obj) for md_obj in filtered_stock_md]

    @staticmethod
    def _filter_for_exchange(exchange: str) -> Callable[[Dict[str, str]], bool]:
        return lambda stock_md: stock_md['Exchange'] == exchange

    @staticmethod
    def _retrieve_ticker() -> Callable[[Dict[str, str]], str]:
        return lambda stock_md: stock_md['Ticker']

    def get_stocks_in_exchange(self, exchange: str) -> Iterable[str]:
        """
        Retrieve all ticker symbols that correspond to an exchange

        Args:

        * `exchange` - a string repr of the exchange for which the tickers should be retrieved

        Returns:

        > A generator over the tickers present in the exchange
        """
        return self._retrieve_stock_metadata(
            self._filter_for_exchange(exchange), self._retrieve_ticker())


class _YahooFinanceMetadataClientTest(unittest.TestCase):
    def setUp(self):
        self.metadata_client = YahooFinanceMetadataClient()

    def test_get_all_rows(self):
        data = self.metadata_client._get_all_stock_metadata()
        for datum in data:
            self.assertIn('Exchange', datum)

    def test_filter_for_exchange(self):
        exchange = 'NYQ'
        filter_exch = self.metadata_client._filter_for_exchange(exchange)
        data = [{'Exchange': 'NYQ'}, {'Exchange': 'NYQA'}]
        self.assertEqual([True, False], [filter_exch(x) for x in data])

    def test_get_all_data_for_exchange(self):
        exchange = 'NYQ'
        filter_exch = self.metadata_client._filter_for_exchange(exchange)
        data = self.metadata_client._retrieve_stock_metadata(filter_exch, lambda x: x)
        for datum in data:
            self.assertEqual(exchange, datum['Exchange'])

    def test_pluck(self):
        pluck_ticker = self.metadata_client._retrieve_ticker()
        data = [{'Ticker': 'NYQ'}, {'Ticker': 'NYQA'}]
        self.assertEqual(['NYQ', 'NYQA'], [pluck_ticker(x) for x in data])

    def test_pluck_for_all_data(self):
        pluck_ticker = self.metadata_client._retrieve_ticker()
        data = self.metadata_client._retrieve_stock_metadata(lambda x: x, pluck_ticker)
        self.assertEqual(len(data), len(self.metadata_client._get_all_stock_metadata()))

    def test_retrieve_data(self):
        data = self.metadata_client.get_stocks_in_exchange('NYQ')
        for datum in data:
            self.assertIsInstance(datum, str)


if __name__ == '__main__':
    unittest.main()