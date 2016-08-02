'''
This module provides helper functions that extract information from stocks and diversify a portfolio
'''

from sklearn.decomposition import RandomizedPCA as PCA
from collections import namedtuple
from typing import List, Dict
import numpy as np
_ndarray = np.ndarray
'''type alias for improved code clarity'''
_nested_ndarray = _ndarray
'''type alias for improved code clarity'''

from ..exceptions import (StockCorrelationException, HistoricalReturnsException, PcaAccuracyException,
    exception_wrapper)

import unittest


PrincipalStockData = namedtuple('PrincipalStockData', ['name', 'data'])
'''immutable container for information regarding a PCA result'''

@exception_wrapper(StockCorrelationException, 'Could not calculate correlation')
def get_correlation(matrix_a: _nested_ndarray, matrix_b: _nested_ndarray) -> _nested_ndarray:
    '''
    Given two matrixes of the same shape, this method calculates the correlation between the two matrices

    Args:

    * `matrix_a` - a matrix where each row corresponds to a stock and the columns in the row
        correspond to historical data
    * `matrix_b` - another matrix similar to `matrix_a`

    Returns:

    > A square matrix of the correlations of matrix_a and matrix_b
    '''

    A = ((matrix_a - np.mean(matrix_a,
                             axis=1).reshape((-1, 1))) /
         np.std(matrix_a,
                axis=1).reshape((-1, 1)))

    B = ((matrix_b - np.mean(matrix_b,
                             axis=1).reshape((-1, 1))) /
         np.std(matrix_b,
                axis=1).reshape((-1, 1)))

    n = float(A.shape[-1])
    C = np.dot(A, B.T) / n
    return C

@exception_wrapper(StockCorrelationException, 'Could not calculate self correlation')
def get_self_correlation(matrix: _nested_ndarray) -> _nested_ndarray:
    '''
    This method calculates the self correlation of a matrix with itself

    Args:

    * `matrix` - a matrix where each row corresponds to a stock and each column corresponds to
        historical data for that stock
    
    Returns:

    > The self correlation of the given matrix with itself
    '''

    num_stocks = len(matrix)
    half_width = int(num_stocks / 2)
    matrix1 = [
        matrix[:half_width].astype('float16'), matrix[half_width:].astype('float16'),
        matrix[:half_width].astype('float16')
    ]

    matrix2 = [
        matrix[:half_width].astype('float16'), matrix[half_width:].astype('float16'),
        matrix[half_width:].astype('float16')
    ]

    self_corr_data = [get_correlation(*matrices)
                      for matrices in zip(matrix1, matrix2)]

    top_half_result = np.hstack((self_corr_data[0], self_corr_data[2]))
    bottom_half_result = np.hstack((self_corr_data[2].T, self_corr_data[1]))
    res = np.vstack((top_half_result, bottom_half_result))
    return res.astype('float16')

@exception_wrapper(HistoricalReturnsException, 'Could not calculate returns')
def get_returns(price_matrix: _nested_ndarray) -> _nested_ndarray:
    '''
    This method calculates the historical returns from an array of stock price data

    Args:

    * `price_matrix` - each row corresponds to a stock, the columns correspond to historical prices for a
        given stock. The latest price is at index 0.
    
    Returns:

    > A matrix where each row corresponds to a stock and the columns correspond to the historical returns
    for the given stock.
    '''
    return (price_matrix[:, :-1] - price_matrix[:, 1:]) / price_matrix[:, 1:]

def perturb_correlations(corr_matrix: _nested_ndarray, index_to_perturb: int,
    perturb_mag: float =1.) -> _nested_ndarray:
    '''
    This method perturbs the correlations of a row and column in a given matrix by a fixed amount
    equal to `perturb_mag`. This has the effect of making that row and column more (or less) important in
    future calculations (such as PCA)

    Args:

    * `corr_matrix` - a square matrix of correlations
    * `index_to_perturb` - the row and column that you wish to perturb.
    * `perturb_mag` - the degree to which you would like to perturb the provided row and column

    Returns:

    > A perturbed `corr_matrix`
    '''

    corr_matrix[:, index_to_perturb] += perturb_mag * np.sign(corr_matrix[
        :, index_to_perturb])
    corr_matrix[index_to_perturb, :] += perturb_mag * np.sign(corr_matrix[
        index_to_perturb, :])
    return corr_matrix

def do_pca(corr_matrix: _nested_ndarray, num_dim: int,
    min_var_explanation: float =0.7) -> _nested_ndarray:
    '''
    This method performs PCA on a self-correlation matrix, reducing the number of columns to `num_dim`.
    If such analysis does not sufficiently explain the underlying variance in the data, an exception is
    thrown.
    
    Args:

    * `corr_matrix` - a square matrix of correlations
    * `num_dim` - the number of dimensions to which the data should be reduced
    * `min_var_explanation` - the minimum fraction of the underlying data variance that should be explained

    Returns:

    > A matrix of the PCA result on `corr_matrix`.
    '''

    num_dim = int(num_dim)
    pca = PCA(n_components=num_dim, random_state=0)
    pca_result = pca.fit_transform(corr_matrix)
    var_ratio = pca.explained_variance_ratio_
    if sum(var_ratio) < min_var_explanation:
        raise PcaAccuracyException(
            'PCA doesn\'t explain enough of the variance in the data')

    return pca_result

def group_pca_result(pca_result: _nested_ndarray,
    stock_names: List[str]) -> Dict[int, List[PrincipalStockData]]:
    '''
    Group the results of PCA, by assigning each stock to the axis that explains most of its variance.

    Args:

    * `pca_result` - a matrix of size N x M, where N is the number of stocks and M is the number of
        dimensions after PCA.
    * `stock_names` - a list of names corresponding to the rows in pca_result

    Returns:

    > A mapping from principal axis to a row in PCA result. The key is the principal axis and it

    :arg pca_result: a matrix of size N x m, where N is the number of stocks and m is the 
        number of dimensions. Note - the max possible number of columns in the pca result.
    :arg stock_names: a list of names corresponding to the rows of the pca_result

    :ret dict: a dict mapping principal axis to rows in `pca_result`.
    '''

    num_cols = len(pca_result[0])  # the total number of columns

    # which principal axis each row in the pca_result belongs to
    grouped_data = np.argmax(np.abs(pca_result), axis = 1)

    groups = {}

    for i in range(num_cols):
        pca_result_axis_i = grouped_data == i

        groups[i] = [
            PrincipalStockData(name, data)
            for name, data, truthyness in zip(stock_names, pca_result, pca_result_axis_i)
            if truthyness
        ]

    return groups

def extract_stocks_from_group(grouped_data: Dict[int, List[PrincipalStockData]],
    stocks_to_extract: List[str], num_to_extract: int) -> Dict[str, _ndarray]:
    '''
    This method diversifies a set of extracted stock data based on the PCA result.

    Args:

    * `grouped_data` - see `group_pca_result`
    * `stocks_to_extract` - a list of stock ticker symbols that must be a part of the diversified set

    Returns:

    > A list of atleast length num_to_extract, of PrincipalStockData corresponding to
        the stocks that lie most on the principal axes
    '''
    total_extracted = 0

    extracted_stock_data = {}

    loop_count = 3

    try:
        while True:
            loop_count -= 1

            for p_axis in range(len(grouped_data)):
                
                data = [
                    x for x in grouped_data[p_axis] if x.name not in extracted_stock_data
                ]
                
                required_stocks = {
                    x.name: x.data for x in data if x.name in stocks_to_extract
                }

                extracted_stock_data = {** extracted_stock_data, ** required_stocks}

                non_required_stocks = [
                    x for x in data if x.name not in extracted_stock_data
                ]

                sorted_data_p_axis = sorted(
                    non_required_stocks,
                    key=lambda stock_data: stock_data.data[p_axis]
                )

                if len(sorted_data_p_axis) == 0:
                    # if we do not have enough stocks in the group to extract
                    # continue to the next iteration of the for loop
                    continue

                neg_stock = sorted_data_p_axis[0]
                pos_stock = sorted_data_p_axis[-1]

                # get two stocks, one that has the most positive overall effect... 
                # and one that has the least positive overall effect
                extracted_stock_data[neg_stock.name] = neg_stock.data
                extracted_stock_data[pos_stock.name] = pos_stock.data

                if len(extracted_stock_data) >= num_to_extract:
                    # if we have enough data extracted, exit the for loop
                   raise StopIteration

            if loop_count < 0:
                # if the loop count has been exhausted, exit the while loop
                raise StopIteration
    
    except StopIteration as exc:
        pass

    return extracted_stock_data


class _DiversificationHelperTest(unittest.TestCase):

    def setUp(self):
        self.data = np.array([[1., 2.1, 3.], [4., 6., 7.], [9.3, 11., 17.5]])
        self.dup_data = np.array([[1., 1., 3.], [1., 1., 3.], [1 + 1e-9, 1. - 1e-9, 3.]])
        self.test_data1 = np.array([[1., 2., 3.], [4., 5., 6.]])
        self.test_data2 = np.array([[6., 9., 8.], [9., 13., 21.]])
        self.rand_data = np.random.rand(100, 100)
        self.test_data3 = np.array([[1., 1., 1.], [1., 1., 1.]])

    def test_get_correlation(self):
        res = get_correlation(self.test_data1, self.test_data1)

        self.assertEqual((2, 2), res.shape)

        for row in range(2):
            for col in range(2):
                self.assertAlmostEqual(res[row][col], 1.)

        res = get_correlation(self.test_data1, self.test_data2)

        self.assertEqual((2, 2), res.shape)

        for row in range(2):
            for col in range(2):
                self.assertNotAlmostEqual(res[row][col], 1.)

    def test_self_corr(self):
        res = get_self_correlation(self.rand_data)

        exact_res = get_correlation(self.rand_data,
            self.rand_data).astype('float16')

        self.assertAlmostEqual(res.shape, exact_res.shape)

        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                self.assertAlmostEqual(res[i][j], exact_res[i][j], places=2)

    def test_returns(self):
        expected_res = [[2. / 1., 3. / 2.], [5. / 4., 6. / 5.]]

        res = get_returns(self.test_data1)

    def test_exc(self):
        with self.assertRaises(StockCorrelationException):
            get_correlation(self.test_data3, None)

        with self.assertRaises(StockCorrelationException):
            get_self_correlation(list(self.test_data3))

        with self.assertRaises(HistoricalReturnsException):
            get_returns(list(self.test_data3))

    def test_perturb_correlations(self):
        perturbed_data = perturb_correlations(self.data, 1, 1)

        expected_result = np.array([[1., 3.1, 3.], [5., 8., 8.], [9.3, 12., 17.5]])

        self.assertEqual(
            (perturbed_data.reshape(1, -1) == expected_result.reshape(1, -1)).all(),
            True
        )

    def test_do_pca(self):
        pca_res = do_pca(self.dup_data, 3)

        for datum in pca_res.reshape(-1, 1):
            self.assertAlmostEqual(datum[0], 0.)

        pca_res = do_pca(self.data, 2).reshape(1, -1)[0]
        expected_pca = PCA(n_components = 2)
        expected_res = expected_pca.fit_transform(self.data).reshape(1, -1)[0]

        for expected, actual in zip(expected_res, pca_res):
            self.assertAlmostEqual(expected, actual)

    def test_group_pca_res(self):
        stock_names = ['a', 'b', 'c']
        pca_res = do_pca(self.data, 2)
        grouped_res = group_pca_result(pca_res, stock_names)

        self.assertEqual(len(grouped_res[1]), 0)
        self.assertEqual(len(grouped_res[0]), 3)

        for group_row, pca_row in zip(grouped_res[0], pca_res):
            self.assertEqual((group_row.data == pca_row).all(), True)

    def test_extract_stocks_from_groups(self):
        for i in range(100):
            data = np.random.rand(300, 52)
            names = [str(x) for x in range(300)]
            grouped_data = group_pca_result(data, names)

            extracted = extract_stocks_from_group(grouped_data, [], 3)
            self.assertTrue(len(extracted) - 3 >= 0.) # atleast 3 stocks
            self.assertTrue(abs(len(extracted) - 4) <= 2.) # give or take two stocks from 4

            extracted = extract_stocks_from_group(grouped_data, [], 4)
            self.assertTrue(len(extracted) - 4 >= 0.) # atleast 4 stocks
            self.assertTrue(abs(len(extracted) - 4) <= 2.) # give or take two stocks from 4

            extracted = extract_stocks_from_group(grouped_data, names[:3], 7)
            self.assertTrue(len(extracted) - 7 >= 0.) # atleast 7 stocks
            self.assertTrue(len(extracted) - 8 <= 2.) # give or take two stocks from 8

            extracted = extract_stocks_from_group(grouped_data, names[:3], 8)
            self.assertTrue(len(extracted) - 8 >= 0.) # atleast 8 stocks
            self.assertTrue(len(extracted) - 8 <= 2.) # give or take two stocks from 8

    def test_workflow(self):
        data = np.random.rand(300, 300)
        names = [str(x) for x in range(300)]

        data = perturb_correlations(data, 30)
        data = perturb_correlations(data, 100)
        data = perturb_correlations(data, 210)

        pca_data = do_pca(data, 30, 0.1)
        grouped_data = group_pca_result(pca_data, names)
        extracted = extract_stocks_from_group(grouped_data, [], 30)
        self.assertTrue(len(extracted) - 30 >= 0.)
        self.assertTrue(len(extracted) - 30 <= 2.)

if __name__ == '__main__':
    unittest.main()