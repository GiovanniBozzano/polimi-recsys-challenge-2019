"""
Created on 10/04/18
@author: Maurizio Ferrari Dacrema
"""
import numpy as np
import scipy.sparse as sps


def okapi_bm_25(data_matrix, K1=1.2, B=0.75):
    """
    Items are assumed to be on rows
    :param data_matrix:
    :param K1:
    :param B:
    :return:
    """
    assert 0 < B < 1, 'okapi_BM_25: B must be in (0,1)'
    assert K1 > 0, 'okapi_BM_25: K1 must be > 0'
    assert np.all(np.isfinite(data_matrix.data)), 'okapi_bm_25: Data matrix contains {} non finite values' \
        .format(np.sum(np.logical_not(np.isfinite(data_matrix.data))))

    # Weighs each row of a sparse matrix by OkapiBM25 weighting
    # calculate idf per term (user)

    data_matrix = sps.coo_matrix(data_matrix)

    N = float(data_matrix.shape[0])
    idf = np.log(N / (1 + np.bincount(data_matrix.col)))

    # calculate length_norm per document
    row_sums = np.ravel(data_matrix.sum(axis=1))

    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    denominator = K1 * length_norm[data_matrix.row] + data_matrix.data
    denominator[denominator == 0.0] += 1e-9

    data_matrix.data = data_matrix.data * (K1 + 1.0) / denominator * idf[data_matrix.col]

    return data_matrix.tocsr()
