"""
Created on 23/10/17
@author: Maurizio Ferrari Dacrema
"""
import sys
import time

import numpy as np
import scipy.sparse as sps

from lib.recommender_utils import check_matrix


class ComputeSimilarityPython:
    def __init__(self, data_matrix, top_k=100, shrink=0, normalize=True, asymmetric_alpha=0.5, tversky_alpha=1.0,
                 tversky_beta=1.0, similarity='cosine', row_weights=None):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param data_matrix:
        :param top_k:
        :param shrink:
        :param normalize:           If True divide the dot product by the product of the norms
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param asymmetric_alpha     Coefficient alpha for the asymmetric cosine
        :param similarity:  'cosine'        computes Cosine similarity
                            'adjusted'      computes Adjusted Cosine, removing the average of the users
                            'asymmetric'    computes Asymmetric Cosine
                            'pearson'       computes Pearson Correlation, removing the average of the items
                            'jaccard'       computes Jaccard similarity for binary interactions using Tanimoto
                            'dice'          computes Dice similarity for binary interactions
                            'tversky'       computes Tversky similarity for binary interactions
                            'tanimoto'      computes Tanimoto coefficient for binary interactions

        Asymmetric Cosine as described in: 
        Aiolli, F. (2013, October). Efficient top-n recommendation for very large scale binary rated datasets.
        In Proceedings of the 7th ACM conference on Recommender systems (pp. 273-280). ACM.
        """
        super(ComputeSimilarityPython, self).__init__()

        self.shrink = shrink
        self.normalize = normalize

        self.n_rows, self.n_columns = data_matrix.shape
        self.top_k = min(top_k, self.n_columns)

        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.dataMatrix = data_matrix.copy()

        self.adjusted_cosine = False
        self.asymmetric_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False
        self.dice_coefficient = False
        self.tversky_coefficient = False

        if similarity == 'adjusted':
            self.adjusted_cosine = True
        elif similarity == 'asymmetric':
            self.asymmetric_cosine = True
        elif similarity == 'pearson':
            self.pearson_correlation = True
        elif similarity == 'jaccard' or similarity == 'tanimoto':
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif similarity == 'dice':
            self.dice_coefficient = True
            self.normalize = False

        elif similarity == 'tversky':
            self.tversky_coefficient = True
            self.normalize = False

        elif similarity == 'cosine':
            pass
        else:
            raise ValueError('Cosine_Similarity: value for parameter "mode" not recognized. Allowed values are: '
                             '"cosine", "pearson", "adjusted", "asymmetric", "jaccard", "tanimoto", "dice", "tversky". '
                             'Passed value was "{}"'.format(similarity))

        self.use_row_weights = False

        if row_weights is not None:

            if data_matrix.shape[0] != len(row_weights):
                raise ValueError('Cosine_Similarity: provided row_weights and dataMatrix have different number of rows.'
                                 'Col_weights has {} columns, dataMatrix has {}.'.format(len(row_weights),
                                                                                         data_matrix.shape[0]))

            self.use_row_weights = True
            self.row_weights = row_weights.copy()
            self.row_weights_diag = sps.diags(self.row_weights)

            self.dataMatrix_weighted = self.dataMatrix.T.dot(self.row_weights_diag).T

    def apply_adjusted_cosine(self):
        """
        Remove from every data point the average for the corresponding row
        :return:
        """
        self.dataMatrix = check_matrix(self.dataMatrix, 'csr')

        interactions_per_row = np.diff(self.dataMatrix.indptr)

        nonzero_rows = interactions_per_row > 0
        sum_per_row = np.asarray(self.dataMatrix.sum(axis=1)).ravel()

        row_average = np.zeros_like(sum_per_row)
        row_average[nonzero_rows] = sum_per_row[nonzero_rows] / interactions_per_row[nonzero_rows]

        # Split in blocks to avoid duplicating the whole data structure
        start_row = 0
        end_row = 0

        block_size = 1000

        while end_row < self.n_rows:
            end_row = min(self.n_rows, end_row + block_size)

            self.dataMatrix.data[self.dataMatrix.indptr[start_row]:self.dataMatrix.indptr[end_row]] -= \
                np.repeat(row_average[start_row:end_row], interactions_per_row[start_row:end_row])

            start_row += block_size

    def apply_pearson_correlation(self):
        """
        Remove from every data point the average for the corresponding column
        :return:
        """
        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        interactions_per_col = np.diff(self.dataMatrix.indptr)

        nonzero_cols = interactions_per_col > 0
        sum_per_col = np.asarray(self.dataMatrix.sum(axis=0)).ravel()

        col_average = np.zeros_like(sum_per_col)
        col_average[nonzero_cols] = sum_per_col[nonzero_cols] / interactions_per_col[nonzero_cols]

        # Split in blocks to avoid duplicating the whole data structure
        start_col = 0
        end_col = 0

        block_size = 1000

        while end_col < self.n_columns:
            end_col = min(self.n_columns, end_col + block_size)

            self.dataMatrix.data[self.dataMatrix.indptr[start_col]:self.dataMatrix.indptr[end_col]] -= \
                np.repeat(col_average[start_col:end_col], interactions_per_col[start_col:end_col])

            start_col += block_size

    def use_only_boolean_interactions(self):
        # Split in blocks to avoid duplicating the whole data structure
        start_pos = 0
        end_pos = 0

        block_size = 1000

        while end_pos < len(self.dataMatrix.data):
            end_pos = min(len(self.dataMatrix.data), end_pos + block_size)

            self.dataMatrix.data[start_pos:end_pos] = np.ones(end_pos - start_pos)

            start_pos += block_size

    def compute_similarity(self, start_col=None, end_col=None, block_size=100):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :param block_size
        :return:
        """
        values = []
        rows = []
        cols = []

        start_time = time.time()
        start_time_print_batch = start_time
        processed_items = 0

        if self.adjusted_cosine:
            self.apply_adjusted_cosine()

        elif self.pearson_correlation:
            self.apply_pearson_correlation()

        elif self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient:
            self.use_only_boolean_interactions()

        # We explore the matrix column-wise
        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        # Compute sum of squared values to be used in normalization
        sum_of_squared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()

        # Tanimoto does not require the square root to be applied
        if not (self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient):
            sum_of_squared = np.sqrt(sum_of_squared)

        if self.asymmetric_cosine:
            sum_of_squared_to_1_minus_alpha = np.power(sum_of_squared, 2 * (1 - self.asymmetric_alpha))
            sum_of_squared_to_alpha = np.power(sum_of_squared, 2 * self.asymmetric_alpha)

        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        start_col_local = 0
        end_col_local = self.n_columns

        if start_col is not None and 0 < start_col < self.n_columns:
            start_col_local = start_col

        if end_col is not None and start_col_local < end_col < self.n_columns:
            end_col_local = end_col

        start_col_block = start_col_local

        this_block_size = 0

        # Compute all similarities for each item using vectorization
        while start_col_block < end_col_local:

            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block - start_col_block

            # All data points for a given item
            item_data = self.dataMatrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray().squeeze()

            # If only 1 feature avoid last dimension to disappear
            if item_data.ndim == 1:
                item_data = np.atleast_2d(item_data)

            if self.use_row_weights:
                this_block_weights = self.dataMatrix_weighted.T.dot(item_data)

            else:
                # Compute item similarities
                this_block_weights = self.dataMatrix.T.dot(item_data)

            for col_index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_column_weights = this_block_weights
                else:
                    this_column_weights = this_block_weights[:, col_index_in_block]

                column_index = col_index_in_block + start_col_block
                this_column_weights[column_index] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:

                    if self.asymmetric_cosine:
                        denominator = sum_of_squared_to_alpha[column_index] * sum_of_squared_to_1_minus_alpha + \
                                      self.shrink + 1e-6
                    else:
                        denominator = sum_of_squared[column_index] * sum_of_squared + self.shrink + 1e-6

                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                # Apply the specific denominator for Tanimoto
                elif self.tanimoto_coefficient:
                    denominator = sum_of_squared[
                                      column_index] + sum_of_squared - this_column_weights + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.dice_coefficient:
                    denominator = sum_of_squared[column_index] + sum_of_squared + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.tversky_coefficient:
                    denominator = this_column_weights + \
                                  (sum_of_squared[column_index] - this_column_weights) * self.tversky_alpha + \
                                  (sum_of_squared - this_column_weights) * self.tversky_beta + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                # If no normalization or tanimoto is selected, apply only shrink
                elif self.shrink != 0:
                    this_column_weights = this_column_weights / self.shrink

                # this_column_weights = this_column_weights.toarray().ravel()

                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(self.top_k - 1)[0:self.top_k]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                not_zeros_mask = this_column_weights[top_k_idx] != 0.0
                num_not_zeros = np.sum(not_zeros_mask)

                values.extend(this_column_weights[top_k_idx][not_zeros_mask])
                rows.extend(top_k_idx[not_zeros_mask])
                cols.extend(np.ones(num_not_zeros) * column_index)

            # Add previous block size
            processed_items += this_block_size

            if time.time() - start_time_print_batch >= 30 or end_col_block == end_col_local:
                columns_per_sec = processed_items / (time.time() - start_time + 1e-9)

                print('similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min'.format(
                    processed_items, processed_items / (end_col_local - start_col_local) * 100, columns_per_sec,
                                     (time.time() - start_time) / 60))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()

            start_col_block += block_size

        # End while on columns

        W_sparse = sps.csr_matrix((values, (rows, cols)),
                                  shape=(self.n_columns, self.n_columns),
                                  dtype=np.float32)

        return W_sparse
