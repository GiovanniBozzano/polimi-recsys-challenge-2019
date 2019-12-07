"""
Created on 06/06/18
@author: Maurizio Ferrari Dacrema
"""
from enum import Enum

import numpy as np
import scipy.sparse as sps

from lib.similarity.compute_similarity_euclidean import ComputeSimilarityEuclidean
from lib.similarity.compute_similarity_python import ComputeSimilarityPython


class SimilarityFunction(Enum):
    COSINE = 'cosine'
    PEARSON = 'pearson'
    JACCARD = 'jaccard'
    TANIMOTO = 'tanimoto'
    ADJUSTED_COSINE = 'adjusted'
    EUCLIDEAN = 'euclidean'


class ComputeSimilarity:
    def __init__(self, data_matrix, use_implementation='density', similarity=None, **args):
        """
        Interface object that will call the appropriate similarity implementation
        :param data_matrix:
        :param use_implementation:      'density' will choose the most efficient implementation automatically
                                        'cython' will use the cython implementation, if available. Most efficient for
                                                 sparse matrix
                                        'python' will use the python implementation. Most efficient for dense matrix
        :param similarity:              the type of similarity to use, see SimilarityFunction enum
        :param args:                    other args required by the specific similarity implementation
        """
        assert np.all(np.isfinite(data_matrix.data)), 'Compute_Similarity: Data matrix contains {} non finite values' \
            .format(np.sum(np.logical_not(np.isfinite(data_matrix.data))))
        self.dense = False
        if similarity == 'euclidean':
            # This is only available here
            self.compute_similarity_object = ComputeSimilarityEuclidean(data_matrix, **args)
        else:
            assert not (data_matrix.shape[0] == 1 and data_matrix.nnz == data_matrix.shape[1]), \
                'Compute_Similarity: data has only 1 feature (shape: {}) with dense values,' \
                ' vector and set based similarities are not defined on 1-dimensional dense data,' \
                ' use Euclidean similarity instead.'.format(data_matrix.shape)
            if similarity is not None:
                args['similarity'] = similarity
            if use_implementation == 'density':
                if isinstance(data_matrix, np.ndarray):
                    self.dense = True
                elif isinstance(data_matrix, sps.spmatrix):
                    shape = data_matrix.shape
                    num_cells = shape[0] * shape[1]
                    sparsity = data_matrix.nnz / num_cells
                    self.dense = sparsity > 0.5
                else:
                    print('Compute_Similarity: matrix type not recognized, calling default...')
                    use_implementation = 'python'
                if self.dense:
                    print('Compute_Similarity: detected dense matrix')
                    use_implementation = 'python'
                else:
                    use_implementation = 'cython'
            if use_implementation == 'cython':
                try:
                    from lib.similarity.compute_similarity_cython import ComputeSimilarityCython
                    self.compute_similarity_object = ComputeSimilarityCython(data_matrix, **args)
                except ImportError:
                    print('Unable to load cython Compute_Similarity, reverting to Python')
                    self.compute_similarity_object = ComputeSimilarityPython(data_matrix, **args)
            elif use_implementation == 'python':
                self.compute_similarity_object = ComputeSimilarityPython(data_matrix, **args)
            else:
                raise ValueError('Compute_Similarity: value for argument "use_implementation" not recognized')

    def compute_similarity(self, **args):
        return self.compute_similarity_object.compute_similarity(**args)
