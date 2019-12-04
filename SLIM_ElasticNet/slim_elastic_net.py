import multiprocessing
import warnings
from functools import partial
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet


class SLIMElasticNet(object):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available
    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        https://www.slideshare.net/MarkLevy/efficient-slides
        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """
    # 0.042518585493269104
    def __init__(self, alpha=0.001, l1_ratio=0.1, fit_intercept=False, copy_X=False, precompute=False,
                 selection='cyclic', max_iter=3, tol=1e-4, top_k=50, positive_only=True,
                 workers=multiprocessing.cpu_count()):

        self.analyzed_items = 0
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.precompute = precompute
        self.selection = selection
        self.max_iter = max_iter
        self.tol = tol
        self.top_k = top_k
        self.positive_only = positive_only
        self.workers = workers
        self.training_urm = None
        self.W_sparse = None

    ''' 
    Fit given to each pool thread, to fit the W_sparse 
    '''
    def _partial_fit(self, current_item, X):

        model = ElasticNet(alpha=self.alpha,
                           l1_ratio=self.l1_ratio,
                           positive=self.positive_only,
                           fit_intercept=self.fit_intercept,
                           copy_X=self.copy_X,
                           precompute=self.precompute,
                           selection=self.selection,
                           max_iter=self.max_iter,
                           tol=self.tol)

        print('Processing item ' + str(current_item))

        # WARNING: make a copy of X to avoid race conditions on column j
        # TODO: We can probably come up with something better here.
        X_j = X.copy()
        # get the target column
        y = X_j[:, current_item].toarray()
        # set the j-th column of X to zero
        X_j.data[X_j.indptr[current_item]:X_j.indptr[current_item + 1]] = 0.0
        # fit one ElasticNet model per column
        model.fit(X_j, y)

        relevant_items_partition = (-model.coef_).argpartition(self.top_k)[0:self.top_k]
        relevant_items_partition_sorting = np.argsort(-model.coef_[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        non_zero_mask = model.coef_[ranking] > 0.0
        ranking = ranking[non_zero_mask]

        values = model.coef_[ranking]
        rows = ranking
        cols = [current_item] * len(ranking)

        return values, rows, cols

    def fit(self, training_urm):
        # Display ConvergenceWarning only once and not for every item it occurs
        warnings.simplefilter('ignore', category=ConvergenceWarning)

        self.training_urm = sps.csc_matrix(training_urm)

        n_items = self.training_urm.shape[1]
        print('Iterating for ' + str(n_items) + 'times')
        # fit item's factors in parallel

        # create a copy of the URM since each _pfit will modify it
        copy_urm = self.training_urm.copy()

        # oggetto riferito alla funzione nel quale predefinisco parte dell'input
        _pfit = partial(self._partial_fit, X=copy_urm)

        # creo un pool con un certo numero di processi
        pool = multiprocessing.Pool(processes=self.workers)

        # avvio il pool passando la funzione (con la parte fissa dell'input)
        # e il rimanente parametro, variabile
        res = pool.map(_pfit, np.arange(n_items))

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    def get_expected_ratings(self, user_id):
        user_profile = self.training_urm[user_id]
        expected_ratings = user_profile.dot(self.W_sparse).toarray().ravel()
        if user_id == 19335:
            print('ELASTICNET RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        """
        maximum = np.abs(expected_ratings).max(axis=0)
        if maximum > 0:
            expected_ratings = expected_ratings / maximum
        """
        return expected_ratings

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        scores = self.get_expected_ratings(user_id)
        user_profile = self.training_urm[user_id].indices
        scores[user_profile] = 0
        recommended_items = np.flip(np.argsort(scores), 0)
        return recommended_items[:at]
