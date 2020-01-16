import multiprocessing
import warnings
from functools import partial

import numpy as np
import scipy.sparse as sps
from sklearn import linear_model
from sklearn.exceptions import ConvergenceWarning

from recommenders.base_recommender import BaseRecommender


class ElasticNet(BaseRecommender):
    name = 'elastic_net'

    def __init__(self, session, user_interactions_threshold=2, item_interactions_threshold=0,
                 alpha=0.001, l1_ratio=0.04, fit_intercept=False, copy_X=False, precompute=False, selection='cyclic',
                 max_iter=50, tol=1e-4, top_k=200, positive_only=True,
                 workers=int(multiprocessing.cpu_count() * 3 / 4)):
        super().__init__(session, user_interactions_threshold, item_interactions_threshold)
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
        self.W_sparse = None

    def _partial_fit(self, current_item, X):
        warnings.simplefilter('ignore', category=ConvergenceWarning)
        model = linear_model.ElasticNet(alpha=self.alpha,
                                        l1_ratio=self.l1_ratio,
                                        positive=self.positive_only,
                                        fit_intercept=self.fit_intercept,
                                        copy_X=self.copy_X,
                                        precompute=self.precompute,
                                        selection=self.selection,
                                        max_iter=self.max_iter,
                                        tol=self.tol,
                                        random_state=np.random.RandomState(self.session.random_seed))
        # WARNING: Make a copy of X to avoid race conditions on column j
        X_j = X.copy()
        # Get the target column
        y = X_j[:, current_item].toarray()
        # Set the j-th column of X to zero
        X_j.data[X_j.indptr[current_item]:X_j.indptr[current_item + 1]] = 0.0
        # Fit one ElasticNet model per column
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
        training_urm = super().fit(training_urm)

        training_urm = training_urm.tocsc()

        n_items = training_urm.shape[1]

        # Create a copy of the URM since each _pfit will modify it
        copy_urm = training_urm.copy()

        _pfit = partial(self._partial_fit, X=copy_urm)

        pool = multiprocessing.Pool(processes=self.workers)

        res = pool.map(_pfit, np.arange(n_items))

        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)

        self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    def get_ratings(self, training_urm, user_id):
        interacted_items = training_urm[user_id]
        ratings = interacted_items.dot(self.W_sparse)
        if np.max(ratings) != 0:
            ratings = ratings / np.max(ratings)
        ratings = ratings.toarray().ravel()
        ratings[interacted_items.indices] = -100
        return ratings
