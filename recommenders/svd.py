import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

from recommenders.recommender import Recommender


class SVD(Recommender):
    name = 'svd'

    # 0.02403678660970901
    def __init__(self, session, n_factors=100):
        super().__init__(session)
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None

    def fit(self, training_urm):
        super().fit(self)

        U, s, V = randomized_svd(training_urm,
                                 n_components=self.n_factors,
                                 n_oversamples=5,
                                 n_iter=6)
        s_V = sparse.diags(s) * V
        self.user_factors = U
        self.item_factors = s_V.T

    def get_ratings(self, training_urm, user_id):
        ratings = np.dot(self.user_factors[user_id], self.item_factors.T)
        ratings = ratings - ratings.min()
        ratings = ratings.reshape(1, -1)
        ratings = normalize(ratings, axis=1, norm='max')
        ratings = ratings.ravel()
        return ratings
