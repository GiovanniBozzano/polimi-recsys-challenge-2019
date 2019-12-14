import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

from recommenders.base_recommender import BaseRecommender


class SVD(BaseRecommender):
    name = 'svd'

    # 0.02403678660970901
    def __init__(self, session, user_interactions_threshold=0, item_interactions_threshold=0,
                 n_factors=100):
        super().__init__(session, user_interactions_threshold, item_interactions_threshold)
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None

    def fit(self, training_urm):
        training_urm = super().fit(training_urm)

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
        interacted_items = training_urm[user_id]
        ratings[interacted_items.indices] = -100
        return ratings
