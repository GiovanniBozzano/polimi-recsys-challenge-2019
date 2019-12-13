import os

import numpy as np
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import normalize

from recommenders.recommender import Recommender


class ALS(Recommender):
    name = 'als'

    # 0.04271062831051826
    def __init__(self, session, factors=448, regularization=100, iterations=30, alpha=21):
        super().__init__(session)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.user_factors = None
        self.item_factors = None

    def fit(self, training_urm):
        super().fit(self)

        sparse_item_user = training_urm.transpose().tocsr()
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        model = AlternatingLeastSquares(factors=self.factors,
                                        regularization=self.regularization,
                                        iterations=self.iterations)
        data_confidence = (sparse_item_user * self.alpha).astype(np.float32)
        model.fit(data_confidence, show_progress=False)
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors

    def get_ratings(self, training_urm, user_id):
        ratings = np.dot(self.user_factors[user_id], self.item_factors.T)
        ratings = ratings - ratings.min()
        ratings = ratings.reshape(1, -1)
        ratings = normalize(ratings, axis=1, norm='max')
        ratings = ratings.ravel()
        return ratings
