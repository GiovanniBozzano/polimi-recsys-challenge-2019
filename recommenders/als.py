import os

import numpy as np
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import normalize


class ALS:

    def __init__(self, factors=448, regularization=100, iterations=30, alpha=21):
        # 0.04271062831051826
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.training_urm = None
        self.user_factors = None
        self.item_factors = None

    def fit(self, training_urm):
        self.training_urm = training_urm
        sparse_item_user = self.training_urm.transpose().tocsr()
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        model = AlternatingLeastSquares(factors=self.factors,
                                        regularization=self.regularization,
                                        iterations=self.iterations)
        data_confidence = (sparse_item_user * self.alpha).astype(np.float32)
        model.fit(data_confidence)
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors

    def get_expected_ratings(self, user_id):
        expected_ratings = np.dot(self.user_factors[user_id], self.item_factors.T)
        expected_ratings = expected_ratings - expected_ratings.min()
        expected_ratings = expected_ratings.reshape(1, -1)
        expected_ratings = normalize(expected_ratings, axis=1, norm='max')
        expected_ratings = expected_ratings.ravel()
        interacted_items = self.training_urm[user_id]
        expected_ratings[interacted_items.indices] = -100
        return expected_ratings

    def recommend(self, user_id, k=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        unseen_items_mask = np.in1d(recommended_items, self.training_urm[user_id].indices, assume_unique=True,
                                    invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[:k]
