import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import normalize


class AlternatingLeastSquare:

    def __init__(self, factors=448, regularization=100, iterations=30, alpha=24):
        # 0.042420263647561435
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
        # Initialize the als model and fit it using the sparse item-user matrix
        model = AlternatingLeastSquares(factors=self.factors,
                                        regularization=self.regularization,
                                        iterations=self.iterations)
        # Calculate the confidence by multiplying it by our alpha value.
        data_confidence = (sparse_item_user * self.alpha).astype(np.float64)
        # Fit the model
        model.fit(data_confidence)
        # Get the user and item vectors from our trained model
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors

    def get_expected_ratings(self, user_id):
        expected_ratings = self.user_factors[user_id].dot(self.item_factors.transpose())
        expected_ratings = expected_ratings - expected_ratings.min()
        expected_ratings = expected_ratings.reshape(1, -1)
        expected_ratings = normalize(expected_ratings, axis=1, norm='l2')
        expected_ratings = expected_ratings.ravel()
        if user_id == 0:
            print('0 ALS RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        if user_id == 1:
            print('1 ALS RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        if user_id == 2:
            print('2 ALS RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        return expected_ratings

    def recommend(self, user_id, k=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        unseen_items_mask = np.in1d(recommended_items, self.training_urm[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[:k]
