import numpy as np
import similaripy
import similaripy.normalization
from sklearn.preprocessing import normalize


class ItemBasedCollaborativeFiltering(object):
    def __init__(self, top_k=10, shrink=400):
        # 0.04711466950872567
        self.top_k = top_k
        self.shrink = shrink
        self.training_urm = None
        self.recommendations = None

    def fit(self, training_urm):
        self.training_urm = training_urm
        self.training_urm = similaripy.normalization.bm25(self.training_urm.transpose().tocsr())
        similarity_matrix = similaripy.dice(self.training_urm, k=self.top_k, shrink=self.shrink, binary=True)
        similarity_matrix = similarity_matrix.transpose().tocsr()
        self.training_urm = self.training_urm.transpose().tocsr()
        self.recommendations = self.training_urm.dot(similarity_matrix)

    def get_expected_ratings(self, user_id):
        expected_ratings = self.recommendations[user_id]
        expected_ratings = normalize(expected_ratings, axis=1, norm='max').tocsr()
        expected_ratings = expected_ratings.toarray().ravel()
        interacted_items = self.training_urm[user_id]
        expected_ratings[interacted_items.indices] = -100
        return expected_ratings

    def recommend(self, user_id, k=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), axis=0)
        unseen_items_mask = np.in1d(recommended_items, self.training_urm[user_id].indices, assume_unique=True,
                                    invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[:k]
