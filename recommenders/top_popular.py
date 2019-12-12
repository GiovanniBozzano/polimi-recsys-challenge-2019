import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize


class TopPopular(object):

    # 0.007290055996715304
    def __init__(self):
        self.training_urm = None
        self.ratings = None

    def fit(self, training_urm):
        self.training_urm = training_urm
        item_popularity = (self.training_urm > 0).sum(axis=0)
        item_popularity = np.array(item_popularity).squeeze()
        zeroes = np.zeros(self.training_urm.shape[1], dtype=np.int)
        self.ratings = sps.coo_matrix((item_popularity, (np.arange(self.training_urm.shape[1]), zeroes)),
                                      shape=(self.training_urm.shape[1], 1), dtype=np.float32).tocsr()

    def get_expected_ratings(self, user_id):
        expected_ratings = normalize(self.ratings, axis=0, norm='max').tocsr()
        expected_ratings = expected_ratings.toarray().ravel()
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
