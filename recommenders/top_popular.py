import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize

from recommenders.recommender import Recommender


class TopPopular(Recommender):
    name = 'top_popular'

    # 0.007290055996715304
    def __init__(self, session):
        super().__init__(session)
        self.ratings = None

    def fit(self, training_urm):
        super().fit(self)

        item_popularity = (training_urm > 0).sum(axis=0)
        item_popularity = np.array(item_popularity).squeeze()
        zeroes = np.zeros(training_urm.shape[1], dtype=np.int)
        self.ratings = sps.coo_matrix((item_popularity, (np.arange(training_urm.shape[1]), zeroes)),
                                      shape=(training_urm.shape[1], 1), dtype=np.float32).tocsr()

    def get_ratings(self, training_urm, user_id):
        ratings = normalize(self.ratings, axis=0, norm='max')
        ratings = ratings.toarray().ravel()
        return ratings
