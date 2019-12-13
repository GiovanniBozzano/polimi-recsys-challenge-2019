import numpy as np


class Recommender(object):

    @property
    def name(self):
        raise NotImplementedError

    def __init__(self, session):
        self.session = session

    def fit(self, training_urm):
        print('=== Fitting ' + self.name)

    def get_ratings(self, training_urm, user_id):
        raise NotImplementedError

    def recommend(self, training_urm, user_id, k=10):
        ratings = self.get_ratings(training_urm, user_id)
        recommended_items = np.flip(np.argsort(ratings), axis=0)
        unseen_items_mask = np.in1d(recommended_items, training_urm[user_id].indices, assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[:k]
