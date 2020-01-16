import numpy as np


class BaseRecommender(object):
    @property
    def name(self):
        raise NotImplementedError

    def __init__(self, session, user_interactions_threshold=0, item_interactions_threshold=0):
        self.session = session
        self.user_interactions_threshold = user_interactions_threshold
        self.item_interactions_threshold = item_interactions_threshold

    def fit(self, training_urm):
        print('=== Fitting ' + self.name)
        training_urm = training_urm.copy()
        for user_id in self.session.user_list_unique:
            if training_urm[user_id].getnnz() <= self.user_interactions_threshold:
                for item_id in training_urm[user_id].indices:
                    training_urm[user_id, item_id] = 0
        training_urm = training_urm.transpose().tocsr()
        for item_id in self.session.item_list_unique:
            if training_urm[item_id].getnnz() <= self.item_interactions_threshold:
                for user_id in training_urm[item_id].indices:
                    training_urm[item_id, user_id] = 0
        training_urm = training_urm.transpose().tocsr()
        training_urm.eliminate_zeros()
        return training_urm

    def get_ratings(self, training_urm, user_id):
        raise NotImplementedError

    def recommend(self, training_urm, user_id, k=10):
        ratings = self.get_ratings(training_urm, user_id)
        recommended_items = np.flip(np.argsort(ratings), axis=0)
        unseen_items_mask = np.in1d(recommended_items, training_urm[user_id].indices, assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[:k]
