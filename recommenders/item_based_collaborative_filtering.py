import numpy as np
import scipy.sparse as sps
import similaripy
import similaripy.normalization

from recommenders.base_recommender import BaseRecommender


class ItemBasedCollaborativeFiltering(BaseRecommender):
    name = 'item_based_collaborative_filtering'

    # 0.0489724229038116

    def __init__(self, session, user_interactions_threshold=0, item_interactions_threshold=1,
                 top_k=20, shrink=500):
        super().__init__(session, user_interactions_threshold, item_interactions_threshold)
        self.top_k = top_k
        self.shrink = shrink
        self.recommendations = None

    def fit(self, training_urm):
        training_urm = super().fit(training_urm)

        items_sub_classes = self.session.get_icm_sub_classes()

        matrix = sps.hstack((training_urm.transpose().tocsr(), items_sub_classes))

        similarity_matrix = similaripy.dice(matrix, k=self.top_k, shrink=self.shrink, binary=True, verbose=False)
        similarity_matrix = similarity_matrix.transpose().tocsr()

        # Epic magic trick of destiny
        training_urm = training_urm.transpose().tocsr()
        training_urm = similaripy.normalization.bm25(training_urm)
        training_urm = training_urm.transpose().tocsr()

        self.recommendations = training_urm.dot(similarity_matrix)

    def get_ratings(self, training_urm, user_id):
        ratings = self.recommendations[user_id]
        # if np.max(ratings) != 0:
        #    ratings = ratings / np.max(ratings)
        ratings = ratings.toarray().ravel()
        if user_id == 0:
            print('I_CF')
            print(np.sort(ratings))
        interacted_items = training_urm[user_id]
        ratings[interacted_items.indices] = -100
        return ratings
