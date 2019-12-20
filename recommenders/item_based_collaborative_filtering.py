import numpy as np
import scipy.sparse as sps
import similaripy
import similaripy.normalization

from recommenders.base_recommender import BaseRecommender


class ItemBasedCollaborativeFiltering(BaseRecommender):
    name = 'item_based_collaborative_filtering'

    # 0.04716062001147311

    # 0.047007276952241155
    # 0.047739194497587414
    # 0.04861113295070362

    # 0.04811303008331077
    # 0.048589832068092906
    # 0.04964753809552925

    # 0.04883036848731215
    def __init__(self, session, user_interactions_threshold=0, item_interactions_threshold=1,
                 top_k=20, shrink=550):
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
        if np.max(ratings) != 0:
            ratings = ratings / np.max(ratings)
        ratings = ratings.toarray().ravel()
        interacted_items = training_urm[user_id]
        ratings[interacted_items.indices] = -100
        return ratings
