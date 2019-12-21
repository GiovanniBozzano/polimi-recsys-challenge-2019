import numpy as np
import scipy.sparse as sps
import similaripy
import similaripy.normalization

from recommenders.base_recommender import BaseRecommender


class UserBasedCollaborativeFiltering(BaseRecommender):
    name = 'user_based_collaborative_filtering'

    # 0.0457084618229362
    def __init__(self, session, user_interactions_threshold=0, item_interactions_threshold=2,
                 top_k=1500, shrink=5):
        super().__init__(session, user_interactions_threshold, item_interactions_threshold)
        self.top_k = top_k
        self.shrink = shrink
        self.recommendations = None

    def fit(self, training_urm):
        training_urm = super().fit(training_urm)

        users_ages = self.session.get_ucm_ages()
        users_regions = self.session.get_ucm_regions()

        interactions = similaripy.normalization.bm25plus(training_urm, axis=1, k1=1.2, b=0.75, delta=0.85, tf_mode='raw', idf_mode='bm25', inplace=False)
        users_ages = similaripy.normalization.bm25(users_ages)
        users_regions = similaripy.normalization.tfidf(users_regions)

        matrix = sps.hstack((interactions, users_ages, users_regions))

        similarity_matrix = similaripy.cosine(matrix, k=self.top_k, shrink=self.shrink, binary=False, verbose=False)
        similarity_matrix = similarity_matrix.transpose().tocsr()

        training_urm = similaripy.normalization.bm25plus(training_urm, axis=1, k1=1.2, b=0.75, delta=0.8, tf_mode='raw', idf_mode='bm25', inplace=False)
        training_urm = similaripy.normalization.bm25plus(training_urm, axis=1, k1=1.2, b=0.75, delta=0.8, tf_mode='raw', idf_mode='bm25', inplace=False)

        self.recommendations = similarity_matrix.dot(training_urm)

    def get_ratings(self, training_urm, user_id):
        ratings = self.recommendations[user_id]
        if np.max(ratings) != 0:
            ratings = ratings / np.max(ratings)
        ratings = ratings.toarray().ravel()
        interacted_items = training_urm[user_id]
        ratings[interacted_items.indices] = -100
        return ratings
