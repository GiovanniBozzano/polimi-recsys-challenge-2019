import similaripy
import similaripy.normalization
from sklearn.preprocessing import normalize

from recommenders.base_recommender import BaseRecommender


class UserBasedCollaborativeFiltering(BaseRecommender):
    name = 'user_based_collaborative_filtering'

    # 0.041997361777218724
    def __init__(self, session, user_interactions_threshold=0, item_interactions_threshold=0,
                 top_k=1000, shrink=5):
        super().__init__(session, user_interactions_threshold, item_interactions_threshold)
        self.top_k = top_k
        self.shrink = shrink
        self.recommendations = None

    def fit(self, training_urm):
        training_urm = super().fit(training_urm)

        training_urm = similaripy.normalization.bm25plus(training_urm)
        similarity_matrix = similaripy.cosine(training_urm, k=self.top_k, shrink=self.shrink, binary=False,
                                              verbose=False)
        similarity_matrix = similarity_matrix.transpose().tocsr()
        self.recommendations = similarity_matrix.dot(training_urm)

    def get_ratings(self, training_urm, user_id):
        ratings = self.recommendations[user_id]
        ratings = normalize(ratings, axis=1, norm='max')
        ratings = ratings.toarray().ravel()
        return ratings
