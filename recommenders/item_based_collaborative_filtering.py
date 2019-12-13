import similaripy
import similaripy.normalization
from sklearn.preprocessing import normalize

from recommenders.recommender import Recommender


class ItemBasedCollaborativeFiltering(Recommender):
    name = 'item_based_collaborative_filtering'

    # 0.04711466950872567
    def __init__(self, session, top_k=10, shrink=400):
        super().__init__(session)
        self.top_k = top_k
        self.shrink = shrink
        self.recommendations = None

    def fit(self, training_urm):
        super().fit(self)

        training_urm = similaripy.normalization.bm25(training_urm.transpose().tocsr())
        similarity_matrix = similaripy.dice(training_urm, k=self.top_k, shrink=self.shrink, binary=True, verbose=False)
        similarity_matrix = similarity_matrix.transpose().tocsr()
        training_urm = training_urm.transpose().tocsr()
        self.recommendations = training_urm.dot(similarity_matrix)

    def get_ratings(self, training_urm, user_id):
        ratings = self.recommendations[user_id]
        ratings = normalize(ratings, axis=1, norm='max')
        ratings = ratings.toarray().ravel()
        return ratings
