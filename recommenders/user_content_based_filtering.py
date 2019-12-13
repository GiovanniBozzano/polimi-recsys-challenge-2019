import similaripy
import similaripy.normalization
from sklearn.preprocessing import normalize

from recommenders.base_recommender import BaseRecommender


class UserContentBasedFiltering(BaseRecommender):
    name = 'user_content_based_filtering'

    # 0.009671896521098483
    def __init__(self, session, user_interactions_threshold=0, item_interactions_threshold=0,
                 top_k_user_age=2000, top_k_user_region=2000, shrink_user_age=40, shrink_user_region=40,
                 weight_user_age=0.4):
        super().__init__(session, user_interactions_threshold, item_interactions_threshold)
        self.top_k_user_age = top_k_user_age
        self.top_k_user_region = top_k_user_region
        self.shrink_user_age = shrink_user_age
        self.shrink_user_region = shrink_user_region
        self.weight_user_age = weight_user_age
        self.similarity_matrix = None

    def fit(self, training_urm):
        super().fit(training_urm)

        users_ages = self.session.get_ucm_ages()
        users_ages = similaripy.normalization.bm25plus(users_ages)
        users_regions = self.session.get_ucm_regions()
        users_regions = similaripy.normalization.bm25plus(users_regions)

        users_ages_similarity_matrix = similaripy.dice(users_ages, k=self.top_k_user_age, shrink=self.shrink_user_age,
                                                       binary=False, verbose=False)
        users_ages_similarity_matrix = users_ages_similarity_matrix.tocsr()

        users_regions_similarity_matrix = similaripy.dice(users_regions, k=self.top_k_user_region,
                                                          shrink=self.shrink_user_region, binary=False, verbose=False)
        users_regions_similarity_matrix = users_regions_similarity_matrix.tocsr()

        self.similarity_matrix = users_ages_similarity_matrix * self.weight_user_age + \
                                 users_regions_similarity_matrix * (1 - self.weight_user_age)

    def get_ratings(self, training_urm, user_id):
        similar_users = self.similarity_matrix[user_id]
        ratings = similar_users.dot(training_urm)
        ratings = normalize(ratings, axis=1, norm='max')
        ratings = ratings.toarray().ravel()
        return ratings
