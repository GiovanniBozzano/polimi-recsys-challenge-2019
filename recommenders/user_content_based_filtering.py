import numpy as np
import similaripy
import similaripy.normalization

from recommenders.base_recommender import BaseRecommender


class UserContentBasedFiltering(BaseRecommender):
    name = 'user_content_based_filtering'

    def __init__(self, session, user_interactions_threshold=0, item_interactions_threshold=0,
                 top_k_user_age=2500, top_k_user_region=2500,
                 shrink_user_age=10, shrink_user_region=10,
                 weight_user_age=0.4):
        super().__init__(session, user_interactions_threshold, item_interactions_threshold)
        self.top_k_user_age = top_k_user_age
        self.top_k_user_region = top_k_user_region
        self.shrink_user_age = shrink_user_age
        self.shrink_user_region = shrink_user_region
        self.weight_user_age = weight_user_age
        self.similarity_matrix = None
        self.training_urm = None

    def fit(self, training_urm):
        super().fit(training_urm)

        users_ages = self.session.get_ucm_ages()
        users_regions = self.session.get_ucm_regions()

        users_ages = similaripy.normalization.bm25plus(users_ages)
        users_regions = similaripy.normalization.tfidf(users_regions)

        users_ages_similarity_matrix = similaripy.dice(users_ages, k=self.top_k_user_age, shrink=self.shrink_user_age,
                                                       binary=False, verbose=False)
        users_ages_similarity_matrix = users_ages_similarity_matrix.tocsr()

        users_regions_similarity_matrix = similaripy.dice(users_regions, k=self.top_k_user_region,
                                                          shrink=self.shrink_user_region, binary=False, verbose=False)
        users_regions_similarity_matrix = users_regions_similarity_matrix.tocsr()

        self.similarity_matrix = users_ages_similarity_matrix * self.weight_user_age + \
                                 users_regions_similarity_matrix * (1 - self.weight_user_age)

        self.training_urm = similaripy.normalization.bm25(training_urm)
        self.training_urm = similaripy.normalization.bm25(self.training_urm)

    def get_ratings(self, training_urm, user_id):
        similar_users = self.similarity_matrix[user_id]
        ratings = similar_users.dot(self.training_urm)
        if np.max(ratings) != 0:
            ratings = ratings / np.max(ratings)
        ratings = ratings.toarray().ravel()
        interacted_items = training_urm[user_id]
        ratings[interacted_items.indices] = -100
        return ratings
