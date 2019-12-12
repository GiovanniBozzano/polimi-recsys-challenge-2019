import numpy as np
import similaripy
import similaripy.normalization
from sklearn.preprocessing import normalize

import session


class UserContentBasedFiltering(object):

    def __init__(self, top_k_user_age=2000, top_k_user_region=2000,
                 shrink_user_age=40, shrink_user_region=40,
                 weight_user_age=0.4):
        # 0.009671896521098483
        self.top_k_user_age = top_k_user_age
        self.top_k_user_region = top_k_user_region
        self.shrink_user_age = shrink_user_age
        self.shrink_user_region = shrink_user_region
        self.weight_user_age = weight_user_age
        self.training_urm = None
        self.similarity_matrix = None

    def fit(self, training_urm):
        self.training_urm = training_urm

        users_ages = session.INSTANCE.get_ucm_ages()
        users_ages = similaripy.normalization.bm25plus(users_ages)
        users_regions = session.INSTANCE.get_ucm_regions()
        users_regions = similaripy.normalization.bm25plus(users_regions)

        users_ages_similarity_matrix = similaripy.dice(users_ages, k=self.top_k_user_age, shrink=self.shrink_user_age,
                                                       binary=False)
        users_ages_similarity_matrix = users_ages_similarity_matrix.tocsr()

        users_regions_similarity_matrix = similaripy.dice(users_regions, k=self.top_k_user_region,
                                                          shrink=self.shrink_user_region, binary=False)
        users_regions_similarity_matrix = users_regions_similarity_matrix.tocsr()

        self.similarity_matrix = users_ages_similarity_matrix * self.weight_user_age + \
                                 users_regions_similarity_matrix * (1 - self.weight_user_age)

    def get_expected_ratings(self, user_id):
        similar_users = self.similarity_matrix[user_id]
        expected_ratings = similar_users.dot(self.training_urm)
        expected_ratings = normalize(expected_ratings, axis=1, norm='max').tocsr()
        expected_ratings = expected_ratings.toarray().ravel()
        interacted_items = self.training_urm[user_id]
        expected_ratings[interacted_items.indices] = -100
        return expected_ratings

    def recommend(self, user_id, k=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), axis=0)
        unseen_items_mask = np.in1d(recommended_items, self.training_urm[user_id].indices, assume_unique=True,
                                    invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[:k]
