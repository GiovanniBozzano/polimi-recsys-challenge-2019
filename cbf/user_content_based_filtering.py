import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

import session
import utils
from Base.Similarity.Compute_Similarity import Compute_Similarity, SimilarityFunction


def compute_similarity(ucm, top_k, shrink, similarity):
    similarity_object = Compute_Similarity(ucm.transpose().tocsr(), topK=top_k, shrink=shrink, similarity=similarity)
    similarity_object = similarity_object.compute_similarity()
    return similarity_object


class UserContentBasedFiltering(object):

    def __init__(self, top_k_user_region=1000, top_k_user_age=1000,
                 shrink_user_region=1, shrink_user_age=1,
                 weight_user_region=0.6):

        # 0.009669101053250731
        self.top_k_user_region = top_k_user_region
        self.top_k_user_age = top_k_user_age
        self.shrink_user_region = shrink_user_region
        self.shrink_user_age = shrink_user_age
        self.weight_user_region = weight_user_region
        self.training_urm = None
        self.similarity_matrix = None

    def fit(self, training_urm):
        self.training_urm = training_urm

        users_regions = session.INSTANCE.get_ucm_regions()
        users_ages = session.INSTANCE.get_ucm_ages()
        users_regions = utils.get_matrix_bm_25(users_regions)
        users_ages = utils.get_matrix_bm_25(users_ages)

        users_regions_similarity_matrix = compute_similarity(users_regions, self.top_k_user_region,
                                                             self.shrink_user_region,
                                                             similarity=SimilarityFunction.COSINE.value)
        users_regions_similarity_matrix = users_regions_similarity_matrix.transpose().tocsr()

        users_ages_similarity_matrix = compute_similarity(users_ages, self.top_k_user_age, self.shrink_user_age,
                                                          similarity=SimilarityFunction.COSINE.value)
        users_ages_similarity_matrix = users_ages_similarity_matrix.transpose().tocsr()

        print(users_ages_similarity_matrix[0].getnnz())
        print(users_regions_similarity_matrix[0].getnnz())

        self.similarity_matrix = users_regions_similarity_matrix * self.weight_user_region + \
            users_ages_similarity_matrix * (1 - self.weight_user_region)

    def get_expected_ratings(self, user_id):
        similar_users = self.similarity_matrix[user_id]
        expected_ratings = similar_users.dot(self.training_urm)
        expected_ratings = normalize(expected_ratings, axis=1, norm='l2').tocsr()
        expected_ratings = expected_ratings.toarray().ravel()
        if user_id == 0:
            print('0 UCBF RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        if user_id == 1:
            print('1 UCBF RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        if user_id == 2:
            print('2 UCBF RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        interacted_items = self.training_urm[user_id]
        expected_ratings[interacted_items.indices] = -100
        return expected_ratings

    def recommend(self, user_id, k=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), axis=0)
        return recommended_items[:k]
