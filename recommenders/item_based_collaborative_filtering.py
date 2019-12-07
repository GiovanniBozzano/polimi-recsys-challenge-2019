import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

import utils
from lib.similarity.compute_similarity import ComputeSimilarity, SimilarityFunction


class ItemBasedCollaborativeFiltering(object):
    def __init__(self, top_k=10, shrink=500, similarity=SimilarityFunction.JACCARD.value):
        # 0.046665268932742816
        self.top_k = top_k
        self.shrink = shrink
        self.similarity = similarity
        self.training_urm = None
        self.recommendations = None

    def generate_similarity_matrix(self):
        similarity_object = ComputeSimilarity(self.training_urm.transpose().tocsr(), top_k=self.top_k,
                                              shrink=self.shrink, similarity=self.similarity)
        similarity_object = similarity_object.compute_similarity()
        return similarity_object

    def fit(self, training_urm):
        self.training_urm = training_urm
        self.training_urm = utils.get_matrix_tfidf(self.training_urm.transpose())
        similarity_matrix = self.generate_similarity_matrix()
        self.training_urm = self.training_urm.transpose().tocsr()
        self.recommendations = self.training_urm.dot(similarity_matrix)

    def get_expected_ratings(self, user_id):
        expected_ratings = self.recommendations[user_id]
        expected_ratings = normalize(expected_ratings, axis=1, norm='l2').tocsr()
        expected_ratings = expected_ratings.toarray().ravel()
        if user_id == 0:
            print('0 I_CF RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        if user_id == 1:
            print('1 I_CF RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        if user_id == 2:
            print('2 I_CF RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        if user_id == 19335:
            print('19335 I_CF RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        return expected_ratings

    def recommend(self, user_id, k=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), axis=0)
        unseen_items_mask = np.in1d(recommended_items, self.training_urm[user_id].indices, assume_unique=True,
                                    invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[:k]
