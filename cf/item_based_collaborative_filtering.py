import numpy as np
import pandas as pd
import utils
from Base.Similarity.Compute_Similarity import Compute_Similarity, SimilarityFunction


class ItemBasedCollaborativeFiltering(object):
    def __init__(self, top_k=10, shrink=500, similarity=SimilarityFunction.JACCARD.value):
        # 0.04607811876661628
        self.top_k = top_k
        self.shrink = shrink
        self.similarity = similarity
        self.training_urm = None
        self.recommendations = None

    def generate_similarity_matrix(self):
        similarity_object = Compute_Similarity(self.training_urm, topK=self.top_k, shrink=self.shrink,
                                               similarity=self.similarity)
        similarity_object = similarity_object.compute_similarity()
        return similarity_object

    def fit(self, training_urm):
        self.training_urm = training_urm
        self.training_urm = utils.get_matrix_tfidf(self.training_urm.transpose())
        self.training_urm = self.training_urm.transpose().tocsr()
        similarity_matrix = self.generate_similarity_matrix()
        self.recommendations = self.training_urm.dot(similarity_matrix)

    def get_expected_ratings(self, user_id):
        expected_ratings = self.recommendations[user_id].toarray().ravel()
        if user_id == 19335:
            print('I_CF RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        maximum = np.abs(expected_ratings).max(axis=0)
        if maximum > 0:
            expected_ratings = expected_ratings / maximum
        return expected_ratings

    def recommend(self, user_id, k=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), axis=0)
        unseen_items_mask = np.in1d(recommended_items, self.training_urm[user_id].indices, assume_unique=True,
                                    invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[:k]
