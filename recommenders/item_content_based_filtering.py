import numpy as np
import similaripy
import similaripy.normalization

from lib.similarity.compute_similarity_euclidean import ComputeSimilarityEuclidean
from recommenders.base_recommender import BaseRecommender


class ItemContentBasedFiltering(BaseRecommender):
    name = 'item_content_based_filtering'

    def __init__(self, session, user_interactions_threshold=0, item_interactions_threshold=0,
                 top_k_item_asset=140, top_k_item_price=140, top_k_item_sub_class=300,
                 shrink_item_asset=1, shrink_item_price=1, shrink_item_sub_class=1, weight_item_asset=0.2,
                 weight_item_price=0.2):
        super().__init__(session, user_interactions_threshold, item_interactions_threshold)
        self.top_k_item_asset = top_k_item_asset
        self.top_k_item_price = top_k_item_price
        self.top_k_item_sub_class = top_k_item_sub_class
        self.shrink_item_asset = shrink_item_asset
        self.shrink_item_price = shrink_item_price
        self.shrink_item_sub_class = shrink_item_sub_class
        self.weight_item_asset = weight_item_asset
        self.weight_item_price = weight_item_price
        self.similarity_matrix = None

    def fit(self, training_urm):
        super().fit(training_urm)

        items_assets = self.session.get_icm_assets()
        items_prices = self.session.get_icm_prices()
        items_sub_classes = self.session.get_icm_sub_classes()
        items_sub_classes = similaripy.normalization.bm25plus(items_sub_classes)

        items_assets_similarity_matrix = ComputeSimilarityEuclidean(items_assets.transpose().tocsr(),
                                                                    top_k=self.top_k_item_asset,
                                                                    shrink=self.shrink_item_asset)
        items_assets_similarity_matrix = items_assets_similarity_matrix.compute_similarity()
        items_assets_similarity_matrix = items_assets_similarity_matrix.transpose().tocsr()

        items_prices_similarity_matrix = ComputeSimilarityEuclidean(items_prices.transpose().tocsr(),
                                                                    top_k=self.top_k_item_price,
                                                                    shrink=self.shrink_item_price)
        items_prices_similarity_matrix = items_prices_similarity_matrix.compute_similarity()
        items_prices_similarity_matrix = items_prices_similarity_matrix.transpose().tocsr()

        items_sub_classes_similarity_matrix = similaripy.cosine(items_sub_classes, k=self.top_k_item_sub_class,
                                                                shrink=self.shrink_item_sub_class, binary=False,
                                                                verbose=False)
        items_sub_classes_similarity_matrix = items_sub_classes_similarity_matrix.tocsr()
        self.similarity_matrix = items_assets_similarity_matrix * self.weight_item_asset + \
                                 items_prices_similarity_matrix * self.weight_item_price + \
                                 items_sub_classes_similarity_matrix * (
                                         1 - self.weight_item_asset - self.weight_item_price)

    def get_ratings(self, training_urm, user_id):
        interacted_items = training_urm[user_id]
        ratings = interacted_items.dot(self.similarity_matrix)
        if np.max(ratings) != 0:
            ratings = ratings / np.max(ratings)
        ratings = ratings.toarray().ravel()
        ratings[interacted_items.indices] = -100
        return ratings
