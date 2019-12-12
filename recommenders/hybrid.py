import numpy as np

from recommenders.als import ALS
from recommenders.elastic_net import ElasticNet
from recommenders.item_based_collaborative_filtering import ItemBasedCollaborativeFiltering
from recommenders.item_content_based_filtering import ItemContentBasedFiltering
from recommenders.lightfm import LightFM
from recommenders.slim_bpr import SLIMBPR
from recommenders.top_popular import TopPopular
from recommenders.user_based_collaborative_filtering import UserBasedCollaborativeFiltering
from recommenders.user_content_based_filtering import UserContentBasedFiltering


class Hybrid(object):
    def __init__(self, weights_cold_start, weights_low_interactions, weights_high_interactions,
                 user_content_based_filtering_parameters, item_content_based_filtering_parameters,
                 user_based_collaborative_filtering_parameters, item_based_collaborative_filtering_parameters,
                 slim_bpr_parameters, als_parameters):
        self.weights_cold_start = weights_cold_start
        self.weights_low_interactions = weights_low_interactions
        self.weights_high_interactions = weights_high_interactions

        self.user_content_based_filtering = \
            UserContentBasedFiltering(top_k_user_age=user_content_based_filtering_parameters['top_k_user_age'],
                                      top_k_user_region=user_content_based_filtering_parameters['top_k_user_region'],
                                      shrink_user_age=user_content_based_filtering_parameters['shrink_user_age'],
                                      shrink_user_region=user_content_based_filtering_parameters['shrink_user_region'],
                                      weight_user_age=user_content_based_filtering_parameters['weight_user_age'])
        self.item_content_based_filtering = \
            ItemContentBasedFiltering(top_k_item_asset=item_content_based_filtering_parameters['top_k_item_asset'],
                                      top_k_item_price=
                                      item_content_based_filtering_parameters['top_k_item_price'],
                                      top_k_item_sub_class=
                                      item_content_based_filtering_parameters['top_k_item_sub_class'],
                                      shrink_item_asset=item_content_based_filtering_parameters['shrink_item_asset'],
                                      shrink_item_price=item_content_based_filtering_parameters['shrink_item_price'],
                                      shrink_item_sub_class=
                                      item_content_based_filtering_parameters['shrink_item_sub_class'],
                                      weight_item_asset=item_content_based_filtering_parameters['weight_item_asset'],
                                      weight_item_price=item_content_based_filtering_parameters['weight_item_price'])

        self.user_based_collaborative_filtering = \
            UserBasedCollaborativeFiltering(top_k=user_based_collaborative_filtering_parameters['top_k'],
                                            shrink=user_based_collaborative_filtering_parameters['shrink'])
        self.item_based_collaborative_filtering = \
            ItemBasedCollaborativeFiltering(top_k=item_based_collaborative_filtering_parameters['top_k'],
                                            shrink=item_based_collaborative_filtering_parameters['shrink'])
        self.slim_bpr = SLIMBPR(epochs=slim_bpr_parameters['epochs'],
                                top_k=slim_bpr_parameters['top_k'])
        self.elastic_net = ElasticNet()
        self.als = ALS(factors=als_parameters['factors'],
                       regularization=als_parameters['regularization'],
                       iterations=als_parameters['iterations'],
                       alpha=als_parameters['alpha'])
        self.lightfm = LightFM()
        # self.svd = SVD()
        # self.nmf = NMF()
        # self.top_popular = TopPopular()

        self.training_urm = None

    def fit(self, training_urm):
        self.training_urm = training_urm

        print('Fitting User Content Based Filtering...')
        self.user_content_based_filtering.fit(self.training_urm)
        print('Fitting Item Content Based Filtering...')
        self.item_content_based_filtering.fit(self.training_urm)
        print('Fitting User Collaborative Filtering...')
        self.user_based_collaborative_filtering.fit(self.training_urm)
        print('Fitting Item Collaborative Filtering...')
        self.item_based_collaborative_filtering.fit(self.training_urm)
        print('Fitting SLIM BPR...')
        self.slim_bpr.fit(self.training_urm)
        print('Fitting ElasticNet...')
        self.elastic_net.fit(self.training_urm)
        print('Fitting ALS...')
        self.als.fit(self.training_urm)
        # print('Fitting LightFM...')
        # self.lightfm.fit(self.training_urm)
        # print('Fitting NMF...')
        # self.nmf.fit(self.training_urm)
        # print('Fitting SVD...')
        # self.svd.fit(self.training_urm)
        # print('Fitting Top Popular...')
        # self.top_popular.fit(self.training_urm)

    def recommend(self, user_id, k=10):
        # DUE TO TIME CONSTRAINT THE CODE STRUCTURE HERE IS REDUNDANT
        # TODO exploit inheritance to reduce code duplications and simple extract ratings, combine them,
        #  simply by iterate over a list of recommenders

        user_content_based_filtering_ratings = self.user_content_based_filtering.get_expected_ratings(user_id)
        item_content_based_filtering_ratings = self.item_content_based_filtering.get_expected_ratings(user_id)
        user_based_collaborative_filtering_ratings = \
            self.user_based_collaborative_filtering.get_expected_ratings(user_id)
        item_based_collaborative_filtering_ratings = \
            self.item_based_collaborative_filtering.get_expected_ratings(user_id)
        slim_bpr_ratings = self.slim_bpr.get_expected_ratings(user_id)
        elastic_net_ratings = self.elastic_net.get_expected_ratings(user_id)
        als_ratings = self.als.get_expected_ratings(user_id)
        # lightfm_ratings = self.lightfm.get_expected_ratings(user_id)
        # nmf_ratings = self.nmf.get_expected_ratings(user_id)
        # svd_ratings = self.svd.get_expected_ratings(user_id)
        # top_popular_ratings = self.top_popular.get_expected_ratings(user_id)

        if self.training_urm[user_id].getnnz() > 10:
            weights = self.weights_high_interactions
        elif self.training_urm[user_id].getnnz() == 0:
            weights = self.weights_cold_start
        else:
            weights = self.weights_low_interactions

        hybrid_ratings = user_content_based_filtering_ratings * weights['user_content_based_filtering']
        hybrid_ratings += item_content_based_filtering_ratings * weights['item_content_based_filtering']
        hybrid_ratings += user_based_collaborative_filtering_ratings * weights['user_based_collaborative_filtering']
        hybrid_ratings += item_based_collaborative_filtering_ratings * weights['item_based_collaborative_filtering']
        hybrid_ratings += slim_bpr_ratings * weights['slim_bpr']
        hybrid_ratings += elastic_net_ratings * weights['elastic_net']
        hybrid_ratings += als_ratings * weights['als']
        # hybrid_ratings += lightfm_ratings * weights['lightfm']
        # hybrid_ratings += nmf_ratings * weights['nmf']
        # hybrid_ratings += svd_ratings * weights['svd']
        # hybrid_ratings += top_popular_ratings * weights['top_popular']

        recommended_items = np.flip(np.argsort(hybrid_ratings), 0)
        unseen_items_mask = np.in1d(recommended_items, self.training_urm[user_id].indices, assume_unique=True,
                                    invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[:k]
