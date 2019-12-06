import numpy as np
import pandas as pd

from SLIM_BPR.slim_bpr import SLIMBPR
from SLIM_ElasticNet.slim_elastic_net import SLIMElasticNet
from cf.item_based_collaborative_filtering import ItemBasedCollaborativeFiltering
from cf.user_based_collaborative_filtering import UserBasedCollaborativeFiltering
from cbf.user_content_based_filtering import UserContentBasedFiltering
from cbf.item_content_based_filtering import ItemContentBasedFiltering
from MF.alternating_least_square import AlternatingLeastSquare


class HybridRecommender(object):
    def __init__(self, weights_short, weights_long, user_cbf_param, item_cbf_param, user_cf_param, item_cf_param,
                 slim_param, svd_param, als_param):

        self.weights_short = weights_short
        self.weights_long = weights_long

        self.user_content_based_filtering = UserContentBasedFiltering(
                                                            top_k_user_region=user_cbf_param['top_k_user_region'],
                                                            top_k_user_age=user_cbf_param['top_k_user_age'],
                                                            shrink_user_region=user_cbf_param['shrink_user_region'],
                                                            shrink_user_age=user_cbf_param['shrink_user_age'],
                                                            weight_user_region=user_cbf_param['weight_user_region'])
        self.item_content_based_filtering = \
            ItemContentBasedFiltering(top_k_item_asset=item_cbf_param['top_k_item_asset'],
                                      top_k_item_price=item_cbf_param['top_k_item_price'],
                                      top_k_item_sub_class=item_cbf_param['top_k_item_sub_class'],
                                      shrink_item_asset=item_cbf_param['shrink_item_asset'],
                                      shrink_item_price=item_cbf_param['shrink_item_price'],
                                      shrink_item_sub_class=item_cbf_param['shrink_item_sub_class'],
                                      weight_item_asset=item_cbf_param['weight_item_asset'],
                                      weight_item_price=item_cbf_param['weight_item_price'])

        self.user_collaborative_filtering = UserBasedCollaborativeFiltering(top_k=user_cf_param['top_k'],
                                                                            shrink=user_cf_param['shrink'],
                                                                            similarity=item_cf_param['similarity'])
        self.item_collaborative_filtering = ItemBasedCollaborativeFiltering(top_k=item_cf_param['top_k'],
                                                                            shrink=item_cf_param['shrink'],
                                                                            similarity=item_cf_param['similarity'])
        self.slim_bpr = SLIMBPR(epochs=slim_param['epochs'], top_k=slim_param['top-k'])
        self.elastic_net = SLIMElasticNet()
        # SVD BASED ON ITEM CONTENT MATRIX #
        # It takes too long to be computed and the increase in quality recommandations is quite low or none
        # self.svd_icm = SVDRec(n_factors=svd_param['n_factors'], knn=svd_param['knn'])
        self.als = AlternatingLeastSquare(factors=als_param['factors'],
                                          regularization=als_param['regularization'],
                                          iterations=als_param['iterations'],
                                          alpha=als_param['alpha'])

        self.training_urm = None

    def fit(self, training_urm):
        self.training_urm = training_urm

        print('Fitting User Content Based Filtering...')
        self.user_content_based_filtering.fit(self.training_urm)

        print('Fitting Item Content Based Filtering...')
        self.item_content_based_filtering.fit(self.training_urm)

        print('Fitting User Collaborative Filtering...')
        self.user_collaborative_filtering.fit(self.training_urm)

        print('Fitting Item Collaborative Filtering...')
        self.item_collaborative_filtering.fit(self.training_urm)

        print('Fitting SLIM BPR...')
        self.slim_bpr.fit(self.training_urm)
        
        print('Fitting Elastic Net...')
        self.elastic_net.fit(self.training_urm)

        # self.svd_icm.fit(self.training_urm)

        print('Fitting ALS')
        self.als.fit(self.training_urm)

    def recommend(self, user_id, k=10):
        user_id = int(user_id)

        # DUE TO TIME CONSTRAINT THE CODE STRUCTURE HERE IS REDUNDANT
        # TODO exploit inheritance to reduce code duplications and simple extract ratings, combine them,
        #  simply by iterate over a list of recommenders

        user_content_based_filtering_ratings = self.user_content_based_filtering.get_expected_ratings(user_id)
        item_content_based_filtering_ratings = self.item_content_based_filtering.get_expected_ratings(user_id)
        user_collaborative_filtering_ratings = self.user_collaborative_filtering.get_expected_ratings(user_id)
        item_collaborative_filtering_ratings = self.item_collaborative_filtering.get_expected_ratings(user_id)
        slim_bpr_ratings = self.slim_bpr.get_expected_ratings(user_id)
        elastic_ratings = self.elastic_net.get_expected_ratings(user_id)
        # svd_icm_ratings = self.svd_icm.get_expected_ratings(user_id)
        als_ratings = self.als.get_expected_ratings(user_id)
        if self.training_urm[user_id].getnnz() > 10:
            weights = self.weights_long
        else:
            weights = self.weights_short

        hybrid_ratings = user_content_based_filtering_ratings * weights['user_cbf']
        hybrid_ratings += item_content_based_filtering_ratings * weights['item_cbf']
        hybrid_ratings += user_collaborative_filtering_ratings * weights['user_cf']
        hybrid_ratings += item_collaborative_filtering_ratings * weights['item_cf']
        hybrid_ratings += slim_bpr_ratings * weights['slim']
        hybrid_ratings += elastic_ratings * weights['elastic']
        # hybrid_ratings += svd_icm_ratings * weights['svd_icm']
        hybrid_ratings += als_ratings * weights['als']

        if user_id == 19335:
            print('HYBRID RATINGS:')
            print(pd.DataFrame(hybrid_ratings).sort_values(by=0, ascending=False))

        recommended_items = np.flip(np.argsort(hybrid_ratings), 0)

        # REMOVING SEEN
        unseen_items_mask = np.in1d(recommended_items, self.training_urm[user_id].indices, assume_unique=True,
                                    invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[:k]
