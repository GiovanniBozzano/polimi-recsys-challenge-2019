import numpy as np
import pandas as pd

from SLIM_BPR.slim_bpr import SLIMBPR
from SLIM_ElasticNet.slim_elastic_net import SLIMElasticNet
from cf.item_based_collaborative_filtering import ItemBasedCollaborativeFiltering
from cf.user_based_collaborative_filtering import UserBasedCollaborativeFiltering
from cbf.content_based_filtering import ContentBasedFiltering
from MF.alternating_least_square import AlternatingLeastSquare


class HybridRecommender(object):
    def __init__(self, weights_short, weights_long, user_cf_param, item_cf_param, cbf_param,
                 slim_param, svd_param, als_param):

        self.weights_short = weights_short
        self.weights_long = weights_long

        # USER CF #
        self.user_collaborative_filtering = UserBasedCollaborativeFiltering(top_k=user_cf_param['top_k'],
                                                                            shrink=user_cf_param['shrink'],
                                                                            similarity=item_cf_param['similarity'])

        # ITEM_CF #
        self.item_collaborative_filtering = ItemBasedCollaborativeFiltering(top_k=item_cf_param['top_k'],
                                                                            shrink=item_cf_param['shrink'],
                                                                            similarity=item_cf_param['similarity'])

        # CBF #
        self.content_based_filtering = ContentBasedFiltering(top_k_item_asset=cbf_param['top_k_item_asset'],
                                                             top_k_item_price=cbf_param['top_k_item_price'],
                                                             top_k_item_sub_class=cbf_param['top_k_item_sub_class'],
                                                             shrink_item_asset=cbf_param['shrink_item_asset'],
                                                             shrink_item_price=cbf_param['shrink_item_price'],
                                                             shrink_item_sub_class=cbf_param['shrink_item_sub_class'],
                                                             weight_item_asset=cbf_param['weight_item_asset'],
                                                             weight_item_price=cbf_param['weight_item_price'])

        self.slim_random = SLIMBPR(epochs=slim_param['epochs'], top_k=slim_param['top-k'])

        self.slim_elastic = SLIMElasticNet()

        # SVD BASED ON ITEM CONTENT MATRIX #
        # It takes too long to be computed and the increase in quality recommandations is quite low or none
        # self.svd_icm = SVDRec(n_factors=svd_param['n_factors'], knn=svd_param['knn'])

        # ALS #
        self.als = AlternatingLeastSquare(factors=als_param['factors'],
                                          regularization=als_param['regularization'],
                                          iterations=als_param['iterations'],
                                          alpha=als_param['alpha'])

        self.training_urm = None

    def fit(self, training_urm):
        self.training_urm = training_urm

        print('Fitting cbf...')
        self.content_based_filtering.fit(self.training_urm)

        print('Fitting user cf...')
        self.user_collaborative_filtering.fit(self.training_urm)

        print('Fitting item cf...')
        self.item_collaborative_filtering.fit(self.training_urm)

        print('Fitting slim...')
        self.slim_random.fit(self.training_urm)
        
        print('Fitting slim elastic net...')
        self.slim_elastic.fit(self.training_urm)

        # self.svd_icm.fit(self.training_urm)

        print('Fitting ALS')
        self.als.fit(self.training_urm)

    def recommend(self, user_id, k=10):
        user_id = int(user_id)

        # DUE TO TIME CONSTRAINT THE CODE STRUCTURE HERE IS REDUNDANT
        # TODO exploit inheritance to reduce code duplications and simple extract ratings, combine them,
        #  simply by iterate over a list of recommenders

        # COMBINE RATINGS IN DIFFERENT WAYS (seq, random short, random long)
        cbf_ratings = self.content_based_filtering.get_expected_ratings(user_id)
        user_cf_ratings = self.user_collaborative_filtering.get_expected_ratings(user_id)
        item_cf_ratings = self.item_collaborative_filtering.get_expected_ratings(user_id)
        slim_ratings = self.slim_random.get_expected_ratings(user_id)
        slim_elastic_ratings = self.slim_elastic.get_expected_ratings(user_id)
        # svd_icm_ratings = self.svd_icm.get_expected_ratings(user_id)
        als_ratings = self.als.get_expected_ratings(user_id)
        if self.training_urm[user_id].getnnz() > 10:
            weights = self.weights_long
        else:
            weights = self.weights_short

        hybrid_ratings = cbf_ratings * weights['cbf']
        hybrid_ratings += user_cf_ratings * weights['user_cf']
        hybrid_ratings += item_cf_ratings * weights['item_cf']
        hybrid_ratings += slim_ratings * weights['slim']
        hybrid_ratings += slim_elastic_ratings * weights['elastic']
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
