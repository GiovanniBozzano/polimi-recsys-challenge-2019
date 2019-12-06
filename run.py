import os
import utils
from tqdm import tqdm
from Base.Similarity.Compute_Similarity import SimilarityFunction
from MF.alternating_least_square import AlternatingLeastSquare
from SLIM_BPR.slim_bpr import SLIMBPR
from SLIM_ElasticNet.slim_elastic_net import SLIMElasticNet
from cbf.item_content_based_filtering import ItemContentBasedFiltering
from cbf.user_content_based_filtering import UserContentBasedFiltering
from cf.item_based_collaborative_filtering import ItemBasedCollaborativeFiltering
from cf.user_based_collaborative_filtering import UserBasedCollaborativeFiltering
from evaluation.evaluator import Evaluator
from hybrid.hybrid import HybridRecommender
from session import Session


def run(recommender, urm_path, urm_users_column, urm_items_column,
        users_amount, items_amount, target_users_path,
        ucm_ages_path, ucm_ages_index_column, ucm_ages_value_column,
        ucm_regions_path, ucm_regions_index_column, ucm_regions_value_column,
        icm_assets_path, icm_assets_index_column, icm_assets_value_column,
        icm_prices_path, icm_prices_index_column, icm_prices_value_column,
        icm_sub_classes_path, icm_sub_classes_index_column, icm_sub_classes_values_column,
        submission_users_column, submission_items_column,
        is_test, leave_one_out=True, test_percentage=0.2, test_interactions_threshold=10, k=10,
        users_usefulness_threshold=None, items_usefulness_threshold=None):

    session = Session(urm_path, urm_users_column, urm_items_column,
                      users_amount, items_amount, target_users_path,
                      ucm_ages_path, ucm_ages_index_column, ucm_ages_value_column,
                      ucm_regions_path, ucm_regions_index_column, ucm_regions_value_column,
                      icm_assets_path, icm_assets_index_column, icm_assets_value_column,
                      icm_prices_path, icm_prices_index_column, icm_prices_value_column,
                      icm_sub_classes_path, icm_sub_classes_index_column, icm_sub_classes_values_column,
                      users_usefulness_threshold, items_usefulness_threshold)
    if is_test:
        print('Starting testing phase..')
        evaluator = Evaluator(session)
        evaluator.split(leave_one_out, test_percentage, test_interactions_threshold)
        score = evaluator.evaluate(recommender, k)
        print('Evaluation completed, score = ' + str(score) + '\n')
        return score
    else:
        print('Starting prediction to be submitted..')
        recommender.fit(session.urm)
        results = {}
        for user in tqdm(session.target_users_list):
            results[user] = recommender.recommend(user, k)
        utils.create_csv(results, users_column=submission_users_column, items_column=submission_items_column)
        print('Saved predictions to file')


# 0.04680114031792625 = 0.03319
# 0.034609182829823344 = 0.03319
# ======================================
# < 10
# ============================
# CBF = 0.011632757451942505
# UCF = 0.046129799270716094
# ICF = 0.04938960303985777
# ALS = 0.04718438984143571
# SLIM = 0.043278637039248266
# ELASTICNET =
# ============================
# 0.05106765051586792
# ======================================
# > 10
# ============================
# CBF = 0.008287904950663514
# UCF = 0.03880122241231024
# ICF = 0.04681579272693917
# ALS = 0.04121776645225262
# SLIM = 0.041645630700148734
# ELASTICNET =
# ============================
# 0.04906610942956304
# ======================================
# 0.04850864702268111 - 1 - 2
weights_short = {
    'user_cbf': 0.1,
    'item_cbf': 0.1,
    'user_cf': 0.2,
    'item_cf': 0.3,
    'slim': 0.2,
    'elastic': 1.5,
    'als': 0.2,

    'icm_svd': 0
}
weights_long = {
    'user_cbf': 0.1,
    'item_cbf': 0.1,
    'user_cf': 0.1,
    'item_cf': 0.4,
    'slim': 0.2,
    'elastic': 1,
    'als': 0.2,

    'icm_svd': 0
}
user_cbf_param = {
    'top_k_user_region': 1000,
    'top_k_user_age': 1000,
    'shrink_user_region': 1,
    'shrink_user_age': 1,
    'weight_user_region': 0.6
}
item_cbf_param = {
    'top_k_item_asset': 50,
    'top_k_item_price': 50,
    'top_k_item_sub_class': 50,
    'shrink_item_asset': 1,
    'shrink_item_price': 1,
    'shrink_item_sub_class': 1,
    'weight_item_asset': 0.2,
    'weight_item_price': 0.2
}
user_cf_param = {
    'top_k': 2000,
    'shrink': 5,
    'similarity': SimilarityFunction.COSINE.value
}
item_cf_param = {
    'top_k': 10,
    'shrink': 500,
    'similarity': SimilarityFunction.JACCARD.value
}
slim_param = {
    'epochs': 80,
    'top-k': 40
}
svd_param = {
    'n_factors': 2000,
    'knn': 100
}
als_param = {
    'factors': 448,
    'regularization': 100,
    'iterations': 30,
    'alpha': 24
}


recommender = HybridRecommender(weights_long=weights_long,
                                weights_short=weights_short,
                                user_cbf_param=user_cbf_param,
                                item_cbf_param=item_cbf_param,
                                user_cf_param=user_cf_param,
                                item_cf_param=item_cf_param,
                                slim_param=slim_param,
                                svd_param=svd_param,
                                als_param=als_param)
# recommender = ItemContentBasedFiltering()
if __name__ == '__main__':
    run(recommender=recommender,
        urm_path=os.path.join(os.getcwd(), './dataset/data_train.csv'),
        urm_users_column='row',
        urm_items_column='col',
        users_amount=30911,
        items_amount=18495,
        target_users_path=os.path.join(os.getcwd(), './dataset/data_target_users_test.csv'),
        ucm_ages_path='./dataset/data_UCM_age.csv',
        ucm_ages_index_column='row',
        ucm_ages_value_column='col',
        ucm_regions_path='./dataset/data_UCM_region.csv',
        ucm_regions_index_column='row',
        ucm_regions_value_column='col',
        icm_assets_path='./dataset/data_ICM_asset.csv',
        icm_assets_index_column='row',
        icm_assets_value_column='data',
        icm_prices_path='./dataset/data_ICM_price.csv',
        icm_prices_index_column='row',
        icm_prices_value_column='data',
        icm_sub_classes_path='./dataset/data_ICM_sub_class.csv',
        icm_sub_classes_index_column='row',
        icm_sub_classes_values_column='col',
        submission_users_column='user_id',
        submission_items_column='item_list',
        is_test=True,
        leave_one_out=True,
        test_percentage=0.2,
        k=10,
        test_interactions_threshold=10,
        users_usefulness_threshold=0,
        items_usefulness_threshold=4)
