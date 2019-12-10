import os

from tqdm import tqdm

import utils
from evaluator import Evaluator
from recommenders.alternating_least_square import AlternatingLeastSquare
from recommenders.hybrid import HybridRecommender
from recommenders.item_based_collaborative_filtering import ItemBasedCollaborativeFiltering
from recommenders.item_content_based_filtering import ItemContentBasedFiltering
from recommenders.svd import SVD
from recommenders.user_based_collaborative_filtering import UserBasedCollaborativeFiltering
from recommenders.user_content_based_filtering import UserContentBasedFiltering
from recommenders.nmf import NMF
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


# 0.050232533421856386
weights_low_interactions = {
    'top_popular': 0.01,
    'user_content_based_filtering': 0.01,  # OK
    'item_content_based_filtering': 0.1,  # OK
    'user_based_collaborative_filtering': 0.2,  # OK
    'item_based_collaborative_filtering': 0.7,  # OK
    'slim_bpr': 0.1,  # OK
    'elastic_net': 1.3,  # OK
    'alternating_least_square': 0.2,  # OK

    'svd': 0
}
weights_high_interactions = {
    'top_popular': 0.01,
    'user_content_based_filtering': 0.01,  # OK
    'item_content_based_filtering': 0.1,  # OK
    'user_based_collaborative_filtering': 0.0,  # OK
    'item_based_collaborative_filtering': 0.4,  # OK
    'slim_bpr': 0.0,  # OK
    'elastic_net': 1.2,  # OK
    'alternating_least_square': 0.4,  # OK

    'svd': 0
}
user_content_based_filtering_parameters = {
    'top_k_user_region': 2000,
    'top_k_user_age': 2000,
    'shrink_user_region': 40,
    'shrink_user_age': 40,
    'weight_user_region': 0.6
}
item_content_based_filtering_parameters = {
    'top_k_item_asset': 140,
    'top_k_item_price': 140,
    'top_k_item_sub_class': 300,
    'shrink_item_asset': 1,
    'shrink_item_price': 1,
    'shrink_item_sub_class': 1,
    'weight_item_asset': 0.2,
    'weight_item_price': 0.2
}
user_based_collaborative_filtering_parameters = {
    'top_k': 1000,
    'shrink': 5
}
item_based_collaborative_filtering_parameters = {
    'top_k': 10,
    'shrink': 400
}
slim_bpr_parameters = {
    'epochs': 80,
    'top_k': 40
}
alternating_least_square_parameters = {
    'factors': 448,
    'regularization': 100,
    'iterations': 30,
    'alpha': 21
}
svd_parameters = {
    'n_factors': 2000,
    'knn': 100
}

recommender = HybridRecommender(weights_high_interactions=weights_high_interactions,
                                weights_low_interactions=weights_low_interactions,
                                user_content_based_filtering_parameters=user_content_based_filtering_parameters,
                                item_content_based_filtering_parameters=item_content_based_filtering_parameters,
                                user_based_collaborative_filtering_parameters=
                                user_based_collaborative_filtering_parameters,
                                item_based_collaborative_filtering_parameters=
                                item_based_collaborative_filtering_parameters,
                                slim_bpr_parameters=slim_bpr_parameters,
                                alternating_least_square_parameters=alternating_least_square_parameters)
# recommender = AlternatingLeastSquare()
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
