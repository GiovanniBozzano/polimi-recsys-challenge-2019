import os

from tqdm import tqdm

import utils
from evaluator import Evaluator
from lib.similarity.compute_similarity import SimilarityFunction
from recommenders.hybrid import HybridRecommender
from recommenders.item_content_based_filtering import ItemContentBasedFiltering
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


weights_short = {
    'user_content_based_filtering': 0.1,
    'item_content_based_filtering': 0.1,
    'user_based_collaborative_filtering': 0.2,
    'item_based_collaborative_filtering': 0.3,
    'slim_bpr': 0.2,
    'elastic_net': 1.5,
    'alternating_least_square': 0.2,

    'svd': 0
}
weights_long = {
    'user_content_based_filtering': 0.1,
    'item_content_based_filtering': 0.1,
    'user_based_collaborative_filtering': 0.1,
    'item_based_collaborative_filtering': 0.4,
    'slim_bpr': 0.2,
    'elastic_net': 1,
    'alternating_least_square': 0.2,

    'svd': 0
}
user_content_based_filtering_parameters = {
    'top_k_user_region': 1000,
    'top_k_user_age': 1000,
    'shrink_user_region': 1,
    'shrink_user_age': 1,
    'weight_user_region': 0.6
}
item_content_based_filtering_parameters = {
    'top_k_item_asset': 50,
    'top_k_item_price': 50,
    'top_k_item_sub_class': 50,
    'shrink_item_asset': 1,
    'shrink_item_price': 1,
    'shrink_item_sub_class': 1,
    'weight_item_asset': 0.2,
    'weight_item_price': 0.2
}
user_based_collaborative_filtering_parameters = {
    'top_k': 2000,
    'shrink': 5,
    'similarity': SimilarityFunction.COSINE.value
}
item_based_collaborative_filtering_parameters = {
    'top_k': 10,
    'shrink': 500,
    'similarity': SimilarityFunction.JACCARD.value
}
slim_bpr_parameters = {
    'epochs': 80,
    'top-k': 40
}
alternating_least_square_parameters = {
    'factors': 448,
    'regularization': 100,
    'iterations': 30,
    'alpha': 24
}
svd_parameters = {
    'n_factors': 2000,
    'knn': 100
}

recommender = HybridRecommender(weights_long=weights_long,
                                weights_short=weights_short,
                                user_content_based_filtering_parameters=user_content_based_filtering_parameters,
                                item_content_based_filtering_parameters=item_content_based_filtering_parameters,
                                user_based_collaborative_filtering_parameters=
                                user_based_collaborative_filtering_parameters,
                                item_based_collaborative_filtering_parameters=
                                item_based_collaborative_filtering_parameters,
                                slim_bpr_parameters=slim_bpr_parameters,
                                alternating_least_square_parameters=alternating_least_square_parameters,
                                svd_parameters=svd_parameters)
recommender = ItemContentBasedFiltering()
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
