import os

from tqdm import tqdm

import utils
from evaluator import Evaluator
from recommenders.hybrid import Hybrid
from session import Session


def run(urm_path, urm_users_column, urm_items_column,
        users_amount, items_amount, target_users_path,
        ucm_ages_path, ucm_ages_index_column, ucm_ages_value_column,
        ucm_regions_path, ucm_regions_index_column, ucm_regions_value_column,
        icm_assets_path, icm_assets_index_column, icm_assets_value_column,
        icm_prices_path, icm_prices_index_column, icm_prices_value_column,
        icm_sub_classes_path, icm_sub_classes_index_column, icm_sub_classes_values_column,
        submission_users_column, submission_items_column,
        is_test, leave_one_out=True, test_percentage=0.2, test_interactions_threshold=10, k=10, random_seed=3333):
    session = Session(urm_path, urm_users_column, urm_items_column,
                      users_amount, items_amount, target_users_path,
                      ucm_ages_path, ucm_ages_index_column, ucm_ages_value_column,
                      ucm_regions_path, ucm_regions_index_column, ucm_regions_value_column,
                      icm_assets_path, icm_assets_index_column, icm_assets_value_column,
                      icm_prices_path, icm_prices_index_column, icm_prices_value_column,
                      icm_sub_classes_path, icm_sub_classes_index_column, icm_sub_classes_values_column, random_seed)

    recommender = Hybrid(session=session,
                         weights_cold_start=weights_cold_start,
                         weights_low_interactions=weights_low_interactions,
                         weights_high_interactions=weights_high_interactions,
                         user_content_based_filtering_parameters=user_content_based_filtering_parameters,
                         item_content_based_filtering_parameters=item_content_based_filtering_parameters,
                         user_based_collaborative_filtering_parameters=user_based_collaborative_filtering_parameters,
                         item_based_collaborative_filtering_parameters=item_based_collaborative_filtering_parameters,
                         slim_bpr_parameters=slim_bpr_parameters,
                         elastic_net_parameters=elastic_net_parameters,
                         als_parameters=als_parameters,
                         lightfm_parameters=lightfm_parameters,
                         svd_parameters=svd_parameters,
                         nmf_parameters=nmf_parameters,
                         top_popular_parameters=top_popular_parameters,
                         spotlight_parameters=spotlight_parameters)
    # recommender = LightFM(session=session)

    if is_test:
        evaluator = Evaluator(session)
        evaluator.split(leave_one_out, test_percentage, test_interactions_threshold)
        mapk = evaluator.evaluate(recommender, k)
        print('map@' + str(k) + ' = ' + str(mapk) + '\n')
    else:
        recommender.fit(session.urm)
        results = {}
        for user in tqdm(session.target_users_list):
            results[user] = recommender.recommend(session.urm, user, k)
        utils.create_csv(results, users_column=submission_users_column, items_column=submission_items_column)


# 0.05048601829284261 lightfm
# 0.05040985635371977
weights_cold_start = {
    'user_content_based_filtering': 1,  # OK
    'item_content_based_filtering': 0,  # OK
    'user_based_collaborative_filtering': 0,  # OK
    'item_based_collaborative_filtering': 0,  # OK
    'slim_bpr': 0,  # OK
    'elastic_net': 0,  # OK
    'als': 0,  # OK
    'lightfm': 1,  # OK
    'nmf': 0,  # OK
    'svd': 0,  # OK
    'top_popular': 0,  # OK
    'spotlight': 0  # OK
}
weights_low_interactions = {
    'user_content_based_filtering': 0,  # OK
    'item_content_based_filtering': 0.1,  # OK
    'user_based_collaborative_filtering': 0.2,  # OK
    'item_based_collaborative_filtering': 0.7,  # OK
    'slim_bpr': 0.1,  # OK
    'elastic_net': 0.7,  # OK
    'als': 0.2,  # OK
    'lightfm': 0,
    'nmf': 0,
    'svd': 0,
    'top_popular': 0,
    'spotlight': 0
}
weights_high_interactions = {
    'user_content_based_filtering': 0,  # OK
    'item_content_based_filtering': 0.1,  # OK
    'user_based_collaborative_filtering': 0.0,  # OK
    'item_based_collaborative_filtering': 0.4,  # OK
    'slim_bpr': 0.0,  # OK
    'elastic_net': 1.2,  # OK
    'als': 0.4,  # OK
    'lightfm': 0,
    'nmf': 0,
    'svd': 0,
    'top_popular': 0,
    'spotlight': 0
}
user_content_based_filtering_parameters = {
    'user_interactions_threshold': 0,
    'item_interactions_threshold': 0,
    'top_k_user_age': 2000,
    'top_k_user_region': 2000,
    'shrink_user_age': 40,
    'shrink_user_region': 40,
    'weight_user_age': 0.6
}
item_content_based_filtering_parameters = {
    'user_interactions_threshold': 0,
    'item_interactions_threshold': 0,
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
    'user_interactions_threshold': 0,
    'item_interactions_threshold': 1,
    'top_k': 1000,
    'shrink': 5
}
item_based_collaborative_filtering_parameters = {
    'user_interactions_threshold': 0,
    'item_interactions_threshold': 0,
    'top_k': 10,
    'shrink': 400
}
slim_bpr_parameters = {
    'user_interactions_threshold': 0,
    'item_interactions_threshold': 1,
    'epochs': 200,
    'top_k': 16
}
elastic_net_parameters = {
    'user_interactions_threshold': 2,
    'item_interactions_threshold': 0
}
als_parameters = {
    'user_interactions_threshold': 0,
    'item_interactions_threshold': 2,
    'factors': 448,
    'regularization': 100,
    'iterations': 30,
    'alpha': 21
}
lightfm_parameters = {
    'user_interactions_threshold': 0,
    'item_interactions_threshold': 0
}
svd_parameters = {
    'user_interactions_threshold': 0,
    'item_interactions_threshold': 0
}
nmf_parameters = {
    'user_interactions_threshold': 0,
    'item_interactions_threshold': 0
}
top_popular_parameters = {
    'user_interactions_threshold': 0,
    'item_interactions_threshold': 0
}
spotlight_parameters = {
    'user_interactions_threshold': 0,
    'item_interactions_threshold': 0
}

if __name__ == '__main__':
    run(urm_path=os.path.join(os.getcwd(), './dataset/data_train.csv'),
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
        random_seed=3333)
