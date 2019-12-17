import os

import numpy as np
import pandas as pd
import scipy.sparse as sps


class Session(object):

    def __init__(self, urm_path, urm_users_column, urm_items_column,
                 users_amount, items_amount, target_users__path,
                 ucm_ages_path, ucm_ages_index_column, ucm_ages_value_column,
                 ucm_regions_path, ucm_regions_index_column, ucm_regions_value_column,
                 icm_assets_path, icm_assets_index_column, icm_assets_value_column,
                 icm_prices_path, icm_prices_index_column, icm_prices_value_column,
                 icm_sub_classes_path, icm_sub_classes_index_column, icm_sub_classes_values_column, random_seed=3333):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        self.urm_dataframe = pd.read_csv(urm_path)

        user_list = np.asarray(list(self.urm_dataframe[urm_users_column]))
        item_list = np.asarray(list(self.urm_dataframe[urm_items_column]))

        self.user_list_unique = set(user_list)
        self.item_list_unique = set(item_list)

        ratings_list = list(np.ones(len(item_list)))
        urm = sps.coo_matrix((ratings_list, (user_list, item_list)), shape=(users_amount, items_amount),
                             dtype=np.float32)
        self.urm = urm.tocsr()

        self.users_amount = users_amount
        self.items_amount = items_amount

        target_file = open(target_users__path)
        row_list = list(target_file)[1:]
        self.target_users_list = []
        for row in row_list:
            self.target_users_list.append(int(row))

        self.ucm_ages_path = ucm_ages_path
        self.ucm_ages_index_column = ucm_ages_index_column
        self.ucm_ages_value_column = ucm_ages_value_column
        self.ucm_regions_path = ucm_regions_path
        self.ucm_regions_index_column = ucm_regions_index_column
        self.ucm_regions_value_column = ucm_regions_value_column
        self.icm_assets_path = icm_assets_path
        self.icm_assets_index_column = icm_assets_index_column
        self.icm_assets_value_column = icm_assets_value_column
        self.icm_prices_path = icm_prices_path
        self.icm_prices_index_column = icm_prices_index_column
        self.icm_prices_value_column = icm_prices_value_column
        self.icm_sub_classes_path = icm_sub_classes_path
        self.icm_sub_classes_index_column = icm_sub_classes_index_column
        self.icm_sub_classes_values_column = icm_sub_classes_values_column

    def get_icm_assets(self):
        assets_data = pd.read_csv(os.path.join(os.getcwd(), self.icm_assets_path))
        items_list = list(assets_data[self.icm_assets_index_column])
        assets_list = list(assets_data[self.icm_assets_value_column])
        zeroes = np.zeros(len(items_list), dtype=np.int32)
        icm_assets = sps.coo_matrix((assets_list, (items_list, zeroes)), shape=(self.items_amount, 1), dtype=np.float32)
        return icm_assets.tocsr()

    def get_icm_prices(self):
        prices_data = pd.read_csv(os.path.join(os.getcwd(), self.icm_prices_path))
        items_list = list(prices_data[self.icm_prices_index_column])
        prices_list = list(prices_data[self.icm_prices_value_column])
        zeroes = np.zeros(len(items_list), dtype=np.int32)
        icm_prices = sps.coo_matrix((prices_list, (items_list, zeroes)), shape=(self.items_amount, 1), dtype=np.float32)
        return icm_prices.tocsr()

    def get_icm_sub_classes(self):
        sub_classes_data = pd.read_csv(os.path.join(os.getcwd(), self.icm_sub_classes_path))
        items_list = list(sub_classes_data[self.icm_sub_classes_index_column])
        sub_classes_list = list(sub_classes_data[self.icm_sub_classes_values_column])
        values_list = list(np.ones(len(sub_classes_list)))
        icm_sub_classes = sps.coo_matrix((values_list, (items_list, sub_classes_list)),
                                         shape=(self.items_amount, max(sub_classes_list) + 1), dtype=np.float32)
        return icm_sub_classes.tocsr()

    def get_ucm_ages(self):
        ages_data = pd.read_csv(os.path.join(os.getcwd(), self.ucm_ages_path))
        users_list = list(ages_data[self.ucm_ages_index_column])
        ages_list = list(ages_data[self.ucm_ages_value_column])
        values_list = list(np.ones(len(ages_list)))
        ucm_ages = sps.coo_matrix((values_list, (users_list, ages_list)),
                                  shape=(self.users_amount, max(ages_list) + 1), dtype=np.float32)
        return ucm_ages.tocsr()

    def get_ucm_regions(self):
        regions_data = pd.read_csv(os.path.join(os.getcwd(), self.ucm_regions_path))
        users_list = list(regions_data[self.ucm_regions_index_column])
        regions_list = list(regions_data[self.ucm_regions_value_column])
        values_list = list(np.ones(len(regions_list)))
        ucm_regions = sps.coo_matrix((values_list, (users_list, regions_list)),
                                     shape=(self.users_amount, max(regions_list) + 1), dtype=np.float32)
        return ucm_regions.tocsr()
