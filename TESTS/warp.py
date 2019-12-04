import os
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from datetime import datetime
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k

import helpers


def create_csv(results, results_dir='./'):
    csv_filename = 'results_'
    csv_filename += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'
    with open(os.path.join(results_dir, csv_filename), 'w') as file:
        file.write('user_id,item_list\n')
        for key, value in results.items():
            file.write(str(key) + ',')
            first = True
            for prediction in value:
                if not first:
                    file.write(' ')
                first = False
                file.write(str(prediction))
            file.write('\n')


def binarize(age, low, high):
    if low < age <= high:
        return 1
    else:
        return 0


dataset = Dataset(user_identity_features=False,
                  item_identity_features=False)


user_features_map = {}
user_features_list = []

age_data = pd.read_csv('dataset/data_UCM_age.csv', sep=',')
age_data = age_data.rename(columns={'row': 'uid', 'col': 'age'})
age_data = age_data[['uid', 'age']]

dataset.fit_partial(users=set([user_id for user_id in age_data['uid']]))

for index, row in age_data.iterrows():
    feature_key = '{}_{}'.format('age', str(row['age'].astype(int)))
    if feature_key not in user_features_list:
        user_features_list.append(feature_key)
    user_features_map.setdefault(row['uid'], []).append(feature_key)

region_data = pd.read_csv('dataset/data_UCM_region.csv', sep=',')
region_data = region_data.rename(columns={'row': 'uid', 'col': 'region'})
region_data = region_data[['uid', 'region']]

dataset.fit_partial(users=set([user_id for user_id in region_data['uid']]))

for index, row in region_data.iterrows():
    feature_key = '{}_{}'.format('region', str(row['region'].astype(int)))
    if feature_key not in user_features_list:
        user_features_list.append(feature_key)
    user_features_map.setdefault(row['uid'], []).append(feature_key)

user_features = []
for item_id, features in user_features_map.items():
    user_features.append((item_id, features))
# print(pd.DataFrame(user_features).sort_values(by=[0]))

dataset.fit_partial(user_features=user_features_list)
user_features = dataset.build_user_features(user_features)


item_features_map = {}
item_features_list = []

sub_class_data = pd.read_csv('dataset/data_ICM_sub_class.csv', sep=',')
sub_class_data = sub_class_data.rename(columns={'row': 'iid', 'col': 'sub_class'})
sub_class_data = sub_class_data[['iid', 'sub_class']]

dataset.fit_partial(items=set([item_id for item_id in sub_class_data['iid']]))

for index, row in sub_class_data.iterrows():
    feature_key = '{}_{}'.format('sub_class', str(row['sub_class'].astype(int)))
    if feature_key not in item_features_list:
        item_features_list.append(feature_key)
    item_features_map.setdefault(row['iid'], []).append(feature_key)

asset_data = pd.read_csv('dataset/data_ICM_asset.csv', sep=',')
asset_data = asset_data.rename(columns={'row': 'iid', 'data': 'asset'})
asset_data = asset_data[['iid', 'asset']]

dataset.fit_partial(items=set([item_id for item_id in asset_data['iid']]))

asset_data['asset_0.5_to_1'] = asset_data['asset'].apply(binarize, low=0.5, high=1)
asset_data['asset_0.2_to_0.5'] = asset_data['asset'].apply(binarize, low=0.2, high=0.5)
asset_data['asset_0.15_to_0.2'] = asset_data['asset'].apply(binarize, low=0.15, high=0.2)
asset_data['asset_0.1_to_0.15'] = asset_data['asset'].apply(binarize, low=0.1, high=0.15)
asset_data['asset_0.09_to_0.1'] = asset_data['asset'].apply(binarize, low=0.09, high=0.1)
asset_data['asset_0.08_to_0.09'] = asset_data['asset'].apply(binarize, low=0.08, high=0.09)
asset_data['asset_0.07_to_0.08'] = asset_data['asset'].apply(binarize, low=0.07, high=0.08)
asset_data['asset_0.06_to_0.07'] = asset_data['asset'].apply(binarize, low=0.06, high=0.07)
asset_data['asset_0.05_to_0.06'] = asset_data['asset'].apply(binarize, low=0.05, high=0.06)
asset_data['asset_0.04_to_0.05'] = asset_data['asset'].apply(binarize, low=0.04, high=0.05)
asset_data['asset_0.03_to_0.04'] = asset_data['asset'].apply(binarize, low=0.03, high=0.04)
asset_data['asset_0.02_to_0.03'] = asset_data['asset'].apply(binarize, low=0.02, high=0.03)
asset_data['asset_0.01_to_0.02'] = asset_data['asset'].apply(binarize, low=0.01, high=0.02)
asset_data['asset_0_to_0.01'] = asset_data['asset'].apply(binarize, low=0, high=0.01)

asset_data_series = asset_data[asset_data.columns[asset_data.isin([0, 1]).all()]]
asset_data_series = pd.Series([''.join(x).strip(', ') for x in np.where(
                                  asset_data_series, ['{}, '.format(x) for x in asset_data_series.columns], ''
                              )],
                              index=asset_data_series.index)

for index, row in asset_data.iterrows():
    feature_key = asset_data_series.get(row['iid'])
    if feature_key is None:
        continue
    if feature_key not in item_features_list:
        item_features_list.append(feature_key)
    item_features_map.setdefault(row['iid'], []).append(feature_key)

price_data = pd.read_csv('dataset/data_ICM_price.csv', sep=',')
price_data = price_data.rename(columns={'row': 'iid', 'data': 'price'})
price_data = price_data[['iid', 'price']]

dataset.fit_partial(items=set([item_id for item_id in price_data['iid']]))

price_data['price_0.5_to_1'] = price_data['price'].apply(binarize, low=0.5, high=1)
price_data['price_0.2_to_0.5'] = price_data['price'].apply(binarize, low=0.2, high=0.5)
price_data['price_0.15_to_0.2'] = price_data['price'].apply(binarize, low=0.15, high=0.2)
price_data['price_0.1_to_0.15'] = price_data['price'].apply(binarize, low=0.1, high=0.15)
price_data['price_0.09_to_0.1'] = price_data['price'].apply(binarize, low=0.09, high=0.1)
price_data['price_0.08_to_0.09'] = price_data['price'].apply(binarize, low=0.08, high=0.09)
price_data['price_0.07_to_0.08'] = price_data['price'].apply(binarize, low=0.07, high=0.08)
price_data['price_0.06_to_0.07'] = price_data['price'].apply(binarize, low=0.06, high=0.07)
price_data['price_0.05_to_0.06'] = price_data['price'].apply(binarize, low=0.05, high=0.06)
price_data['price_0.04_to_0.05'] = price_data['price'].apply(binarize, low=0.04, high=0.05)
price_data['price_0.03_to_0.04'] = price_data['price'].apply(binarize, low=0.03, high=0.04)
price_data['price_0.02_to_0.03'] = price_data['price'].apply(binarize, low=0.02, high=0.03)
price_data['price_0.01_to_0.02'] = price_data['price'].apply(binarize, low=0.01, high=0.02)
price_data['price_0_to_0.01'] = price_data['price'].apply(binarize, low=0, high=0.01)

price_data_series = price_data[price_data.columns[price_data.isin([0, 1]).all()]]
price_data_series = pd.Series([''.join(x).strip(', ') for x in np.where(
                                  price_data_series, ['{}, '.format(x) for x in price_data_series.columns], ''
                              )],
                              index=price_data_series.index)

for index, row in price_data.iterrows():
    feature_key = price_data_series.get(row['iid'])
    if feature_key is None:
        continue
    if feature_key not in item_features_list:
        item_features_list.append(feature_key)
    item_features_map.setdefault(row['iid'], []).append(feature_key)

item_features = []
for item_id, features in item_features_map.items():
    item_features.append((item_id, features))
# print(pd.DataFrame(item_features).sort_values(by=[0]))

dataset.fit_partial(item_features=item_features_list)
item_features = dataset.build_item_features(item_features)


interactions_data = pd.read_csv('dataset/data_train.csv', sep=',')
interactions_data = interactions_data.rename(columns={'row': 'uid', 'col': 'iid'})
interactions_data = interactions_data[['uid', 'iid']]
interactions_data = helpers.threshold_interactions_df(interactions_data, 'uid', 'iid', 5, 5)

interactions_list = []
for index, row in interactions_data.iterrows():
    interactions_list.append((row['uid'], row['iid']))

(interactions, weights) = dataset.build_interactions(interactions_list)
# print(pd.DataFrame(interactions.toarray()))

(train, test) = random_train_test_split(interactions, test_percentage=0.2, random_state=np.random.RandomState(3333))
# print(pd.DataFrame(train.toarray()))
# print(pd.DataFrame(test.toarray()))


model = LightFM(loss='warp',
                no_components=30,
                max_sampled=5,
                learning_rate=0.05,
                learning_schedule='adagrad',
                user_alpha=0.0001,
                item_alpha=0.0001,
                random_state=np.random.RandomState(3333))

old_epoch = 0
for epoch in range(1, 500, 1):
    epochs = epoch - old_epoch
    model.fit_partial(train,
                      user_features=user_features,
                      item_features=item_features,
                      epochs=epochs)
    old_epoch = epoch
    print('=========================')
    print('map@10 test: ' +
          str(precision_at_k(model, test,
                             train_interactions=None,
                             user_features=user_features,
                             item_features=item_features,
                             k=10).mean()))
    print('map@10 test: ' +
          str(precision_at_k(model, test,
                             train_interactions=train,
                             user_features=user_features,
                             item_features=item_features,
                             k=10).mean()))
