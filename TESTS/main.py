import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from lightfm import LightFM
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


def print_log(row, header=False, spacing=14):
    top = ''
    middle = ''
    bottom = ''
    for r in row:
        top += '+{}'.format('-' * spacing)
        if isinstance(r, str):
            middle += '| {0:^{1}} '.format(r, spacing - 2)
        elif isinstance(r, int):
            middle += '| {0:^{1}} '.format(r, spacing - 2)
        elif (isinstance(r, float)
              or isinstance(r, np.float32)
              or isinstance(r, np.float64)):
            middle += '| {0:^{1}.5f} '.format(r, spacing - 2)
        bottom += '+{}'.format('=' * spacing)
    top += '+'
    middle += '|'
    bottom += '+'
    if header:
        print(top)
        print(middle)
        print(bottom)
    else:
        print(middle)
        print(top)


def binarize(age, low, high):
    if low < age <= high:
        return 1
    else:
        return 0


df = pd.read_csv('dataset/data_train.csv',
                 sep=',')
df = df.rename(columns={'row': 'uid', 'col': 'iid'})
df = df[['uid', 'iid']]

df = helpers.threshold_interactions_df(df, 'uid', 'iid', 5, 5)

interactions, uid_to_idx, idx_to_uid, iid_to_idx, idx_to_iid = helpers.df_to_matrix(df, 'uid', 'iid')

train, test, user_index = helpers.train_test_split(interactions, 1, fraction=0.2)


region_data = pd.read_csv('dataset/data_UCM_region.csv', sep=',')
del region_data['data']
region_data = region_data.rename(columns={'row': 'uid', 'col': 'region'})

user_features = [{} for _ in idx_to_uid]
for index, row in region_data.iterrows():
    feature_key = '{}_{}'.format('region', str(row.region.astype(int)))
    idx = uid_to_idx.get(row.uid)
    if idx is not None:
        user_features[idx][feature_key] = 1

age_data = pd.read_csv('dataset/data_UCM_age.csv', sep=',')
del age_data['data']
age_data = age_data.rename(columns={'row': 'uid', 'col': 'age'})

for index, row in age_data.iterrows():
    feature_key = '{}_{}'.format('age', str(row.age.astype(int)))
    idx = uid_to_idx.get(row.uid)
    if idx is not None:
        user_features[idx][feature_key] = 1

user_features = DictVectorizer().fit_transform(user_features)

eye = sp.eye(user_features.shape[0], user_features.shape[0]).tocsr()
user_features_concat = sp.hstack((eye, user_features))
user_features_concat = user_features_concat.tocsr().astype(np.float32)


category_data = pd.read_csv('dataset/data_ICM_sub_class.csv', sep=',')
del category_data['data']
category_data = category_data.rename(columns={'row': 'iid', 'col': 'category'})

item_features = [{} for _ in idx_to_iid]
for index, row in category_data.iterrows():
    feature_key = '{}_{}'.format('category', str(row.category.astype(int)))
    idx = iid_to_idx.get(row.iid)
    if idx is not None:
        item_features[idx][feature_key] = 1

price_data = pd.read_csv('dataset/data_ICM_price.csv', sep=',')
price_data = price_data.rename(columns={'row': 'iid', 'data': 'price'})
del price_data['col']

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
    feature_key = price_data_series.get(row.iid)
    idx = iid_to_idx.get(row.iid)
    if idx is not None:
        item_features[idx][str(feature_key)] = 1

asset_data = pd.read_csv('dataset/data_ICM_asset.csv', sep=',')
asset_data = asset_data.rename(columns={'row': 'iid', 'data': 'asset'})
del asset_data['col']

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
    feature_key = asset_data_series.get(row.iid)
    idx = iid_to_idx.get(row.iid)
    if idx is not None:
        item_features[idx][str(feature_key)] = 1

item_features = DictVectorizer().fit_transform(item_features)

eye = sp.eye(item_features.shape[0], item_features.shape[0]).tocsr()
item_features_concat = sp.hstack((eye, item_features))
item_features_concat = item_features_concat.tocsr().astype(np.float32)

eval_train = train.copy()
non_eval_users = list(set(range(train.shape[0])) - set(user_index))

eval_train = eval_train.tolil()
for u in non_eval_users:
    eval_train[u, :] = 0.0
eval_train = eval_train.tocsr()


print(pd.DataFrame(train.toarray()))
print(pd.DataFrame(test.toarray()))
print(pd.DataFrame(user_features_concat.toarray()))
print(pd.DataFrame(item_features_concat.toarray()))


model = LightFM(loss='warp',
                no_components=30,
                max_sampled=5,
                learning_rate=0.05,
                learning_schedule='adagrad',
                user_alpha=0.001,
                item_alpha=0.001,
                random_state=np.random.RandomState(3333))

old_epoch = 0
train_mapk = []
test_mapk = []
headers = ['Epoch', 'train map@10', 'test map@10']
print_log(headers, header=True)
for epoch in range(1, 500, 1):
    epochs = epoch - old_epoch
    model.fit_partial(train,
                      user_features=user_features_concat,
                      item_features=item_features_concat,
                      epochs=epochs)
    old_epoch = epoch
    train_pk = precision_at_k(model, eval_train,
                              train_interactions=None,
                              user_features=user_features_concat,
                              item_features=item_features_concat,
                              k=10)
    test_pk = precision_at_k(model, test,
                             train_interactions=None,
                             user_features=user_features_concat,
                             item_features=item_features_concat,
                             k=10)
    train_mapk.append(train_pk.mean())
    test_mapk.append(test_pk.mean())
    row = [epoch, train_mapk[-1], test_mapk[-1]]
    print_log(row)

to_predict = pd.read_table('dataset/data_target_users_test.csv', delimiter=',')
to_predict.columns = ['user']

results = {}

for user in to_predict['user']:
    scores = model.predict(user_ids=user,
                           item_ids=category_data['iid'].tolist())
    print(np.argsort(-scores)[:10])
    top_items = category_data['iid'][np.argsort(-scores)[:10]]
    results[user] = top_items

create_csv(results)
