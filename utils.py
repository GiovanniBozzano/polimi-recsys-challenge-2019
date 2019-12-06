import os
from datetime import datetime

import scipy.sparse as sp
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import feature_extraction
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from Base.IR_feature_weighting import okapi_BM_25


def get_icm():
    tracks_data = pd.read_csv(os.path.join(os.getcwd(), '../data/tracks.csv'))
    artists = tracks_data.reindex(columns=['track_id', 'artist_id'])
    artists.sort_values(by='track_id', inplace=True)  # this seems not useful, values are already ordered
    artists_list = [[a] for a in artists['artist_id']]
    icm_artists = MultiLabelBinarizer(sparse_output=True).fit_transform(artists_list)
    icm_artists_csr = icm_artists.tocsr()

    albums = tracks_data.reindex(columns=['track_id', 'album_id'])
    albums.sort_values(by='track_id', inplace=True)  # this seems not useful, values are already ordered
    albums_list = [[a] for a in albums['album_id']]
    icm_albums = MultiLabelBinarizer(sparse_output=True).fit_transform(albums_list)
    icm_albums_csr = icm_albums.tocsr()

    durations = tracks_data.reindex(columns=['track_id', 'duration_sec'])
    durations.sort_values(by='track_id', inplace=True)  # this seems not useful, values are already ordered
    durations_list = [[d] for d in durations['duration_sec']]
    icm_durations = MultiLabelBinarizer(sparse_output=True).fit_transform(durations_list)
    icm_durations_csr = icm_durations.tocsr()

    icm = sparse.hstack((icm_albums_csr, icm_artists_csr, icm_durations_csr))
    return icm.tocsr()


def get_matrix_bm_25(matrix):
    return okapi_BM_25(matrix)


def get_matrix_tfidf(matrix):
    matrix_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(matrix)
    return matrix_tfidf.tocsr()


def mean_average_precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(len(is_relevant)))
    return np.sum(p_at_k) / np.min([len(relevant_items), len(is_relevant)])


def categorize(value, category_map):
    for zone in category_map:
        if zone[1] < value <= zone[2]:
            return zone[0]
    return 0


def train_test_split(interactions, test_percentage, split_count):
    train = interactions.copy().tocoo()
    test = sp.lil_matrix(train.shape)
    try:
        user_index = np.random.choice(np.where(np.bincount(train.row) >= split_count * 2)[0], replace=False,
                                      size=np.int64(np.floor(test_percentage * train.shape[0]))
        ).tolist()
    except:
        print('Not enough users with > {} interactions for fraction of {}'.format(2 * split_count, test_percentage))
        raise
    train = train.tolil()
    for user in user_index:
        test_interactions = np.random.choice(interactions.getrow(user).indices, size=split_count, replace=False)
        train[user, test_interactions] = 0
        test[user, test_interactions] = interactions[user, test_interactions]
    assert(train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr(), user_index


def train_test_split_leave_one_out(urm, test_users):
    training_urm = urm.copy().tocoo()
    test_urm = sp.lil_matrix(training_urm.shape)
    training_urm = training_urm.tolil()
    for user in test_users:
        test_item = np.random.choice(urm.getrow(user).indices, replace=False)
        training_urm[user, test_item] = 0
        test_urm[user, test_item] = urm[user, test_item]
    assert(training_urm.multiply(test_urm).nnz == 0)
    return training_urm.tocsr(), test_urm.tocsr()


def create_csv(results, users_column, items_column, results_directory='./'):
    csv_filename = 'results_'
    csv_filename += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'
    with open(os.path.join(results_directory, csv_filename), 'w') as file:
        file.write(users_column + ',' + items_column + '\n')
        for key, value in results.items():
            file.write(str(key) + ',')
            first = True
            for prediction in value:
                if not first:
                    file.write(' ')
                first = False
                file.write(str(prediction))
            file.write('\n')
