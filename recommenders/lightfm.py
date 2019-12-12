import lightfm
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
from tqdm import tqdm

import session


class LightFM(object):

    # 0.017361384106018098
    def __init__(self):
        self.training_urm = None
        self.ucm = None
        self.icm = None
        self.model = None

    def fit(self, training_urm):
        self.training_urm = training_urm

        user_list = np.arange(session.INSTANCE.users_amount)
        ones = np.ones(session.INSTANCE.users_amount, dtype=np.int)
        user_identity_matrix = sps.coo_matrix((ones, (user_list, user_list)),
                                              shape=(session.INSTANCE.users_amount, session.INSTANCE.users_amount),
                                              dtype=np.int).tocsr()
        item_list = np.arange(session.INSTANCE.items_amount)
        ones = np.ones(session.INSTANCE.items_amount, dtype=np.int)
        item_identity_matrix = sps.coo_matrix((ones, (item_list, item_list)),
                                              shape=(session.INSTANCE.items_amount, session.INSTANCE.items_amount),
                                              dtype=np.int).tocsr()

        self.ucm = sps.hstack((user_identity_matrix, session.INSTANCE.get_ucm_ages(),
                               session.INSTANCE.get_ucm_regions())).tocsr()
        self.icm = sps.hstack((item_identity_matrix, session.INSTANCE.get_icm_assets(),
                               session.INSTANCE.get_icm_prices(), session.INSTANCE.get_icm_sub_classes())).tocsr()

        self.model = lightfm.LightFM(loss='warp',
                                     no_components=64,
                                     learning_rate=0.05,
                                     learning_schedule='adagrad',
                                     user_alpha=1e-5,
                                     item_alpha=1e-5,
                                     random_state=np.random.RandomState(session.INSTANCE.random_seed))
        for _ in tqdm(np.arange(20)):
            self.model.fit_partial(self.training_urm,
                                   user_features=self.ucm,
                                   item_features=self.icm,
                                   epochs=1)

    def get_expected_ratings(self, user_id):
        expected_ratings = self.model.predict(user_ids=user_id, item_ids=np.arange(session.INSTANCE.items_amount),
                                              user_features=self.ucm, item_features=self.icm)
        expected_ratings = expected_ratings - expected_ratings.min()
        expected_ratings = expected_ratings.reshape(1, -1)
        expected_ratings = normalize(expected_ratings, axis=1, norm='max')
        expected_ratings = expected_ratings.ravel()
        interacted_items = self.training_urm[user_id]
        expected_ratings[interacted_items.indices] = -100
        return expected_ratings

    def recommend(self, user_id, k=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        unseen_items_mask = np.in1d(recommended_items, self.training_urm[user_id].indices, assume_unique=True,
                                    invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[:k]
