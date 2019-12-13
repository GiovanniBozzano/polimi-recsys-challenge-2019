import lightfm
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
from tqdm import tqdm

from recommenders.base_recommender import BaseRecommender


class LightFM(BaseRecommender):
    name = 'lightfm'

    # 0.017361384106018098
    def __init__(self, session, user_interactions_threshold=0, item_interactions_threshold=0):
        super().__init__(session, user_interactions_threshold, item_interactions_threshold)
        self.ucm = None
        self.icm = None
        self.model = None

    def fit(self, training_urm):
        training_urm = super().fit(training_urm)

        user_list = np.arange(self.session.users_amount)
        ones = np.ones(self.session.users_amount, dtype=np.int32)
        user_identity_matrix = sps.coo_matrix((ones, (user_list, user_list)),
                                              shape=(self.session.users_amount, self.session.users_amount),
                                              dtype=np.int32).tocsr()
        item_list = np.arange(self.session.items_amount)
        ones = np.ones(self.session.items_amount, dtype=np.int32)
        item_identity_matrix = sps.coo_matrix((ones, (item_list, item_list)),
                                              shape=(self.session.items_amount, self.session.items_amount),
                                              dtype=np.int32).tocsr()

        self.ucm = sps.hstack((user_identity_matrix, self.session.get_ucm_ages(),
                               self.session.get_ucm_regions())).tocsr()
        self.icm = sps.hstack((item_identity_matrix, self.session.get_icm_assets(),
                               self.session.get_icm_prices(), self.session.get_icm_sub_classes())).tocsr()

        self.model = lightfm.LightFM(loss='warp',
                                     no_components=64,
                                     learning_rate=0.05,
                                     learning_schedule='adagrad',
                                     user_alpha=1e-5,
                                     item_alpha=1e-5,
                                     random_state=np.random.RandomState(self.session.random_seed))
        for _ in tqdm(np.arange(20)):
            self.model.fit_partial(interactions=training_urm,
                                   user_features=self.ucm,
                                   item_features=self.icm,
                                   epochs=1)

    def get_ratings(self, training_urm, user_id):
        ratings = self.model.predict(user_ids=user_id, item_ids=np.arange(self.session.items_amount),
                                     user_features=self.ucm, item_features=self.icm)
        ratings = ratings - ratings.min()
        ratings = ratings.reshape(1, -1)
        ratings = normalize(ratings, axis=1, norm='max')
        ratings = ratings.ravel()
        return ratings
