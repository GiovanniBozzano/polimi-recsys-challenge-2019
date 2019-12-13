import numpy as np
from sklearn.preprocessing import normalize
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.interactions import Interactions

from recommenders.recommender import Recommender


class Spotlight(Recommender):
    name = 'spotlight'

    # 0.010533919799192217
    def __init__(self, session):
        super().__init__(session)
        self.model = None

    def fit(self, training_urm):
        super().fit(self)

        training_urm = training_urm.tocoo()
        interactions = Interactions(user_ids=training_urm.row, item_ids=training_urm.col, ratings=training_urm.data)
        self.model = ImplicitFactorizationModel(loss='bpr',
                                                embedding_dim=2048,
                                                n_iter=2,
                                                batch_size=1024,
                                                learning_rate=1e-2,
                                                use_cuda=True,
                                                random_state=np.random.RandomState(self.session.random_seed))
        self.model.fit(interactions, verbose=True)

    def get_ratings(self, training_urm, user_id):
        ratings = self.model.predict(user_id)
        ratings = ratings - ratings.min()
        ratings = ratings.reshape(1, -1)
        ratings = normalize(ratings, axis=1, norm='max')
        ratings = ratings.ravel()
        return ratings
