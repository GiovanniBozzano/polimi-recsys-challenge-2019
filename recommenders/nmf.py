import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import normalize

from recommenders.base_recommender import BaseRecommender


class NMF(BaseRecommender):
    name = 'nmf'

    def __init__(self, session, user_interactions_threshold=0, item_interactions_threshold=0,
                 n_factors=100, l1_ratio=0.5, solver='mu', init_type='random', beta_loss='frobenius'):
        super().__init__(session, user_interactions_threshold, item_interactions_threshold)
        self.n_factors = n_factors
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.init_type = init_type
        self.beta_loss = beta_loss
        self.user_factors = None
        self.item_factors = None

    def fit(self, training_urm):
        training_urm = super().fit(training_urm)

        model = decomposition.NMF(n_components=self.n_factors,
                                  init=self.init_type,
                                  solver=self.solver,
                                  beta_loss=self.beta_loss,
                                  l1_ratio=self.l1_ratio,
                                  max_iter=500)
        model.fit(training_urm)
        self.item_factors = model.components_.copy().T
        self.user_factors = model.transform(training_urm)

    def get_ratings(self, training_urm, user_id):
        ratings = np.dot(self.user_factors[user_id], self.item_factors.T)
        ratings = ratings - ratings.min()
        ratings = ratings.reshape(1, -1)
        ratings = normalize(ratings, axis=1, norm='max')
        ratings = ratings.ravel()
        interacted_items = training_urm[user_id]
        ratings[interacted_items.indices] = -100
        return ratings
