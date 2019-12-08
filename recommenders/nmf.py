import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import normalize


class NMF(object):

    # 0.02005880674290138
    def __init__(self, n_factors=100,
                 l1_ratio=0.5,
                 solver='mu',
                 init_type='random',
                 beta_loss='frobenius'):
        self.n_factors = n_factors
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.init_type = init_type
        self.beta_loss = beta_loss
        self.training_urm = None
        self.user_factors = None
        self.item_factors = None

    def fit(self, training_urm):
        self.training_urm = training_urm

        model = decomposition.NMF(n_components=self.n_factors,
                                  init=self.init_type,
                                  solver=self.solver,
                                  beta_loss=self.beta_loss,
                                  l1_ratio=self.l1_ratio,
                                  max_iter=500)

        model.fit(self.training_urm)

        self.item_factors = model.components_.copy().T
        self.user_factors = model.transform(self.training_urm)

    def get_expected_ratings(self, user_id):
        expected_ratings = np.dot(self.user_factors[user_id], self.item_factors.T)
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
