import numpy as np
import pandas as pd
import scipy.sparse as sps
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import normalize

from recommenders.base_recommender import BaseRecommender


class FPGrowth(BaseRecommender):
    name = 'fpgrowth'

    def __init__(self, session, user_interactions_threshold=0, item_interactions_threshold=0):
        super().__init__(session, user_interactions_threshold, item_interactions_threshold)
        self.results = None

    def fit(self, training_urm):
        training_urm = super().fit(training_urm)

        training_urm = training_urm.tolil()
        interactions = []
        for row in training_urm.rows:
            interactions.append(row)

        transaction_encoder = TransactionEncoder()
        transaction_encoder_array = transaction_encoder.fit(interactions).transform(interactions)
        dataframe = pd.DataFrame(transaction_encoder_array, columns=transaction_encoder.columns_)

        self.results = fpgrowth(dataframe, min_support=0.005, use_colnames=True).sort_values('support', ascending=False)
        single_rows = []
        for index, row in self.results.iterrows():
            if len(row['itemsets']) <= 1:
                single_rows.append(index)
        for row in single_rows:
            self.results = self.results.drop(row)

        pd.set_option('display.max_rows', None)

    def get_ratings(self, training_urm, user_id):
        interacted_items = training_urm[user_id]
        zeroes = np.zeros(self.session.items_amount, dtype=np.int32)
        ratings = sps.coo_matrix((zeroes, (np.arange(self.session.items_amount), zeroes)),
                                 shape=(self.session.items_amount, 1),
                                 dtype=np.float32).tocsr()
        for item_id in interacted_items[0].indices:
            for itemset in self.results['itemsets']:
                if item_id in list(itemset):
                    for recommended_item_id in list(itemset):
                        ratings[recommended_item_id, 0] += 1
        ratings = ratings.reshape(1, -1)
        ratings = normalize(ratings, axis=0, norm='max')
        ratings = ratings.toarray().ravel()
        interacted_items = training_urm[user_id]
        ratings[interacted_items.indices] = -100
        return ratings
