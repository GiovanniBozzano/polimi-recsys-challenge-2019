import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from lib.similarity.compute_similarity import ComputeSimilarity, SimilarityFunction
import session
import utils


# top-k = Oggetti simili da mappare per ogni oggetto.
#         Aumentando top-k si considerano più oggetti simili per ciascun oggetto, aumentando la probabilità di trovare
#         similitudini comuni a più oggetti con cui l'utente ha interagito.
# shrink = Reduces the importance of average quantities with a small support.
def compute_similarity(icm, top_k, shrink, similarity):
    similarity_object = ComputeSimilarity(icm.transpose().tocsr(), topK=top_k, shrink=shrink, similarity=similarity)
    similarity_object = similarity_object.compute_similarity()
    """
    for i in [2103]:
        test = pd.DataFrame(similarity_object.transpose().toarray()).loc[[i]]
        test = test.loc[:, (test != 0).any(axis=0)]
        print(test)
    """
    return similarity_object


class ItemContentBasedFiltering(object):
    """
    Crea una similarity matrix che rappresenta quanto ogni oggetto è simile a top-k altri oggetti.
    La similarity matrix è la somma pesata di tre similarity matrix, ciascuna rappresentante una feature diversa con
    valori da 0 a 1. I pesi dovrebbero avere somma 1 in modo da mantenere i valori normalizzati anche nella similarity
    matrix complessiva.

    La similarity matrix complessiva viene moltiplicata per la matrice delle interazioni per ottenere una matrice di
    oggetti simili a quelli con cui l'utente ha interagito, ciascuno con il suo punteggio di similarità.

    Un altro approccio è calcolare i punteggi di similarità con ogni similarity matrix separatamente e poi fare la somma
    pesata, ma in questo modo i punteggi non sono normalizzati.
    """

    def __init__(self, top_k_item_asset=1000, top_k_item_price=1000, top_k_item_sub_class=1000,
                 shrink_item_asset=1, shrink_item_price=1, shrink_item_sub_class=1,
                 weight_item_asset=0.2, weight_item_price=0.2):

        # 0.01147660397247628
        self.top_k_item_asset = top_k_item_asset
        self.top_k_item_price = top_k_item_price
        self.top_k_item_sub_class = top_k_item_sub_class
        self.shrink_item_asset = shrink_item_asset
        self.shrink_item_price = shrink_item_price
        self.shrink_item_sub_class = shrink_item_sub_class
        self.weight_item_asset = weight_item_asset
        self.weight_item_price = weight_item_price
        self.training_urm = None
        self.similarity_matrix = None

    def fit(self, training_urm):
        self.training_urm = training_urm

        items_assets = session.INSTANCE.get_icm_assets()
        items_prices = session.INSTANCE.get_icm_prices()
        items_sub_classes = session.INSTANCE.get_icm_sub_classes()
        # Imposta i pesi che verranno usati con lo shrink.
        items_sub_classes = utils.get_matrix_bm_25(items_sub_classes)

        items_assets_similarity_matrix = compute_similarity(items_assets, self.top_k_item_asset,
                                                            self.shrink_item_asset,
                                                            similarity=SimilarityFunction.EUCLIDEAN.value)
        items_assets_similarity_matrix = items_assets_similarity_matrix.transpose().tocsr()
        items_prices_similarity_matrix = compute_similarity(items_prices, self.top_k_item_price,
                                                            self.shrink_item_price,
                                                            similarity=SimilarityFunction.EUCLIDEAN.value)
        items_prices_similarity_matrix = items_prices_similarity_matrix.transpose().tocsr()
        items_sub_classes_similarity_matrix = compute_similarity(items_sub_classes, self.top_k_item_sub_class,
                                                                 self.shrink_item_sub_class,
                                                                 similarity=SimilarityFunction.COSINE.value)
        items_sub_classes_similarity_matrix = items_sub_classes_similarity_matrix.transpose().tocsr()
        self.similarity_matrix = items_assets_similarity_matrix * self.weight_item_asset + \
            items_prices_similarity_matrix * self.weight_item_price + \
            items_sub_classes_similarity_matrix * (1 - self.weight_item_asset - self.weight_item_price)
        """
        for i in [2103, 3741, 6885, 10144, 10807, 10808, 11752, 12638, 17594, 18053]:
            test = pd.DataFrame(self.similarity_matrix.toarray()).loc[[i]]
            test = test.loc[:, (test != 0).any(axis=0)]
            print(test)
        """

    def get_expected_ratings(self, user_id):
        interacted_items = self.training_urm[user_id]
        # Si selezionano gli oggetti dalla similarity matrix in base a quelli con cui l'utente ha interagito,
        # sommando i punteggi se presenti più volte. Ad esempio un oggetto identico a due oggetti con cui l'utente
        # ha interagito avrà punteggio 2.0.
        expected_ratings = interacted_items.dot(self.similarity_matrix)
        expected_ratings = normalize(expected_ratings, axis=1, norm='max').tocsr()
        expected_ratings = expected_ratings.toarray().ravel()
        if user_id == 0:
            print('0 ICBF RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        if user_id == 1:
            print('1 ICBF RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        if user_id == 2:
            print('2 ICBF RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        expected_ratings[interacted_items.indices] = -100
        return expected_ratings

    def recommend(self, user_id, k=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), axis=0)
        return recommended_items[:k]
