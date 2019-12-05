import numpy as np
import pandas as pd

import session
import utils
from Base.Similarity.Compute_Similarity import Compute_Similarity, SimilarityFunction


# top-k = Utenti simili da mappare per ogni utente.
#         Aumentando top-k si considerano più utenti simili per ciascun utente, aumentando la probabilità di trovare
#         similitudini comuni a più utenti.
# shrink = Reduces the importance of average quantities with a small support.
def compute_similarity(icm, top_k, shrink, similarity):
    similarity_object = Compute_Similarity(icm.transpose().tocsr(), topK=top_k, shrink=shrink, similarity=similarity)
    similarity_object = similarity_object.compute_similarity()
    """
    for i in [2103]:
        test = pd.DataFrame(similarity_object.transpose().toarray()).loc[[i]]
        test = test.loc[:, (test != 0).any(axis=0)]
        print(test)
    """
    return similarity_object


class UserContentBasedFiltering(object):
    """
    Crea una similarity matrix che rappresenta quanto ogni utente è simile a top-k altri utenti.
    La similarity matrix è la somma pesata di due similarity matrix, ciascuna rappresentante una feature diversa.
    I pesi dovrebbero avere somma 1 in modo da mantenere i valori normalizzati anche nella similarity
    matrix complessiva.

    La similarity matrix complessiva viene moltiplicata per la matrice delle interazioni per ottenere una matrice di
    utenti simili ciascuno con il suo punteggio di similarità.

    Un altro approccio è calcolare i punteggi di similarità con ogni similarity matrix separatamente e poi fare la somma
    pesata, ma in questo modo i punteggi non sono normalizzati.
    """

    def __init__(self, top_k_user_region=10000, top_k_user_age=10000,
                 shrink_user_region=8, shrink_user_age=8,
                 weight_user_region=0.5, weight_user_age=0.5):
        # 0.009160063829849168
        self.top_k_user_region = top_k_user_region
        self.top_k_user_age = top_k_user_age
        self.shrink_user_region = shrink_user_region
        self.shrink_user_age = shrink_user_age
        self.weight_user_region = weight_user_region
        self.weight_user_age = weight_user_age
        self.training_urm = None
        self.similarity_matrix = None

    def fit(self, training_urm):
        self.training_urm = training_urm

        users_regions = session.INSTANCE.get_ucm_regions()
        users_ages = session.INSTANCE.get_ucm_ages()
        # Imposta i pesi che verranno usati con lo shrink.
        users_regions = utils.get_matrix_bm_25(users_regions)
        users_ages = utils.get_matrix_bm_25(users_ages)

        users_regions_similarity_matrix = compute_similarity(users_regions, self.top_k_user_region,
                                                             self.shrink_user_region,
                                                             similarity=SimilarityFunction.COSINE.value)

        users_ages_similarity_matrix = compute_similarity(users_ages, self.top_k_user_age,
                                                          self.shrink_user_age,
                                                          similarity=SimilarityFunction.COSINE.value)

        self.similarity_matrix = users_regions_similarity_matrix * self.weight_user_region + \
                                 users_ages_similarity_matrix * self.weight_user_age

        self.recommendations = self.similarity_matrix.dot(self.training_urm)
        """
        for i in [2103, 3741, 6885, 10144, 10807, 10808, 11752, 12638, 17594, 18053]:
            test = pd.DataFrame(self.similarity_matrix.toarray()).loc[[i]]
            test = test.loc[:, (test != 0).any(axis=0)]
            print(test)
        """

    def get_expected_ratings(self, user_id):
        # Si selezionano gli oggetti dalla similarity matrix in base a quelli con cui l'utente ha interagito,
        # sommando i punteggi se presenti più volte. Ad esempio un oggetto identico a due oggetti con cui l'utente
        # ha interagito avrà punteggio 2.0.
        expected_ratings = self.recommendations[user_id].toarray().ravel()
        if user_id == 19335:
            print('CBF RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        """
        maximum = np.abs(expected_ratings).max(axis=0)
        if maximum > 0:
            expected_ratings = expected_ratings / maximum
        """
        return expected_ratings

    def recommend(self, user_id, k=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), axis=0)
        return recommended_items[:k]
