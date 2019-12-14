import similaripy
import similaripy.normalization
from sklearn.preprocessing import normalize

from lib.similarity.compute_similarity_euclidean import ComputeSimilarityEuclidean
from recommenders.base_recommender import BaseRecommender


class ItemContentBasedFiltering(BaseRecommender):
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

    name = 'item_content_based_filtering'

    # 0.013040420717911071
    def __init__(self, session, user_interactions_threshold=0, item_interactions_threshold=0,
                 top_k_item_asset=140, top_k_item_price=140, top_k_item_sub_class=300,
                 shrink_item_asset=1, shrink_item_price=1, shrink_item_sub_class=1, weight_item_asset=0.2,
                 weight_item_price=0.2):
        super().__init__(session, user_interactions_threshold, item_interactions_threshold)
        self.top_k_item_asset = top_k_item_asset
        self.top_k_item_price = top_k_item_price
        self.top_k_item_sub_class = top_k_item_sub_class
        self.shrink_item_asset = shrink_item_asset
        self.shrink_item_price = shrink_item_price
        self.shrink_item_sub_class = shrink_item_sub_class
        self.weight_item_asset = weight_item_asset
        self.weight_item_price = weight_item_price
        self.similarity_matrix = None

    def fit(self, training_urm):
        super().fit(training_urm)

        items_assets = self.session.get_icm_assets()
        items_prices = self.session.get_icm_prices()
        items_sub_classes = self.session.get_icm_sub_classes()
        # Imposta i pesi che verranno usati con lo shrink.
        items_sub_classes = similaripy.normalization.bm25plus(items_sub_classes)

        items_assets_similarity_matrix = ComputeSimilarityEuclidean(items_assets.transpose().tocsr(),
                                                                    top_k=self.top_k_item_asset,
                                                                    shrink=self.shrink_item_asset)
        items_assets_similarity_matrix = items_assets_similarity_matrix.compute_similarity()
        items_assets_similarity_matrix = items_assets_similarity_matrix.transpose().tocsr()

        items_prices_similarity_matrix = ComputeSimilarityEuclidean(items_prices.transpose().tocsr(),
                                                                    top_k=self.top_k_item_price,
                                                                    shrink=self.shrink_item_price)
        items_prices_similarity_matrix = items_prices_similarity_matrix.compute_similarity()
        items_prices_similarity_matrix = items_prices_similarity_matrix.transpose().tocsr()

        items_sub_classes_similarity_matrix = similaripy.cosine(items_sub_classes, k=self.top_k_item_sub_class,
                                                                shrink=self.shrink_item_sub_class, binary=False,
                                                                verbose=False)
        items_sub_classes_similarity_matrix = items_sub_classes_similarity_matrix.tocsr()
        self.similarity_matrix = items_assets_similarity_matrix * self.weight_item_asset + \
                                 items_prices_similarity_matrix * self.weight_item_price + \
                                 items_sub_classes_similarity_matrix * (
                                         1 - self.weight_item_asset - self.weight_item_price)

    def get_ratings(self, training_urm, user_id):
        interacted_items = training_urm[user_id]
        # Si selezionano gli oggetti dalla similarity matrix in base a quelli con cui l'utente ha interagito,
        # sommando i punteggi se presenti più volte. Ad esempio un oggetto identico a due oggetti con cui l'utente
        # ha interagito avrà punteggio 2.0.
        ratings = interacted_items.dot(self.similarity_matrix)
        ratings = normalize(ratings, axis=1, norm='max')
        ratings = ratings.toarray().ravel()
        ratings[interacted_items.indices] = -100
        return ratings
