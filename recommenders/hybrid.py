from recommenders.als import ALS
from recommenders.elastic_net import ElasticNet
from recommenders.item_based_collaborative_filtering import ItemBasedCollaborativeFiltering
from recommenders.item_content_based_filtering import ItemContentBasedFiltering
from recommenders.recommender import Recommender
from recommenders.slim_bpr import SLIMBPR
from recommenders.user_based_collaborative_filtering import UserBasedCollaborativeFiltering
from recommenders.user_content_based_filtering import UserContentBasedFiltering


class Hybrid(Recommender):
    name = 'hybrid'

    def __init__(self, session, weights_cold_start, weights_low_interactions, weights_high_interactions,
                 user_content_based_filtering_parameters, item_content_based_filtering_parameters,
                 user_based_collaborative_filtering_parameters, item_based_collaborative_filtering_parameters,
                 slim_bpr_parameters, als_parameters):
        super().__init__(session)

        self.weights_cold_start = weights_cold_start
        self.weights_low_interactions = weights_low_interactions
        self.weights_high_interactions = weights_high_interactions

        self.recommenders = [
            UserContentBasedFiltering(session=session,
                                      top_k_user_age=user_content_based_filtering_parameters['top_k_user_age'],
                                      top_k_user_region=user_content_based_filtering_parameters['top_k_user_region'],
                                      shrink_user_age=user_content_based_filtering_parameters['shrink_user_age'],
                                      shrink_user_region=user_content_based_filtering_parameters['shrink_user_region'],
                                      weight_user_age=user_content_based_filtering_parameters['weight_user_age']),
            ItemContentBasedFiltering(session=session,
                                      top_k_item_asset=item_content_based_filtering_parameters['top_k_item_asset'],
                                      top_k_item_price=item_content_based_filtering_parameters['top_k_item_price'],
                                      top_k_item_sub_class=item_content_based_filtering_parameters[
                                          'top_k_item_sub_class'],
                                      shrink_item_asset=item_content_based_filtering_parameters['shrink_item_asset'],
                                      shrink_item_price=item_content_based_filtering_parameters['shrink_item_price'],
                                      shrink_item_sub_class=item_content_based_filtering_parameters[
                                          'shrink_item_sub_class'],
                                      weight_item_asset=item_content_based_filtering_parameters['weight_item_asset'],
                                      weight_item_price=item_content_based_filtering_parameters['weight_item_price']),
            UserBasedCollaborativeFiltering(session=session,
                                            top_k=user_based_collaborative_filtering_parameters['top_k'],
                                            shrink=user_based_collaborative_filtering_parameters['shrink']),
            ItemBasedCollaborativeFiltering(session=session,
                                            top_k=item_based_collaborative_filtering_parameters['top_k'],
                                            shrink=item_based_collaborative_filtering_parameters['shrink']),
            SLIMBPR(session=session,
                    epochs=slim_bpr_parameters['epochs'],
                    top_k=slim_bpr_parameters['top_k']), ElasticNet(session=session),
            ALS(session=session,
                factors=als_parameters['factors'],
                regularization=als_parameters['regularization'],
                iterations=als_parameters['iterations'],
                alpha=als_parameters['alpha']),
            # LightFM(session=session)
            # SVD(session=session),
            # NMF(session=session),
            # TopPopular(session=session),
            # Spotlight(session=session)
        ]

    def fit(self, training_urm):
        for recommender in self.recommenders:
            recommender.fit(training_urm.copy())

    def get_ratings(self, training_urm, user_id):
        if training_urm[user_id].getnnz() > 10:
            weights = self.weights_high_interactions
        elif training_urm[user_id].getnnz() == 0:
            weights = self.weights_cold_start
        else:
            weights = self.weights_low_interactions

        hybrid_ratings = [self.session.items_amount]
        for recommender in self.recommenders:
            hybrid_ratings += recommender.get_ratings(training_urm, user_id) * weights[recommender.name]

        return hybrid_ratings
