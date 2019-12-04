import numpy as np

import utils
from tqdm import tqdm


class Evaluator(object):

    def __init__(self, session):
        self.session = session
        self.training_urm = None
        self.test_urm = None
        self.target_users = None

    def split(self, leave_one_out=True, test_percentage=0.2, test_interaction_threshold=10):
        if leave_one_out:
            test_users2 = []
            for user_id in self.session.user_list_unique:
                if self.session.urm[user_id].getnnz() > 10:
                    test_users2.append(user_id)
            self.target_users = self.session.user_list_unique
            self.training_urm, self.test_urm = utils.train_test_split_leave_one_out(self.session.urm,
                                                                                    self.session.user_list_unique)
        else:
            self.training_urm, self.test_urm, self.target_users = utils.train_test_split(self.session.urm,
                                                                                         test_percentage,
                                                                                         test_interaction_threshold)
        training_urm = self.training_urm.copy().tolil()
        for user_id in self.session.user_list_unique:
            if self.training_urm[user_id].getnnz() < self.session.users_usefulness_threshold:
                for item_id in self.training_urm[user_id].indices:
                    training_urm[user_id, item_id] = 0
        self.training_urm = training_urm.transpose().tocsr()
        training_urm = training_urm.transpose().tocsr()
        for item_id in self.session.item_list_unique:
            if self.training_urm[item_id].getnnz() < self.session.items_usefulness_threshold:
                for user_id in self.training_urm[item_id].indices:
                    training_urm[item_id, user_id] = 0
        self.training_urm = training_urm.transpose().tocsr()

    def evaluate(self, recommender, k):
        mean_average_precision_final = 0
        recommender.fit(self.training_urm)
        for target_user in tqdm(self.target_users):
            recommended_items = recommender.recommend(target_user, k)
            relevant_items = self.test_urm[target_user].indices
            mean_average_precision = utils.mean_average_precision(recommended_items, relevant_items)
            mean_average_precision_final += mean_average_precision
        mean_average_precision_final /= len(self.target_users)
        return mean_average_precision_final
