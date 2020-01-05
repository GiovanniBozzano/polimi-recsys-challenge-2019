import numpy as np
from tqdm import tqdm
import scipy.sparse as sps
import utils


class Evaluator(object):

    def __init__(self, session):
        self.session = session
        self.training_urm = None
        self.test_urm = None
        self.target_users = None

    def split(self, leave_one_out=True, test_percentage=0.2, test_interaction_threshold=10):
        if leave_one_out:
            self.target_users = self.session.user_list_unique
            self.training_urm, self.test_urm = utils.train_test_split_leave_one_out(self.session.urm,
                                                                                    self.target_users)
        else:
            self.training_urm, self.test_urm, self.target_users = utils.train_test_split(self.session.urm,
                                                                                         test_percentage,
                                                                                         test_interaction_threshold)

    def evaluate(self, recommender, k):
        mean_average_precision_final = 0
        recommender.fit(self.training_urm)
        for target_user in tqdm(self.target_users):
            recommended_items = recommender.recommend(self.training_urm, target_user, k)
            relevant_items = self.test_urm[target_user].indices
            mean_average_precision = utils.mean_average_precision(recommended_items, relevant_items)
            mean_average_precision_final += mean_average_precision
        mean_average_precision_final /= len(self.target_users)
        return mean_average_precision_final
