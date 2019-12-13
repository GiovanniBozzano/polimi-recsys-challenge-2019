"""
Created on 07/09/17
@author: Maurizio Ferrari Dacrema
"""
import os
import sys

from sklearn.preprocessing import normalize

from lib.recommender_utils import similarity_matrix_top_k, check_matrix
from recommenders.base_recommender import BaseRecommender


def estimate_required_mb(items_amount, symmetric):
    required_mb = 8 * items_amount ** 2 / 1e+06
    if symmetric:
        required_mb /= 2
    return required_mb


def get_ram_status():
    try:
        data_list = os.popen('free -t -m').readlines()[1].split()
        total_memory = float(data_list[1])
        used_memory = float(data_list[2])
        available_memory = float(data_list[6])
    except Exception as exception:
        print('Unable to read memory status: {}'.format(str(exception)))
        total_memory, used_memory, available_memory = None, None, None
    return total_memory, used_memory, available_memory


class SLIMBPR(BaseRecommender):
    name = 'slim_bpr'

    # 0.04119579049133264
    def __init__(self, session, user_interactions_threshold=0, item_interactions_threshold=1,
                 final_model_sparse_weights=True, train_with_sparse_weights=False, symmetric=False,
                 epochs=200, batch_size=1, lambda_i=0.05, lambda_j=0.005, learning_rate=0.001, top_k=16,
                 sgd_mode='adagrad', gamma=0.995, beta_1=0.9, beta_2=0.999):
        super().__init__(session, user_interactions_threshold, item_interactions_threshold)
        self.final_model_sparse_weights = final_model_sparse_weights
        self.train_with_sparse_weights = train_with_sparse_weights
        if self.train_with_sparse_weights:
            self.final_model_sparse_weights = True
        self.symmetric = symmetric
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate
        self.top_k = top_k
        self.sgd_mode = sgd_mode
        self.gamma = gamma
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.W_sparse = None
        self.S_incremental = None
        self.cython_epoch = None
        self.recommendations = None

    def fit(self, training_urm):

        training_urm = super().fit(training_urm)

        from recommenders.slim_bpr_epoch import SLIMBPREpoch

        if self.train_with_sparse_weights is None:
            required_memory = estimate_required_mb(training_urm.shape[1], self.symmetric)
            total_memory, _, available_memory = get_ram_status()
            if total_memory is not None:
                string = 'Automatic selection of fastest train mode. Available RAM is {:.2f} MB ({:.2f}%) of {:.2f} ' \
                         'MB, required is {:.2f} MB.'.format(available_memory, available_memory / total_memory * 100,
                                                             total_memory, required_memory)
            else:
                string = 'Automatic selection of fastest train mode. Unable to get current RAM status, you may be ' \
                         'using a non-Linux operating system.'
            if total_memory is None or required_memory / available_memory < 0.5:
                print(string + 'Using dense matrix.')
                self.train_with_sparse_weights = False
            else:
                print(string + 'Using sparse matrix.')
                self.train_with_sparse_weights = True

        training_urm_positive = training_urm.copy()

        self.cython_epoch = SLIMBPREpoch(training_urm_positive,
                                         train_with_sparse_weights=self.train_with_sparse_weights,
                                         final_model_sparse_weights=self.final_model_sparse_weights,
                                         top_k=self.top_k,
                                         learning_rate=self.learning_rate,
                                         li_reg=self.lambda_i,
                                         lj_reg=self.lambda_j,
                                         batch_size=self.batch_size,
                                         symmetric=self.symmetric,
                                         sgd_mode=self.sgd_mode,
                                         gamma=self.gamma,
                                         beta_1=self.beta_1,
                                         beta_2=self.beta_2)

        self._initialize_incremental_model()
        current_epoch = 0

        while current_epoch < self.epochs:
            self._run_epoch()
            self._update_best_model()
            current_epoch += 1

        self.get_S_incremental_and_set_W()
        self.cython_epoch._dealloc()
        sys.stdout.flush()

        self.recommendations = training_urm.dot(self.W_sparse)

    def _initialize_incremental_model(self):
        self.S_incremental = self.cython_epoch.get_S()
        self.S_best = self.S_incremental.copy()

    def _update_best_model(self):
        self.S_best = self.S_incremental.copy()

    def _run_epoch(self):
        self.cython_epoch.epoch_iteration_cython()

    def get_S_incremental_and_set_W(self):
        self.S_incremental = self.cython_epoch.get_S()
        if self.train_with_sparse_weights:
            self.W_sparse = self.S_incremental
            self.W_sparse = check_matrix(self.W_sparse, format='csr')
        else:
            self.W_sparse = similarity_matrix_top_k(self.S_incremental, k=self.top_k)
            self.W_sparse = check_matrix(self.W_sparse, format='csr')

    def get_ratings(self, training_urm, user_id):
        ratings = self.recommendations[user_id]
        ratings = normalize(ratings, axis=1, norm='max')
        ratings = ratings.toarray().ravel()
        return ratings
