"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from lib.Recommender_utils import similarityMatrixTopK, check_matrix


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


class SLIMBPR(object):
    # 0.03988547317661249
    def __init__(self, positive_threshold=1, final_model_sparse_weights=True, train_with_sparse_weights=False,
                 symmetric=False, epochs=80, batch_size=1, lambda_i=0.03, lambda_j=0.003, learning_rate=0.01, top_k=40,
                 sgd_mode='adagrad', gamma=0.995, beta_1=0.9, beta_2=0.999):

        self.positive_threshold = positive_threshold
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
        self.training_urm = None
        self.W_sparse = None
        self.S_incremental = None
        self.cython_epoch = None
        self.recommendations = None

    def fit(self, training_urm):
        from recommenders.slim_bpr_epoch import SLIMBPREpoch

        if self.train_with_sparse_weights is None:
            required_memory = estimate_required_mb(self.training_urm.shape[1], self.symmetric)
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

        self.training_urm = training_urm

        training_urm_positive = self.training_urm.copy()
        if self.positive_threshold is not None:
            training_urm_positive.data = training_urm_positive.data >= self.positive_threshold
            training_urm_positive.eliminate_zeros()
            assert training_urm_positive.nnz > 0, 'SLIM BPR: training_urm_positive is empty, positive threshold is ' \
                                                  'too high'

        self.cython_epoch = SLIM_BPR_Cython_Epoch(training_urm_positive,
                                                  train_with_sparse_weights=self.train_with_sparse_weights,
                                                  final_model_sparse_weights=self.final_model_sparse_weights,
                                                  topK=self.top_k,
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

        self.recommendations = self.training_urm.dot(self.W_sparse)

    def _initialize_incremental_model(self):
        self.S_incremental = self.cython_epoch.get_S()
        self.S_best = self.S_incremental.copy()

    def _update_best_model(self):
        self.S_best = self.S_incremental.copy()

    def _run_epoch(self):
        self.cython_epoch.epochIteration_Cython()

    def get_S_incremental_and_set_W(self):
        self.S_incremental = self.cython_epoch.get_S()
        if self.train_with_sparse_weights:
            self.W_sparse = self.S_incremental
            self.W_sparse = check_matrix(self.W_sparse, format='csr')
        else:
            self.W_sparse = similarityMatrixTopK(self.S_incremental, k=self.top_k)
            self.W_sparse = check_matrix(self.W_sparse, format='csr')

    def get_expected_ratings(self, user_id):
        user_id = int(user_id)
        expected_ratings = self.recommendations[user_id]
        expected_ratings = normalize(expected_ratings, axis=1, norm='max').tocsr()
        expected_ratings = expected_ratings.toarray().ravel()
        if user_id == 0:
            print('0 SLIM BPR RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        if user_id == 1:
            print('1 SLIM BPR RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        if user_id == 2:
            print('2 SLIM BPR RATINGS:')
            print(pd.DataFrame(expected_ratings).sort_values(by=0, ascending=False))
        return expected_ratings

    def recommend(self, user_id, at=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.training_urm[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]
