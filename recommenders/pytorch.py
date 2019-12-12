import sys
import time

import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch.autograd import Variable
from torch.utils.data import DataLoader

import session


class PyTorch(object):

    def __init__(self):
        self.training_urm = None
        self.ucm = None
        self.icm = None
        self.model = None

    def seconds_to_biggest_unit(self, time_in_seconds, data_array=None):

        conversion_factor = [
            ("sec", 60),
            ("min", 60),
            ("hour", 24),
            ("day", 365),
        ]

        terminate = False
        unit_index = 0

        new_time_value = time_in_seconds
        new_time_unit = "sec"

        while not terminate:

            next_time = new_time_value / conversion_factor[unit_index][1]

            if next_time >= 1.0:
                new_time_value = next_time

                if data_array is not None:
                    data_array /= conversion_factor[unit_index][1]

                unit_index += 1
                new_time_unit = conversion_factor[unit_index][0]

            else:
                terminate = True

        if data_array is not None:
            return new_time_value, new_time_unit, data_array

        else:
            return new_time_value, new_time_unit

    def _train_with_early_stopping(self, epochs_max, epochs_min=0,
                                   validation_every_n=None, stop_on_validation=False,
                                   validation_metric=None, lower_validations_allowed=None, evaluator_object=None,
                                   algorithm_name="Incremental_Training_Early_Stopping"):

        assert epochs_max >= 0, "{}: Number of epochs_max must be >= 0, passed was {}".format(algorithm_name,
                                                                                              epochs_max)
        assert epochs_min >= 0, "{}: Number of epochs_min must be >= 0, passed was {}".format(algorithm_name,
                                                                                              epochs_min)
        assert epochs_min <= epochs_max, "{}: epochs_min must be <= epochs_max, passed are epochs_min {}, epochs_max {}".format(
            algorithm_name, epochs_min, epochs_max)

        # Train for max number of epochs with no validation nor early stopping
        # OR Train for max number of epochs with validation but NOT early stopping
        # OR Train for max number of epochs with validation AND early stopping
        assert evaluator_object is None or \
               (
                       evaluator_object is not None and not stop_on_validation and validation_every_n is not None and validation_metric is not None) or \
               (
                       evaluator_object is not None and stop_on_validation and validation_every_n is not None and validation_metric is not None and lower_validations_allowed is not None), \
            "{}: Inconsistent parameters passed, please check the supported uses".format(algorithm_name)

        start_time = time.time()

        self.best_validation_metric = None
        lower_validatons_count = 0
        convergence = False

        self.epochs_best = 0

        epochs_current = 0

        while epochs_current < epochs_max and not convergence:

            self._run_epoch(epochs_current)

            # If no validation required, always keep the latest
            if evaluator_object is None:

                self.epochs_best = epochs_current

            # Determine whether a validaton step is required
            elif (epochs_current + 1) % validation_every_n == 0:

                print("{}: Validation begins...".format(algorithm_name))

                self._prepare_model_for_validation()

                # If the evaluator validation has multiple cutoffs, choose the first one
                results_run, results_run_string = evaluator_object.evaluateRecommender(self)
                results_run = results_run[list(results_run.keys())[0]]

                print("{}: {}".format(algorithm_name, results_run_string))

                # Update optimal model
                current_metric_value = results_run[validation_metric]

                if not np.isfinite(current_metric_value):
                    assert False, "{}: metric value is not a finite number, terminating!".format(self.RECOMMENDER_NAME)

                if self.best_validation_metric is None or self.best_validation_metric < current_metric_value:

                    print("{}: New best model found! Updating.".format(algorithm_name))

                    self.best_validation_metric = current_metric_value

                    self._update_best_model()

                    self.epochs_best = epochs_current + 1
                    lower_validatons_count = 0

                else:
                    lower_validatons_count += 1

                if stop_on_validation and lower_validatons_count >= lower_validations_allowed and epochs_current >= epochs_min:
                    convergence = True

                    elapsed_time = time.time() - start_time
                    new_time_value, new_time_unit = self.seconds_to_biggest_unit(elapsed_time)

                    print(
                        "{}: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                            algorithm_name, epochs_current + 1, validation_metric, self.epochs_best,
                            self.best_validation_metric, new_time_value, new_time_unit))

            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = self.seconds_to_biggest_unit(elapsed_time)

            print("{}: Epoch {} of {}. Elapsed time {:.2f} {}".format(
                algorithm_name, epochs_current + 1, epochs_max, new_time_value, new_time_unit))

            epochs_current += 1

            sys.stdout.flush()
            sys.stderr.flush()

        # If no validation required, keep the latest
        if evaluator_object is None:
            self._prepare_model_for_validation()
            self._update_best_model()

        # Stop when max epochs reached and not early-stopping
        if not convergence:
            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = self.seconds_to_biggest_unit(elapsed_time)

            if evaluator_object is not None and self.best_validation_metric is not None:
                print(
                    "{}: Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                        algorithm_name, epochs_current, validation_metric, self.epochs_best,
                        self.best_validation_metric, new_time_value, new_time_unit))
            else:
                print("{}: Terminating at epoch {}. Elapsed time {:.2f} {}".format(
                    algorithm_name, epochs_current, new_time_value, new_time_unit))

    def fit(self, training_urm):
        self.training_urm = training_urm

        self.n_factors = 10

        # Select only positive interactions
        URM_train_positive = self.training_urm.copy()

        URM_train_positive.data = URM_train_positive.data >= 4
        URM_train_positive.eliminate_zeros()

        self.batch_size = 128
        self.learning_rate = 0.001

        ########################################################################################################
        #
        #                                SETUP PYTORCH MODEL AND DATA READER
        #
        ########################################################################################################

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("MF_MSE_PyTorch: Using CUDA")
        else:
            self.device = torch.device('cpu')
            print("MF_MSE_PyTorch: Using CPU")

        from recommenders.pytorch_model import MF_MSE_PyTorch_model, DatasetIterator_URM

        n_users, n_items = self.training_urm.shape

        self.pyTorchModel = MF_MSE_PyTorch_model(n_users, n_items, self.n_factors).to(self.device)

        # Choose loss
        self.lossFunction = torch.nn.MSELoss(size_average=False)
        # self.lossFunction = torch.nn.BCELoss(size_average=False)
        self.optimizer = torch.optim.Adagrad(self.pyTorchModel.parameters(), lr=self.learning_rate)

        dataset_iterator = DatasetIterator_URM(self.training_urm)

        self.train_data_loader = DataLoader(dataset=dataset_iterator,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            # num_workers = 2,
                                            )

        ########################################################################################################

        self._train_with_early_stopping(30)

        self.ITEM_factors = self.W_best.copy()
        self.USER_factors = self.H_best.copy()

        print("Computing NMF decomposition... Done!")

    def _initialize_incremental_model(self):

        self.W_incremental = self.pyTorchModel.get_W()
        self.W_best = self.W_incremental.copy()

        self.H_incremental = self.pyTorchModel.get_H()
        self.H_best = self.H_incremental.copy()

    def _update_incremental_model(self):

        self.W_incremental = self.pyTorchModel.get_W()
        self.H_incremental = self.pyTorchModel.get_H()

        self.W = self.W_incremental.copy()
        self.H = self.H_incremental.copy()

    def _update_best_model(self):

        self.W_best = self.W_incremental.copy()
        self.H_best = self.H_incremental.copy()

    def _run_epoch(self, num_epoch):

        for num_batch, (input_data, label) in enumerate(self.train_data_loader, 0):

            if num_batch % 1000 == 0:
                print("num_batch: {}".format(num_batch))

            # On windows requires int64, on ubuntu int32
            # input_data_tensor = Variable(torch.from_numpy(np.asarray(input_data, dtype=np.int64))).to(self.device)
            input_data_tensor = Variable(input_data).to(self.device)

            label_tensor = Variable(label).to(self.device)

            user_coordinates = input_data_tensor[:, 0]
            item_coordinates = input_data_tensor[:, 1]

            # FORWARD pass
            prediction = self.pyTorchModel(user_coordinates, item_coordinates)

            # Pass prediction and label removing last empty dimension of prediction
            loss = self.lossFunction(prediction.view(-1), label_tensor)

            # BACKWARD pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_expected_ratings(self, user_id):
        expected_ratings = self.model.predict(user_ids=user_id, item_ids=np.arange(session.INSTANCE.items_amount),
                                              user_features=self.ucm, item_features=self.icm)
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
