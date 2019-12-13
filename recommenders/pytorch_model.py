"""
Created on 05/07/18
@author: Maurizio Ferrari Dacrema
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class PyTorchModel(torch.nn.Module):

    def __init__(self, users_amount, items_amount, factors_amount):
        super(PyTorchModel, self).__init__()

        self.users_amount = users_amount
        self.items_amount = items_amount
        self.factors_amount = factors_amount

        self.user_factors = torch.nn.Embedding(num_embeddings=self.users_amount, embedding_dim=self.factors_amount)
        self.item_factors = torch.nn.Embedding(num_embeddings=self.items_amount, embedding_dim=self.factors_amount)

        self.layer_1 = torch.nn.Linear(in_features=self.factors_amount, out_features=1)

        self.activation_function = torch.nn.ReLU()

    def forward(self, user_coordinates, item_coordinates):
        current_user_factors = self.user_factors(user_coordinates)
        current_item_factors = self.item_factors(item_coordinates)

        prediction = torch.mul(current_user_factors, current_item_factors)

        prediction = self.layer_1(prediction)
        prediction = self.activation_function(prediction)

        return prediction

    def get_W(self):
        return self.user_factors.weight.detach().cpu().numpy()

    def get_H(self):
        return self.item_factors.weight.detach().cpu().numpy()


class DatasetIterator(Dataset):

    def __init__(self, urm):
        urm = urm.tocoo()

        self.n_data_points = urm.nnz

        self.user_item_coordinates = np.empty((self.n_data_points, 2))

        self.user_item_coordinates[:, 0] = urm.row.copy()
        self.user_item_coordinates[:, 1] = urm.col.copy()
        self.rating = urm.data.copy().astype(np.float)

        self.user_item_coordinates = torch.Tensor(self.user_item_coordinates).type(torch.LongTensor)
        self.rating = torch.Tensor(self.rating)

    def __getitem__(self, index):
        return self.user_item_coordinates[index, :], self.rating[index]

    def __len__(self):
        return self.n_data_points
