import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch.autograd import Variable
from torch.utils.data import DataLoader

from recommenders.pytorch_model import PyTorchModel, DatasetIterator
from recommenders.recommender import Recommender


class PyTorch(Recommender):
    name = 'pytorch'

    def __init__(self, session):
        super().__init__(session)
        self.n_factors = 32
        self.epochs = 10
        self.batch_size = 20
        self.learning_rate = 0.0005
        self.device = None
        self.loss_function = None
        self.optimizer = None
        self.pytorch_model = None
        self.train_data_loader = None
        self.user_factors = None
        self.item_factors = None

    def fit(self, training_urm):
        super().fit(self)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("PyTorch: Using CUDA")
        else:
            self.device = torch.device('cpu')
            print("PyTorch: Using CPU")

        self.pytorch_model = PyTorchModel(self.session.users_amount, self.session.items_amount,
                                          self.n_factors).to(self.device)

        self.loss_function = torch.nn.MSELoss(size_average=False)

        self.optimizer = torch.optim.Adagrad(self.pytorch_model.parameters(), lr=self.learning_rate)

        dataset_iterator = DatasetIterator(training_urm)

        self.train_data_loader = DataLoader(dataset=dataset_iterator,
                                            batch_size=self.batch_size,
                                            shuffle=True)

        for epoch in range(self.epochs):
            self.run_epoch(epoch)

        self.user_factors = self.pytorch_model.get_W()
        self.item_factors = self.pytorch_model.get_H()

    def run_epoch(self, num_epoch):

        print('Starting epoch: ' + str(num_epoch))

        for num_batch, (input_data, label) in enumerate(self.train_data_loader, 0):

            # On windows requires int64, on ubuntu int32
            # input_data_tensor = Variable(torch.from_numpy(np.asarray(input_data, dtype=np.int64))).to(self.device)
            input_data_tensor = Variable(input_data).to(self.device)

            label_tensor = Variable(label).to(self.device)

            user_coordinates = input_data_tensor[:, 0]
            item_coordinates = input_data_tensor[:, 1]

            # FORWARD pass
            prediction = self.pytorch_model(user_coordinates, item_coordinates)

            # Pass prediction and label removing last empty dimension of prediction
            loss = self.loss_function(prediction.view(-1), label_tensor)

            if num_batch % 100 == 0:
                print("Batch {} of {}, loss {:.4f}".format(num_batch, len(self.train_data_loader), loss.data.item()))

            # BACKWARD pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_ratings(self, training_urm, user_id):
        ratings = np.dot(self.user_factors[user_id], self.item_factors.T)
        ratings = normalize(ratings, axis=1, norm='max')
        ratings = ratings.ravel()
        return ratings
