import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import f1_score


class Trainer:
    def __init__(self, model, optimizer, device=None):
        """
            Initialize the trainer.
            :param model: Instance of the model to train.
            :param optimizer: The optimizer to train with.
            :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device

        if self.device:
            self.model.to(self.device)

    def fit(self,
            dl_train: DataLoader,
            dl_dev: DataLoader,
            num_epochs):
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_dev: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param print_every: print every
        """
        self.model.train()
        for epoch in range(num_epochs):
            # Forward pass: Compute predicted y by passing
            # x to the model
            for batch_ndx, sample in enumerate(dl_train):
                self.optimizer.zero_grad()
                x_train, y_train = sample
                y_prob = self.model(x_train)

                # Compute and print loss
                # if self.loss_fn_string == "binary_cross_entropy":
                # TODO: let the model get the loss
                loss = torch.nn.functional.nll_loss(y_prob, y_train.long())
                # Zero gradients, perform a backward pass,
                # and update the weights.
                loss.backward()
                self.optimizer.step()

            f1 = self.eval(dl_dev)
            print('epoch {}, loss {}'.format(epoch, loss.item()))
            print('epoch {}, f1 {}'.format(epoch, f1))

    def eval(self, dl_dev: DataLoader):
        """
        Args:
            dl_dev:
        Returns: f1_score
        """
        f_score = 0
        for batch_ndx, sample in enumerate(dl_dev):
            self.model.eval()
            x_dev, y_dev = sample
            y_pred = self.model(x_dev)
            y_pred = y_pred[:, 0] < y_pred[:, 1]
            f1 = f1_score(y_dev, y_pred, average='binary', pos_label=True)
            f_score += f1
        return f_score / len(dl_dev)

    def test(self, dl_test: DataLoader):
        """
        Args:
            dl_dev:
        Returns: f1_score
        """
        predictions = []
        for batch_ndx, sample in enumerate(dl_test):
            self.model.eval()
            x_test = sample
            y_pred = self.model(x_test)
            y_pred = y_pred[:, 0] < y_pred[:, 1]
            # for i in range (y_pred)
        return
