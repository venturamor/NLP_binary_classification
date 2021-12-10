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

    # def test_batch(self, batch):
        # x_train, y_train = batch
        # pred = self.model(x_train)
        # binary_pred = torch.max(pred)
        # num_correct = (binary_pred == y_train).sum().item()
        # acc = (num_correct * 100) / y_train.size()[0]
        # return acc

    # def test_epoch(self, dl_dev: DataLoader):
    #     """
    #     Evaluate model once over a test set (single epoch).
    #     :return: An EpochResult for the epoch.
    #     """
    #     with torch.no_grad():
    #         f_score = 0
    #         self.model.eval()
    #         for batch in dl_dev:
    #             for batch_ndx, sample in enumerate(batch):
    #                 self.model.eval()
    #                 x_dev, y_dev = sample
    #                 y_pred = self.model(x_dev)
    #                 y_pred = y_pred[:, 0] < y_pred[:, 1]
    #                 f1 = f1_score(y_dev, y_pred, average='binary', pos_label=True)
    #                 f_score += f1
    #         return f_score / (len(batch) * batch_ndx)

    def fit(self,
            dl_train: DataLoader,
            dl_dev: DataLoader,
            num_epochs,
            print_every=1):
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_dev: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param print_every: print every
        """
        for epoch in range(num_epochs):
            # Forward pass: Compute predicted y by passing
            # x to the model
            for batch_ndx, sample in enumerate(dl_train):
                self.optimizer.zero_grad()
                x_train, y_train = sample
                y_prob = self.model(x_train)
                # print("x_train: ", x_train, " y_dev: ", y_pred, " y_pred: ", y_pred)
                # Compute and print loss
                # if self.loss_fn_string == "binary_cross_entropy":
                # TODO: let the model get the loss
                # loss = torch.nn.functional.nll_loss(y_prob, y_train.long())  # Michael
                loss = torch.nn.functional.binary_cross_entropy(y_prob, y_train.float())
                # Zero gradients, perform a backward pass,
                # and update the weights.
                loss.backward()
                self.optimizer.step()

            acc = self.eval(dl_dev)
            print('epoch {}, loss {}'.format(epoch, loss.item()))
            print('epoch {}, f1 {}'.format(epoch, acc))

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
            y_pred = torch.argmax(y_pred, dim=1)
            y_dev = torch.argmax(y_dev, dim=1)
            # y_pred = y_pred[:, 0] < y_pred[:, 1]
            f1 = f1_score(y_dev.numpy(), y_pred.numpy(), average='binary', pos_label=True)
            f_score += f1
        return f_score / len(dl_dev)
