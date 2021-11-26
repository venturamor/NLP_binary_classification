import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F


class Trainer():
    def __init__(self, model, loss_fn_string, optimizer, device=None):
        """
            Initialize the trainer.
            :param model: Instance of the model to train.
            :param loss_fn: The loss function to evaluate with.
            :param optimizer: The optimizer to train with.
            :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn_string = loss_fn_string
        self.optimizer = optimizer
        self.device = device

        if self.device:
            self.model.to(self.device)

    def test_batch(self, batch):
        x_train, y_train = batch
        pred = self.model(x_train)
        binary_pred = [pred > 0.5]
        num_correct = (binary_pred[0].reshape(-1) == y_train).sum().item()
        acc = (num_correct * 100) / y_train.size()[0]
        return acc

    def test_epoch(self, dl_dev: DataLoader):
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :return: An EpochResult for the epoch.
        """
        with torch.no_grad():
            acc_list = []
            self.model.eval()
            for batch in dl_dev:
                acc = self.test_batch(batch)
                acc_list.append(acc)
            return (np.asarray(acc_list)).mean()

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
                x_train, y_train = sample
                y_pred = self.model(x_train)
                # Compute and print loss
                if self.loss_fn_string == "binary_cross_entropy":  # TODO fix
                    loss = F.binary_cross_entropy(y_pred.reshape(-1), y_train.float())
                    # Zero gradients, perform a backward pass,
                    # and update the weights.
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            acc = self.test_epoch(dl_dev)
            print('epoch {}, loss {}'.format(epoch, loss.item()))
            print('epoch {}, acc {}'.format(epoch, acc))
