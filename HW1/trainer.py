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
        binary_pred = F.softmax(pred)
        num_correct = (binary_pred.reshape(-1) == y_train).sum().item()
        acc = (num_correct * 100) / y_train.size()[0]
        return acc

    def test_epoch(self, dl_dev: DataLoader):
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :return: An EpochResult for the epoch.
        """
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
                    loss = F.binary_cross_entropy((torch.sigmoid(y_pred)).reshape(-1), y_train.float())
                    # Zero gradients, perform a backward pass,
                    # and update the weights.
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            acc = self.test_epoch(dl_dev)
            print('epoch {}, loss {}'.format(epoch, loss.item()))
            print('epoch {}, acc {}'.format(epoch, acc))

    #
    #         actual_num_epochs = 0
    #         train_loss, train_acc, test_loss, test_acc = [], [], [], []
    #
    #         best_acc = None
    #         epochs_without_improvement = 0
    #
    #         for epoch in range(num_epochs):
    #             if epoch % print_every == 0 or epoch == num_epochs - 1:
    #                 self._print(f"--- EPOCH {epoch + 1}/{num_epochs} ---")
    #
    #             #   Train & evaluate for one epoch
    #             #  - Use the train/test_epoch methods.
    #             #  - Save losses and accuracies in the lists above.
    #             #  - Implement early stopping. This is a very useful and
    #             #    simple regularization technique that is highly recommended.
    #             #  - Optional: Implement checkpoints. You can use torch.save() to
    #             #    save the model to the file specified by the checkpoints
    #             #    argument.
    #             # ====== YOUR CODE: ======
    #
    #             # train epoch:
    #             EpochResultTrain = self.train_epoch(dl_train)
    #             train_loss.append(torch.mean(torch.FloatTensor(EpochResultTrain.losses)))
    #             train_acc.append(EpochResultTrain.accuracy)
    #
    #             # test:
    #
    #             EpochResultTest = self.test_epoch(dl_test, **kw)
    #             test_loss.append(torch.mean(torch.FloatTensor(EpochResultTest.losses)))
    #             test_acc.append(EpochResultTest.accuracy)
    #
    #             if checkpoints and (test_acc > best_acc):
    #                 best_acc = test_acc
    #                 state = {'net': model.state_dict(), 'epoch': epoch}
    #                 torch.save(state, checkpoints)
    #
    #             if epoch > 0 and test_loss[epoch - 1] < test_loss[epoch]:
    #                 epochs_without_improvement += 1
    #
    #             else:
    #                 epochs_without_improvement = 0
    #
    #             if early_stopping != None and epochs_without_improvement >= early_stopping:
    #                 break
    #
    #         #             raise NotImplementedError()
    #         # ========================
    #
    #         return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)
    #
    # def fit(self, x_train, y_train):
    #     for epoch in range(500):
    #         # Forward pass: Compute predicted y by passing
    #         # x to the model
    #         pred_y = our_model(x_data)
    #
    #         # Compute and print loss
    #         loss = criterion(pred_y, y_data)
    #
    #         # Zero gradients, perform a backward pass,
    #         # and update the weights.
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         print('epoch {}, loss {}'.format(epoch, loss.item()))
    #
    #     new_var = Variable(torch.Tensor([[4.0]]))
    #     pred_y = our_model(new_var)
    #     print("predict (after training)", 4, our_model(new_var).item())

