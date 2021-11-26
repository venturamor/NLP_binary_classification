import numpy
import dataset
import torch
from first_model import First_Model
from second_model import Second_model
from trainer import Trainer
from torch.utils.data import DataLoader


def run_first_model(dataset_train, dataset_dev):
    # first model - train
    first_model = First_Model()
    x_train = list(dataset_train.dict_words2embedd.values())
    y_train = list(dataset_train.dict_words2tags.values())
    best_clf = first_model.train(x_train, y_train)
    # eval on dev
    x_dev = list(dataset_dev.dict_words2embedd.values())
    y_dev = list(dataset_dev.dict_words2tags.values())
    y_prob, y_pred = first_model.test(x_dev)
    first_model.model_performance(x_dev, y_dev, y_pred, y_prob, best_clf)

    print("done")


def run_second_model(dataset_train, dataset_dev):
    data_size = dataset_train.__getitem__(0)[0].__len__()
    batch_size = 100
    dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dl_dev = DataLoader(dataset_dev, batch_size=batch_size, shuffle=True)
    num_epochs = 30

    second_model = Second_model(inputSize=data_size, outputSize=1)
    loss_fn_string = "binary_cross_entropy"
    optimizer = torch.optim.SGD(second_model.parameters(), lr=0.01)
    trainer = Trainer(model=second_model, loss_fn_string=loss_fn_string, optimizer=optimizer, device=None)

    trainer.fit(dl_train=dl_train, dl_dev=dl_dev, num_epochs=num_epochs)


if __name__ == '__main__':
    # load dataset
    train_path = "data/train.tagged"
    dev_path = "data/dev.tagged"
    dataset_train = dataset.EntityDataSet(train_path)
    dataset_dev = dataset.EntityDataSet(dev_path)

    run_first_model(dataset_train, dataset_dev)
    # run_second_model(dataset_train, dataset_dev)
