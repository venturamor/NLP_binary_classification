import numpy
import dataset
import torch
from first_model import First_Model
from second_model import Second_model
from trainer import Trainer
from torch.utils.data import DataLoader


def run_first_model(dataset_train):
    # first model - Classic
    first_model = First_Model()
    x_train = list(dataset_train.dict_words2embedd.values())
    y_train = list(dataset_train.dict_words2tags.values())
    first_model.train(x_train, y_train)

    eval1 = first_model.eval(x_train, y_train)
    print("eval1: ", eval1)
    print("done")


def run_second_model(dataset_train, dataset_dev):
    data_size = dataset_train.__getitem__(0)[0].__len__()
    batch_size = 300
    dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dl_dev = DataLoader(dataset_dev, batch_size=batch_size, shuffle=True)
    num_epochs = 100

    second_model = Second_model(inputSize=data_size, outputSize=1)
    loss_fn_string = "binary_cross_entropy"
    optimizer = torch.optim.Adam(second_model.parameters(), lr=0.1)
    trainer = Trainer(model=second_model, loss_fn_string=loss_fn_string, optimizer=optimizer, device=None)

    trainer.fit(dl_train=dl_train, dl_dev=dl_dev, num_epochs=num_epochs)


if __name__ == '__main__':
    # load dataset
    train_path = "data/train.tagged"
    dev_path = "data/dev.tagged"
    dataset_train = dataset.EntityDataSet(train_path)
    dataset_dev = dataset.EntityDataSet(dev_path)

    # run_first_model(dataset_train)
    run_second_model(dataset_train, dataset_dev)
