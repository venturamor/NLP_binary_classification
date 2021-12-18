import dataset
import torch
import random
from first_model import First_Model
from second_model import Second_model
from trainer import Trainer
from torch.utils.data import DataLoader
import gensim
from gensim import downloader
import pickle
import os
from dataset import data_imbalance_fix


def run_first_model(dataset_train, dataset_dev, dataset_test, data_balance, pickle_path="first_model_ver2.pickle"):
    """
    :param pickle_path:
    :param data_balance:
    :param dataset_train:
    :param dataset_dev:
    :return:
    """
    # first model - train
    first_model = First_Model()
    x_train, y_train = dataset_train.split()

    # data imbalance fix
    if data_balance:
        new_x_train_cpy, new_y_train_cpy = data_imbalance_fix(x_train, y_train)
        # shuffle
        zipped_ = list(zip(new_x_train_cpy, new_y_train_cpy))
        random.shuffle(zipped_)
        new_x_train_cpy, new_y_train_cpy = zip(*zipped_)
    else:
        new_x_train_cpy, new_y_train_cpy = x_train, y_train

    # train please ('choo chooooo!')
    print('start training first model')
    best_clf_shuffle = first_model.train(new_x_train_cpy, new_y_train_cpy)
    print('done training first model')
    # eval on dev
    x_dev, y_dev = dataset_dev.split()

    print('start testing first model')
    y_prob, y_pred = first_model.test(x_dev)
    print('done testing first model')

    print('start evaluating first model')
    first_model.eval(x_dev, y_dev)
    first_model.model_performance(x_dev, y_dev, y_pred, y_prob)  # , best_clf_shuffle)
    print('done evaluating first model')
    print("First model done with f1: ")

    # Store data (serialize)+
    with open(pickle_path, 'wb') as handle:
        pickle.dump(first_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('done run first model')


def run_second_model(dataset_train, dataset_dev, dataset_test):
    """
    :param dataset_train:
    :param dataset_dev:
    :return:
    """

    torch.torch.manual_seed(3133)
    # first model - train
    first_model = First_Model()
    x_train, y_train = dataset_train.split()
    # data imbalance fix

    new_x_train_cpy, new_y_train_cpy = data_imbalance_fix(x_train, y_train)
    dataset_train = dataset.ListDataSet(new_x_train_cpy, new_y_train_cpy)

    # Hyperparameters
    batch_size = 1024
    num_epochs = 16
    learning_rate = 0.00036

    data_size = dataset_train.__getitem__(0)[0].__len__()

    dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dl_dev = DataLoader(dataset_dev, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    second_model = Second_model(inputSize=data_size, outputSize=2)
    print(second_model)
    optimizer = torch.optim.Adam(second_model.parameters(), lr=learning_rate)
    trainer = Trainer(model=second_model, optimizer=optimizer, device=None)

    trainer.fit(dl_train=dl_train, dl_dev=dl_dev, num_epochs=num_epochs)
    f1 = trainer.eval(dl_dev=dl_dev)
    print("Second model done with f1: ", f1)

    # Specify a path
    PATH = "second_model.pt"

    # Save
    torch.save(second_model.state_dict(), PATH)

    trainer.test(dl_test)


if __name__ == '__main__':

    glove_path = 'glove-twitter-25'  # 100
    # glove_path = 'glove-twitter-50'
    embedding_size = int(glove_path.split('-')[-1])

    # load dataset
    train_path = "data/train.tagged"
    dev_path = "data/dev.tagged"
    test_path = "data/test.untagged"

    print("loading gensim model")
    # download if there is no pickle
    gensim_model_path = 'gensim_model_25.pickle'  #  'gensim_model.pickle'
    if os.path.isfile(gensim_model_path):
        with open(gensim_model_path, 'rb') as handle:
            gensim_model = pickle.load(handle)
    else:
        gensim_model = gensim.downloader.load(glove_path)
        with open(gensim_model_path, 'wb') as handle:
            pickle.dump(gensim_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("gensim model downloaded")

    # Hyper parameter
    window_size = 1

    dataset_train = dataset.EntityDataSet(train_path, model=gensim_model, embedding_size=embedding_size, window_size=window_size)
    dataset_dev = dataset.EntityDataSet(dev_path, model=gensim_model, embedding_size=embedding_size, window_size=window_size)
    dataset_test = dataset.EntityDataSet(test_path, model=gensim_model, embedding_size=embedding_size,
                                         window_size=window_size, is_test=True)
    print('done creating datasets')

    run_first_model(dataset_train, dataset_dev, dataset_test, data_balance=False, pickle_path="first_model_glove25_with_balance.pickle")
    run_second_model(dataset_train, dataset_dev, dataset_test)
