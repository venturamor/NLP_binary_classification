import numpy
import dataset
import torch
import pandas as pd
import numpy as np
import random
from first_model import First_Model
from second_model import Second_model
from trainer import Trainer
from torch.utils.data import DataLoader
import gensim
from gensim import downloader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

WORD_2_VEC_PATH = 'word2vec-google-news-300'
GLOVE_PATH = 'glove-twitter-25'
embedding_size = int(GLOVE_PATH.split('-')[-1])


def data_imbalance_fix(x_train, y_train):
    """

    :param x_train:
    :param y_train:
    :return:
    """
    # try to deal with data imbalance - resample minority (True)
    # TODO: move as function to dataset.py
    true_indices = [index for index, element in enumerate(y_train) if element]
    false_indices = [index for index, element in enumerate(y_train) if not element]
    x_train_true_embedding = [x_train[ind] for ind in true_indices]
    x_train_false_embedding = [x_train[ind] for ind in false_indices]
    num_to_duplicate = np.int8(np.floor(len(x_train_false_embedding) / len(x_train_true_embedding)))
    new_x_train = x_train_false_embedding
    new_y_train = [False] * len(new_x_train)
    y_true = [True] * len(x_train_true_embedding)
    for dup in range(num_to_duplicate):
        new_x_train.extend(x_train_true_embedding)
        new_y_train.extend(y_true)
    # shuffle
    new_x_train_cpy = new_x_train.copy()
    new_y_train_cpy = new_y_train.copy()

    return new_x_train_cpy, new_y_train_cpy


def run_first_model(dataset_train, dataset_dev):
    """

    :param dataset_train:
    :param dataset_dev:
    :return:
    """
    # first model - train
    first_model = First_Model()
    x_train = list(dataset_train.dict_words2embedd.values())
    y_train = list(dataset_train.dict_words2tags.values())

    # data imbalance fix

    new_x_train_cpy, new_y_train_cpy = data_imbalance_fix(x_train, y_train)

    # shuffle
    zipped_ = list(zip(new_x_train_cpy, new_y_train_cpy))
    random.shuffle(zipped_)
    new_x_train_cpy, new_y_train_cpy = zip(*zipped_)

    # train please ('choo chooooo!')
    print('start training first model')
    best_clf_shuffle = first_model.train(new_x_train_cpy, new_y_train_cpy)
    print('done training first model')
    # best_clf = first_model.train(x_train, y_train)
    # eval on dev
    x_dev = list(dataset_dev.dict_words2embedd.values())
    y_dev = list(dataset_dev.dict_words2tags.values())
    print('start testing first model')
    y_prob, y_pred = first_model.test(x_dev)
    print('done testing first model')

    print('start evaluating first model')
    first_model.eval(x_dev, y_dev)
    first_model.model_performance(x_dev, y_dev, y_pred, y_prob, best_clf_shuffle)
    print('done evaluating first model')
    print("First model done with f1: ")


def run_second_model(dataset_train, dataset_dev):
    """
    :param dataset_train:
    :param dataset_dev:
    :return:
    """

    # first model - train
    first_model = First_Model()
    x_train = list(dataset_train.dict_words2embedd.values())
    y_train = list(dataset_train.dict_words2tags.values())

    # data imbalance fix

    new_x_train_cpy, new_y_train_cpy = data_imbalance_fix(x_train, y_train)
    dataset_train = dataset.ListDataSet(new_x_train_cpy, new_y_train_cpy)



    data_size = dataset_train.__getitem__(0)[0].__len__()
    batch_size = 300
    dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dl_dev = DataLoader(dataset_dev, batch_size=batch_size, shuffle=True)
    num_epochs = 10

    second_model = Second_model(inputSize=data_size, outputSize=2)
    print(second_model)
    optimizer = torch.optim.Adam(second_model.parameters(), lr=0.1)
    trainer = Trainer(model=second_model, optimizer=optimizer, device=None)

    trainer.fit(dl_train=dl_train, dl_dev=dl_dev, num_epochs=num_epochs)
    f1 = trainer.eval(dl_dev=dl_dev)
    print("Second model done with f1: ", f1)


if __name__ == '__main__':
    # load dataset
    train_path = "data/train.tagged"
    dev_path = "data/dev.tagged"

    print("loading model")
    model = gensim.downloader.load(GLOVE_PATH)
    print("model downloaded")

    dataset_train = dataset.EntityDataSet(train_path, model=model)
    dataset_dev = dataset.EntityDataSet(dev_path, model=model)
    print('done creating datasets')

    # run_first_model(dataset_train, dataset_dev)
    run_second_model(dataset_train, dataset_dev)
