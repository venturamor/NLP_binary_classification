import pickle
from second_model import Second_model
from first_model import First_Model
from dataset import EntityDataSet
from torch.utils.data import DataLoader
from trainer import Trainer
import trainer
from gensim import downloader
import gensim
import torch
import second_model
import os

class main_run_class:
    def __init__(self):
        self.first_trained_model = None
        self.second_trained_model = None
        self.dataset_test = None
        self.dataloader_test = None
        self.predictions_first_model = None
        self.predictions_second_model = None
        self.words = None

    def load_first_model(self):
        """
        load first model from pickle
        :return:
        """
        # Load data (deserialize)
        first_model_pickle_path = 'first_model.pickle'
        with open(first_model_pickle_path, 'rb') as handle:
            first_model = pickle.load(handle)
            self.first_trained_model = first_model

    def load_second_model(self):
        """

        :return: load second model from state dict
        """
        second_model_path = "second_model.pt"
        data_size = self.dataset_test.__getitem__(0).__len__()
        model = second_model.Second_model(inputSize=data_size, outputSize=2)
        model.load_state_dict(torch.load(second_model_path))
        self.second_trained_model = model

    def create_dataset_test(self):

        test_data_path = "data/test.untagged"
        glove_path = 'glove-twitter-100'
        embedding_size = int(glove_path.split('-')[-1])
        window_size = 1

        # print("loading embedding model " + glove_path)
        # glove_model = gensim.downloader.load(glove_path)
        # print(glove_path + " - model downloaded")

        print("loading gensim model")
        # download if there is no pickle
        gensim_model_path = 'gensim_model.pickle'
        if os.path.isfile(gensim_model_path):
            gensim_model_path = 'gensim_model.pickle'
            with open(gensim_model_path, 'rb') as handle:
                gensim_model = pickle.load(handle)
        else:
            gensim_model = gensim.downloader.load(glove_path)
            with open('gensim_model.pickle', 'wb') as handle:
                pickle.dump(gensim_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("gensim model downloaded")

        self.dataset_test = EntityDataSet(test_data_path,
                                          model=gensim_model,
                                          embedding_size=embedding_size,
                                          window_size=window_size,
                                          is_test=True)

    def run_first_model(self):
        # eval on dev
        x_test = list(self.dataset_test.dict_words2embedd.values())
        y_test = list(self.dataset_test.dict_words2tags.values())
        print('start testing first model')
        y_prob, y_pred = self.first_trained_model.test(x_test)
        print('done testing first model')
        self.predictions_first_model = y_pred

        # create test tagged

    def run_second_model(self):
        batch_size = 1024
        learning_rate = 0.00036

        # create data loader
        self.dataloader_test = DataLoader(self.dataset_test, batch_size=batch_size, shuffle=False)
        trainer = Trainer(model=self.second_trained_model)
        # save dict as self.second_model_dict
        self.predictions_second_model = trainer.test(self.dataloader_test)

    def save_test_tagged(self, run_name):

        self.words = [item for sublist in self.dataset_test.words_lists for item in sublist]
        # if run_name == 'first_model':
        #     f = open("comp_m1_313177412.tagged", "w", encoding="utf8")
        #     for word in words:
        #         embedding = self.dataset_test.dict_words2embedd[word.lower()]
        #         pred_of_word = self.predictions_second_model[idx_of_word]
        #         f.write(word + "\t" + pred_of_word + "\n")
        #
        #     # words = [item for sublist in self.dataset_test.words_lists for item in sublist]
        #
        #     for sentence in self.dataset_test.words_lists:
        #         for word in sentence:

                # TODO save test embeddings words in a list: self.word_embeddings

        f = open("comp_m2_313177412.tagged", "w", encoding="utf8")
        for idx, word in enumerate(self.words):
            prediction = self.predictions_second_model[idx]
            f.write(word + "    " + str(prediction) + "\n")

    def run(self):
        print("create test dataset")
        self.create_dataset_test()
        print("loading first model")
        # self.load_first_model()
        print("loading second model")
        self.load_second_model()
        print("run first model on test")
        # self.run_first_model()
        print("save test_tagged by first model")
        # self.save_test_tagged('first_model')
        print("run second model on test")
        self.run_second_model()
        print("save test_tagged by second model")
        self.save_test_tagged('second_model')


if __name__ == '__main__':
    main_run = main_run_class()
    main_run.run()


