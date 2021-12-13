import pickle
from second_model import Second_model
from first_model import First_Model
from dataset import EntityDataSet
import trainer
from gensim import downloader
import gensim
import torch


class main_run_class():

    def __init__(self):
        self.first_trained_model = None
        self.second_trained_model = None
        self.dataset_test = None

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

    def load_second_model(self, second_model_path):
        """

        :return: load second model from state dict
        """
        model = Second_model()
        model.load_state_dict(torch.load(second_model_path))
        self.second_trained_model = model

    def create_dataset_test(self):

        test_path = "data/test.untagged"
        GLOVE_PATH = 'glove-twitter-100'
        embedding_size = int(GLOVE_PATH.split('-')[-1])
        window_size = 2

        print("loading embedding model " + GLOVE_PATH)
        glove_model = gensim.downloader.load(GLOVE_PATH)
        print(GLOVE_PATH + " - model downloaded")

        self.dataset_test = EntityDataSet(test_path,
                                          model=glove_model,
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


        # create test tagged

    def run_second_model(self):
        trainer.test(self.dataset_test)

    def save_test_tagged(self):
        # TODO
        print('yello')

    def run(self):
        print("loading first model")
        self.load_first_model()
        print("loading second model")
        # self.load_second_model()
        print("create test dataset")
        self.create_dataset_test()
        print("run first model on test")
        self.run_first_model()
        print("save test_tagged by first model")
        self.save_test_tagged()
        print("run second model on test")
        self.run_second_model()
        print("save test_tagged by second model")
        self.save_test_tagged()



if __name__ == '__main__':

    main_run = main_run_class()
    main_run.run()


