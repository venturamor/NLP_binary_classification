import pickle
from second_model import Second_model
from first_model import First_Model
from dataset import EntityDataSet
import trainer
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
        with open('filename.pickle', 'rb') as handle:
            unserialized_data = pickle.load(handle)
            self.first_trained_model = unserialized_data

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
        dl_test = DataLoader(self.dataset_test, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(second_model.parameters(), lr=learning_rate)
        trainer = Trainer(model=second_model, optimizer=optimizer, device=None)

        dict = trainer.test(dl_test)
        # TODO save dict as self.second_model_dict
        # self.save_test_tagged(dict)

    def save_test_tagged(self):
        # TODO save test embeddings words in a list: self.word_embeddings
        f = open("test_output_model_1.txt", "r")
        for word in self.words:
            prediction = self.second_model_dict[self.word_embeddings]
            f.write(word + "    " + prediction + "\n")

    def run(self):
        print("loading first model")
        self.load_first_model()
        print("loading second model")
        second_model_path = "second_model.pt"
        self.load_second_model(second_model_path)
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


