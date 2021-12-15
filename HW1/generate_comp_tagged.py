import pickle
from dataset import EntityDataSet
from torch.utils.data import DataLoader
from trainer import Trainer
from gensim import downloader
import gensim
import torch
import second_model
import os

class main_run_class:
    def __init__(self):
        """
        main class for testing
        """
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
        """
        # Load data (deserialize)
        # first_model_pickle_path = 'first_model.pickle'
        first_model_pickle_path = 'first_model_ver2.pickle'  # glove 50 no balance
        with open(first_model_pickle_path, 'rb') as handle:
            first_model = pickle.load(handle)
            self.first_trained_model = first_model

    def load_second_model(self):
        """
        load second model from state dict
        """
        second_model_path = "second_model.pt"
        data_size = self.dataset_test.__getitem__(0).__len__()
        model = second_model.Second_model(inputSize=data_size, outputSize=2)
        model.load_state_dict(torch.load(second_model_path))
        self.second_trained_model = model

    def create_dataset_test(self, run_name):
        """
        creating dataset of test ta
        :param run_name: model name to run
        :return:
        """

        model_to_glove = {'first_model': ['glove-twitter-50', 'gensim_model_50.pickle'],
                          'second_model': ['glove-twitter-100', 'gensim_model.pickle'] }

        test_data_path = "data/test.untagged"
        glove_path = model_to_glove[run_name]
        embedding_size = int(glove_path.split('-')[-1])
        window_size = 1

        print("loading gensim model")
        # download if there is no pickle
        gensim_model_path = model_to_glove[run_name][1]

        if os.path.isfile(gensim_model_path):
            with open(gensim_model_path, 'rb') as handle:
                gensim_model = pickle.load(handle)
        else:
            gensim_model = gensim.downloader.load(glove_path)
            with open(gensim_model, 'wb') as handle:
                pickle.dump(gensim_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("gensim model downloaded")

        self.dataset_test = EntityDataSet(test_data_path,
                                          model=gensim_model,
                                          embedding_size=embedding_size,
                                          window_size=window_size,
                                          is_test=True)

    def run_first_model(self):
        """
        run first trained model on test dataset
        """
        # eval on dev
        x_test, y_test = self.dataset_test.split()
        print('start testing first model')
        y_prob, y_pred = self.first_trained_model.test(x_test)
        print('done testing first model')
        self.predictions_first_model = y_pred

        # create test tagged

    def run_second_model(self):
        batch_size = 1024
        learning_rate = 0.00036

        """
        run second trained model on test dataset
        """
        batch_size = 128
        learning_rate = 0.0001
        # create data loader
        self.dataloader_test = DataLoader(self.dataset_test, batch_size=batch_size, shuffle=False)
        trainer = Trainer(model=self.second_trained_model)
        # save dict as self.second_model_dict
        self.predictions_second_model = trainer.test(self.dataloader_test)

    def save_test_tagged(self, run_name):
        """
        saving the model predictions to test.tagged files
        :param run_name: model name
        :return: comp_mX_313177412.tagged files (x [1,2,3])
        """
        test_tagged__names = {'first_model': ["comp_m1_313177412.tagged", self.predictions_first_model],
                              'second_model': ["comp_m2_313177412.tagged", self.predictions_second_model],
                              'competition_model': "comp_m3_313177412.tagged"}

        f = open(test_tagged__names[run_name][0], "w", encoding="utf8")
        for sentence in self.dataset_test.words_lists_orig:
            for idx, word in enumerate(sentence):
                prediction = test_tagged__names[run_name][1][idx]
                # prediction = self.predictions_first_model[idx]
                if not prediction:
                    prediction = 'O'
                f.write(word + '\t' + str(prediction) + "\n")
            f.write("\n")

    def run(self):
        """
        running the flow of creating dataset->
        loading pre trained model->
        run model->
        save test tagged file
        """
        run_name_model = ['first_model', 'second_model', 'competitive_model']

        print("create test dataset")
        self.create_dataset_test(run_name_model[0])
        print("loading first model")
        self.load_first_model()
        print("create test dataset")
        self.create_dataset_test(run_name_model[1])
        print("loading second model")
        self.load_second_model()
        print("run first model on test")
        self.run_first_model()
        print("save test_tagged by first model")
        self.save_test_tagged(run_name_model[0])
        print("run second model on test")
        self.run_second_model()
        print("save test_tagged by second model")
        self.save_test_tagged(run_name_model[1])

        # TODO: competitive_model


if __name__ == '__main__':
    main_run = main_run_class()
    main_run.run()


