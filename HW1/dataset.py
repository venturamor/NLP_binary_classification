import torch
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import gensim
from gensim import downloader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#
WORD_2_VEC_PATH = 'word2vec-google-news-300'
GLOVE_PATH = 'gensim-model-glove-twitter-25'


class EntityDataSet(Dataset):
    def __init__(self, file_path, window_size_prev=1, window_size_next=1, tokenizer=None):
        """
        :param file_path:
        :param tokenizer:
        :param wind_size: prev + curr word + next window
        """
        # padding is essential for representing the neighbors of the words in the edges
        # padding_word = '*'

        # open and read the file content
        self.file_path = file_path
        data = open(file_path, "r").read().lower()

        # prepare the data
        tagged_sentences = data.split('\n\n')[:-1]
        tagged_words_lists = [sentence.split('\n') for sentence in tagged_sentences]

        self.words_lists = \
            [[tagged_word.split('\t')[0] for tagged_word in tagged_word_list] for tagged_word_list in tagged_words_lists]
        self.tags_lists = \
            [[(tagged_word.split('\t')[1]) for tagged_word in tagged_word_list] for tagged_word_list in tagged_words_lists]
        self.bin_tags_lists =\
            [[tag != 'O' for tag in tags_list] for tags_list in self.tags_lists]

        # # padding  before tokenize the sentence
        # left_padding = [padding_word] * window_size_prev
        # right_padding = [padding_word] * window_size_next
        # self.words_lists_with_padding = [left_padding + words_list + right_padding for words_list in self.words_lists]

        # create a list of tokenized sentences

        model = gensim.models.Word2Vec.load(GLOVE_PATH)

        model.build_vocab(self.words_lists, update=True)
        model.train(self.words_lists, total_examples=model.corpus_count, epochs=model.epochs)

            # Create labeled data from the tokenizer


        self.vocabulary_size = len(self.tokenizer.vocabulary_)
        self.items = []
        # as if we have self.tokenized_words:
        # {idx : [tokenized_word for tokenized_word in self.tokenized_words }

    def __getitem__(self, item):
        """
        param: item index for word
        """
        return self.items[item]

# class SpamDataSet(Dataset):
#
#     def __init__(self, file_path, tokenizer=None):
#         self.file_path = file_path
#         data = pd.read_csv(self.file_path)
#         self.sentences = data['reviewText'].tolist()
#         self.labels = data['label'].tolist()
#         self.tags_to_idx = {tag: idx for idx, tag in enumerate(sorted(list(set(self.labels))))}
#         self.idx_to_tag = {idx: tag for tag, idx in self.tags_to_idx.items()}
#         if tokenizer is None:
#             self.tokenizer = TfidfVectorizer(lowercase=True, stop_words=None)
#             self.tokenized_sen = self.tokenizer.fit_transform(self.sentences)
#         else:
#             self.tokenizer = tokenizer
#             self.tokenized_sen = self.tokenizer.transform(self.sentences)
#         self.vocabulary_size = len(self.tokenizer.vocabulary_)
#
#     def __getitem__(self, item):
#         cur_sen = self.tokenized_sen[item]
#         cur_sen = torch.FloatTensor(cur_sen.toarray()).squeeze()
#         label = self.labels[item]
#         label = self.tags_to_idx[label]
#         # label = torch.Tensor(label)
#         data = {"input_ids": cur_sen, "labels": label}
#         return data
#
#     def __len__(self):
#         return len(self.sentences)
