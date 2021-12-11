import torch
from torch.utils.data import Dataset
from gensim.models import Word2Vec

import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
# Updated

class ListDataSet(Dataset):
    def __init__(self, data_list, label_list):
        self.data = data_list
        self.label = label_list

    def __getitem__(self, item):
        '''

        :param item: for idx of word in our corpus
        :return: tuple of (embeddings, tag)
        '''
        return self.data[item], self.label[item]

    def __len__(self):
        """

        :return:
        """
        return self.data.__len__()


class EntityDataSet(Dataset):
    def __init__(self, file_path, model, embedding_size, use_window=True, window_size=1):
        """
        :param file_path:
        :param wind_size: prev + curr word + next window
        """

        # open and read the file content
        # new embedding with window embedding neighbors
        padding_word_start = 'MorVentura'
        padding_word_end = 'MichaelToker'

        self.file_path = file_path
        data = open(file_path, "r").read()
        data_lower = data.lower()


        # prepare data with Big letters

        # prepare the data
        self.words_lists, self.tags_lists, self.bin_tags_lists = self.prepare_data(data_lower)
        words_lists_orig, tags_lists_orig, bin_tags_lists_orig = self.prepare_data(data)

        list_updated = self.words_lists
        # unique dict words to embedd
        words = [item for sublist in list_updated for item in sublist]

        if use_window:
            # padding  before tokenize the sentence
            left_padding = [padding_word_start] * window_size
            right_padding = [padding_word_end] * window_size
            self.words_lists_padd = [left_padding + words_list + right_padding for words_list in self.words_lists]
            list_updated = self.words_lists_padd
        # load pre-trained model

        # list of lists of embeddings
        embeddings_lists = self.get_embeddings(model, list_updated, padding_word_start, padding_word_end, embedding_size)
        if use_window:
            new_embeddings_lists = []
            for sentence in embeddings_lists:
                new_embedding_sentence = []
                for idx, emb in enumerate(sentence[1:-1]):
                    embed1 = np.concatenate([sentence[idx - 1], emb, sentence[idx + 1]])
                    new_embedding_sentence.append(embed1)
                new_embeddings_lists.append(new_embedding_sentence)
            embeddings_lists = new_embeddings_lists

        embeddings = [item for sublist in embeddings_lists for item in sublist]

        # dicts
        tags = [item for sublist in self.bin_tags_lists for item in sublist]
        self.dict_words2embedd = {}
        self.dict_words2tags = {}
        self.dict_idx2tuple = {}   # the main dict - tuple of tag and embedding
        self.dict_words2tuple = {}
        self.define_dicts(words, tags, embeddings)

    def prepare_data(self, data):
        tagged_sentences = data.split('\n\n')[:-1]
        tagged_words_lists = [sentence.split('\n') for sentence in tagged_sentences]

        words_lists = \
            [[tagged_word.split('\t')[0] for tagged_word in tagged_word_list] for tagged_word_list in tagged_words_lists]
        tags_lists = \
            [[(tagged_word.split('\t')[1]) for tagged_word in tagged_word_list] for tagged_word_list in tagged_words_lists]
        bin_tags_lists =\
            [[tag != 'o' for tag in tags_list] for tags_list in self.tags_lists]

        return words_lists, tags_lists, bin_tags_lists

    def get_embeddings(self, model, words_lists, word_start, word_end, embedding_size):
        """
        :param model:
        :param words_lists:
        :param word_start:
        :param word_end:
        :return:
        """

        symbols = ["@", "http", "#"]
        embeddings = []
        for sentence in words_lists:
            embedd_sentence = []
            for word in sentence:
                ind_symbol = [ind for ind, x in enumerate([(symbol in word) for symbol in symbols]) if x]
                if ind_symbol:
                    word = symbols[ind_symbol[0]]
                try:
                    embedding = model[word]
                except KeyError:
                    if word == word_start:
                        embedding = 0.1 * np.zeros(embedding_size)
                    elif word == word_end:
                        embedding = 0.9 * np.ones(embedding_size)
                    else:
                        embedding = np.zeros(embedding_size)
                embedd_sentence.append(embedding)
            embeddings.append(embedd_sentence)
        return embeddings

    def define_dicts(self, words, tags, embeddings):
        """

        :param words:
        :param tags:
        :param embeddings:
        :return:
        """
        dict_index = 0
        for idx, word in enumerate(words):
            if word not in self.dict_words2embedd.keys():
                # assumption : for 2 identical words - same tag
                self.dict_words2tags[word] = tags[idx]
                self.dict_words2embedd[word] = embeddings[idx]
                # dict_embedd2tags[embeddings[idx, :]] = tags[idx]
                self.dict_words2tuple[word] = (embeddings[idx], tags[idx])
                self.dict_idx2tuple[dict_index] = (embeddings[idx], tags[idx])
                dict_index += 1
            else:
                continue

    def __getitem__(self, item):
        '''

        :param item: for idx of word in our corpus
        :return: tuple of (embeddings, tag)
        '''
        return self.dict_idx2tuple[item]

    def __len__(self):
        """

        :return:
        """
        return self.dict_idx2tuple.__len__() - 1
