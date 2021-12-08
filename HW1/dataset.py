import torch
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import gensim
from gensim import downloader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


WORD_2_VEC_PATH = 'word2vec-google-news-300'
GLOVE_PATH = 'glove-twitter-25'
embedding_size = 25

embedding_size = int(GLOVE_PATH.split('-')[-1])

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
        data = open(file_path, "r").read()
        data_lower = data.lower()

        # prepare the data
        tagged_sentences = data_lower.split('\n\n')[:-1]
        tagged_words_lists = [sentence.split('\n') for sentence in tagged_sentences]

        self.words_lists = \
            [[tagged_word.split('\t')[0] for tagged_word in tagged_word_list] for tagged_word_list in tagged_words_lists]
        self.tags_lists = \
            [[(tagged_word.split('\t')[1]) for tagged_word in tagged_word_list] for tagged_word_list in tagged_words_lists]
        self.bin_tags_lists =\
            [[tag != 'o' for tag in tags_list] for tags_list in self.tags_lists]

        # # padding  before tokenize the sentence
        # left_padding = [padding_word] * window_size_prev
        # right_padding = [padding_word] * window_size_next
        # self.words_lists_with_padding = [left_padding + words_list + right_padding for words_list in self.words_lists]


        # load pre-trained model

        # model = gensim.models.Word2Vec.load(GLOVE_PATH)
        # model.build_vocab(self.words_lists, update=True)
        # model.train(self.words_lists, total_examples=model.corpus_count, epochs=model.epochs)
        print("loading model")
        model = gensim.downloader.load(GLOVE_PATH)
        print("model downloaded")


        # train model on our corpus
        # vector_size = 100  # 50
        # model = Word2Vec(sentences=self.words_lists, vector_size=vector_size, window=5, min_count=1, workers=1, epochs=1)
        # model.save("word2vec.model") #  model.wv is the embedding

        # as if we have self.tokenized_words:

        # unique dict words to embedd
        words = [item for sublist in self.words_lists for item in sublist]
        # embeddings = model.wv[words]

        # if word is not in vocab - use zeros representation
        # if word contains  "@" / "http" / "#" - word will be embedded as those symbols
        embeddings = []
        symbols = ["@", "http", "#"]
        for i, word in enumerate(words):
            ind_symbol = [ind for ind, x in enumerate([(symbol in word) for symbol in symbols]) if x]
            if ind_symbol:
                word = symbols[ind_symbol[0]]
            try:
                embedding = model[word]
            except KeyError:
                embedding = np.zeros(embedding_size)
            embeddings.append(embedding)


        # new embedding with window embedding neighbors

        # dicts
        tags = [item for sublist in self.bin_tags_lists for item in sublist]
        self.dict_words2embedd = {}
        self.dict_words2tags = {}
        self.dict_idx2tuple = {}   # the main dict - tuple of tag and embedding
        self.dict_words2tuple = {}
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
        return self.dict_idx2tuple.__len__() - 1
