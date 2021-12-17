from torch.utils.data import Dataset
import numpy as np


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
    def __init__(self, file_path, model, embedding_size, use_window=True, window_size=1, is_test=False):
        """
        :param file_path:
        :param wind_size: prev + curr word + next window
        """

        # open and read the file content
        # new embedding with window embedding neighbors
        padding_word_start = 'morVentura'
        padding_word_end = 'michaelToker'

        self.file_path = file_path
        self.is_test = is_test
        data = open(file_path, "r", encoding="utf8").read()
        data_lower = data.lower()

        # prepare the data
        self.words_lists, self.tags_lists, self.bin_tags_lists = self.prepare_data(data_lower)
        self.words_lists_orig, tags_lists_orig, bin_tags_lists_orig = self.prepare_data(data)

        list_updated = self.words_lists_orig
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
        # embeddings_lists_with_indicator =

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
        if self.is_test:
            tags = []
        else:
            tags = [item for sublist in self.bin_tags_lists for item in sublist]
        self.dict_words2embedd = {}
        self.dict_words2tags = {}
        self.dict_idx2tuple = {}   # the main dict - tuple of tag and embedding
        self.dict_words2tuple = {}
        # New dict for test
        self.dict_idx2embedd = {}
        # dicts for get item using idx of corpus (not unique)
        self.dict_idxCorpus_2embedd = {}
        self.dict_idxCorpus2tuple = {}

        self.define_dicts(words, tags, embeddings)
        print('done dataset config')


    def prepare_data(self, data):
        tagged_sentences = data.split('\n\n')[:-1]
        tagged_words_lists = [sentence.split('\n') for sentence in tagged_sentences]

        # If is_test is True: the data has only the data without tagging
        if self.is_test:
            words_lists = tagged_words_lists
            tags_lists = []
            bin_tags_lists = []
        else:
            words_lists = \
                [[tagged_word.split('\t')[0] for tagged_word in tagged_word_list] for tagged_word_list in tagged_words_lists]
            tags_lists = \
                [[(tagged_word.split('\t')[1]) for tagged_word in tagged_word_list] for tagged_word_list in tagged_words_lists]
            bin_tags_lists =\
                [[tag != 'o' for tag in tags_list] for tags_list in tags_lists]

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
            for idx, word in enumerate(sentence):
                ind_symbol = [ind for ind, x in enumerate([(symbol in word) for symbol in symbols]) if x]
                if ind_symbol:
                    word = symbols[ind_symbol[0]]
                try:
                    embedding = model[word.lower()]
                except KeyError:
                    if word == word_start:
                        # embedding = 0.1 * np.ones(embedding_size)
                        embedding = np.random.rand(embedding_size) * 0.1
                    elif word == word_end:
                        # embedding = 0.5 * np.ones(embedding_size)
                        embedding = np.random.rand(embedding_size) * 0.5
                    else:
                        # embedding = np.zeros(embedding_size)
                        embedding = np.random.rand(embedding_size)
                is_capital = int(word[0].isupper())
                # is_capital = 0
                embedding = np.append(embedding, is_capital)
                embedding = np.append(embedding, idx * 0.1)
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
            # for get item
            self.dict_idxCorpus_2embedd[idx] = embeddings[idx]
            if not self.is_test:
                self.dict_idxCorpus2tuple[idx] = (embeddings[idx], tags[idx])
            # for working with unique
            if word not in self.dict_words2embedd.keys():
                # assumption : for 2 identical words - same tag
                if self.is_test:
                    self.dict_idx2embedd[dict_index] = embeddings[idx]
                else:
                    self.dict_words2tags[word] = tags[idx]
                    self.dict_words2tuple[word] = (embeddings[idx], tags[idx])
                    self.dict_idx2tuple[dict_index] = (embeddings[idx], tags[idx])
                self.dict_words2embedd[word] = embeddings[idx]
                # dict_embedd2tags[embeddings[idx, :]] = tags[idx]
                dict_index += 1
            else:
                continue

    def __getitem__(self, item):
        '''

        :param item: for idx of word in our corpus
        :return: tuple of (embeddings, tag)
        '''
        if self.is_test:
            # return self.dict_idx2embedd[item]
            return self.dict_idxCorpus_2embedd[item]
        else:
            # return self.dict_idx2tuple[item]
            return self.dict_idxCorpus2tuple[item]

    def __len__(self):
        """

        :return:
        """
        if self.is_test:
            return self.dict_idxCorpus_2embedd.__len__()
        else:
            return self.dict_idxCorpus2tuple.__len__()

    def split(self):

        x = list(self.dict_idxCorpus_2embedd.values())
        y = [item[1] for item in self.dict_idxCorpus2tuple.values()]

        return x, y


# -------- help function
def data_imbalance_fix(x_train, y_train):
    """
    fix the imbalance between minority tags (True) and majority - duplicate to uniform disribution
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