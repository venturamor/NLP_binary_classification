import torch
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import gensim.downloader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class EntityDataSet(Dataset):
    def __init__(self, file_path, prev_wind_size = 1, next_wind_size = 1, tokenizer=None):
        """
        :param file_path:
        :param tokenizer:
        :param wind_size: prev + curr word + next window
        """
        self.wind_size.prev = prev_wind_size
        self.wind_size.next = next_wind_size

        self.file_path = file_path
        data = open(file_path, "r").read()
        tagged_sentences = data.split('\n\n')
        tagged_words_lists = [sentence.split('\n') for sentence in tagged_sentences]
        self.words_lists = [[tagged_word.split('\t')[0] for tagged_word in tagged_word_list] for tagged_word_list in tagged_words_lists]
        self.tags_lists = [[tagged_word.split('\t')[1] for tagged_word in tagged_word_list] for tagged_word_list in tagged_words_lists]
        self.bin_tags_lists = [[tag != 'O' for tag in tags_list] for tags_list in self.tags_lists]  # tag != 'O' for tag in tags_list

        # padding  before we tokenize the sentence
        padding_word = '*****'


        # if tokenizer is None:   # default word2vec
            # self.tokenizer = gensim.downloader.load('word2vec-google-news-300')
            # print(self.tokenizer.most_similar('shavasana'))
            # self.tokenized_words = self.tokenizer.wv(self.words_lists)

            # model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
            # model.save("word2vec.model")

        # else:
        #     self.tokenizer = tokenizer
        #     self.tokenized_sen = self.tokenizer(self.words_lists)


        self.vocabulary_size = len(self.tokenizer.vocabulary_)

        # as if we have self.tokenized_words:


        {idx : [tokenized_word for tokenized_word in self.tokenized_words }

    def __getitem__(self, item):
    """
    param: item index for word
    """
        # return self.___[item]





if __name__ == '__main__':

    file_path = "data/train_fixed.tagged"
    dataset = EntityDataSet(file_path)
    print("done")



#
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
