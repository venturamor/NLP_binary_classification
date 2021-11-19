import numpy
import dataset

if __name__ == '__main__':
    # import gensim.downloader as api
    # model = api.load("glove-twitter-25")  # load glove vectors
    # model.most_similar("cat")  # show words that similar to word 'cat'
    # exit(0)
    file_path = "data/train_fixed.tagged"
    dataset = dataset.EntityDataSet(file_path)
    print("done")
