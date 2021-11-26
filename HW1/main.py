import numpy
import dataset
import first_model

if __name__ == '__main__':

    # dataset
    file_path = "data/train.tagged"
    dataset_train = dataset.EntityDataSet(file_path)

    # first model
    first_model = first_model.First_Model()
    x_train = list(dataset_train.dict_words2embedd.values())
    y_train = list(dataset_train.dict_words2tags.values())
    first_model.train(x_train, y_train)

    eval1 = first_model.eval(x_train, y_train)
    print("done")
