import numpy as np
from sklearn.metrics import pairwise
import pickle
import pandas as pd
from sklearn import preprocessing


class ELM:
    def __init__(self, c=1, weighted=False, kernel='linear'):
        super(self.__class__, self).__init__()

        assert kernel in ["rbf", "linear"]
        self.x_train = []
        self.C = c
        self.weighted = weighted
        self.beta = []
        self.kernel = kernel

    def fit(self, x_train, y_train):
        """
        Calculate beta using kernel.
        :param x_train: features of train set
        :param y_train: labels of train set
        :return:
        """
        self.x_train = x_train
        class_num = max(y_train) + 1
        n = len(x_train)
        y_one_hot = np.eye(class_num)[y_train]

        if self.kernel == 'rbf':
            kernel_func = pairwise.rbf_kernel(x_train)
        else:  # kernel == linear
            kernel_func = pairwise.linear_kernel(x_train)

        if self.weighted:
            W = np.zeros((n, n))
            hist = np.zeros(class_num)
            for label in y_train:
                hist[label] += 1
            hist = 1 / hist
            for i in range(len(y_train)):
                W[i, i] = hist[y_train[i]]
            beta = np.matmul(np.linalg.inv(np.matmul(W, kernel_func) +
                                           np.identity(n) / self.C), np.matmul(W, y_one_hot))
        else:
            beta = np.matmul(np.linalg.inv(kernel_func + np.identity(n) / self.C), y_one_hot)
        self.beta = beta

    def predict(self, x_test):
        """
        Predict label probabilities of new data using calculated beta.
        :param x_test: features of new data
        :return: class probabilities of new data
        """
        if self.kernel == 'rbf':
            kernel_func = pairwise.rbf_kernel(x_test, self.x_train)
        else:  # kernel == linear
            kernel_func = pairwise.linear_kernel(x_test, self.x_train)
        pred = np.matmul(kernel_func, self.beta)
        return pred


if __name__ == "__main__":
    model = ELM(c=1, weighted=False, kernel="linear")
    with open("data/embeddings_pickle/bert_train_words_300pca_fv.pickle", "rb") as f:
        train_data = pickle.load(f)
    with open("data/embeddings_pickle/bert_devel_words_300pca_fv.pickle", "rb") as f:
        devel_data = pickle.load(f)

    df = pd.read_csv("data/labels_csv/train_labels.csv")
    labels = df["L1"].values
    label_encoder = preprocessing.LabelEncoder()
    train_labels = label_encoder.fit_transform(labels)

    model.fit(train_data, train_labels)
    prediction = model.predict(devel_data)
    print(prediction)
