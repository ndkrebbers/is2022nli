from typing import Tuple

# import src.elm_kernel as elm
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import pairwise
import statistics
from operator import itemgetter


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

    # def normalize_data(self, feature_level: str = None, value_level: str = None,
    #                    instance_level: str = None, a: float = 1):
    #     """
    #     Cascaded normalization of data.
    #     :param feature_level: normalization over columns
    #     :param value_level: all data cells
    #     :param instance_level: normalization over rows
    #     :param a: tuning parameter for power normalization
    #     :return:
    #     """
    #     if feature_level == "z":
    #         scaler = preprocessing.StandardScaler()
    #         scaler.fit(self.x_train)
    #         self.x_train = scaler.transform(self.x_train)
    #         self.x_test = scaler.transform(self.x_test)
    #
    #     if value_level == "power":
    #         if a != -1:
    #             self.x_train = self.power_norm(self.x_train, a)
    #             self.x_test = self.power_norm(self.x_test, a)
    #
    #     if instance_level == "l2":
    #         self.x_train = preprocessing.normalize(self.x_train, norm=instance_level)
    #         self.x_test = preprocessing.normalize(self.x_test, norm=instance_level)
    #
    # @staticmethod
    # def power_norm(z, a):
    #     """
    #     Power normalization of float.
    #     :param z: float
    #     :param a: optimization parameter
    #     :return: normalized float
    #     """
    #     return np.multiply(np.sign(z), np.power(np.abs(z), a))

def score_fusion(y_true: list, cp_a: np.array, cp_b: np.array):
    best_uar = 0
    best_gamma = 0
    for gamma in range(0, 105, 5):
        gamma = gamma / 100
        fused_confidences = gamma * cp_a + (1 - gamma) * cp_b
        y_pred = fused_confidences.argmax(axis=-1)
        uar = recall_score(y_true, y_pred, average="macro")
        if uar > best_uar:
            best_uar = uar
            best_gamma = gamma

    print(f"Best score of {round(best_uar, 4) * 100}%  UAR is found using gamma={best_gamma}.")

def test_score_fusion(cp_a: np.array, cp_b: np.array, gamma):
    fused_confidences = gamma * cp_a + (1 - gamma) * cp_b
    return fused_confidences.argmax(axis=-1)

def svm_scorefusion(y_true: list, cp_a: np.array, cp_b: np.array, y_true_train, cp_a_train, cp_b_train):
    # data = np.concatenate((cp_a_train, cp_b_train), axis=1)
    # clf = SVC()
    # clf.fit(data, y_true_train)
    #
    # test_data = np.concatenate((cp_a, cp_b), axis=1)
    # y_pred = clf.predict(test_data)
    # uar = recall_score(y_true, y_pred, average="macro")

    data = np.concatenate((cp_a, cp_b), axis=1)
    clf = SVC()
    scores = cross_val_score(clf, data, y_true, cv=5)
    print(statistics.mean(scores))


class PFI:
    def __init__(self, ELM, x_dev, y_dev, old_score, gmm_components, pca_comp):
        self.model = ELM
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.gmm_comp = gmm_components
        self.pca_comp = pca_comp
        self.old_score = old_score
        self.order = []

    def feature_performance(self):
        feature_importances = []
        for comp in range(self.gmm_comp):
            mutated_set = self.shuffle_featureset(comp)
            score = self.calculate_score(mutated_set)
            print(score)
            feature_importances.append((comp, score))

        feature_dict = dict(feature_importances)
        features_ordered = [k for k, v in sorted(feature_dict.items(), key=lambda item: item[1])]
        self.order = features_ordered

    def update_data(self, data_loc):
        with open(data_loc, "rb") as f:
            data = pickle.load(f)
        organized_data = self.re_organize_data(data)
        with open(data_loc, "wb") as f:
            pickle.dump(organized_data, f)

    def re_organize_data(self, dataset):
        organized_data = []
        for comp in self.order:
            start_i = 2 * comp * self.pca_comp
            stop_i = start_i + 2 * self.pca_comp
            organized_data.append(dataset[:, start_i:stop_i])
        organized_data = np.concatenate(organized_data, axis=1)
        return organized_data

    def shuffle_featureset(self, gmm_comp):
        arr = self.x_dev.copy()
        start_i = 2 * self.pca_comp * gmm_comp
        stop_i = start_i + 2 * self.pca_comp
        np.random.shuffle(arr[:, start_i:stop_i])
        return arr

    def calculate_score(self, x_mutated):
        pred_probss = self.model.predict(x_mutated)
        predd = np.argmax(pred_probss, axis=1)
        new_score = recall_score(self.y_dev, predd, average="macro")
        score_dif = self.old_score - new_score
        return score_dif


class CascadedNormalizer:
    def __init__(self, x_train: np.array, x_test: np.array, feature_lvl: str = None, value_lvl: str = None, instance_lvl: str = None, power_gamma: float = 1):
        self.x_train = x_train
        self.x_test = x_test

        self.feature_lvl = feature_lvl
        self.value_lvl = value_lvl
        self.instance_lvl = instance_lvl

        self.power_gamma = power_gamma

    def normalize(self) -> Tuple[np.array, np.array]:
        self.__feature_norm()
        self.__value_norm()
        self.__instance_norm()
        return self.x_train, self.x_test

    def __feature_norm(self):
        if self.feature_lvl == "z":
            scaler = preprocessing.StandardScaler()
            scaler.fit(self.x_train)
            self.x_train = scaler.transform(self.x_train)
            self.x_test = scaler.transform(self.x_test)
        else:
            pass

    def __value_norm(self):
        if self.value_lvl == "power" and self.power_gamma != -1:
            self.x_train = np.multiply(np.sign(self.x_train), np.power(np.abs(self.x_train), self.power_gamma))
            self.x_test = np.multiply(np.sign(self.x_test), np.power(np.abs(self.x_test), self.power_gamma))
        else:
            pass

    def __instance_norm(self):
        if self.instance_lvl == "l2":
            self.x_train = preprocessing.normalize(self.x_train, norm=self.instance_lvl)
            self.x_test = preprocessing.normalize(self.x_test, norm=self.instance_lvl)
        else:
            pass


class DataLoader:
    def __init__(self, train_set: str, test_set: str, ling_model: str = "", linguistic_utt: str = "",
                 acoustic_utt: str = "", utt_functionals: str = ""):
        self.train_set = train_set
        self.test_set = test_set

        self.ling_model = ling_model
        self.linguistic_utt = linguistic_utt
        self.acoustic_utt = acoustic_utt
        self.utt_functionals = utt_functionals
        assert self.linguistic_utt or self.acoustic_utt or self.utt_functionals, "There is no data to extract."

        self.label_encoder = preprocessing.LabelEncoder()
        if self.utt_functionals:
            self.selected_cols = self.__select_5300()

    def construct_feature_set(self):
        if self.train_set == "train_devel":
            x_train = self.__get_features("train")
            x_devel = self.__get_features("devel")
            x_train = np.concatenate((x_train, x_devel), axis=0)

            y_train = self.__read_labels("train")
            y_devel = self.__read_labels("devel")
            y_train = np.concatenate((y_train, y_devel), axis=0)
        else:
            x_train = self.__get_features(self.train_set)
            y_train = self.__read_labels(self.train_set)

        x_test = self.__get_features(self.test_set)
        y_test = self.__read_labels(self.test_set)
        return x_train, x_test, y_train, y_test

    def __get_features(self, t_d_t: str) -> np.array:
        nr_of_rows = self.__determine_size(t_d_t)
        features = np.empty((nr_of_rows, 0))

        if self.linguistic_utt:
            ling_f = self.__read_fv_features(self.ling_model, t_d_t, self.linguistic_utt)
            features = np.concatenate((features, ling_f), axis=1)
        if self.acoustic_utt:
            acou_f = self.__read_fv_features("acoustic", t_d_t, self.acoustic_utt)
            features = np.concatenate((features, acou_f), axis=1)
        if self.utt_functionals:
            func_f = self.__read_utt_functionals(t_d_t)
            features = np.concatenate((features, func_f), axis=1)

        return features

    @staticmethod
    def __read_fv_features(model: str, t_d_t: str, specs: str) -> np.array:
        file_loc = f"data/embeddings_pickle/{model}_{t_d_t}_{specs}.pickle"
        with open(file_loc, "rb") as f:
            features = pickle.load(f)
        return features

    def __read_utt_functionals(self, t_d_t: str) -> np.array:
        if self.utt_functionals == "compare":
            file_loc = f"data/features_csv/{self.utt_functionals}_{t_d_t}.csv"
            df = pd.read_csv(file_loc)
            features = df.values
            features = features[:, self.selected_cols]
        else:
            df = pd.read_csv("data/features_csv/mfcc_rastaplpc_10functionals.csv", header=None)  # Slightly inefficient
            df = df.drop(df.columns[0], axis=1)
            if t_d_t == "train":
                features = df.tail(3300)
            elif t_d_t == "devel":
                features = df.head(965)
            else:  # t_d_t == "test"
                features = df.iloc[965:965 + 867, ]
        return features

    def __read_labels(self, t_d_t: str) -> np.array:
        file_loc = f"data/labels_csv/{t_d_t}_labels.csv"
        if t_d_t != "test":
            df = pd.read_csv(file_loc)
        else:
            df = pd.read_csv(file_loc, delimiter=";")
        labels = df["L1"].values

        if t_d_t == "train":
            labels = self.label_encoder.fit_transform(labels)
        else:
            labels = self.label_encoder.transform(labels)
        return labels

    @staticmethod
    def __determine_size(t_d_t):
        if t_d_t == "train":
            size = 3300
        elif t_d_t == "devel":
            size = 965
        elif t_d_t == "train_devel":
            size = 3300 + 965
        elif t_d_t == "test":
            size = 867
        else:
            size = 0
        assert size != 0
        return size

    @staticmethod
    def __select_5300():
        cols = pd.read_csv("data/features_csv/feature_ranking_5300.csv", header=None).values
        cols = cols.transpose()[0]
        return cols


if __name__ == "__main__":
    dl = DataLoader(train_set="train", test_set="devel", ling_model="bert", linguistic_utt="words_400pca_64gmm_fv",
                    acoustic_utt="", utt_functionals="")
    a, b, c, d = dl.construct_feature_set()
    a, b = CascadedNormalizer(a, b, "z", "power", "l2", 0.5).normalize()
    model = ELM(c=4)
    model.fit(a, c)

    #TODO: include permutation feature importance in pipeline
    pred_probs = model.predict(b)
    pred = np.argmax(pred_probs, axis=1)
    pred_score = recall_score(d, pred, average="macro")
    print(pred_score)

    pfi = PFI(model, b, d, pred_score, 64, 400)
    pfi.feature_performance()
    pfi.update_data("data/embeddings_pickle/bert_train_words_400pca_64gmm_fv.pickle")
    pfi.update_data("data/embeddings_pickle/bert_devel_words_400pca_64gmm_fv.pickle")
    pfi.update_data("data/embeddings_pickle/bert_test_words_400pca_64gmm_fv.pickle")

    pass
    # ml = ModelLearning(bert_model="sbert", test="devel", data_extension="sentences", c=0.8, acoustic_functionals=True)
    # ml.normalize_data(feature_level="z", value_level="power", instance_level="l2", a=0.6)
    # pred = ml.fit_predict()

    # print(recall_score(ml.y_test, pred, average="macro"))
    # score_fusion([0, 0, 0, 0, 0, 0, 0, 0], np.random.rand(8, 4), np.random.rand(8, 4))


